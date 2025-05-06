import logging
import json
import uuid
import scripts.pipeline_tasks as pt
from scripts.llm_client import get_llm_strategy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from sqlalchemy.orm import Session
from crud.generated_ucs_raw import add_generated_ucs_raw
from crud.knowledge_unit_evaluations_batch import add_knowledge_unit_evaluations_batch

def check_batch_status(batch_id: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Consulta status do batch via LLM strategy e retorna (status, output_file_id, error_file_id)."""
    llm = get_llm_strategy()
    return llm.get_batch_status(batch_id)

# --- Template Method para processar batches ---
class BaseBatchProcessor(ABC):
    """Template Method para processar batches OpenAI."""
    def __init__(
        self,
        batch_id: str,
        output_file_id: str,
        error_file_id: str,
        stage_output_dir,
        output_filename: str,
        run_id: str
    ):
        self.batch_id = batch_id
        self.output_file_id = output_file_id
        self.error_file_id = error_file_id
        self.stage_output_dir = stage_output_dir
        self.output_filename = output_filename
        self.run_id = run_id

    def process(self, db: Session) -> bool:
        """
        Processa resultados do batch usando LLM strategy e persiste no banco
        usando a SESSÃO fornecida. Retorna True se bem sucedido.
        """
        llm = get_llm_strategy()
        logging.info(f"Processando resultados do Batch {self.batch_id} (Output File: {self.output_file_id}) para run_id {self.run_id}...")
        processed_data: List[Dict[str, Any]] = []
        errors_in_batch = 0
        all_ok = True # Indica se o processamento do ARQUIVO foi ok
        save_needed = False # Indica se há dados para salvar no DB

        try:
            if self.error_file_id:
                self._log_error_file()

            # Baixa conteúdo via LLM strategy
            content_bytes = llm.read_file(self.output_file_id)
            result_content = content_bytes.decode('utf-8')
            logging.info(f"Arquivo de resultado {self.output_file_id} baixado.")

            for line in result_content.strip().split('\n'):
                errors_in_batch += self._process_line(line, processed_data)

            logging.info(f"Processamento arquivo concluído. {len(processed_data)} registros processados. {errors_in_batch} erros em linhas.")

            if processed_data:
                save_needed = True # Há dados para salvar no banco
            elif errors_in_batch > 0:
                logging.error("Nenhum dado processado com sucesso do batch.")
                all_ok = False
            else:
                logging.warning("Nenhum dado encontrado no arquivo de resultados do batch.")
                all_ok = True # Arquivo vazio não é erro, mas nada a salvar

            # chamar o método de salvamento passando a sessão db
            if all_ok and save_needed:
                try:
                    logging.info(f"Adicionando {len(processed_data)} registros ao banco (pendente na sessão)...")
                    # Passa a sessão db recebida
                    self._save_to_db(db, processed_data)
                except Exception as db_err:
                    # Se o salvamento FALHAR (ex: erro no add_records),
                    # logamos e retornamos False para que o chamador faça rollback.
                    logging.exception("Falha ao adicionar dados do batch à sessão do banco de dados.")
                    return False

            # Se chegamos aqui, ou tudo correu bem, ou o arquivo estava vazio/sem dados válidos.
            # Retorna True se o processamento do arquivo não teve erros fatais.
            return all_ok

        except Exception as file_err:
            # Falha ao baixar ou erro inesperado no processamento do arquivo
            logging.exception(f"Falha crítica ao baixar/processar arquivo {self.output_file_id} do batch {self.batch_id}.")
            return False


    def _log_error_file(self):
        """Loga o conteúdo de erros do batch usando LLM strategy."""
        llm = get_llm_strategy()
        try:
            error_bytes = llm.read_file(self.error_file_id)
            error_content = error_bytes.decode('utf-8')
            logging.warning(f"Erros individuais reportados no arquivo de erro {self.error_file_id} do batch {self.batch_id}:\n{error_content[:1000]}...")
        except Exception as ef:
            logging.error(f"Não foi possível ler o arquivo de erro {self.error_file_id}: {ef}")

    def _process_line(self, line: str, processed_data: List[Dict[str, Any]]) -> int:

        errors = 0
        try:
            line_data = json.loads(line)
            custom_id = line_data.get("custom_id", "unknown_custom_id")

            origin_id = self._extract_origin_id(custom_id)
            error = line_data.get("error")
            if error:
                errors += 1
                logging.error(f"Erro na linha do batch (custom_id: {custom_id}, run_id: {self.run_id}): {error.get('message')}")
                return errors
            response = line_data.get("response")
            if not response or response.get("status_code") != 200:
                errors += 1
                logging.warning(f"Request não OK na linha do batch (custom_id: {custom_id}, run_id: {self.run_id}): {response}")
                return errors
            body = response.get("body", {})
            choice = (body.get("choices") or [{}])[0]
            message_content = choice.get("message", {}).get("content")
            if not message_content:
                errors += 1
                logging.warning(f"Resposta OK sem conteúdo na linha do batch (custom_id: {custom_id}, run_id: {self.run_id})")
                return errors
            content_cleaned = message_content.strip()
            if content_cleaned.startswith("```json"):
                content_cleaned = content_cleaned[7:-3].strip()
            elif content_cleaned.startswith("```"):
                content_cleaned = content_cleaned[3:-3].strip()
            try:
                inner_data = json.loads(content_cleaned)
            except json.JSONDecodeError as e:
                errors += 1
                logging.error(f"Erro JSON decode interno na linha do batch (custom_id: {custom_id}, run_id: {self.run_id}): {e}")
                return errors
            # Passar origin_id e run_id se necessário para parse_inner
            items, errs = self.parse_inner(inner_data, origin_id, self.run_id)
            processed_data.extend(items)
            errors += errs
        except Exception as e:
            errors += 1
            logging.error(f"Erro processando linha do batch (run_id: {self.run_id}): {e}. Linha: {line[:100]}...")
        return errors


    def _extract_origin_id(self, custom_id: str) -> str:
        if custom_id.startswith("gen_req_"):
            parts = custom_id.split("_")
            return "_".join(parts[2:-1])
        return custom_id # Retorna o custom_id se não for padrão gen_req

    @abstractmethod
    def parse_inner(self, inner_data: Dict[str, Any], origin_id: str, run_id: str) -> Tuple[List[Dict[str, Any]], int]:
        """Extrai itens processados de inner_data e retorna (itens, numero_de_erros)."""
        pass

    @abstractmethod
    def _save_to_db(self, db: Session, processed_data: List[Dict[str, Any]]) -> None:
        """
        Hook para subclasses persistirem processed_data no banco usando a SESSÃO fornecida.
        Esta função NÃO deve fazer commit.
        Levanta exceção em caso de falha na adição à sessão.
        """
        raise NotImplementedError

class GenerationBatchProcessor(BaseBatchProcessor):
    def parse_inner(self, inner_data: Dict[str, Any], origin_id: str, run_id: str) -> Tuple[List[Dict[str, Any]], int]:
        items = []
        errors = 0
        units = inner_data.get("generated_units", [])
        if not isinstance(units, list) or len(units) != 6:
            errors += 1
            logging.warning(f"JSON interno {origin_id} (run: {run_id}) != 6 UCs: {units}")
            return items, errors
        for unit in units:
            if isinstance(unit, dict) and "bloom_level" in unit and "uc_text" in unit:
                record = unit.copy()
                record["uc_id"] = str(uuid.uuid4()) # Gerar ID único da UC
                record["origin_id"] = origin_id
                # run_id será adicionado por add_records
                items.append(record)
            else:
                errors += 1
                logging.warning(f"Formato UC inválido para {origin_id} (run: {run_id}): {unit}")
        return items, errors

    def _save_to_db(self, db: Session, processed_data: List[Dict[str, Any]]) -> None:
        """Persiste generated UCs raw no banco usando a sessão db fornecida."""
        try:
            add_generated_ucs_raw(db, self.run_id, processed_data)
        except Exception as e:
            logging.error(f"Erro em add_generated_ucs_raw para run_id {self.run_id}: {e}")
            raise


class DifficultyBatchProcessor(BaseBatchProcessor):
    def parse_inner(self, inner_data: Dict[str, Any], origin_id: str, run_id: str) -> Tuple[List[Dict[str, Any]], int]:
        items = []
        errors = 0
        assessments = inner_data.get("difficulty_assessments", [])
        if not isinstance(assessments, list):
            errors += 1
            logging.warning(f"JSON interno (run: {run_id}) sem lista 'difficulty_assessments': {inner_data}")
            return items, errors
        for assessment in assessments:
            uc_id = assessment.get("uc_id")
            score = assessment.get("difficulty_score")
            justification = assessment.get("justification", "")
            if isinstance(uc_id, str) and isinstance(score, int) and isinstance(justification, str):
                record = {
                    "knowledge_unit_id": uc_id,
                    "difficulty_score": score,
                    "justification": justification
                }
                items.append(record)
            else:
                errors += 1
                logging.warning(f"Formato assessment inválido (run: {run_id}): {assessment}")
        return items, errors

    def _save_to_db(self, db: Session, processed_data: List[Dict[str, Any]]) -> None:
        """Persiste raw UC evaluations no banco usando a sessão db fornecida."""
        try:
            add_knowledge_unit_evaluations_batch(db, self.run_id, processed_data)
        except Exception as e:
            logging.error(f"Erro em add_knowledge_unit_evaluations_batch para run_id {self.run_id}: {e}")
            raise

def process_batch_results(
    batch_id: str,
    output_file_id: str,
    error_file_id: str,
    stage_output_dir,
    output_filename: str,
    run_id: str,
    db: Session
) -> bool:
    """
    Delegates batch processing to the appropriate processor, using the provided db Session.
    Returns True if processing and adding to the session were successful, False otherwise.
    Does NOT commit the transaction.
    """
    processor: Optional[BaseBatchProcessor] = None
    # Seleciona o processor conforme o tipo de saída
    if output_filename == pt.GENERATED_UCS_RAW:
        processor = GenerationBatchProcessor(
            batch_id, output_file_id, error_file_id, stage_output_dir, output_filename, run_id
        )
    elif output_filename == pt.UC_EVALUATIONS_RAW:
        processor = DifficultyBatchProcessor(
            batch_id, output_file_id, error_file_id, stage_output_dir, output_filename, run_id
        )
    else:
        logging.error(f"process_batch_results: output_filename inválido '{output_filename}' para run_id {run_id}")
        return False

    success = processor.process(db)

    return success