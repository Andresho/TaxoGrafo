import logging
import json
import uuid
import scripts.pipeline_tasks as pt
from scripts.llm_client import get_llm_strategy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from scripts.llm_client import get_llm_strategy
from pathlib import Path
from db import get_session
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
        output_filename: str
    ):
        self.batch_id = batch_id
        self.output_file_id = output_file_id
        self.error_file_id = error_file_id
        self.stage_output_dir = stage_output_dir
        self.output_filename = output_filename

    def process(self) -> bool:
        """Processa resultados do batch usando LLM strategy e persiste no banco de dados."""
        llm = get_llm_strategy()
        logging.info(f"Processando resultados do Batch {self.batch_id} (Output File: {self.output_file_id})...")
        processed_data: List[Dict[str, Any]] = []
        errors_in_batch = 0
        all_ok = True
        try:
            if self.error_file_id:
                self._log_error_file()
            # Baixa conteúdo via LLM strategy
            content_bytes = llm.read_file(self.output_file_id)
            result_content = content_bytes.decode('utf-8')
            logging.info(f"Arquivo de resultado {self.output_file_id} baixado.")
            for line in result_content.strip().split('\n'):
                errors_in_batch += self._process_line(line, processed_data)
            logging.info(f"Processamento arquivo concluído. {len(processed_data)} registros. {errors_in_batch} erros.")
            if processed_data:
                # Persiste dados processados no banco
                try:
                    self._save_to_db(processed_data)
                except Exception:
                    logging.exception("Falha ao persistir batch processado no banco de dados.")
                    return False
                all_ok = True
            elif errors_in_batch > 0:
                logging.error("Nenhum dado processado com sucesso do batch.")
                all_ok = False
            else:
                logging.warning("Nenhum dado encontrado no arquivo de resultados do batch.")
                all_ok = True
        except Exception:
            logging.exception(f"Falha ao baixar/processar arquivo {self.output_file_id} do batch {self.batch_id}.")
            all_ok = False
        return all_ok

    def _log_error_file(self):
        """Loga o conteúdo de erros do batch usando LLM strategy."""
        llm = get_llm_strategy()
        try:
            error_bytes = llm.read_file(self.error_file_id)
            error_content = error_bytes.decode('utf-8')
            logging.warning(f"Erros individuais no batch {self.batch_id}:\n{error_content[:1000]}...")
        except Exception as ef:
            logging.error(f"Não leu arq erro {self.error_file_id}: {ef}")

    def _process_line(self, line: str, processed_data: List[Dict[str, Any]]) -> int:
        errors = 0
        try:
            line_data = json.loads(line)
            custom_id = line_data.get("custom_id", "unknown_custom_id")
            origin_id = self._extract_origin_id(custom_id)
            error = line_data.get("error")
            if error:
                errors += 1
                logging.error(f"Erro batch {custom_id}: {error.get('message')}")
                return errors
            response = line_data.get("response")
            if not response or response.get("status_code") != 200:
                errors += 1
                logging.warning(f"Request {custom_id} status não OK: {response}")
                return errors
            body = response.get("body", {})
            choice = (body.get("choices") or [{}])[0]
            message_content = choice.get("message", {}).get("content")
            if not message_content:
                errors += 1
                logging.warning(f"Resposta OK sem conteúdo {custom_id}")
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
                logging.error(f"Erro JSON decode interno {custom_id}: {e}")
                return errors
            items, errs = self.parse_inner(inner_data, origin_id)
            processed_data.extend(items)
            errors += errs
        except Exception as e:
            errors += 1
            logging.error(f"Erro processando linha: {e}. Linha: {line[:100]}...")
        return errors

    def _extract_origin_id(self, custom_id: str) -> str:
        if custom_id.startswith("gen_req_"):
            parts = custom_id.split("_")
            return "_".join(parts[2:-1])
        return custom_id

    @abstractmethod
    def parse_inner(self, inner_data: Dict[str, Any], origin_id: str) -> Tuple[List[Dict[str, Any]], int]:
        """Extrai itens processados de inner_data e retorna (itens, numero_de_erros)."""
        ...
    def _save_to_db(self, processed_data: List[Dict[str, Any]]) -> None:
        """Hook para subclasses persistirem processed_data no banco. Deve ser sobrescrito."""
        raise NotImplementedError

class GenerationBatchProcessor(BaseBatchProcessor):
    def parse_inner(self, inner_data: Dict[str, Any], origin_id: str) -> Tuple[List[Dict[str, Any]], int]:
        items = []
        errors = 0
        units = inner_data.get("generated_units", [])
        if not isinstance(units, list) or len(units) != 6:
            errors += 1
            logging.warning(f"JSON interno {origin_id} != 6 UCs")
            return items, errors
        for unit in units:
            if isinstance(unit, dict) and "bloom_level" in unit and "uc_text" in unit:
                record = unit.copy()
                record["uc_id"] = str(uuid.uuid4())
                record["origin_id"] = origin_id
                items.append(record)
            else:
                errors += 1
                logging.warning(f"Formato UC inválido para {origin_id}: {unit}")
        return items, errors
    def _save_to_db(self, processed_data: List[Dict[str, Any]]) -> None:
        """Persiste generated UCs raw no banco."""
        # Extrai run_id da estrutura de diretórios: AIRFLOW_DATA_DIR/run_id/pipeline_workdir/2_generated_ucs
        run_id = Path(self.stage_output_dir).parent.parent.name
        with get_session() as db:
            add_generated_ucs_raw(db, run_id, processed_data)

class DifficultyBatchProcessor(BaseBatchProcessor):
    def parse_inner(self, inner_data: Dict[str, Any], origin_id: str) -> Tuple[List[Dict[str, Any]], int]:
        items = []
        errors = 0
        assessments = inner_data.get("difficulty_assessments", [])
        if not isinstance(assessments, list):
            errors += 1
            logging.warning(f"JSON interno {origin_id} sem lista 'difficulty_assessments'")
            return items, errors
        for assessment in assessments:
            if isinstance(assessment, dict) and "uc_id" in assessment and "difficulty_score" in assessment:
                assessment["knowledge_unit_id"] = assessment.pop("uc_id")
                items.append(assessment)
            else:
                errors += 1
                logging.warning(f"Formato assessment inválido para {origin_id}: {assessment}")
        return items, errors
    def _save_to_db(self, processed_data: List[Dict[str, Any]]) -> None:
        """Persiste raw UC evaluations no banco."""
        # Extrai run_id da estrutura de diretórios: AIRFLOW_DATA_DIR/run_id/pipeline_workdir/4_difficulty_evals
        run_id = Path(self.stage_output_dir).parent.parent.name
        with get_session() as db:
            add_knowledge_unit_evaluations_batch(db, run_id, processed_data)

def process_batch_results(
    batch_id: str,
    output_file_id: str,
    error_file_id: str,
    stage_output_dir,
    output_filename: str
) -> bool:
    """Delegates batch processing to the appropriate processor."""
    # Seleciona o processor conforme o tipo de saída
    if output_filename == pt.GENERATED_UCS_RAW:
        processor = GenerationBatchProcessor(
            batch_id, output_file_id, error_file_id, stage_output_dir, output_filename
        )
    elif output_filename == pt.UC_EVALUATIONS_RAW:
        processor = DifficultyBatchProcessor(
            batch_id, output_file_id, error_file_id, stage_output_dir, output_filename
        )
    else:
        raise ValueError(f"process_batch_results: output_filename inválido '{output_filename}'")
    return processor.process()