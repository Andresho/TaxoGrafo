import logging
import json
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session

from app.scripts.llm_core.models import GenericLLMResponse
from app.scripts.llm_client import get_llm_strategy, LLMClient
from app.scripts.constants import GENERATED_UCS_RAW, UC_EVALUATIONS_RAW

from app.crud.generated_ucs_raw import add_generated_ucs_raw
from app.crud.knowledge_unit_evaluations_batch import add_knowledge_unit_evaluations_batch


def check_batch_status(batch_id: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Queries the status of an LLM batch job.
    Returns a tuple: (status, output_file_id, error_file_id).
    """
    llm: LLMClient = get_llm_strategy() # Agora llm é do tipo LLMClient
    return llm.get_batch_status(batch_id)


class BaseBatchProcessor(ABC):
    def __init__(
            self,
            batch_id: str, # LLM provider's batch ID
            output_file_id: str, # LLM provider's output file ID
            error_file_id: Optional[str], # LLM provider's error file ID
            stage_output_dir: Any, # Path object, para salvar debug local se necessário
            output_filename_key: str, # Chave para identificar o tipo de output (ex: GENERATED_UCS_RAW)
            run_id: str
    ):
        self.batch_id = batch_id
        self.output_file_id = output_file_id
        self.error_file_id = error_file_id
        self.stage_output_dir = stage_output_dir
        self.output_filename_key = output_filename_key
        self.run_id = run_id
        self.llm: LLMClient = get_llm_strategy()

    def process(self, db: Session) -> bool:
        logging.info(
            f"Processing results for Batch ID: {self.batch_id}, Run ID: {self.run_id}, Output File: {self.output_file_id}, Type: {self.output_filename_key}")
        processed_data_for_db: List[Dict[str, Any]] = []
        total_line_errors = 0
        lines_resulting_in_data = 0
        overall_success = True

        try:
            if self.error_file_id:
                self._log_error_file_content()

            logging.debug(f"Attempting to read file: {self.output_file_id} from LLM provider.")
            content_bytes = self.llm.read_file(self.output_file_id)
            result_content_str = content_bytes.decode('utf-8')
            logging.info(f"Successfully downloaded result file {self.output_file_id} for batch {self.batch_id}.")

            lines = result_content_str.strip().split('\n')
            if not lines or (len(lines) == 1 and not lines[0].strip()):
                logging.warning(f"Result file for batch {self.batch_id} is empty or contains only whitespace.")
                return True

            for line_number, line_content_str in enumerate(lines, 1):
                if not line_content_str.strip():
                    logging.debug(f"Skipping blank line {line_number} in batch {self.batch_id}.")
                    continue

                items_from_line, errors_in_line = self._process_single_line_wrapper(line_content_str, line_number)
                if items_from_line:
                    processed_data_for_db.extend(items_from_line)
                    lines_resulting_in_data += 1
                if errors_in_line > 0:
                    total_line_errors += errors_in_line

            logging.info(
                f"Batch file processing complete for batch {self.batch_id} (Type: {self.output_filename_key}). "
                f"Extracted {len(processed_data_for_db)} items for DB from {lines_resulting_in_data} lines. "
                f"Encountered {total_line_errors} errors in individual lines."
            )

            if processed_data_for_db:
                self._save_to_db(db, processed_data_for_db)
            elif total_line_errors > 0 and lines_resulting_in_data == 0:
                logging.error(
                    f"No data successfully processed from batch {self.batch_id} (Type: {self.output_filename_key}) due to errors in all relevant lines.")
                overall_success = False

        except Exception:
            logging.exception(
                f"Critical error during file download or main processing for batch {self.batch_id}, output file {self.output_file_id}.")
            overall_success = False
        return overall_success

    def _log_error_file_content(self):
        if not self.error_file_id: return
        logging.info(f"Attempting to read error file {self.error_file_id} for batch {self.batch_id}.")
        try:
            error_bytes = self.llm.read_file(self.error_file_id)
            error_content = error_bytes.decode('utf-8')
            logging.warning(
                f"Content from error file {self.error_file_id} for batch {self.batch_id} (first 1000 chars):\n"
                f"{error_content[:1000]}"
            )
        except Exception:
            logging.exception(f"Failed to read or decode error file {self.error_file_id} for batch {self.batch_id}.")

    def _process_single_line_wrapper(self, line_content_str: str, line_number: int) -> Tuple[List[Dict[str, Any]], int]:
        try:
            return self._process_line_content(line_content_str, line_number)
        except Exception:
            logging.exception(
                f"Fatal error processing line {line_number} of batch {self.batch_id} (run_id: {self.run_id}, type: {self.output_filename_key}). "
                f"Line (first 200 chars): '{line_content_str[:200]}'"
            )
            return [], 1

    def _process_line_content(self, line_content_str: str, line_number: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        Usa o parser do LLMClient para obter uma GenericLLMResponse,
        e então delega para _parse_llm_response_content_wrapper com o conteúdo genérico.
        """
        generic_response: GenericLLMResponse = self.llm.parse_llm_batch_line(line_content_str)

        request_metadata = generic_response['request_metadata']
        original_provider_custom_id = generic_response.get('raw_response_data', {}).get('custom_id', 'unknown_provider_custom_id')

        if generic_response['error_message']:
            logging.error(
                f"LLM API error or provider parsing issue on line {line_number} "
                f"(Provider custom_id: {original_provider_custom_id}, "
                f"parsed metadata: {request_metadata}, run: {self.run_id}): {generic_response['error_message']}"
            )
            return [], 1

        llm_response_content_str = generic_response['response_content']

        context_id_for_inner_parse = self._extract_context_id_from_metadata(request_metadata)
        if context_id_for_inner_parse is None:
            logging.error(f"Não foi possível extrair ID de contexto do metadata: {request_metadata} para a linha {line_number}, batch {self.batch_id}.")
            return [], 1


        return self._parse_llm_response_content_wrapper(
            llm_response_content_str,
            request_metadata,
            context_id_for_inner_parse,
            line_number
        )

    def _extract_context_id_from_metadata(self, request_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extrai o ID de contexto relevante do request_metadata.
        Este ID é usado por `parse_inner` para associar os itens processados.
        """

        req_type = request_metadata.get("type")
        if req_type == "uc_generation":
            return str(request_metadata.get("origin_id", "unknown_origin_id_in_meta"))
        elif req_type == "difficulty_assessment":
            return str(request_metadata.get("comparison_group_id", "unknown_comp_group_id_in_meta"))

        logging.warning(f"Tipo de metadata desconhecido '{req_type}' ou ID de contexto ausente: {request_metadata}")
        return str(request_metadata.get("id_context", None))


    def _parse_llm_response_content_wrapper(self, llm_message_content_str: Optional[str],
                                            request_metadata_from_line: Dict[str, Any],
                                            id_for_item_context: str,
                                            line_number: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        Wrapper para _parse_llm_response_content que lida com llm_message_content_str sendo None.
        """
        if llm_message_content_str is None:
            logging.warning(
                f"Nenhum 'content' na mensagem LLM (metadata: {request_metadata_from_line}, run: {self.run_id}, line: {line_number}). "
                f"Chamando parse_inner com inner_data=None."
            )
            items_from_inner, inner_errors = self.parse_inner(None, id_for_item_context, self.run_id)
            if not items_from_inner and inner_errors == 0:
                 logging.warning(f"parse_inner não retornou itens nem erros para conteúdo LLM None. Linha {line_number}, metadata {request_metadata_from_line}")
            return items_from_inner, inner_errors


        parsed_items: List[Dict[str, Any]] = []
        parsing_errors = 0
        content_cleaned = llm_message_content_str.strip()

        if content_cleaned.startswith("```json"):
            content_cleaned = content_cleaned[len("```json"):-len("```")].strip()
        elif content_cleaned.startswith("```"):
            content_cleaned = content_cleaned[len("```"):-len("```")].strip()

        try:
            inner_data = json.loads(content_cleaned)
        except json.JSONDecodeError:
            logging.error(
                f"Falha ao decode JSON interno da LLM response (metadata: {request_metadata_from_line}, run: {self.run_id}, line: {line_number}). "
                f"Cleaned content (first 200 chars): '{content_cleaned[:200]}'"
            )
            return parsed_items, 1

        try:
            items_from_inner, inner_errors = self.parse_inner(inner_data, id_for_item_context, self.run_id)
            if items_from_inner:
                parsed_items.extend(items_from_inner)
            parsing_errors += inner_errors
        except Exception:
            logging.exception(
                f"Erro inesperado em 'parse_inner' (metadata: {request_metadata_from_line}, run: {self.run_id}, line: {line_number}).")
            parsing_errors += 1

        return parsed_items, parsing_errors


    @abstractmethod
    def parse_inner(self, inner_data: Optional[Dict[str, Any]],
                    id_for_item_context: str, run_id: str) -> Tuple[List[Dict[str, Any]], int]:
        pass

    @abstractmethod
    def _save_to_db(self, db: Session, processed_data: List[Dict[str, Any]]) -> None:
        pass


class GenerationBatchProcessor(BaseBatchProcessor):
    def parse_inner(self, inner_data: Optional[Dict[str, Any]], origin_id_for_uc: str, run_id: str) -> Tuple[List[Dict[str, Any]], int]:
        items = []
        errors = 0
        if inner_data is None:
            logging.warning(f"parse_inner (Generation) recebeu inner_data=None para origin_id '{origin_id_for_uc}' (run: {run_id}). Nenhum UC gerado.")
            return items, 1

        generated_units = inner_data.get("generated_units")
        if not isinstance(generated_units, list):
            logging.warning(
                f"Expected 'generated_units' to be a list in inner_data for origin_id '{origin_id_for_uc}' (run: {run_id}). "
                f"Found: {type(generated_units)}. Data (first 200 chars): {str(inner_data)[:200]}"
            )
            return items, 1

        for unit_idx, unit_data in enumerate(generated_units):
            if isinstance(unit_data, dict) and "bloom_level" in unit_data and "uc_text" in unit_data:
                record = {
                    "uc_id": str(uuid.uuid4()),
                    "origin_id": origin_id_for_uc,
                    "bloom_level": unit_data["bloom_level"],
                    "uc_text": unit_data["uc_text"],
                    "pipeline_run_id": run_id
                }
                items.append(record)
            else:
                errors += 1
                logging.warning(
                    f"Invalid UC format in 'generated_units' at index {unit_idx} for "
                    f"origin_id '{origin_id_for_uc}' (run: {run_id}). Unit data (first 100): {str(unit_data)[:100]}"
                )
        return items, errors

    def _save_to_db(self, db: Session, processed_data: List[Dict[str, Any]]) -> None:
        logging.debug(f"Saving {len(processed_data)} generated UCs for run_id {self.run_id} (Type: {self.output_filename_key}).")
        add_generated_ucs_raw(db, self.run_id, processed_data)


class DifficultyBatchProcessor(BaseBatchProcessor):
    def parse_inner(self, inner_data: Optional[Dict[str, Any]], comparison_group_id: str, run_id: str) -> Tuple[List[Dict[str, Any]], int]:
        items = []
        errors = 0
        if inner_data is None:
            logging.warning(f"parse_inner (Difficulty) recebeu inner_data=None para comparison_group_id '{comparison_group_id}' (run: {run_id}). Nenhuma avaliação processada.")
            return items, 1

        difficulty_assessments = inner_data.get("difficulty_assessments")
        if not isinstance(difficulty_assessments, list):
            logging.warning(
                f"Expected 'difficulty_assessments' to be a list in inner_data "
                f"(for comparison_group_id: {comparison_group_id}, run: {run_id}). "
                f"Found: {type(difficulty_assessments)}. Data (first 200): {str(inner_data)[:200]}"
            )
            return items, 1

        for assessment_idx, assessment_data in enumerate(difficulty_assessments):
            uc_id = assessment_data.get("uc_id")
            difficulty_score = assessment_data.get("difficulty_score")
            justification = assessment_data.get("justification", "")

            if not isinstance(uc_id, str) or not uc_id:
                errors += 1
                logging.warning(f"Missing or invalid 'uc_id' at index {assessment_idx} for comp_group '{comparison_group_id}'. Data: {assessment_data}")
                continue
            if not isinstance(difficulty_score, int) or not (0 <= difficulty_score <= 100):
                errors += 1
                logging.warning(f"Invalid 'difficulty_score' for uc_id '{uc_id}' (comp_group: {comparison_group_id}). Score: {difficulty_score}. Data: {assessment_data}")
                continue # Pular este registro problemático
            if not isinstance(justification, str):
                justification = str(justification) # Tentar converter para string
                logging.debug(f"Justification for uc_id '{uc_id}' converted to string.")


            record = {
                "comparison_group_id": comparison_group_id,
                "knowledge_unit_id": uc_id,
                "difficulty_score": difficulty_score,
                "justification": justification,
                "pipeline_run_id": run_id
            }
            items.append(record)
        return items, errors

    def _save_to_db(self, db: Session, processed_data: List[Dict[str, Any]]) -> None:
        logging.debug(f"Saving {len(processed_data)} difficulty assessments for run_id {self.run_id} (Type: {self.output_filename_key}).")
        add_knowledge_unit_evaluations_batch(db, self.run_id, processed_data)

def process_batch_results(
        batch_id: str,
        output_file_id: str,
        error_file_id: Optional[str],
        stage_output_dir: Any,
        output_filename_key: str,
        run_id: str,
        db: Session
) -> bool:
    logging.debug(
        f"process_batch_results called for: batch_id='{batch_id}', output_filename_key='{output_filename_key}', run_id='{run_id}'"
    )
    processor: Optional[BaseBatchProcessor] = None

    if output_filename_key == GENERATED_UCS_RAW:
        processor = GenerationBatchProcessor(
            batch_id, output_file_id, error_file_id, stage_output_dir, output_filename_key, run_id
        )
    elif output_filename_key == UC_EVALUATIONS_RAW:
        processor = DifficultyBatchProcessor(
            batch_id, output_file_id, error_file_id, stage_output_dir, output_filename_key, run_id
        )
    else:
        logging.error(f"Unsupported 'output_filename_key': {output_filename_key} for batch processing (run_id: {run_id}).")
        return False

    return processor.process(db)