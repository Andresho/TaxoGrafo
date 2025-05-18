import logging
import json
import uuid
from app.scripts.llm_client import get_llm_strategy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from app.crud.generated_ucs_raw import add_generated_ucs_raw
from app.crud.knowledge_unit_evaluations_batch import add_knowledge_unit_evaluations_batch
from app.scripts.constants import GENERATED_UCS_RAW, UC_EVALUATIONS_RAW


def check_batch_status(batch_id: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Queries the status of an LLM batch job.
    Returns a tuple: (status, output_file_id, error_file_id).
    """
    llm = get_llm_strategy()
    return llm.get_batch_status(batch_id)


class BaseBatchProcessor(ABC):
    """
    Abstract Base Class for processing LLM batch job results.
    Utilizes a template method pattern for the main `process` logic.
    """

    def __init__(
            self,
            batch_id: str,
            output_file_id: str,
            error_file_id: Optional[str],
            stage_output_dir: Any,  # Type can be Path; currently not strictly used after refactor
            output_filename: str,
            run_id: str
    ):
        self.batch_id = batch_id
        self.output_file_id = output_file_id
        self.error_file_id = error_file_id
        self.stage_output_dir = stage_output_dir
        self.output_filename = output_filename
        self.run_id = run_id
        self.llm = get_llm_strategy()

    def process(self, db: Session) -> bool:
        """
        Main processing logic for batch results.
        Downloads the output file, processes each line, and prepares data for DB saving.
        Actual DB saving (add to session) is delegated to `_save_to_db`.
        The provided `db` Session is used for DB operations; this method does NOT commit.
        Returns True if file processing and DB preparation were successful, False otherwise.
        """
        logging.info(
            f"Processing results for Batch ID: {self.batch_id}, Run ID: {self.run_id}, Output File: {self.output_file_id}")
        processed_data: List[Dict[str, Any]] = []
        total_line_errors = 0
        lines_resulting_in_data = 0
        overall_success = True

        try:
            if self.error_file_id:
                self._log_error_file_content()

            logging.debug(f"Attempting to read file: {self.output_file_id}")
            content_bytes = self.llm.read_file(self.output_file_id)
            result_content = content_bytes.decode('utf-8')
            logging.info(f"Successfully downloaded result file {self.output_file_id} for batch {self.batch_id}.")

            lines = result_content.strip().split('\n')
            if not lines or (len(lines) == 1 and not lines[0].strip()):
                logging.warning(f"Result file for batch {self.batch_id} is empty or contains only whitespace.")
                return True

            for line_number, line_content in enumerate(lines, 1):
                if not line_content.strip():
                    logging.debug(f"Skipping blank line {line_number} in batch {self.batch_id}.")
                    continue

                items_from_line, errors_in_line = self._process_single_line_wrapper(line_content, line_number)
                if items_from_line:
                    processed_data.extend(items_from_line)
                    lines_resulting_in_data += 1
                if errors_in_line > 0:
                    total_line_errors += errors_in_line

            logging.info(
                f"Batch file processing complete for batch {self.batch_id}. "
                f"Extracted {len(processed_data)} items from {lines_resulting_in_data} lines. "
                f"Encountered {total_line_errors} errors in individual lines."
            )

            if processed_data:
                try:
                    logging.info(
                        f"Adding {len(processed_data)} processed items to the database session (run_id: {self.run_id})...")
                    self._save_to_db(db, processed_data)  # Uses the provided db session
                except Exception:  # Catch specific DB errors if possible
                    logging.exception(
                        f"Database error while trying to save processed data for batch {self.batch_id}, run_id {self.run_id}.")
                    overall_success = False  # DB save failure means overall failure
            elif total_line_errors > 0 and lines_resulting_in_data == 0:
                logging.error(
                    f"No data successfully processed from batch {self.batch_id} due to errors in all relevant lines.")
                overall_success = False  # All lines had errors, no data
            # If no processed_data and no total_line_errors, it means lines were valid but parse_inner found no items (can be valid).

        except Exception:
            logging.exception(
                f"Critical error during file download or main processing for batch {self.batch_id}, output file {self.output_file_id}.")
            overall_success = False

        return overall_success

    def _log_error_file_content(self):
        """Logs the content of the error file associated with the batch, if it exists."""
        if not self.error_file_id:
            return
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

    def _process_single_line_wrapper(self, line_content: str, line_number: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        Wrapper for `_process_line_content` to catch any unexpected fatal errors
        during the processing of a single line, allowing the batch processing to continue.
        Returns a tuple: (list_of_processed_items_from_line, number_of_errors_in_line).
        """
        try:
            return self._process_line_content(line_content, line_number)
        except Exception:
            logging.exception(
                f"Fatal error processing line {line_number} of batch {self.batch_id} (run_id: {self.run_id}). "
                f"Line (first 200 chars): '{line_content[:200]}'"
            )
            return [], 1 

    def _process_line_content(self, line_content: str, line_number: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        Parses the JSON structure of a single line from the batch output file,
        validates the response, and delegates to `_parse_llm_response_content`.
        Returns a tuple: (list_of_processed_items_from_line, number_of_errors_in_line).
        Can raise JSONDecodeError if the line itself is not valid JSON.
        """
        line_errors = 0
        items_from_line: List[Dict[str, Any]] = []

        line_data = json.loads(line_content)  # May raise JSONDecodeError
        custom_id = line_data.get("custom_id", f"unknown_custom_id_line_{line_number}")

        # `_extract_origin_id` is called here to potentially get a more specific ID
        # (e.g., an entity ID) if the `custom_id` format allows for it.
        # This `extracted_id` is then passed to `_parse_llm_response_content` and `parse_inner`.
        extracted_id = self._extract_origin_id(custom_id)

        error_payload = line_data.get("error")
        if error_payload:
            msg = error_payload.get('message', str(error_payload))
            logging.error(f"LLM API error on line {line_number} (custom_id: {custom_id}, run_id: {self.run_id}): {msg}")
            return items_from_line, 1

        response_payload = line_data.get("response")
        if not response_payload or response_payload.get("status_code") != 200:
            status = response_payload.get('status_code', 'N/A') if response_payload else 'N/A'
            logging.warning(
                f"Non-200 status ({status}) on line {line_number} (custom_id: {custom_id}, run_id: {self.run_id}). Response: {str(response_payload)[:200]}"
            )
            return items_from_line, 1

        body = response_payload.get("body", {})
        choices = body.get("choices")
        if not choices or not isinstance(choices, list) or not choices[0]:
            logging.warning(
                f"Missing or invalid 'choices' in response body on line {line_number} (custom_id: {custom_id}, run_id: {self.run_id}).")
            return items_from_line, 1

        message = choices[0].get("message", {})
        message_content_str = message.get("content")

        if not message_content_str:
            logging.warning(
                f"No 'content' in LLM message on line {line_number} (custom_id: {custom_id}, run_id: {self.run_id}).")
            return items_from_line, 1

        parsed_items, parsing_errors = self._parse_llm_response_content(message_content_str, custom_id, extracted_id)
        if parsed_items:
            items_from_line.extend(parsed_items)
        line_errors += parsing_errors

        return items_from_line, line_errors

    def _parse_llm_response_content(self, message_content_str: str, custom_id_from_line: str,
                                    extracted_id_for_inner: str) -> Tuple[List[Dict[str, Any]], int]:
        """
        Cleans the LLM message string, parses it as JSON, and calls `parse_inner`
        for specific data extraction logic.
        `custom_id_from_line` is the original custom_id from the batch line for logging.
        `extracted_id_for_inner` is the (potentially processed) ID to be passed to `parse_inner`.
        Returns a tuple: (list_of_processed_items, number_of_parsing_errors).
        """
        parsed_items: List[Dict[str, Any]] = []
        parsing_errors = 0

        content_cleaned = message_content_str.strip()
        if content_cleaned.startswith("```json"):
            content_cleaned = content_cleaned[len("```json"):-len("```")].strip()
        elif content_cleaned.startswith("```"):
            content_cleaned = content_cleaned[len("```"):-len("```")].strip()

        try:
            inner_data = json.loads(content_cleaned)
        except json.JSONDecodeError:
            logging.error(
                f"Failed to decode internal JSON from LLM response (custom_id: {custom_id_from_line}, run_id: {self.run_id}). "
                f"Cleaned content (first 200 chars): '{content_cleaned[:200]}'"
            )
            return parsed_items, 1

        try:
            # Pass the `extracted_id_for_inner` to `parse_inner`.
            # This `extracted_id_for_inner` will be the entity/community ID for generation tasks,
            # or the original batch request custom_id for difficulty tasks.
            items_from_inner, inner_errors = self.parse_inner(inner_data, extracted_id_for_inner, self.run_id)
            if items_from_inner:
                parsed_items.extend(items_from_inner)
            parsing_errors += inner_errors
        except Exception:  # Catch unexpected errors within parse_inner
            logging.exception(
                f"Unexpected error in 'parse_inner' for custom_id: {custom_id_from_line}, run_id: {self.run_id}.")
            parsing_errors += 1

        return parsed_items, parsing_errors

    def _extract_origin_id(self, custom_id: str) -> str:
        """
        Extracts a more specific ID (e.g., entity/community ID for generation tasks)
        from a formatted `custom_id` string if it matches the 'gen_req_' pattern.
        Otherwise, returns the original `custom_id`. This returned ID is then
        passed to `parse_inner`.
        """
        prefix = "gen_req_"
        if custom_id.startswith(prefix):
            # Process the string part *after* the prefix
            id_part_with_suffix = custom_id[len(prefix):]
            # Split by the last underscore, assuming it separates the ID from a suffix (e.g., index)
            parts = id_part_with_suffix.rsplit('_', 1)
            if len(parts) > 0 and parts[0]:  # Ensure there's an ID part and it's not empty
                actual_id = parts[0]
                return actual_id
            else:  # The format after "gen_req_" was not as expected (e.g., "gen_req_JustID", "gen_req__0")
                logging.warning(
                    f"custom_id '{custom_id}' starts with '{prefix}' but "
                    f"could not reliably extract an ID before a suffix. Using full custom_id."
                )
                return custom_id  # Fallback to full custom_id for safety
        return custom_id  # Default: return the original custom_id

    @abstractmethod
    def parse_inner(self, inner_data: Dict[str, Any], id_for_item_context: str, run_id: str) -> Tuple[
        List[Dict[str, Any]], int]:
        """
        Abstract method to be implemented by subclasses.
        Parses the `inner_data` (JSON content from LLM) to extract relevant items.
        `id_for_item_context` provides a contextual ID (e.g., origin_id for UCs, or
        the batch request's custom_id for difficulty assessments) for associating or logging items.
        Returns a tuple: (list_of_extracted_items, number_of_errors_during_extraction).
        """
        pass

    @abstractmethod
    def _save_to_db(self, db: Session, processed_data: List[Dict[str, Any]]) -> None:
        """
        Abstract method for subclasses to persist `processed_data` to the database
        using the provided `db` Session. This method should NOT commit the transaction.
        It should raise an exception if the database operation fails.
        """
        raise NotImplementedError


class GenerationBatchProcessor(BaseBatchProcessor):
    """Processes batch results for UC generation."""

    def parse_inner(self, inner_data: Dict[str, Any], origin_id_for_uc: str, run_id: str) -> Tuple[
        List[Dict[str, Any]], int]:
        """
        Parses `inner_data` expecting 'generated_units'.
        `origin_id_for_uc` is the ID of the original entity/community for these UCs.
        """
        items = []
        errors = 0
        generated_units = inner_data.get("generated_units")

        if not isinstance(generated_units, list):
            logging.warning(
                f"Expected 'generated_units' to be a list in inner_data for origin_id '{origin_id_for_uc}' (run: {run_id}). "
                f"Found: {type(generated_units)}. Data (first 200 chars): {str(inner_data)[:200]}"
            )
            return items, 1  # Count as one structural error for this inner_data

        if not (0 < len(generated_units) <= 6):  # LLM should ideally return all 6
            logging.warning(
                f"Expected 1-6 'generated_units' for origin_id '{origin_id_for_uc}' (run: {run_id}), "
                f"but found {len(generated_units)}. Data: {str(generated_units)[:200]}"
            )
        for unit_idx, unit_data in enumerate(generated_units):
            if isinstance(unit_data, dict) and "bloom_level" in unit_data and "uc_text" in unit_data:
                record = {
                    "uc_id": str(uuid.uuid4()),
                    "origin_id": origin_id_for_uc,
                    "bloom_level": unit_data["bloom_level"],
                    "uc_text": unit_data["uc_text"]
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
        """Saves generated UCs to the raw UCs table."""
        logging.debug(f"Saving {len(processed_data)} generated UCs for run_id {self.run_id}.")
        add_generated_ucs_raw(db, self.run_id, processed_data)


class DifficultyBatchProcessor(BaseBatchProcessor):
    """Processes batch results for difficulty assessment."""

    def _process_line_content(self, line_content: str, line_number: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        Parses the JSON structure of a single line from the batch output file,
        validates the response, extracts comparison_group_id from custom_id,
        and delegates to _parse_llm_response_content.
        Returns a tuple: (list_of_processed_items_from_line, number_of_errors_in_line).
        """
        line_errors = 0
        items_from_line: List[Dict[str, Any]] = []

        line_data = json.loads(line_content)

        batch_line_custom_id = line_data.get("custom_id", f"unknown_custom_id_line_{line_number}")

        comparison_group_id_for_items: Optional[str] = None
        if batch_line_custom_id.startswith("comp_group="):
            parts = batch_line_custom_id.split("=", 1)
            if len(parts) > 1 and parts[1]:
                comparison_group_id_for_items = parts[1]
            else:
                logging.error(
                    f"Malformed custom_id from OpenAI (empty after 'comp_group='): '{batch_line_custom_id}' "
                    f"on line {line_number} of batch {self.batch_id} (run_id: {self.run_id})."
                )
                return items_from_line, 1
        else:
            logging.error(
                f"Unexpected custom_id format from OpenAI: '{batch_line_custom_id}'. Expected 'comp_group=ID'. "
                f"Line {line_number}, batch {self.batch_id}, run_id {self.run_id}."
            )
            return items_from_line, 1

        if not comparison_group_id_for_items:
            logging.error(
                f"Failed to extract a valid comparison_group_id from custom_id: '{batch_line_custom_id}' "
                f"on line {line_number} of batch {self.batch_id} (run_id: {self.run_id})."
            )
            return items_from_line, 1

        error_payload = line_data.get("error")
        if error_payload:
            msg = error_payload.get('message', str(error_payload))
            logging.error(
                f"LLM API error on line {line_number} (OpenAI custom_id: {batch_line_custom_id}, "
                f"parsed comp_group_id: {comparison_group_id_for_items}, run_id: {self.run_id}): {msg}")
            return items_from_line, 1

        response_payload = line_data.get("response")
        if not response_payload or response_payload.get("status_code") != 200:
            status = response_payload.get('status_code', 'N/A') if response_payload else 'N/A'
            logging.warning(
                f"Non-200 status ({status}) on line {line_number} (OpenAI custom_id: {batch_line_custom_id}, "
                f"parsed comp_group_id: {comparison_group_id_for_items}, run_id: {self.run_id}). Response: {str(response_payload)[:200]}"
            )
            return items_from_line, 1

        body = response_payload.get("body", {})
        choices = body.get("choices")
        if not choices or not isinstance(choices, list) or not choices[0]:
            logging.warning(
                f"Missing or invalid 'choices' in response body on line {line_number} "
                f"(OpenAI custom_id: {batch_line_custom_id}, parsed comp_group_id: {comparison_group_id_for_items}, run_id: {self.run_id}).")
            return items_from_line, 1

        message = choices[0].get("message", {})
        message_content_str = message.get("content")

        if not message_content_str:
            logging.warning(
                f"No 'content' in LLM message on line {line_number} "
                f"(OpenAI custom_id: {batch_line_custom_id}, parsed comp_group_id: {comparison_group_id_for_items}, run_id: {self.run_id}).")
            return items_from_line, 1

        parsed_items, parsing_errors = self._parse_llm_response_content(
            message_content_str,
            batch_line_custom_id,
            comparison_group_id_for_items
        )
        if parsed_items:
            items_from_line.extend(parsed_items)
        line_errors += parsing_errors

        return items_from_line, line_errors

    def parse_inner(self, inner_data: Dict[str, Any], comparison_group_id: str, run_id: str) -> Tuple[
        List[Dict[str, Any]], int]:
        """
        Parses `inner_data` expecting 'difficulty_assessments'.
        `comparison_group_id` is the ID of the comparison group this assessment belongs to.
        The uc_id for each specific assessment is expected inside each item of 'difficulty_assessments'.
        """
        items = []
        errors = 0
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

            if isinstance(uc_id, str) and isinstance(difficulty_score, int) and isinstance(justification, str):
                record = {
                    "comparison_group_id": comparison_group_id,
                    "knowledge_unit_id": uc_id,
                    "difficulty_score": difficulty_score,
                    "justification": justification
                }
                items.append(record)
            else:
                errors += 1
                logging.warning(
                    f"Invalid difficulty assessment format at index {assessment_idx} "
                    f"(for comparison_group_id: {comparison_group_id}, run: {run_id}). "
                    f"Assessment data (first 100): {str(assessment_data)[:100]}"
                )
        return items, errors

    def _save_to_db(self, db: Session, processed_data: List[Dict[str, Any]]) -> None:
        """Saves difficulty assessments to the batch evaluations table."""
        logging.debug(f"Saving {len(processed_data)} difficulty assessments for run_id {self.run_id}.")
        add_knowledge_unit_evaluations_batch(db, self.run_id, processed_data)


def process_batch_results(
        batch_id: str,
        output_file_id: str,
        error_file_id: Optional[str],
        stage_output_dir: Any,
        output_filename: str,
        run_id: str,
        db: Session
) -> bool:
    """
    Main entry point to process batch results.
    Selects the appropriate processor based on `output_filename`,
    invokes its `process` method, and returns the success status.
    The provided `db` session is used for all database operations within the processor.
    This function does NOT commit the transaction; the caller is responsible.
    """
    logging.debug(
        f"process_batch_results called for: batch_id='{batch_id}', output_filename='{output_filename}', run_id='{run_id}'"
    )
    processor: Optional[BaseBatchProcessor] = None

    if output_filename == GENERATED_UCS_RAW:
        processor = GenerationBatchProcessor(
            batch_id, output_file_id, error_file_id, stage_output_dir, output_filename, run_id
        )
    elif output_filename == UC_EVALUATIONS_RAW:
        processor = DifficultyBatchProcessor(
            batch_id, output_file_id, error_file_id, stage_output_dir, output_filename, run_id
        )
    else:
        logging.error(f"Unsupported 'output_filename': {output_filename} for batch processing (run_id: {run_id}).")
        return False  

    return processor.process(db)