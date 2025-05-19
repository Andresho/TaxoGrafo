import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from app.scripts.llm_core.models import (
    GenericLLMRequest, GenericLLMResponse,
    IBatchRequestFormatter, IBatchResponseParser
)
from app.scripts.constants import LLM_MODEL


class OpenAIBatchRequestFormatter(IBatchRequestFormatter):
    def format_requests_to_file(
        self,
        generic_requests: List[GenericLLMRequest],
        output_file_path: Path,
        batch_endpoint_url: str = "/v1/chat/completions"
    ) -> None:
        openai_batch_requests = []
        for req in generic_requests:
            custom_id_str = f"gr_meta::{json.dumps(req['request_metadata'])}"

            body = {
                "model": req['config'].get('model_name') or LLM_MODEL,
                "messages": req['messages'],
            }
            if req['config'].get('temperature') is not None:
                body["temperature"] = req['config']['temperature']
            if req['config'].get('response_format'):
                body["response_format"] = req['config']['response_format']
            if req['config'].get('max_tokens'):
                 body["max_tokens"] = req['config']['max_tokens']

            openai_batch_requests.append({
                "custom_id": custom_id_str,
                "method": "POST",
                "url": batch_endpoint_url,
                "body": body
            })

        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for item in openai_batch_requests:
                f.write(json.dumps(item) + '\n')
        logging.info(f"Salvo JSONL para OpenAI Batch API em {output_file_path} com {len(openai_batch_requests)} requests.")


class OpenAIBatchResponseParser(IBatchResponseParser):
    def parse_batch_output_line(
        self,
        line_content: str
    ) -> GenericLLMResponse:
        line_data = json.loads(line_content)
        openai_custom_id = line_data.get("custom_id")
        response_payload = line_data.get("response")
        error_payload = line_data.get("error")

        parsed_request_metadata: Dict[str, Any] = {}
        if openai_custom_id and openai_custom_id.startswith("gr_meta::"):
            try:
                parsed_request_metadata = json.loads(openai_custom_id[len("gr_meta::"):])
            except json.JSONDecodeError:
                logging.error(f"Falha ao desserializar request_metadata do custom_id: {openai_custom_id}")
                parsed_request_metadata = {"error": "failed_to_parse_custom_id", "original_custom_id": openai_custom_id}
        elif openai_custom_id:
             parsed_request_metadata = {"original_custom_id": openai_custom_id}

        if error_payload:
            return GenericLLMResponse(
                request_metadata=parsed_request_metadata,
                response_content=None,
                error_message=error_payload.get('message', str(error_payload)),
                raw_response_data=line_data
            )

        if not response_payload or response_payload.get("status_code") != 200:
            status_code = response_payload.get('status_code', 'N/A') if response_payload else 'N/A'
            error_msg_detail = response_payload.get('body', {}).get('error', {}).get('message', str(response_payload)[:200])
            return GenericLLMResponse(
                request_metadata=parsed_request_metadata,
                response_content=None,
                error_message=f"Non-200 status ({status_code}). Detail: {error_msg_detail}",
                raw_response_data=line_data
            )

        body = response_payload.get("body", {})
        choices = body.get("choices")
        if not choices or not isinstance(choices, list) or not choices[0]:
            return GenericLLMResponse(
                request_metadata=parsed_request_metadata,
                response_content=None,
                error_message="Missing or invalid 'choices' in response body.",
                raw_response_data=line_data
            )

        message = choices[0].get("message", {})
        message_content_str = message.get("content")

        return GenericLLMResponse(
            request_metadata=parsed_request_metadata,
            response_content=message_content_str,
            error_message=None,
            raw_response_data=line_data
        )