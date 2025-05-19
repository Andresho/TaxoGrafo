import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from app.scripts.llm_core.models import (
    GenericLLMRequest, GenericLLMResponse,
    IBatchRequestFormatter, IBatchResponseParser
)
from app.scripts.llm_providers.openai_utils import (
    OpenAIBatchRequestFormatter, OpenAIBatchResponseParser
)


OPENAI_CLIENT_INSTANCE: Optional[OpenAI] = None 

class LLMClient(ABC):
    @abstractmethod
    def prepare_and_upload_batch_file(
        self,
        generic_requests: List[GenericLLMRequest],
        batch_input_file_path: Path,
        batch_endpoint_url: str
    ) -> str:
        """
        Formata as requisições genéricas para o formato de batch do provedor,
        salva o arquivo, faz o upload e retorna o file_id do provedor.
        """
        pass

    @abstractmethod
    def create_batch_job(self, input_file_id: str, endpoint: str, metadata: dict) -> str:
        """Cria job de batch e retorna batch_id."""
        pass

    @abstractmethod
    def get_batch_status(self, batch_id: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Consulta status do batch: retorna (status, output_file_id, error_file_id)."""
        pass

    @abstractmethod
    def read_file(self, file_id: str) -> bytes:
        """Lê conteúdo bruto de um file_id do provedor."""
        pass

    @abstractmethod
    def parse_llm_batch_line(self, line_content: str) -> GenericLLMResponse:
        """
        Usa o parser específico do provedor para transformar uma linha de resultado
        em uma GenericLLMResponse.
        """
        pass


class OpenAIBatchClient(LLMClient):
    def __init__(self, client_override=None):
        if client_override is not None:
            self.client: OpenAI = client_override
        else:
            if OpenAI is None:
                raise ImportError("OpenAI SDK não está instalado. Execute `pip install openai`.")
            self.client = OpenAI()

        self.request_formatter: IBatchRequestFormatter = OpenAIBatchRequestFormatter()
        self.response_parser: IBatchResponseParser = OpenAIBatchResponseParser()

    def prepare_and_upload_batch_file(
        self,
        generic_requests: List[GenericLLMRequest],
        batch_input_file_path: Path,
        batch_endpoint_url: str = "/v1/chat/completions"
    ) -> str:
        self.request_formatter.format_requests_to_file(
            generic_requests,
            batch_input_file_path,
            batch_endpoint_url
        )

        with open(batch_input_file_path, 'rb') as f:
            file_obj = self.client.files.create(file=f, purpose='batch')
        return file_obj.id

    def create_batch_job(self, input_file_id: str, endpoint: str, metadata: dict) -> str:
        batch_job = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window='24h',
            metadata=metadata
        )
        return batch_job.id

    def get_batch_status(self, batch_id: str) -> Tuple[str, Optional[str], Optional[str]]:
        try:
            batch_job = self.client.batches.retrieve(batch_id)
            return batch_job.status, batch_job.output_file_id, batch_job.error_file_id
        except Exception as e:
            logging.error(f"Erro ao verificar status do batch OpenAI {batch_id}: {e}")

            return "api_error", None, None


    def read_file(self, file_id: str) -> bytes:
        return self.client.files.content(file_id).read()

    def parse_llm_batch_line(self, line_content: str) -> GenericLLMResponse:
        return self.response_parser.parse_batch_output_line(line_content)


def get_llm_strategy() -> LLMClient:
    """Cria e retorna uma estratégia LLM."""
    if OPENAI_CLIENT_INSTANCE is not None:
        return OpenAIBatchClient(client_override=OPENAI_CLIENT_INSTANCE)
    return OpenAIBatchClient()
