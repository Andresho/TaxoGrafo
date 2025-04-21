"""
Interface para abstração de cliente LLM (Batch API, etc).
"""
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from pathlib import Path

def get_llm_strategy() -> "LLMClient":
    """Cria e retorna uma estratégia LLM a partir de OPENAI_CLIENT."""
    try:
        from scripts.pipeline_tasks import OPENAI_CLIENT
    except ImportError:
        raise ValueError("OpenAI client não inicializado")
    if OPENAI_CLIENT is None:
        raise ValueError("OpenAI client não inicializado")
    return OpenAIBatchClient(OPENAI_CLIENT)

class LLMClient(ABC):
    """Interface para comunicação com LLMs suportadas."""
    @abstractmethod
    def upload_batch_file(self, file_path: Path) -> str:
        """Faz upload de arquivo para batch e retorna file_id."""
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
        """Lê conteúdo bruto de um file_id."""
        pass

class OpenAIBatchClient(LLMClient):
    """Implementação de LLMClient usando OpenAI Batch API."""
    def __init__(self, openai_client):
        self.client = openai_client

    def upload_batch_file(self, file_path: Path) -> str:
        with open(file_path, 'rb') as f:
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
        """Consulta status do batch e retorna (status, output_file_id, error_file_id)."""
        try:
            batch_job = self.client.batches.retrieve(batch_id)
            return batch_job.status, batch_job.output_file_id, batch_job.error_file_id
        except Exception as e:
            logging.error(f"Erro ao verificar status do batch {batch_id}: {e}")
            return "API_ERROR", None, None

    def read_file(self, file_id: str) -> bytes:
        # Retorna bytes do arquivo batch
        return self.client.files.content(file_id).read()