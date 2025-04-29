"""
Interface para abstração de cliente LLM (Batch API, etc).
"""
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from pathlib import Path

# Permite injeção de cliente para testes (dummy, etc.)
OPENAI_CLIENT: Optional[object] = None

try:
    from openai import OpenAI  # Importa cliente OpenAI padrão para Batch API
except ImportError:
    OpenAI = None

def get_llm_strategy() -> "LLMClient":
    """Cria e retorna uma estratégia LLM: usa OPENAI_CLIENT injetado ou o cliente OpenAI padrão."""
    # Se há cliente injetado para testes, usa-o
    if OPENAI_CLIENT is not None:
        return OpenAIBatchClient(OPENAI_CLIENT)
    # Senão, usa cliente real
    if OpenAI is None:
        raise ValueError("OpenAI client não inicializado")
    return OpenAIBatchClient()

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
    """Implementação de LLMClient usando OpenAI Batch API, com injeção opcional de cliente para testes."""
    def __init__(self, client=None):
        # Permite injeção de cliente (e.g., DummyClient) em testes
        if client is not None:
            self.client = client
        else:
            if OpenAI is None:
                raise ValueError("OpenAI client não inicializado")
            self.client = OpenAI()

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