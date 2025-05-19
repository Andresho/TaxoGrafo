from typing import List, Dict, Any, Optional, TypedDict
from abc import ABC, abstractmethod
from pathlib import Path

class GenericLLMMessage(TypedDict):
    role: str
    content: str

class GenericLLMRequestConfig(TypedDict, total=False):
    model_name: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    response_format: Optional[Dict[str, str]]

class GenericLLMRequest(TypedDict):
    request_metadata: Dict[str, Any]
    messages: List[GenericLLMMessage]
    config: GenericLLMRequestConfig

class GenericLLMResponse(TypedDict):
    request_metadata: Dict[str, Any]
    response_content: Optional[str]
    error_message: Optional[str]
    raw_response_data: Optional[Dict[str, Any]]

class IBatchRequestFormatter(ABC):
    @abstractmethod
    def format_requests_to_file(
            self,
            generic_requests: List[GenericLLMRequest],
            output_file_path: Path,
            batch_endpoint_url: str
    ) -> None:
        """
        Formata uma lista de GenericLLMRequest para o formato de arquivo de batch
        específico do provedor e salva no output_file_path.
        """
        pass

class IBatchResponseParser(ABC):
    @abstractmethod
    def parse_batch_output_line(
            self,
            line_content: str
    ) -> GenericLLMResponse:
        """
        Parseia uma linha do arquivo de resultado do batch do provedor
        e a transforma em uma GenericLLMResponse.
        """
        pass
