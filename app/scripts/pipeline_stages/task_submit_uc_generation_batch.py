from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import datetime
from app.db import get_session

import app.crud.knowledge_unit_origins as crud_knowledge_unit_origins

from app.scripts.origins_utils import (
    DefaultSelector,
    HubNeighborSelector,
)

from app.scripts.constants import (
    MAX_ORIGINS_FOR_TESTING,
    PROMPT_UC_GENERATION_FILE,
    LLM_MODEL,
    LLM_TEMPERATURE_GENERATION,
    AIRFLOW_DATA_DIR,
    get_dirs
)

from app.scripts.llm_core.models import (
    GenericLLMRequest, GenericLLMMessage, GenericLLMRequestConfig
)
from app.scripts.llm_client import get_llm_strategy, LLMClient, OpenAIBatchClient

def task_submit_uc_generation_batch(run_id: str) -> Optional[str]:
    """
    Prepara GenericLLMRequests e submete batch de geração UC para um run_id.
    Retorna o llm_batch_id do provedor, ou None se nada foi submetido.
    """
    logging.info(f"--- LOGIC: submit_uc_generation_batch (run_id={run_id}) ---")

    with get_session() as db:
        ku_origins_records = crud_knowledge_unit_origins.get_knowledge_unit_origins(db, run_id)

    if not ku_origins_records:
        logging.warning(f"Nenhuma KU Origin encontrada no DB para run_id={run_id}. Nada a submeter para geração.")
        return None

    graphrag_base_dir_for_selector = Path(AIRFLOW_DATA_DIR) / run_id / "output"
    if MAX_ORIGINS_FOR_TESTING is not None and MAX_ORIGINS_FOR_TESTING > 0:
        selector = HubNeighborSelector(MAX_ORIGINS_FOR_TESTING, graphrag_base_dir_for_selector)
    else:
        selector = DefaultSelector(None)
    origins_to_process = selector.select(ku_origins_records)

    if not origins_to_process:
        logging.warning(f"Nenhuma origem selecionada para processar para run_id={run_id}. Nada a submeter.")
        return None

    try:
        with open(PROMPT_UC_GENERATION_FILE, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except Exception as e:
        logging.error(f"Erro crítico lendo prompt de geração UC '{PROMPT_UC_GENERATION_FILE}': {e}", exc_info=True)
        raise ValueError(f"Falha ao ler prompt de geração: {e}") from e

    generic_llm_requests: List[GenericLLMRequest] = []
    for i, origin_data in enumerate(origins_to_process):
        origin_id = str(origin_data.get("origin_id"))
        title = origin_data.get("title", "N/A")
        context_text = origin_data.get("context", "")

        formatted_prompt = (prompt_template
                            .replace("{{CONCEPT_TITLE}}", title)
                            .replace("{{CONTEXT}}", context_text if context_text else "N/A"))

        request_meta: Dict[str, Any] = {
            "type": "uc_generation",
            "origin_id": origin_id,
            "iteration_index": i,
            "run_id": run_id
        }

        messages: List[GenericLLMMessage] = [
            {"role": "system",
             "content": "Você é um especialista em educação, capaz de gerar Unidades de Conhecimento (UCs) abrangentes e personalizadas com base na Taxonomia de Bloom Revisada."},
            {"role": "user", "content": formatted_prompt}
        ]

        config: GenericLLMRequestConfig = {
            "model_name": LLM_MODEL,
            "temperature": LLM_TEMPERATURE_GENERATION,
            "response_format": {"type": "json_object"}
        }

        generic_llm_requests.append(
            GenericLLMRequest(request_metadata=request_meta, messages=messages, config=config)
        )

    if not generic_llm_requests:
        logging.warning(
            f"Nenhum request LLM genérico preparado para geração de UC (run_id: {run_id}). Nada a submeter.")
        return None

    _, _, batch_files_dir_for_run, _, _, _, _, _ = get_dirs(run_id) 
    batch_files_dir_for_run.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_intermediate_file_path = batch_files_dir_for_run / f"openai_uc_generation_batch_{timestamp}_{run_id}.jsonl"

    llm_client_instance: LLMClient = get_llm_strategy()

    try:
        provider_file_id = llm_client_instance.prepare_and_upload_batch_file(
            generic_requests=generic_llm_requests,
            batch_input_file_path=batch_intermediate_file_path,
            batch_endpoint_url="/v1/chat/completions"
        )
        logging.info(f"Arquivo de batch preparado e upload concluído. Provider File ID: {provider_file_id}")

        batch_job_id = llm_client_instance.create_batch_job(
            input_file_id=provider_file_id,
            endpoint="/v1/chat/completions",
            metadata={'description': f'UC Generation Batch for run_id {run_id}'}
        )
        logging.info(f"Batch job de geração criado. Provider Batch ID: {batch_job_id}")
        return batch_job_id
    except Exception as e:
        logging.exception(
            f"Falha ao preparar, fazer upload ou submeter batch de geração de UC ao LLM (run_id: {run_id})")
        raise
