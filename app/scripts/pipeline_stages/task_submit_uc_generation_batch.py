from pathlib import Path
from typing import Optional
import logging
import datetime
from app.db import get_session

import app.crud.knowledge_unit_origins as crud_knowledge_unit_origins

from app.scripts.llm_client import get_llm_strategy

from app.scripts.data_lake import DataLake
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
)

# --- Helpers de Diretórios por run_id ---
def _get_dirs(run_id: str):
    """Retorna tupla de paths: base_input, work_dir e batch_dir conforme run_id."""
    root = Path(AIRFLOW_DATA_DIR) / run_id
    base_input_for_graphrag = root / 'output'
    work_dir_for_batches = root / 'pipeline_workdir'
    batch_files_dir = work_dir_for_batches / 'batch_files'

    s1 = work_dir_for_batches / '1_origins'
    s2 = work_dir_for_batches / '2_generated_ucs'
    s3 = work_dir_for_batches / '3_relationships'
    s4 = work_dir_for_batches / '4_difficulty_evals'
    s5 = work_dir_for_batches / '5_final_outputs'

    return base_input_for_graphrag, work_dir_for_batches, batch_files_dir, s1, s2, s3, s4, s5

def task_submit_uc_generation_batch(run_id: str) -> Optional[str]:
    """
    Prepara JSONL e submete batch de geração UC para um run_id.
    Retorna o llm_batch_id do provedor, ou None se nada foi submetido.
    A idempotência de "já existe UCs geradas?" foi movida para a API que gerencia PipelineBatchJob.
    Esta função agora foca em:
    1. Pegar as origens do DB.
    2. Preparar o JSONL.
    3. Submeter ao LLM.
    """
    logging.info(f"--- LOGIC: submit_uc_generation_batch (run_id={run_id}) ---")
    llm = get_llm_strategy()

    with get_session() as db:
        ku_origins_records = crud_knowledge_unit_origins.get_knowledge_unit_origins(db, run_id)

    if not ku_origins_records:
        logging.warning(
            f"Nenhuma Knowledge Unit Origin encontrada no DB para run_id={run_id}. Nada a submeter para geração de UC.")
        return None

    all_origins = ku_origins_records

    graphrag_base_dir_for_selector = Path(
        AIRFLOW_DATA_DIR) / run_id / "output"

    if MAX_ORIGINS_FOR_TESTING is not None and MAX_ORIGINS_FOR_TESTING > 0:
        selector = HubNeighborSelector(MAX_ORIGINS_FOR_TESTING, graphrag_base_dir_for_selector)
    else:
        selector = DefaultSelector(None)

    origins_to_process = selector.select(all_origins)

    if not origins_to_process:
        logging.warning(
            f"Nenhuma origem selecionada para processar para run_id={run_id} após filtragem/seleção. Nada a submeter.")
        return None

    try:
        with open(PROMPT_UC_GENERATION_FILE, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except Exception as e:
        logging.error(f"Erro crítico lendo prompt de geração de UC '{PROMPT_UC_GENERATION_FILE}': {e}", exc_info=True)
        raise ValueError(f"Falha ao ler prompt de geração: {e}")

    _, _, batch_files_dir_for_run, _, _, _, _, _ = _get_dirs(run_id)
    batch_files_dir_for_run.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_input_filename = f"uc_generation_batch_{timestamp}_{run_id}.jsonl"
    batch_input_path = batch_files_dir_for_run / batch_input_filename

    llm_requests = []
    for i, origin_data in enumerate(origins_to_process):
        origin_id = origin_data.get("origin_id")
        req_custom_id = f"gen_req_{origin_id}_{i}"

        title = origin_data.get("title", "N/A")
        context_text = origin_data.get("context", "")

        formatted_prompt = (
            prompt_template
            .replace("{{CONCEPT_TITLE}}", title)
            .replace("{{CONTEXT}}", context_text if context_text else "N/A")
        )
        request_body = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "Você é um especialista em educação..."},
                {"role": "user", "content": formatted_prompt}
            ],
            "temperature": LLM_TEMPERATURE_GENERATION,
            "response_format": {"type": "json_object"}
        }
        llm_requests.append({
            "custom_id": req_custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request_body
        })

    if not llm_requests:
        logging.warning(f"Nenhum request LLM preparado para geração de UC (run_id: {run_id}). Nada a submeter.")
        return None

    DataLake.write_jsonl(llm_requests, batch_input_path)
    logging.info(f"Arquivo batch de geração criado: {batch_input_path} ({len(llm_requests)} requests)")

    try:
        logging.info(f"Fazendo upload de {batch_input_path} via LLM strategy...")
        file_id = llm.upload_batch_file(batch_input_path)
        logging.info(f"Upload concluído. File ID: {file_id}")

        logging.info("Criando batch job via LLM strategy...")
        batch_job_id = llm.create_batch_job(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            metadata={'description': f'UC Generation Batch for run_id {run_id}'}
        )
        logging.info(f"Batch job de geração criado. LLM Batch ID: {batch_job_id}")
        return batch_job_id
    except Exception as e:
        logging.exception(f"Falha ao submeter batch de geração de UC ao LLM (run_id: {run_id})")
        raise
