from typing import List, Dict, Optional, Any, Tuple
import logging
from collections import defaultdict
import datetime
import uuid

from app.db import get_session
import app.models as models

import app.crud.knowledge_unit_origins as crud_knowledge_unit_origins
import app.crud.generated_ucs_raw as crud_generated_ucs_raw

from app.scripts.llm_client import get_llm_strategy
from app.scripts.data_lake import DataLake

from app.scripts.difficulty_utils import _format_difficulty_prompt
from app.scripts.difficulty_scheduler import OriginDifficultyScheduler
from app.scripts.constants import (
    MIN_EVALUATIONS_PER_UC,
    DIFFICULTY_BATCH_SIZE,
    PROMPT_UC_DIFFICULTY_FILE,
    BLOOM_ORDER,
    LLM_MODEL,
    LLM_TEMPERATURE_DIFFICULTY,
    get_dirs
)

def task_submit_difficulty_batch(run_id: str) -> Optional[str]:
    """
    Prepara e submete batch de avaliação de dificuldade.
    Retorna o llm_batch_id do provedor, ou None se nada foi submetido.
    A idempotência de alto nível é tratada pela API.
    """
    logging.info(f"--- LOGIC: submit_difficulty_batch (run_id={run_id}) ---")

    with get_session() as db_read:
        generated_ucs_raw_list = crud_generated_ucs_raw.get_generated_ucs_raw(db_read, run_id)
        if not generated_ucs_raw_list:
            logging.warning(
                f"Nenhuma UC gerada (raw) encontrada no DB para run_id={run_id} para avaliação de dificuldade. Nada a submeter.")
            return None

        all_knowledge_origins_list = crud_knowledge_unit_origins.get_knowledge_unit_origins(db_read, run_id)
        if not all_knowledge_origins_list:
            logging.error(
                f"Knowledge Origins não encontradas para run_id={run_id}, mas UCs geradas (raw) existem. Crítico. Não é possível submeter para dificuldade.")
            return None

    with get_session() as db_for_groups_write:
        ucs_by_origin_then_bloom: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
        for uc_raw_dict in generated_ucs_raw_list:
            origin_id = str(uc_raw_dict.get('origin_id'))
            bloom_level = uc_raw_dict.get('bloom_level')
            if origin_id and bloom_level:
                ucs_by_origin_then_bloom[origin_id][bloom_level] = uc_raw_dict

        scheduler = OriginDifficultyScheduler(
            all_knowledge_origins=all_knowledge_origins_list,
            min_evaluations_per_origin=MIN_EVALUATIONS_PER_UC,
            difficulty_batch_size=DIFFICULTY_BATCH_SIZE
        )
        paired_origin_sets_with_coherence = scheduler.generate_origin_pairings()

        if not paired_origin_sets_with_coherence:
            logging.info(
                f"Nenhum conjunto de origens foi pareado pelo scheduler para avaliação de dificuldade (run_id: {run_id}). Nada a submeter.")
            return None

        try:
            with open(PROMPT_UC_DIFFICULTY_FILE, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except Exception as e:
            logging.error(f"Erro crítico lendo prompt de dificuldade '{PROMPT_UC_DIFFICULTY_FILE}': {e}", exc_info=True)
            raise ValueError(f"Falha ao ler prompt de dificuldade: {e}")

        requests_for_llm_batch: List[Dict[str, Any]] = []
        num_groups_created_for_llm = 0

        for pairing_info in paired_origin_sets_with_coherence:
            origin_ids_in_current_pairing: Tuple[str, ...] = pairing_info["origin_ids"]
            coherence_level_of_pairing: str = pairing_info["coherence_level"]
            seed_origin_id_for_pairing: str = pairing_info["seed_id_for_batch"]

            for current_bloom_level in BLOOM_ORDER:
                ucs_for_this_llm_request: List[Dict[str, str]] = []
                can_form_valid_llm_request = True

                for origin_id_in_group in origin_ids_in_current_pairing:
                    uc_data = ucs_by_origin_then_bloom.get(str(origin_id_in_group), {}).get(current_bloom_level)
                    if uc_data and uc_data.get('uc_id') and uc_data.get('uc_text'):
                        ucs_for_this_llm_request.append({
                            "uc_id": str(uc_data['uc_id']),
                            "uc_text": str(uc_data['uc_text'])
                        })
                    else:
                        can_form_valid_llm_request = False
                        break

                if can_form_valid_llm_request and len(ucs_for_this_llm_request) == DIFFICULTY_BATCH_SIZE:
                    num_groups_created_for_llm += 1
                    generated_comparison_group_id = str(uuid.uuid4())
                    openai_llm_custom_id = f"comp_group={generated_comparison_group_id}"

                    new_db_comparison_group = models.DifficultyComparisonGroup(
                        pipeline_run_id=run_id,
                        comparison_group_id=generated_comparison_group_id,
                        bloom_level=current_bloom_level,
                        coherence_level=coherence_level_of_pairing,
                        llm_batch_request_custom_id=openai_llm_custom_id
                    )
                    db_for_groups_write.add(new_db_comparison_group)
                    db_for_groups_write.flush()

                    association_entries = [
                        {
                            "pipeline_run_id": run_id,
                            "comparison_group_id": generated_comparison_group_id,
                            "origin_id": origin_id_in_group_item,
                            "is_seed_origin": (origin_id_in_group_item == seed_origin_id_for_pairing)
                        } for origin_id_in_group_item in origin_ids_in_current_pairing
                    ]
                    if association_entries:
                        db_for_groups_write.execute(models.difficulty_group_origin_association.insert(),
                                                    association_entries)

                    formatted_prompt_text_for_llm = _format_difficulty_prompt(ucs_for_this_llm_request, prompt_template)
                    request_body_for_llm = {
                        "model": LLM_MODEL,
                        "messages": [{"role": "system", "content": "Você é um especialista em educação..."},
                                     {"role": "user", "content": formatted_prompt_text_for_llm}],
                        "temperature": LLM_TEMPERATURE_DIFFICULTY,
                        "response_format": {"type": "json_object"}
                    }
                    requests_for_llm_batch.append({
                        "custom_id": openai_llm_custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": request_body_for_llm
                    })

        if not requests_for_llm_batch:
            logging.info(
                f"Nenhum request LLM preparado para avaliação de dificuldade (run_id: {run_id}). Nada a submeter.")
            if num_groups_created_for_llm > 0: db_for_groups_write.rollback()
            return None

        try:
            db_for_groups_write.commit()
            logging.info(
                f"Successfully committed {num_groups_created_for_llm} DifficultyComparisonGroups for run_id {run_id}.")
        except Exception as e_commit:
            db_for_groups_write.rollback()
            logging.error(f"Failed to commit DifficultyComparisonGroups for run_id {run_id}: {e_commit}", exc_info=True)
            raise

    llm = get_llm_strategy()
    _, work_dir_for_run, batch_files_dir_for_run, _, _, _, _, _ = get_dirs(run_id)
    batch_files_dir_for_run.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_input_filename = f"uc_difficulty_batch_input_{timestamp}_{run_id}.jsonl"
    batch_input_path = batch_files_dir_for_run / batch_input_filename

    DataLake.write_jsonl(requests_for_llm_batch, batch_input_path)
    logging.info(f"Arquivo batch de dificuldade ({len(requests_for_llm_batch)} requests) criado: {batch_input_path}")

    try:
        file_id = llm.upload_batch_file(batch_input_path)
        logging.info(f"Upload do arquivo de batch de dificuldade concluído. File ID: {file_id}")

        batch_job_id_from_provider = llm.create_batch_job(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            metadata={'description': f'UC Difficulty Evaluation Batch for run_id {run_id}'}
        )
        logging.info(f"Batch job de dificuldade submetido. LLM Batch ID: {batch_job_id_from_provider}")
        return batch_job_id_from_provider
    except Exception as e:
        logging.exception(f"Falha ao submeter batch de dificuldade ao LLM (run_id: {run_id})")
        raise