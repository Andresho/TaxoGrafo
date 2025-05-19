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

from app.crud import difficulty_comparison_group as crud_dcg  # Criar este CRUD
from app.crud import difficulty_group_origin_association as crud_dga  # Criar este CRUD

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

from app.scripts.llm_core.models import ( # Importar de onde você os definiu
    GenericLLMRequest, GenericLLMMessage, GenericLLMRequestConfig
)
from app.scripts.llm_client import get_llm_strategy, LLMClient, OpenAIBatchClient

def task_submit_difficulty_batch(run_id: str) -> Optional[str]:
    """
    Prepara GenericLLMRequests e submete batch de avaliação de dificuldade.
    Retorna o llm_batch_id do provedor, ou None se nada foi submetido.
    """
    logging.info(f"--- LOGIC: submit_difficulty_batch (run_id={run_id}) ---")

    with get_session() as db_read:
        generated_ucs_raw_list = crud_generated_ucs_raw.get_generated_ucs_raw(db_read, run_id)
        if not generated_ucs_raw_list:
            logging.warning(f"Nenhuma UC gerada (raw) encontrada para avaliação de dificuldade (run_id: {run_id}).")
            return None
        all_knowledge_origins_list = crud_knowledge_unit_origins.get_knowledge_unit_origins(db_read, run_id)
        if not all_knowledge_origins_list:
            logging.error(
                f"KU Origins não encontradas para run_id={run_id}, mas UCs existem. Não pode submeter para dificuldade.")
            return None

    # Mapear UCs para acesso rápido
    ucs_by_origin_then_bloom: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for uc_raw_dict in generated_ucs_raw_list:
        origin_id = str(uc_raw_dict.get('origin_id'))
        bloom_level = uc_raw_dict.get('bloom_level')
        if origin_id and bloom_level:
            ucs_by_origin_then_bloom[origin_id][bloom_level] = uc_raw_dict

    # Usar DifficultyScheduler
    scheduler = OriginDifficultyScheduler(
        all_knowledge_origins=all_knowledge_origins_list,
        min_evaluations_per_origin=MIN_EVALUATIONS_PER_UC,
        difficulty_batch_size=DIFFICULTY_BATCH_SIZE
    )
    paired_origin_sets_with_coherence = scheduler.generate_origin_pairings()

    if not paired_origin_sets_with_coherence:
        logging.info(f"Nenhum conjunto de origens pareado para avaliação de dificuldade (run_id: {run_id}).")
        return None

    try:
        with open(PROMPT_UC_DIFFICULTY_FILE, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except Exception as e:
        logging.error(f"Erro crítico lendo prompt de dificuldade '{PROMPT_UC_DIFFICULTY_FILE}': {e}", exc_info=True)
        raise ValueError(f"Falha ao ler prompt de dificuldade: {e}") from e

    generic_llm_requests: List[GenericLLMRequest] = []

    # Para salvar os grupos de comparação no DB ANTES de submeter ao LLM
    comparison_groups_to_save_in_db: List[Dict[str, Any]] = []
    group_origin_associations_to_save_in_db: List[Dict[str, Any]] = []

    for pairing_info in paired_origin_sets_with_coherence:
        origin_ids_in_pairing: Tuple[str, ...] = pairing_info["origin_ids"]
        coherence_level: str = pairing_info["coherence_level"]
        seed_origin_id: str = pairing_info["seed_id_for_batch"]

        for current_bloom_level in BLOOM_ORDER:
            ucs_for_this_llm_request_payload: List[Dict[str, str]] = []  # Para o {{BATCH_OF_UCS}}
            valid_group_for_llm = True

            for origin_id_in_group in origin_ids_in_pairing:
                uc_data = ucs_by_origin_then_bloom.get(str(origin_id_in_group), {}).get(current_bloom_level)
                if uc_data and uc_data.get('uc_id') and uc_data.get('uc_text'):
                    ucs_for_this_llm_request_payload.append({
                        "uc_id": str(uc_data['uc_id']),
                        "uc_text": str(uc_data['uc_text'])
                    })
                else:
                    valid_group_for_llm = False
                    break  # Um UC faltando no grupo/nível invalida este request específico

            if valid_group_for_llm and len(ucs_for_this_llm_request_payload) == DIFFICULTY_BATCH_SIZE:
                prompt_input_text_for_llm = ""
                for uc_item in ucs_for_this_llm_request_payload:
                    prompt_input_text_for_llm += f"- ID: {uc_item['uc_id']}\n  Texto: {uc_item['uc_text']}\n"

                final_prompt_for_llm = prompt_template.replace("{{BATCH_OF_UCS}}", prompt_input_text_for_llm.strip())

                generated_comparison_group_id = str(uuid.uuid4())

                request_meta: Dict[str, Any] = {
                    "type": "difficulty_assessment",
                    "comparison_group_id": generated_comparison_group_id,
                    "run_id": run_id,
                    "bloom_level": current_bloom_level,
                    "coherence_level": coherence_level
                }

                messages: List[GenericLLMMessage] = [
                    {"role": "system",
                     "content": "Você é um especialista em educação, experiente em analisar a dificuldade intrínseca de unidades de conhecimento (UCs) para aprendizes em geral."},
                    {"role": "user", "content": final_prompt_for_llm}
                ]

                config: GenericLLMRequestConfig = {
                    "model_name": LLM_MODEL,
                    "temperature": LLM_TEMPERATURE_DIFFICULTY,
                    "response_format": {"type": "json_object"}
                }

                generic_llm_requests.append(
                    GenericLLMRequest(request_metadata=request_meta, messages=messages, config=config)
                )

                openai_llm_custom_id_placeholder = f"comp_group={generated_comparison_group_id}" 

                comparison_groups_to_save_in_db.append({
                    "pipeline_run_id": run_id,
                    "comparison_group_id": generated_comparison_group_id,
                    "bloom_level": current_bloom_level,
                    "coherence_level": coherence_level,
                    "llm_batch_request_custom_id": openai_llm_custom_id_placeholder
                })
                for origin_id_in_group_item in origin_ids_in_pairing:
                    group_origin_associations_to_save_in_db.append({
                        "pipeline_run_id": run_id,
                        "comparison_group_id": generated_comparison_group_id,
                        "origin_id": str(origin_id_in_group_item),
                        "is_seed_origin": (str(origin_id_in_group_item) == seed_origin_id)
                    })

    if not generic_llm_requests:
        logging.info(f"Nenhum request LLM genérico preparado para avaliação de dificuldade (run_id: {run_id}).")
        return None

    # Salvar os grupos de comparação e associações ANTES de submeter ao LLM
    if comparison_groups_to_save_in_db or group_origin_associations_to_save_in_db:
        with get_session() as db_write_groups: # Nova sessão para esta operação atômica
            try:

                if comparison_groups_to_save_in_db:
                     crud_dcg.add_difficulty_comparison_groups_raw(db_write_groups, comparison_groups_to_save_in_db)
                if group_origin_associations_to_save_in_db:
                     crud_dga.add_difficulty_group_origin_associations_raw(db_write_groups, group_origin_associations_to_save_in_db)

                db_write_groups.commit()
                logging.info("Grupos de comparação e associações salvos no DB com sucesso.")
            except Exception as e_db_commit:
                db_write_groups.rollback()
                logging.error(f"Falha ao salvar grupos de comparação/associações no DB para run_id {run_id}: {e_db_commit}", exc_info=True)
                raise # Impede a submissão ao LLM se não puder salvar os grupos
    else:
        logging.warning("Nenhum grupo de comparação ou associação para salvar no DB.")

    _, _, batch_files_dir_for_run, _, _, _, _, _ = get_dirs(run_id)
    batch_files_dir_for_run.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_intermediate_file_path = batch_files_dir_for_run / f"openai_uc_difficulty_batch_{timestamp}_{run_id}.jsonl"

    llm_client_instance: LLMClient = get_llm_strategy()

    try:
        provider_file_id = llm_client_instance.prepare_and_upload_batch_file(
            generic_requests=generic_llm_requests,
            batch_input_file_path=batch_intermediate_file_path,
            batch_endpoint_url="/v1/chat/completions"
        )
        logging.info(
            f"Arquivo de batch de dificuldade preparado e upload concluído. Provider File ID: {provider_file_id}")

        batch_job_id = llm_client_instance.create_batch_job(
            input_file_id=provider_file_id,
            endpoint="/v1/chat/completions",
            metadata={'description': f'UC Difficulty Evaluation Batch for run_id {run_id}'}
        )
        logging.info(f"Batch job de dificuldade criado. Provider Batch ID: {batch_job_id}")
        return batch_job_id
    except Exception as e:
        logging.exception(f"Falha ao preparar, fazer upload ou submeter batch de dificuldade ao LLM (run_id: {run_id})")
        raise
