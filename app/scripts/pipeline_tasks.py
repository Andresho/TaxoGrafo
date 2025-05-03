# scripts/pipeline_tasks.py

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from collections import defaultdict
import random
from dotenv import load_dotenv
import time
import datetime

from scripts.llm_client import get_llm_strategy
from scripts.io_utils import save_dataframe, load_dataframe
from scripts.data_lake import DataLake
from scripts.origins_utils import (
    prepare_uc_origins,
    _get_sort_key,
    _select_origins_for_testing,
    DefaultSelector,
    HubNeighborSelector,
)
from scripts.difficulty_utils import _format_difficulty_prompt, _calculate_final_difficulty_from_raw
from scripts.batch_utils import check_batch_status, process_batch_results
from scripts.rel_utils import _add_relationships_avoiding_duplicates, _create_expands_links, _prepare_expands_lookups
from scripts.difficulty_utils import _format_difficulty_prompt, _calculate_final_difficulty_from_raw
from scripts.rel_builders import RequiresBuilder, ExpandsBuilder
from scripts.constants import (
    MAX_ORIGINS_FOR_TESTING,
    BLOOM_ORDER,
    BLOOM_ORDER_MAP,
    PROMPT_UC_GENERATION_FILE,
    PROMPT_UC_DIFFICULTY_FILE,
    LLM_MODEL,
    LLM_TEMPERATURE_GENERATION,
    LLM_TEMPERATURE_DIFFICULTY,
    DIFFICULTY_BATCH_SIZE,
    MIN_EVALUATIONS_PER_UC,
    BATCH_FILES_DIR,
    AIRFLOW_DATA_DIR,
    BASE_INPUT_DIR,
    PIPELINE_WORK_DIR,
    BATCH_FILES_DIR,
    stage1_dir,
    stage2_output_ucs_dir,
    stage3_dir,
    stage4_input_batch_dir,
    stage4_output_eval_dir,
    stage5_dir,
    GENERATED_UCS_RAW,
    UC_EVALUATIONS_RAW,
    REL_TYPE_REQUIRES,
    REL_TYPE_EXPANDS,
    REL_INTERMEDIATE,
    FINAL_UC_FILE,
    FINAL_REL_FILE,
)

# Ingest Graphrag output Parquet files into the database
# Use a DB session and CRUD modules for each output table
from db import get_session
import crud.graphrag_communities as crud_graphrag_communities
import crud.graphrag_community_reports as crud_graphrag_community_reports
import crud.graphrag_documents as crud_graphrag_documents
import crud.graphrag_entities as crud_graphrag_entities
import crud.graphrag_relationships as crud_graphrag_relationships
import crud.graphrag_text_units as crud_graphrag_text_units

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helpers de Diretórios por run_id ---
def _get_dirs(run_id: str):
    """Retorna tupla de paths: base_input, stage1..5 e batch_dir conforme run_id ou valores padrão."""
    # Base de dados (concatena run_id se fornecido)
    root = Path(AIRFLOW_DATA_DIR) / run_id
    base_input = root / 'output'
    work_dir = root / 'pipeline_workdir'
    batch_dir = work_dir / 'batch_files'
    s1 = work_dir / '1_origins'
    s2 = work_dir / '2_generated_ucs'
    s3 = work_dir / '3_relationships'
    s4 = work_dir / '4_difficulty_evals'
    s5 = work_dir / '5_final_outputs'

    return base_input, s1, s2, s3, s4, s5, batch_dir

DEFAULT_OUTPUT_COLUMNS = {
    GENERATED_UCS_RAW: ["uc_id", "origin_id", "bloom_level", "uc_text"],
    UC_EVALUATIONS_RAW: ["uc_id", "difficulty_score", "justification"]
}

# --- Funções de Tarefa do DAG ---
def task_prepare_origins(run_id: str, **context):
    """Tarefa 1: Prepara e salva uc_origins.parquet para um run_id (ou padrão)."""
    logging.info(f"--- TASK: prepare_origins (run_id={run_id}) ---")
    try:
        base_input, s1, *_ = _get_dirs(run_id)
        entities_df = load_dataframe(base_input, "entities")
        reports_df = load_dataframe(base_input, "community_reports")
        if entities_df is None and reports_df is None:
            raise ValueError("Inputs entities/reports não carregados")
        origins = prepare_uc_origins(entities_df, reports_df)
        if not origins:
            logging.warning("Nenhuma origem preparada.")
            origins = []
        df_origins = pd.DataFrame(origins)
        save_dataframe(df_origins, s1, "uc_origins")

        # Base output directory for this run
        base_input, *_ = _get_dirs(run_id)
        output_dir = base_input
        # Map table names to CRUD functions
        table_crud_map = {
            'communities': crud_graphrag_communities.add_communities,
            'community_reports': crud_graphrag_community_reports.add_community_reports,
            'documents': crud_graphrag_documents.add_documents,
            'entities': crud_graphrag_entities.add_entities,
            'relationships': crud_graphrag_relationships.add_relationships,
            'text_units': crud_graphrag_text_units.add_text_units,
        }
        with get_session() as db:
            for table_name, add_func in table_crud_map.items():
                parquet_path = output_dir / f"{table_name}.parquet"
                if parquet_path.is_file():
                    try:
                        df = pd.read_parquet(parquet_path)
                        records = df.to_dict('records')
                        add_func(db, run_id, records)
                    except Exception:
                        logging.exception(f"Falha ao ingestar '{table_name}' para run_id={run_id}")
    except Exception:
        logging.exception("Falha na task_prepare_origins")
        raise

def task_submit_uc_generation_batch(run_id: str, **context):
    """Tarefa 2: Prepara JSONL e submete batch de geração UC para um run_id."""
    logging.info(f"--- TASK: submit_uc_generation_batch (run_id={run_id}) ---")
    llm = get_llm_strategy()
    batch_job_id = None
    try:
        _, s1, *_ = _get_dirs(run_id)
        origins_df = load_dataframe(s1, "uc_origins")
        if origins_df is None or origins_df.empty:
            logging.warning("Nenhuma origem para gerar UCs. Pulando submissão.")
            return None

        all_origins = origins_df.to_dict('records')
        # Seleção de origens via Strategy Pattern
        if MAX_ORIGINS_FOR_TESTING is not None and MAX_ORIGINS_FOR_TESTING > 0:
            selector = HubNeighborSelector(MAX_ORIGINS_FOR_TESTING, BASE_INPUT_DIR)
        else:
            selector = DefaultSelector(None)
        origins_to_process = selector.select(all_origins)
        if not origins_to_process:
            logging.warning("Nenhuma origem selecionada para processar. Pulando submissão.")
            return None

        try:
            with open(PROMPT_UC_GENERATION_FILE, 'r', encoding='utf-8') as f: prompt_template = f.read()
        except Exception as e: raise ValueError(f"Erro lendo prompt UC Gen: {e}")

        # Prepara registros de batch e salva como JSONL via DataLake
        _, _, _, _, _, _, batch_dir = _get_dirs(run_id)
        batch_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_input_filename = f"uc_generation_batch_{timestamp}.jsonl"
        batch_input_path = batch_dir / batch_input_filename
        records = []
        for i, origin in enumerate(origins_to_process):
            origin_id = origin.get("origin_id")
            req_custom_id = f"gen_req_{origin_id}_{i}"  # ID único por request
            title = origin.get("title", "N/A"); context_text = origin.get("context", "")
            formatted_prompt = (
                prompt_template
                .replace("{{CONCEPT_TITLE}}", title)
                .replace("{{CONTEXT}}", context_text if context_text else "N/A")
            )
            request_body = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": formatted_prompt}
                ],
                "temperature": LLM_TEMPERATURE_GENERATION,
                "response_format": {"type": "json_object"}
            }
            records.append({
                "custom_id": req_custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body
            })
        DataLake.write_jsonl(records, batch_input_path)
        logging.info(f"Arquivo batch de geração criado: {batch_input_path} ({len(records)} requests)")

        # Envia batch via LLM strategy
        logging.info(f"Fazendo upload de {batch_input_path} via LLM strategy...")
        file_id = llm.upload_batch_file(batch_input_path)
        logging.info(f"Upload concluído. File ID: {file_id}")
        logging.info("Criando batch job via LLM strategy...")
        batch_job_id = llm.create_batch_job(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            metadata={'description': 'UC Generation Batch'}
        )
        logging.info(f"Batch job criado. Batch ID: {batch_job_id}")

    except Exception as e:
        logging.exception("Falha na task_submit_uc_generation_batch")
        raise # Falha a tarefa do Airflow
    return batch_job_id # Retorna para XCom

def task_wait_and_process_batch_generic(batch_id_key: str, output_dir: Path, output_filename: str, **context):
    """Tarefa Genérica: Espera e processa resultados de um batch job da OpenAI."""
    ti = context['ti'] # TaskInstance para XComs
    batch_id = ti.xcom_pull(task_ids=f'submit_{batch_id_key}_batch', key='return_value')

    if not batch_id:
        logging.warning(f"Nenhum batch_id encontrado para {batch_id_key} via XCom. Pulando.")
        # Garante arquivo vazio para downstream
        output_dir.mkdir(parents=True, exist_ok=True)
        # Gera DataFrame vazio com colunas definidas em DEFAULT_OUTPUT_COLUMNS
        empty_cols = DEFAULT_OUTPUT_COLUMNS.get(output_filename, [])
        save_dataframe(pd.DataFrame(columns=empty_cols), output_dir, output_filename)
        return # Considera "sucesso" pois não havia nada a fazer

    logging.info(f"--- TASK: wait_and_process_{batch_id_key}_results (Batch ID: {batch_id}) ---")

    polling_interval_seconds = 60; max_polling_attempts = 120; attempts = 0
    while attempts < max_polling_attempts:
        attempts += 1; logging.info(f"Verificando status do batch {batch_id} (Tentativa {attempts})...")
        try:
            status, output_file_id, error_file_id = check_batch_status(batch_id)
            if status == 'completed':
                if output_file_id:
                    if process_batch_results(batch_id, output_file_id, error_file_id, output_dir, output_filename):
                        logging.info(f"Processamento de {batch_id} concluído.")
                        return # Sucesso
                    else: raise ValueError(f"Falha ao processar resultados do batch {batch_id}")
                else: raise ValueError(f"Batch {batch_id} completo mas sem output_file_id")
            elif status in ['failed', 'expired', 'cancelled', 'API_ERROR']:
                raise ValueError(f"Batch job {batch_id} falhou (Status: {status}).")
            else: logging.info(f"Status: {status}. Aguardando {polling_interval_seconds}s..."); time.sleep(polling_interval_seconds)
        except Exception as e: logging.exception(f"Erro no polling/processamento do batch {batch_id}"); raise
    raise TimeoutError(f"Polling para batch {batch_id} excedeu {max_polling_attempts} tentativas.")

# Alias para compatibilidade com DAG: nome sem sufixo _generic
task_wait_and_process_batch = task_wait_and_process_batch_generic


def task_define_relationships(run_id: str, **context):
    """Tarefa: Define relações REQUIRES e EXPANDS usando Builders para um run_id."""
    logging.info(f"--- TASK: define_relationships (run_id={run_id}) ---")
    try:
        _, _, s2, s3, *_ = _get_dirs(run_id)
        # Carrega UCs geradas
        generated_df = load_dataframe(s2, GENERATED_UCS_RAW)
        if generated_df is None or generated_df.empty:
            logging.warning("Nenhuma UC para definir relações.")
            # Salva DataFrame vazio com colunas mínimas
            save_dataframe(pd.DataFrame(columns=["source", "target", "type"]), s3, REL_INTERMEDIATE)
            return
        generated_ucs = generated_df.to_dict('records')
        # Carrega dados para EXPANDS
        relationships_df = load_dataframe(BASE_INPUT_DIR, "relationships")
        entities_df = load_dataframe(BASE_INPUT_DIR, "entities")
        # Contexto para builders
        ctx = {
            'generated_ucs': generated_ucs,
            'relationships_df': relationships_df,
            'entities_df': entities_df
        }
        # Encadeia builders: REQUIRES -> EXPANDS
        builder = RequiresBuilder()
        builder.set_next(ExpandsBuilder())
        all_rels = builder.build([], ctx)
        # Salva resultados
        if all_rels:
            save_dataframe(pd.DataFrame(all_rels), s3, REL_INTERMEDIATE)
        else:
            logging.warning("Nenhuma relação definida.")
            save_dataframe(pd.DataFrame(columns=["source", "target", "type"]), s3, REL_INTERMEDIATE)
    except Exception:
        logging.exception("Falha na task_define_relationships")
        raise

def task_submit_difficulty_batch(run_id: str, **context):
    """Tarefa: Prepara e submete batch de avaliação de dificuldade para um run_id."""
    # ... (lógica como antes, lendo de stage2_output_ucs_dir) ...
    logging.info("--- TASK: submit_difficulty_batch ---")
    # Garante que o LLM strategy está configurado
    llm = get_llm_strategy()
    batch_job_id = None
    try:
        _, _, s2, _, s4, _, batch_dir = _get_dirs(run_id)
        generated_ucs_df = load_dataframe(s2, "generated_ucs_raw")
        if generated_ucs_df is None or generated_ucs_df.empty: logging.warning("Nenhuma UC para avaliar."); return None
        generated_ucs = generated_ucs_df.to_dict('records')
        try:
            with open(PROMPT_UC_DIFFICULTY_FILE, 'r', encoding='utf-8') as f: prompt_template = f.read()
        except Exception as e: raise ValueError(f"Erro lendo prompt diff: {e}")
        # Prepara registros de batch de dificuldade e salva JSONL via DataLake
        batch_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_input_filename = f"uc_difficulty_batch_{timestamp}.jsonl"
        batch_input_path = batch_dir / batch_input_filename
        records = []
        # Agrupa UCs por nivel de Bloom
        ucs_by_bloom: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for uc in generated_ucs:
            level = uc.get("bloom_level")
            if level in BLOOM_ORDER_MAP:
                ucs_by_bloom[level].append(uc)
        # Cria requests por batch
        for bloom_level, ucs_in_level in ucs_by_bloom.items():
            indices = list(range(len(ucs_in_level)))
            random.shuffle(indices)
            for batch_idx, start in enumerate(range(0, len(indices), DIFFICULTY_BATCH_SIZE)):
                batch_indices = indices[start:start + DIFFICULTY_BATCH_SIZE]
                batch_ucs_data = [ucs_in_level[i] for i in batch_indices]
                if not batch_ucs_data:
                    continue
                formatted_prompt = _format_difficulty_prompt(batch_ucs_data, prompt_template)
                custom_batch_id = f"diff_eval_{bloom_level}_{batch_idx}"
                request_body = {
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "..."},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    "temperature": LLM_TEMPERATURE_DIFFICULTY,
                    "response_format": {"type": "json_object"}
                }
                records.append({
                    "custom_id": custom_batch_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_body
                })
        DataLake.write_jsonl(records, batch_input_path)
        logging.info(f"Arquivo batch de dificuldade criado: {batch_input_path} ({len(records)} requests)")
        logging.info(f"Fazendo upload de {batch_input_path} via LLM strategy...")
        file_id = llm.upload_batch_file(batch_input_path)
        logging.info(f"Upload concluído. File ID: {file_id}")
        logging.info("Criando batch job de dificuldade via LLM strategy...")
        batch_job_id = llm.create_batch_job(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            metadata={'description': 'UC Difficulty Evaluation Batch'}
        )
        logging.info(f"Batch job criado. Batch ID: {batch_job_id}")
    except Exception as e: logging.exception("Falha na task_submit_difficulty_batch"); raise
    return batch_job_id


def task_finalize_outputs(run_id: str, **context):
    """Tarefa: Combina UCs com avaliações e salva outputs finais para um run_id."""
    # ... (lógica como antes, lendo de stage2, stage3, stage4, salvando em stage5) ...
    # Adapta _calculate_final_difficulty_from_raw se necessário
    logging.info("--- TASK: finalize_outputs ---")
    try:
        _, _, s2, s3, s4, s5, _ = _get_dirs(run_id)
        ucs_raw_df = load_dataframe(s2, GENERATED_UCS_RAW)
        rels_intermed_df = load_dataframe(s3, REL_INTERMEDIATE)
        evals_raw_df = load_dataframe(s4, UC_EVALUATIONS_RAW)

        if ucs_raw_df is None: raise ValueError("UCs brutas não encontradas.")

        final_ucs_list: List[Dict[str, Any]] = []
        generated_ucs_list = ucs_raw_df.to_dict('records') # Lista original

        if evals_raw_df is not None and not evals_raw_df.empty:
            logging.info("Recalculando scores finais de dificuldade...")
            raw_evals_list = evals_raw_df.to_dict('records')
            # Usa avaliações brutas para calcular scores finais
            final_ucs_list, evaluated_count, min_evals_met_count = _calculate_final_difficulty_from_raw(
                generated_ucs_list, raw_evals_list
            )
        else:
            logging.warning("Avaliações brutas não encontradas. UCs finais sem scores.")
            final_ucs_list = generated_ucs_list # Usa a lista original
            for uc in final_ucs_list: uc["difficulty_score"] = None; uc["difficulty_justification"] = "Não avaliado"; uc["evaluation_count"] = 0

        if final_ucs_list:
            final_ucs_df = pd.DataFrame(final_ucs_list)
            # Garante tipos corretos
            for col in ["difficulty_score", "evaluation_count"]:
                 if col in final_ucs_df.columns:
                      try: final_ucs_df[col] = final_ucs_df[col].astype('Int64')
                      except: final_ucs_df[col] = final_ucs_df[col].fillna(0).astype(int)
                 else: final_ucs_df[col] = 0; final_ucs_df[col] = final_ucs_df[col].astype(int)
            final_ucs_df["difficulty_justification"] = final_ucs_df["difficulty_justification"].fillna("Não avaliado")
            save_dataframe(final_ucs_df, s5, FINAL_UC_FILE)
        else: raise ValueError("Lista final de UCs vazia.")

        if rels_intermed_df is not None:
            save_dataframe(rels_intermed_df, s5, FINAL_REL_FILE)
        else:
            logging.warning("Relações intermediárias não encontradas.")

    except Exception as e:
        logging.exception("Falha na task_finalize_outputs")
        raise
    

def task_process_uc_generation_batch(run_id: str, batch_id: str, **context) -> bool:
    """Processa resultados de geração UC para um run_id e batch_id."""
    logging.info(f"--- TASK: process_uc_generation_batch (run_id={run_id}, batch_id={batch_id}) ---")
    # Define diretórios de saída
    _, _, s2, _, _, _, _ = _get_dirs(run_id)
    # Checa status do batch
    status, output_file_id, error_file_id = check_batch_status(batch_id)
    if status != 'completed':
        raise ValueError(f"Batch {batch_id} not completed (status: {status})")
    # Processa e salva parquet de UCs geradas
    return process_batch_results(
        batch_id,
        output_file_id,
        error_file_id,
        s2,
        GENERATED_UCS_RAW,
    )

def task_process_difficulty_batch(run_id: str, batch_id: str, **context) -> bool:
    """Processa resultados de avaliação de dificuldade para um run_id e batch_id."""
    logging.info(f"--- TASK: process_difficulty_batch (run_id={run_id}, batch_id={batch_id}) ---")
    # Define diretórios de saída de avaliação
    _, _, _, _, s4, _, _ = _get_dirs(run_id)
    # Checa status do batch
    status, output_file_id, error_file_id = check_batch_status(batch_id)
    if status != 'completed':
        raise ValueError(f"Batch {batch_id} not completed (status: {status})")
    # Processa e salva parquet de avaliações brutas
    return process_batch_results(
        batch_id,
        output_file_id,
        error_file_id,
        s4,
        UC_EVALUATIONS_RAW,
    )
        