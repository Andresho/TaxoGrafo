# scripts/pipeline_tasks.py

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Set
import logging
import os
import json
import uuid
from collections import defaultdict, Counter
import random
try:
    from openai import OpenAI  # Importa cliente OpenAI padrão para Batch API
except ImportError:
    OpenAI = None
from dotenv import load_dotenv
import math
import time
import datetime

from scripts.io_utils import save_dataframe, load_dataframe
from scripts.origins_utils import (
    prepare_uc_origins,
    _get_sort_key,
    _select_origins_for_testing,
    DefaultSelector,
    HubNeighborSelector,
)
from scripts.rel_utils import _prepare_expands_lookups, _create_expands_links, _add_relationships_avoiding_duplicates
from scripts.difficulty_utils import _format_difficulty_prompt, _calculate_final_difficulty_from_raw
from scripts.batch_utils import check_batch_status, process_batch_results

# --- Configurações Globais ---
# Carrega variáveis de ambiente ANTES de usá-las
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parâmetros configuráveis (poderiam vir de variáveis de ambiente ou config do Airflow)
MAX_ORIGINS_FOR_TESTING: Optional[int] = os.environ.get('MAX_ORIGINS_FOR_TESTING', None)
if MAX_ORIGINS_FOR_TESTING: MAX_ORIGINS_FOR_TESTING = int(MAX_ORIGINS_FOR_TESTING)

BLOOM_ORDER = ["Lembrar", "Entender", "Aplicar", "Analisar", "Avaliar", "Criar"]
BLOOM_ORDER_MAP = {level: i for i, level in enumerate(BLOOM_ORDER)}
PROMPT_UC_GENERATION_FILE = Path("/opt/airflow/scripts/prompt_uc_generation.txt") # Caminho dentro do container
PROMPT_UC_DIFFICULTY_FILE = Path("/opt/airflow/scripts/prompt_uc_difficulty.txt") # Caminho dentro do container
LLM_MODEL = os.environ.get('LLM_MODEL', "gpt-4o-mini")
LLM_TEMPERATURE_GENERATION = float(os.environ.get('LLM_TEMPERATURE_GENERATION', 0.2))
LLM_TEMPERATURE_DIFFICULTY = float(os.environ.get('LLM_TEMPERATURE_DIFFICULTY', 0.1))
DIFFICULTY_BATCH_SIZE = int(os.environ.get('DIFFICULTY_BATCH_SIZE', 5))
MIN_EVALUATIONS_PER_UC = int(os.environ.get('MIN_EVALUATIONS_PER_UC', 1)) # Simplificado para 1
# BATCH API não usa concorrência configurável do nosso lado
# UC_GENERATION_BATCH_SIZE = 20 # Não relevante para Batch API (tamanho do arquivo é o limite)

# Diretórios (dentro do container Airflow)
AIRFLOW_DATA_DIR = Path("/opt/airflow/data")
BASE_INPUT_DIR = AIRFLOW_DATA_DIR / "graphrag_outputs"
PIPELINE_WORK_DIR = AIRFLOW_DATA_DIR / "pipeline_workdir"
BATCH_FILES_DIR = PIPELINE_WORK_DIR / "batch_files"
stage1_dir = PIPELINE_WORK_DIR / "1_origins"
stage2_output_ucs_dir = PIPELINE_WORK_DIR / "2_generated_ucs"
stage3_dir = PIPELINE_WORK_DIR / "3_relationships"
stage4_input_batch_dir = BATCH_FILES_DIR # Reutiliza para dificuldade
stage4_output_eval_dir = PIPELINE_WORK_DIR / "4_difficulty_evals"
stage5_dir = PIPELINE_WORK_DIR / "5_final_outputs"

# --- Inicialização Cliente OpenAI Padrão ---
OPENAI_CLIENT: Optional[OpenAI] = None
try:
    OPENAI_CLIENT = OpenAI() # Pega API key do env var OPENAI_API_KEY por padrão
    logging.info("Cliente OpenAI padrão inicializado.")
except Exception as e:
    logging.error(f"Falha ao inicializar cliente OpenAI: {e}. Verifique API Key.")

# Constantes para evitar magic strings e configuração centralizada
GENERATED_UCS_RAW = "generated_ucs_raw"
UC_EVALUATIONS_RAW = "uc_evaluations_aggregated_raw"
REL_TYPE_REQUIRES = "REQUIRES"
REL_TYPE_EXPANDS = "EXPANDS"
# DEFAULT_OUTPUT_COLUMNS: para DataFrames vazios de batch
DEFAULT_OUTPUT_COLUMNS = {
    GENERATED_UCS_RAW: ["uc_id", "origin_id", "bloom_level", "uc_text"],
    UC_EVALUATIONS_RAW: ["uc_id", "difficulty_score", "justification"]
}

# Nomes de arquivos intermediários e finais
REL_INTERMEDIATE = "knowledge_relationships_intermediate"
FINAL_UC_FILE = "final_knowledge_units"
FINAL_REL_FILE = "final_knowledge_relationships"

# --- Funções Auxiliares de Lógica ---


# --- Funções de Tarefa do DAG ---

def task_prepare_origins(**context):
    """Tarefa 1: Prepara e salva uc_origins.parquet."""
    logging.info("--- TASK: prepare_origins ---")
    try:
        entities_df = load_dataframe(BASE_INPUT_DIR, "entities")
        reports_df = load_dataframe(BASE_INPUT_DIR, "community_reports")
        if entities_df is None and reports_df is None:
            raise ValueError("Inputs entities/reports não carregados")
        origins = prepare_uc_origins(entities_df, reports_df)
        if not origins:
            logging.warning("Nenhuma origem preparada.")
            origins = [] # Garante lista vazia
        save_dataframe(pd.DataFrame(origins), stage1_dir, "uc_origins")
    except Exception as e:
        logging.exception("Falha na task_prepare_origins")
        raise

def task_submit_uc_generation_batch(**context):
    """Tarefa 2: Prepara JSONL e submete batch de geração UC."""
    logging.info("--- TASK: submit_uc_generation_batch ---")
    if OPENAI_CLIENT is None: raise ValueError("Cliente OpenAI não inicializado")
    batch_job_id = None
    try:
        origins_df = load_dataframe(stage1_dir, "uc_origins")
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

        BATCH_FILES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_input_filename = f"uc_generation_batch_{timestamp}.jsonl"
        batch_input_path = BATCH_FILES_DIR / batch_input_filename
        request_count = 0
        logging.info(f"Criando arquivo de batch: {batch_input_path}")
        with open(batch_input_path, 'w', encoding='utf-8') as f_out:
            for i, origin in enumerate(origins_to_process):
                origin_id = origin.get("origin_id")
                req_custom_id = f"gen_req_{origin_id}_{i}" # ID único por request
                title = origin.get("title", "N/A"); context = origin.get("context", "")
                formatted_prompt = prompt_template.replace("{{CONCEPT_TITLE}}", title).replace("{{CONTEXT}}", context if context else "N/A")
                request_body = {"model": LLM_MODEL, "messages": [{"role":"system", "content":"..."}, {"role":"user", "content":formatted_prompt}], "temperature": LLM_TEMPERATURE_GENERATION, "response_format": {"type": "json_object"}}
                request_line = {"custom_id": req_custom_id, "method": "POST", "url": "/v1/chat/completions", "body": request_body}
                f_out.write(json.dumps(request_line) + '\n')
                request_count += 1
        logging.info(f"Arquivo criado com {request_count} requests.")

        logging.info(f"Fazendo upload de {batch_input_path}...")
        with open(batch_input_path, "rb") as f: batch_input_file = OPENAI_CLIENT.files.create(file=f, purpose="batch")
        logging.info(f"Upload concluído. File ID: {batch_input_file.id}")

        logging.info("Criando batch job na OpenAI...")
        batch_job = OPENAI_CLIENT.batches.create(input_file_id=batch_input_file.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={'description': 'UC Generation Batch'})
        batch_job_id = batch_job.id
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
    if OPENAI_CLIENT is None: raise ValueError("Cliente OpenAI não inicializado")

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


def task_define_relationships(**context):
    """Tarefa: Define relações REQUIRES e EXPANDS."""
    # ... (lógica como antes, lendo de stage2_output_ucs_dir, salvando em stage3_dir) ...
    logging.info("--- TASK: define_relationships ---")
    try:
        generated_ucs_df = load_dataframe(stage2_output_ucs_dir, GENERATED_UCS_RAW)
        if generated_ucs_df is None or generated_ucs_df.empty:
            logging.warning("Nenhuma UC para definir relações.")
            save_dataframe(pd.DataFrame(columns=["source", "target", "type"]), stage3_dir, REL_INTERMEDIATE)
            return
        generated_ucs = generated_ucs_df.to_dict('records')
        all_relationships: List[Dict[str, Any]] = []
        ucs_by_origin: Dict[str, List[Dict]] = defaultdict(list)
        for uc in generated_ucs:
            if uc.get("origin_id"): ucs_by_origin[uc.get("origin_id")].append(uc)
        new_requires_rels: List[Dict[str, Any]] = []
        for origin_id, ucs_in_group in ucs_by_origin.items():
            sorted_ucs = sorted(
                ucs_in_group,
                key=lambda uc: BLOOM_ORDER_MAP.get(uc.get("bloom_level"), 99)
            )
            for i in range(len(sorted_ucs) - 1):
                s_uc, t_uc = sorted_ucs[i], sorted_ucs[i + 1]
                s_idx = BLOOM_ORDER_MAP.get(s_uc.get("bloom_level"))
                t_idx = BLOOM_ORDER_MAP.get(t_uc.get("bloom_level"))
                if s_idx is not None and t_idx is not None and t_idx == s_idx + 1:
                    new_requires_rels.append({
                        "source": s_uc.get("uc_id"),
                        "target": t_uc.get("uc_id"),
                        "type": "REQUIRES",
                        "origin_id": origin_id
                    })
        all_relationships = _add_relationships_avoiding_duplicates(all_relationships, new_requires_rels)
        relationships_df = load_dataframe(BASE_INPUT_DIR, "relationships")
        entities_df = load_dataframe(BASE_INPUT_DIR, "entities")
        if relationships_df is not None and entities_df is not None:
            entity_name_to_id, ucs_by_origin_level = _prepare_expands_lookups(entities_df, generated_ucs)
            if entity_name_to_id: new_expands_rels = _create_expands_links(relationships_df, entity_name_to_id, ucs_by_origin_level); all_relationships = _add_relationships_avoiding_duplicates(all_relationships, new_expands_rels)
            else: logging.warning("Pulando EXPANDS (mapa nome->ID falhou).")
        else: logging.warning("Pulando EXPANDS (inputs não carregados).")
        if all_relationships:
            save_dataframe(pd.DataFrame(all_relationships), stage3_dir, REL_INTERMEDIATE)
        else:
            logging.warning("Nenhuma relação definida.")
            save_dataframe(pd.DataFrame(columns=["source", "target", "type"]), stage3_dir, REL_INTERMEDIATE)
    except Exception as e: logging.exception("Falha na task_define_relationships"); raise

def task_submit_difficulty_batch(**context):
    """Tarefa: Prepara e submete batch de avaliação de dificuldade (1 passada)."""
    # ... (lógica como antes, lendo de stage2_output_ucs_dir) ...
    logging.info("--- TASK: submit_difficulty_batch ---")
    if OPENAI_CLIENT is None: raise ValueError("Cliente OpenAI não inicializado")
    batch_job_id = None
    try:
        generated_ucs_df = load_dataframe(stage2_output_ucs_dir, "generated_ucs_raw")
        if generated_ucs_df is None or generated_ucs_df.empty: logging.warning("Nenhuma UC para avaliar."); return None
        generated_ucs = generated_ucs_df.to_dict('records')
        try:
            with open(PROMPT_UC_DIFFICULTY_FILE, 'r', encoding='utf-8') as f: prompt_template = f.read()
        except Exception as e: raise ValueError(f"Erro lendo prompt diff: {e}")
        BATCH_FILES_DIR.mkdir(parents=True, exist_ok=True); timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'); batch_input_filename = f"uc_difficulty_batch_{timestamp}.jsonl"; batch_input_path = BATCH_FILES_DIR / batch_input_filename; request_count = 0
        ucs_by_bloom: Dict[str, List[Dict]] = defaultdict(list)
        for uc in generated_ucs:
            if uc.get("bloom_level") in BLOOM_ORDER_MAP: ucs_by_bloom[uc.get("bloom_level")].append(uc)
        with open(batch_input_path, 'w', encoding='utf-8') as f_out:
            for bloom_level, ucs_in_level in ucs_by_bloom.items():
                indices = list(range(len(ucs_in_level)))
                random.shuffle(indices)
                for i in range(0, len(indices), DIFFICULTY_BATCH_SIZE):
                    batch_indices = indices[i:i + DIFFICULTY_BATCH_SIZE]
                    batch_ucs_data = [ucs_in_level[idx] for idx in batch_indices]
                    if not batch_ucs_data:
                        continue
                    formatted_prompt = _format_difficulty_prompt(batch_ucs_data, prompt_template)
                    custom_batch_id = f"diff_eval_{bloom_level}_{i // DIFFICULTY_BATCH_SIZE}"
                    request_body = {
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": "..."},
                            {"role": "user", "content": formatted_prompt}
                        ],
                        "temperature": LLM_TEMPERATURE_DIFFICULTY,
                        "response_format": {"type": "json_object"}
                    }
                    request_line = {
                        "custom_id": custom_batch_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": request_body
                    }
                    f_out.write(json.dumps(request_line) + '\n')
                    request_count += 1
        logging.info(f"Arquivo batch diff {batch_input_path} criado ({request_count} requests).")
        logging.info(f"Fazendo upload de {batch_input_path}...")
        with open(batch_input_path, "rb") as f: batch_input_file = OPENAI_CLIENT.files.create(file=f, purpose="batch"); logging.info(f"Upload concluído. File ID: {batch_input_file.id}")
        logging.info("Criando batch job de dificuldade..."); batch_job = OPENAI_CLIENT.batches.create(input_file_id=batch_input_file.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={'description': 'UC Difficulty Evaluation Batch'}); batch_job_id = batch_job.id; logging.info(f"Batch job criado. Batch ID: {batch_job_id}")
    except Exception as e: logging.exception("Falha na task_submit_difficulty_batch"); raise
    return batch_job_id


def task_finalize_outputs(**context):
    """Tarefa: Combina UCs com avaliações (recalculadas) e salva outputs finais."""
    # ... (lógica como antes, lendo de stage2, stage3, stage4, salvando em stage5) ...
    # Adapta _calculate_final_difficulty_from_raw se necessário
    logging.info("--- TASK: finalize_outputs ---")
    try:
        ucs_raw_df = load_dataframe(stage2_output_ucs_dir, GENERATED_UCS_RAW)
        rels_intermed_df = load_dataframe(stage3_dir, REL_INTERMEDIATE)
        evals_raw_df = load_dataframe(stage4_output_eval_dir, UC_EVALUATIONS_RAW)  # Lê avaliações brutas

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
            save_dataframe(final_ucs_df, stage5_dir, FINAL_UC_FILE)
        else: raise ValueError("Lista final de UCs vazia.")

        if rels_intermed_df is not None:
            save_dataframe(rels_intermed_df, stage5_dir, FINAL_REL_FILE)
        else:
            logging.warning("Relações intermediárias não encontradas.")

    except Exception as e:
        logging.exception("Falha na task_finalize_outputs")
        raise
        