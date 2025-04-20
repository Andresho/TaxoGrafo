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

# --- Funções Auxiliares de Persistência ---
def save_dataframe(df: pd.DataFrame, stage_dir: Path, filename: str):
    """Salva um DataFrame em formato Parquet no diretório do estágio."""
    try:
        stage_dir.mkdir(parents=True, exist_ok=True)
        output_path = stage_dir / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        logging.info(f"Salvo {len(df)} linhas em {output_path}")
    except Exception as e:
        logging.exception(f"Falha ao salvar DataFrame em {stage_dir}/{filename}.parquet")
        raise

def load_dataframe(stage_dir: Path, filename: str) -> Optional[pd.DataFrame]:
    """Carrega um DataFrame Parquet de um diretório de estágio."""
    file_path = stage_dir / f"{filename}.parquet"
    if not file_path.is_file():
        logging.error(f"Arquivo de input não encontrado: {file_path}")
        return None
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Carregado {len(df)} linhas de {file_path}")
        return df
    except Exception as e:
        logging.exception(f"Falha ao carregar DataFrame de {file_path}")
        return None

# --- Funções Auxiliares de Lógica ---

def prepare_uc_origins(
    entities_df: Optional[pd.DataFrame],
    reports_df: Optional[pd.DataFrame]
) -> List[Dict[str, Any]]:
    """Prepara a lista de 'origens' para a geração de UCs."""
    # ... (código inalterado da versão anterior) ...
    uc_origins = []; logging.info("Preparando origens de UC...")
    if entities_df is not None:
        logging.info(f"Processando {len(entities_df)} entidades...")
        req_cols = ['id', 'title', 'description', 'frequency', 'degree', 'type']
        if all(c in entities_df.columns for c in req_cols):
            for r in entities_df.itertuples(index=False):
                freq = int(r.frequency) if pd.notna(r.frequency) else 0
                deg = int(r.degree) if pd.notna(r.degree) else 0
                entity_type = r.type if pd.notna(r.type) else "unknown"
                uc_origins.append({
                    "origin_id": r.id, "origin_type": "entity", "title": r.title,
                    "context": r.description if pd.notna(r.description) else "",
                    "frequency": freq, "degree": deg, "entity_type": entity_type })
        else: logging.warning(f"Colunas faltando em entities.parquet: {[c for c in req_cols if c not in entities_df.columns]}")
    if reports_df is not None:
        logging.info(f"Processando {len(reports_df)} resumos de comunidade...")
        req_cols = ['id', 'community', 'title', 'summary', 'level']
        if all(c in reports_df.columns for c in req_cols):
            for r in reports_df.itertuples(index=False):
                level = int(r.level) if pd.notna(r.level) else 99
                uc_origins.append({
                    "origin_id": r.id, "origin_type": "community_report", "title": r.title,
                    "context": r.summary if pd.notna(r.summary) else "",
                    "community_human_id": r.community, "level": level })
        else: logging.warning(f"Colunas faltando em community_reports.parquet: {[c for c in req_cols if c not in reports_df.columns]}")
    logging.info(f"Total {len(uc_origins)} origens preparadas.")
    return uc_origins

def _get_sort_key(origin: Dict[str, Any]) -> Tuple[int, int]:
    """Calcula a chave de ordenação para uma origem."""
    # ... (código inalterado da versão anterior) ...
    origin_type = origin.get("origin_type"); score = 0; type_priority = 2
    if origin_type == "community_report": level = origin.get("level", 99); score = 10000 - level; type_priority = 1
    elif origin_type == "entity":
        degree = origin.get("degree", 0); freq = origin.get("frequency", 0); entity_type = origin.get("entity_type", "unknown").lower()
        score = degree * 10 + freq
        if entity_type == "person": type_priority = 3
        elif entity_type in ["organization", "geo", "event", "unknown"]: type_priority = 2
        else: type_priority = 1
    return (type_priority, -score)

def _select_origins_for_testing(
    all_origins: List[Dict[str, Any]],
    graphrag_output_dir: Path,
    max_origins: int
) -> List[Dict[str, Any]]:
    """Seleciona origens para teste, focando em conexões."""
    # ... (código inalterado da versão anterior) ...
    logging.warning(f"--- MODO DE TESTE ATIVO (Foco em Conexões, Max: {max_origins}) ---")
    if len(all_origins) <= max_origins: return all_origins
    entity_origins = [o for o in all_origins if o.get("origin_type") == "entity"]
    if not entity_origins: logging.warning("Nenhuma origem 'entity' para teste. Usando as primeiras gerais."); return sorted(all_origins, key=_get_sort_key)[:max_origins]
    entity_origins.sort(key=_get_sort_key); hub_origin = entity_origins[0]; hub_id = hub_origin.get("origin_id")
    logging.info(f"Hub selecionado: ID={hub_id}, Title='{hub_origin.get('title')[:50]}...'")
    neighbor_ids: Set[str] = set()
    relationships_df = load_dataframe(graphrag_output_dir, "relationships"); entities_df = load_dataframe(graphrag_output_dir, "entities")
    if relationships_df is not None and entities_df is not None:
        entity_name_to_id: Dict[str, str] = {}
        if 'title' in entities_df.columns and 'id' in entities_df.columns: entity_name_to_id = pd.Series(entities_df.id.values, index=entities_df.title).to_dict()
        if entity_name_to_id and 'source' in relationships_df.columns and 'target' in relationships_df.columns:
            logging.info(f"Buscando vizinhos do Hub (ID: {hub_id})...")
            for row in relationships_df.itertuples(index=False):
                s_id = entity_name_to_id.get(row.source); t_id = entity_name_to_id.get(row.target)
                if s_id == hub_id and t_id and t_id != hub_id: neighbor_ids.add(t_id)
                elif t_id == hub_id and s_id and s_id != hub_id: neighbor_ids.add(s_id)
        else: logging.warning("Não buscou vizinhos (mapa nome->ID ou colunas).")
    else: logging.warning("Não carregou relationships/entities para buscar vizinhos.")
    logging.info(f"Encontrados {len(neighbor_ids)} vizinhos únicos.")
    final_ids_to_process = {hub_id}; neighbors_to_add = list(neighbor_ids)[:max_origins - 1]; final_ids_to_process.update(neighbors_to_add)
    logging.info(f"Conjunto final teste: {len(final_ids_to_process)} IDs.")
    selected_origins = [o for o in all_origins if o.get("origin_id") in final_ids_to_process]
    selected_origins.sort(key=_get_sort_key)
    return selected_origins

def _prepare_expands_lookups(
    entities_df: Optional[pd.DataFrame],
    generated_ucs: List[Dict[str, Any]]
) -> Tuple[Dict[str, str], Dict[str, Dict[str, List[str]]]]:
    """Prepara os dicionários de lookup necessários para definir relações EXPANDS."""
    # ... (código inalterado da versão anterior) ...
    entity_name_to_id: Dict[str, str] = {}
    if entities_df is not None and 'title' in entities_df.columns and 'id' in entities_df.columns: entity_name_to_id = pd.Series(entities_df.id.values, index=entities_df.title).to_dict(); logging.info(f"Criado mapa nome->ID ({len(entity_name_to_id)} entidades).")
    else: logging.warning("Não foi possível criar mapa nome->ID para EXPANDS.")
    ucs_by_origin_level: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for uc in generated_ucs:
        origin_id=uc.get("origin_id"); bloom_level=uc.get("bloom_level"); uc_id=uc.get("uc_id")
        if origin_id and bloom_level and uc_id and bloom_level in BLOOM_ORDER_MAP: ucs_by_origin_level[origin_id][bloom_level].append(uc_id)
    logging.info(f"Criado mapa UC por origem/nível ({len(ucs_by_origin_level)} origens).")
    return entity_name_to_id, ucs_by_origin_level

def _create_expands_links(
    relationships_df: pd.DataFrame,
    entity_name_to_id: Dict[str, str],
    ucs_by_origin_level: Dict[str, Dict[str, List[str]]]
) -> List[Dict[str, Any]]:
    """Cria as relações EXPANDS com base nas relações do GraphRAG."""
    # ... (código inalterado da versão anterior) ...
    new_expands_rels = []; processed_graphrag_rels = 0; skipped_missing_entity = 0; LEVELS_TO_CONNECT = ["Lembrar", "Entender"]
    logging.info(f"Processando {len(relationships_df)} relações GraphRAG para EXPANDS (Níveis: {LEVELS_TO_CONNECT})...")
    if not ('source' in relationships_df.columns and 'target' in relationships_df.columns): logging.error("'source'/'target' faltando em relationships.parquet."); return []
    for row in relationships_df.itertuples(index=False):
        s_name=row.source; t_name=row.target; weight_val = getattr(row, 'weight', 1.0); rel_weight = float(weight_val) if pd.notna(weight_val) else 1.0; rel_desc = getattr(row, 'description', None); desc_clean = rel_desc if pd.notna(rel_desc) else None
        s_id=entity_name_to_id.get(s_name); t_id=entity_name_to_id.get(t_name)
        if not s_id or not t_id: skipped_missing_entity+=1; continue
        if s_id == t_id: continue
        if s_id in ucs_by_origin_level and t_id in ucs_by_origin_level:
            processed_graphrag_rels += 1
            for bloom_level in LEVELS_TO_CONNECT:
                s_ucs=ucs_by_origin_level[s_id].get(bloom_level,[]); t_ucs=ucs_by_origin_level[t_id].get(bloom_level,[])
                if s_ucs and t_ucs:
                    for s_uc_id in s_ucs:
                        for t_uc_id in t_ucs:
                            rel_ab={"source":s_uc_id,"target":t_uc_id,"type":"EXPANDS","weight":rel_weight,"graphrag_rel_desc":desc_clean}; new_expands_rels.append(rel_ab)
                            rel_ba={"source":t_uc_id,"target":s_uc_id,"type":"EXPANDS","weight":rel_weight,"graphrag_rel_desc":desc_clean}; new_expands_rels.append(rel_ba)
    logging.info(f"Processadas {processed_graphrag_rels} relações GraphRAG com UCs.");
    if skipped_missing_entity > 0: logging.warning(f"{skipped_missing_entity} relações puladas (entidade não mapeada).");
    logging.info(f"Candidatas a {len(new_expands_rels)} novas relações EXPANDS.")
    return new_expands_rels

def _add_relationships_avoiding_duplicates(
    existing_rels: List[Dict[str, Any]],
    new_rels: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Adiciona novas relações a uma lista existente, evitando duplicatas."""
    # ... (código inalterado da versão anterior) ...
    # Se não há novas relações, retorna lista existente
    if not new_rels:
        return existing_rels

    # Prepara estruturas para evitar duplicatas
    updated_rels = list(existing_rels)
    existing_rel_tuples = {(r.get("source"), r.get("target"), r.get("type")) for r in updated_rels}
    added_count = 0

    # Adiciona apenas relações novas
    for rel in new_rels:
        rel_tuple = (rel.get("source"), rel.get("target"), rel.get("type"))
        if rel_tuple not in existing_rel_tuples:
            updated_rels.append(rel)
            existing_rel_tuples.add(rel_tuple)
            added_count += 1

    logging.info(f"{added_count} novas relações adicionadas ({len(new_rels) - added_count} duplicatas evitadas).")
    return updated_rels

def _format_difficulty_prompt(
    batch_ucs_data: List[Dict[str, Any]],
    prompt_template: str
) -> str:
    """Formata o prompt de avaliação de dificuldade para um batch."""
    # ... (código inalterado da versão anterior) ...
    prompt_input_text = "";
    for uc_data in batch_ucs_data: uc_id=uc_data.get('uc_id','N/A'); uc_text=uc_data.get('uc_text','N/A'); prompt_input_text += f"- ID: {uc_id}\n  Texto: {uc_text}\n"
    return prompt_template.replace("{{BATCH_OF_UCS}}", prompt_input_text.strip())

def _calculate_final_difficulty_from_raw(
    generated_ucs: List[Dict[str, Any]],
    raw_evaluations: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Calcula o score final a partir das avaliações brutas do batch."""
    # ... (código inalterado da versão anterior) ...
    logging.info("Calculando scores finais de dificuldade a partir dos resultados do batch...")
    uc_scores: Dict[str, List[int]] = defaultdict(list); uc_justifications: Dict[str, List[str]] = defaultdict(list); uc_evaluation_count: Counter = Counter()
    for evaluation in raw_evaluations:
        uc_id = evaluation.get("uc_id"); score = evaluation.get("difficulty_score"); justification = evaluation.get("justification")
        if uc_id and isinstance(score, int) and 0 <= score <= 100: uc_scores[uc_id].append(score);
        if justification: uc_justifications[uc_id].append(justification); uc_evaluation_count[uc_id] += 1
    updated_ucs_list = []; evaluated_count = 0; min_evals_met_count = 0
    for original_uc in generated_ucs:
        uc = original_uc.copy(); uc_id = uc.get("uc_id"); scores = uc_scores.get(uc_id); eval_count = uc_evaluation_count.get(uc_id, 0)
        if scores: final_score = round(sum(scores)/len(scores)); justification_text = " | ".join(uc_justifications.get(uc_id,["N/A"])); uc["difficulty_score"]=final_score; uc["difficulty_justification"]=justification_text; uc["evaluation_count"]=eval_count; evaluated_count+=1
        if eval_count >= MIN_EVALUATIONS_PER_UC: min_evals_met_count +=1
        else: uc["difficulty_score"]=None; uc["difficulty_justification"]="Não avaliado"; uc["evaluation_count"]=0
        updated_ucs_list.append(uc)
    logging.info(f"  {evaluated_count}/{len(generated_ucs)} UCs receberam score."); logging.info(f"  {min_evals_met_count}/{len(generated_ucs)} UCs atingiram {MIN_EVALUATIONS_PER_UC} avaliações.")
    return updated_ucs_list, evaluated_count, min_evals_met_count

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
        origins_to_process = all_origins
        if MAX_ORIGINS_FOR_TESTING is not None and MAX_ORIGINS_FOR_TESTING > 0:
            origins_to_process = _select_origins_for_testing(all_origins, BASE_INPUT_DIR, MAX_ORIGINS_FOR_TESTING)
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

def check_batch_status(batch_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Consulta status do batch e retorna (status, output_file_id, error_file_id)."""
    if not OPENAI_CLIENT: raise ValueError("OpenAI client não inicializado")
    try:
        batch_job = OPENAI_CLIENT.batches.retrieve(batch_id)
        logging.info(f"Status do Batch {batch_id}: {batch_job.status}")
        return batch_job.status, batch_job.output_file_id, batch_job.error_file_id
    except Exception as e:
        logging.error(f"Erro ao verificar status do batch {batch_id}: {e}")
        # Retornar um status especial para erro de API pode ser útil
        return "API_ERROR", None, None

def process_batch_results(
    batch_id: str,
    output_file_id: str,
    error_file_id: Optional[str],
    stage_output_dir: Path,
    output_filename: str # e.g., "generated_ucs_raw" ou "uc_evaluations_aggregated_raw"
) -> bool:
    """Baixa e processa o arquivo de resultados do batch."""
    if not OPENAI_CLIENT: raise ValueError("OpenAI client não inicializado")
    logging.info(f"Processando resultados do Batch {batch_id} (Output File: {output_file_id})...")
    processed_data = []
    errors_in_batch = 0
    all_ok = True

    try:
        # Logar erros do batch se existirem
        if error_file_id:
             try: error_content = OPENAI_CLIENT.files.content(error_file_id).read().decode('utf-8'); logging.warning(f"Erros individuais no batch {batch_id}:\n{error_content[:1000]}...")
             except Exception as ef: logging.error(f"Não leu arq erro {error_file_id}: {ef}")

        # Baixar e processar resultados
        result_content_bytes = OPENAI_CLIENT.files.content(output_file_id).read()
        result_content = result_content_bytes.decode('utf-8')
        logging.info(f"Arquivo de resultado {output_file_id} baixado.")

        for line in result_content.strip().split('\n'):
            try:
                line_data = json.loads(line)
                custom_id = line_data.get("custom_id", "unknown_custom_id")
                origin_id = "_".join(custom_id.split("_")[2:-1]) if custom_id.startswith("gen_req_") else custom_id # Extrai origin_id ou usa custom_id
                response = line_data.get("response"); error = line_data.get("error")

                if error: errors_in_batch += 1; logging.error(f"Erro batch {custom_id}: {error.get('message')}"); continue
                if not response or response.get("status_code") != 200: logging.warning(f"Request {custom_id} status não OK: {response}"); errors_in_batch += 1; continue

                body = response.get("body", {}); message_content = body.get("choices", [{}])[0].get("message", {}).get("content")
                if not message_content: logging.warning(f"Resposta OK sem conteúdo {custom_id}"); errors_in_batch += 1; continue

                content_cleaned = message_content.strip()
                if content_cleaned.startswith("```json"): content_cleaned = content_cleaned[7:-3].strip()
                elif content_cleaned.startswith("```"): content_cleaned = content_cleaned[3:-3].strip()

                try:
                    inner_data = json.loads(content_cleaned)
                    # Processamento específico da etapa
                    if output_filename == GENERATED_UCS_RAW:
                        units = inner_data.get("generated_units", [])
                        if isinstance(units, list) and len(units) == 6:
                            for unit in units:
                                if isinstance(unit,dict) and "bloom_level" in unit and "uc_text" in unit: unit["uc_id"]=str(uuid.uuid4()); unit["origin_id"]=origin_id; processed_data.append(unit)
                        else: logging.warning(f"JSON interno {custom_id} != 6 UCs"); errors_in_batch += 1
                    elif output_filename == UC_EVALUATIONS_RAW:
                         assessments = inner_data.get("difficulty_assessments", [])
                         if isinstance(assessments, list):
                              for assessment in assessments:
                                   if isinstance(assessment, dict) and 'uc_id' in assessment and 'difficulty_score' in assessment: processed_data.append(assessment)
                                   else: logging.warning(f"Formato assessment inválido para {custom_id}"); errors_in_batch += 1
                         else: logging.warning(f"JSON interno {custom_id} sem lista 'difficulty_assessments'"); errors_in_batch += 1

                except json.JSONDecodeError as e: logging.error(f"Erro JSON decode interno {custom_id}: {e}"); errors_in_batch += 1
            except Exception as e: logging.error(f"Erro processando linha: {e}. Linha: {line[:100]}..."); errors_in_batch += 1

        logging.info(f"Processamento arquivo concluído. {len(processed_data)} registros. {errors_in_batch} erros.")

        if processed_data:
            save_dataframe(pd.DataFrame(processed_data), stage_output_dir, output_filename)
            if errors_in_batch > 0: logging.warning("Processamento concluído, mas com erros individuais no batch.")
            # Considerar sucesso mesmo com erros individuais? Sim.
            all_ok = True
        elif errors_in_batch > 0:
             logging.error("Nenhum dado processado com sucesso do batch.")
             all_ok = False
        else:
             logging.warning("Nenhum dado encontrado no arquivo de resultados do batch.")
             # Considerar sucesso se o arquivo estava vazio mas o batch completou? Sim.
             all_ok = True

    except Exception as e:
        logging.exception(f"Falha ao baixar/processar arquivo {output_file_id} do batch {batch_id}.")
        all_ok = False

    return all_ok


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
            sorted_ucs = sorted(ucs_in_group, key=lambda uc: BLOOM_ORDER_MAP.get(uc.get("bloom_level"), 99))
            for i in range(len(sorted_ucs) - 1):
                 s_uc, t_uc = sorted_ucs[i], sorted_ucs[i+1]; s_idx, t_idx = BLOOM_ORDER_MAP.get(s_uc.get("bloom_level")), BLOOM_ORDER_MAP.get(t_uc.get("bloom_level"))
                 if s_idx is not None and t_idx is not None and t_idx == s_idx + 1: new_requires_rels.append({"source":s_uc.get("uc_id"),"target":t_uc.get("uc_id"),"type":"REQUIRES","origin_id":origin_id})
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


# --- Bloco de Execução Principal (REMOVIDO) ---
# O if __name__ == "__main__": com a orquestração sequencial foi removido.
# A execução agora será gerenciada pelo Airflow (ou outro orquestrador)
# chamando as funções task_* definidas acima.