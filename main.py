import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any, TypedDict, Tuple, Set
# LangGraph imports removed
import logging
import os
import json
import uuid
from collections import defaultdict, Counter
import random
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from dotenv import load_dotenv
import math
import time

# --- Configurações Globais ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MAX_ORIGINS_FOR_TESTING: Optional[int] = 10
BLOOM_ORDER = ["Lembrar", "Entender", "Aplicar", "Analisar", "Avaliar", "Criar"]
BLOOM_ORDER_MAP = {level: i for i, level in enumerate(BLOOM_ORDER)}
PROMPT_UC_GENERATION_FILE = Path("prompt_uc_generation.txt")
PROMPT_UC_DIFFICULTY_FILE = Path("prompt_uc_difficulty.txt")
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE_GENERATION = 0.2
LLM_TEMPERATURE_DIFFICULTY = 0.1
DIFFICULTY_BATCH_SIZE = 5
MIN_EVALUATIONS_PER_UC = 3
MAX_DIFFICULTY_ITERATIONS = 50
UC_GENERATION_BATCH_SIZE = 20
UC_GENERATION_MAX_CONCURRENCY = 5
BASE_INPUT_DIR = Path("./graphrag_outputs")
PIPELINE_WORK_DIR = Path("./pipeline_workdir")

# Carrega variáveis de ambiente
load_dotenv()

# --- Inicialização Centralizada do Cliente LLM ---
LLM_CLIENT: Optional[BaseChatModel] = None
try:
    LLM_CLIENT = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE_DIFFICULTY)
    logging.info(f"Cliente LLM ({LLM_MODEL}) inicializado globalmente.")
except Exception as e:
    logging.error(f"Falha ao inicializar cliente LLM global: {e}.")

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
    uc_origins = []
    logging.info("Preparando origens de UC...")

    if entities_df is not None:
        logging.info(f"Processando {len(entities_df)} entidades...")
        req_cols = ['id', 'title', 'description', 'frequency', 'degree', 'type']
        if all(c in entities_df.columns for c in req_cols):
            for r in entities_df.itertuples(index=False):
                freq = int(r.frequency) if pd.notna(r.frequency) else 0
                deg = int(r.degree) if pd.notna(r.degree) else 0
                entity_type = r.type if pd.notna(r.type) else "unknown"
                uc_origins.append({
                    "origin_id": r.id,
                    "origin_type": "entity",
                    "title": r.title,
                    "context": r.description if pd.notna(r.description) else "",
                    "frequency": freq,
                    "degree": deg,
                    "entity_type": entity_type
                })
        else:
            missing_cols = [c for c in req_cols if c not in entities_df.columns]
            logging.warning(f"Colunas faltando em entities.parquet: {missing_cols}")

    if reports_df is not None:
        logging.info(f"Processando {len(reports_df)} resumos de comunidade...")
        req_cols = ['id', 'community', 'title', 'summary', 'level']
        if all(c in reports_df.columns for c in req_cols):
            for r in reports_df.itertuples(index=False):
                level = int(r.level) if pd.notna(r.level) else 99
                uc_origins.append({
                    "origin_id": r.id,
                    "origin_type": "community_report",
                    "title": r.title,
                    "context": r.summary if pd.notna(r.summary) else "",
                    "community_human_id": r.community,
                    "level": level
                })
        else:
            missing_cols = [c for c in req_cols if c not in reports_df.columns]
            logging.warning(f"Colunas faltando em community_reports.parquet: {missing_cols}")

    logging.info(f"Total {len(uc_origins)} origens preparadas.")
    return uc_origins

def _get_sort_key(origin: Dict[str, Any]) -> Tuple[int, int]:
    """Calcula a chave de ordenação para uma origem."""
    origin_type = origin.get("origin_type")
    score = 0
    type_priority = 2 # Prioridade média por padrão

    if origin_type == "community_report":
        level = origin.get("level", 99)
        score = 10000 - level
        type_priority = 1
    elif origin_type == "entity":
        degree = origin.get("degree", 0)
        freq = origin.get("frequency", 0)
        entity_type = origin.get("entity_type", "unknown").lower()
        score = degree * 10 + freq
        if entity_type == "person":
            type_priority = 3
        elif entity_type in ["organization", "geo", "event", "unknown"]:
            type_priority = 2
        else:
            type_priority = 1

    # Retorna tipo (menor = maior prio), score (decrescente)
    return (type_priority, -score)

def _select_origins_for_testing(
    all_origins: List[Dict[str, Any]],
    graphrag_output_dir: Path,
    max_origins: int
) -> List[Dict[str, Any]]:
    """Seleciona origens para teste, focando em conexões."""
    logging.warning(f"--- MODO DE TESTE ATIVO (Foco em Conexões, Max: {max_origins}) ---")
    if len(all_origins) <= max_origins:
        return all_origins

    entity_origins = [o for o in all_origins if o.get("origin_type") == "entity"]
    if not entity_origins:
        logging.warning("Nenhuma origem 'entity' para teste. Usando as primeiras gerais.")
        sorted_origins = sorted(all_origins, key=_get_sort_key)
        return sorted_origins[:max_origins]

    entity_origins.sort(key=_get_sort_key)
    hub_origin = entity_origins[0]
    hub_id = hub_origin.get("origin_id")
    logging.info(f"Hub selecionado: ID={hub_id}, Title='{hub_origin.get('title')[:50]}...'")

    neighbor_ids: Set[str] = set()
    relationships_df = load_dataframe(graphrag_output_dir, "relationships")
    entities_df = load_dataframe(graphrag_output_dir, "entities")

    if relationships_df is not None and entities_df is not None:
        entity_name_to_id: Dict[str, str] = {}
        if 'title' in entities_df.columns and 'id' in entities_df.columns:
            entity_name_to_id = pd.Series(entities_df.id.values, index=entities_df.title).to_dict()

        if entity_name_to_id and 'source' in relationships_df.columns and 'target' in relationships_df.columns:
            logging.info(f"Buscando vizinhos do Hub (ID: {hub_id})...")
            for row in relationships_df.itertuples(index=False):
                s_id = entity_name_to_id.get(row.source)
                t_id = entity_name_to_id.get(row.target)
                if s_id == hub_id and t_id and t_id != hub_id:
                    neighbor_ids.add(t_id)
                elif t_id == hub_id and s_id and s_id != hub_id:
                    neighbor_ids.add(s_id)
        else:
            logging.warning("Não buscou vizinhos (mapa nome->ID ou colunas).")
    else:
        logging.warning("Não carregou relationships/entities para buscar vizinhos.")

    logging.info(f"Encontrados {len(neighbor_ids)} vizinhos únicos.")
    final_ids_to_process = {hub_id}
    neighbors_to_add = list(neighbor_ids)[:max_origins - 1]
    final_ids_to_process.update(neighbors_to_add)
    logging.info(f"Conjunto final teste: {len(final_ids_to_process)} IDs.")

    selected_origins = [o for o in all_origins if o.get("origin_id") in final_ids_to_process]
    selected_origins.sort(key=_get_sort_key) # Reordena o subconjunto
    return selected_origins

def _call_llm_with_json_parsing(
    messages: List[BaseMessage],
    expected_keys: Optional[List[str]] = None,
    temperature: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Chama o LLM global, parseia JSON e faz verificações básicas."""
    if LLM_CLIENT is None:
        logging.error("LLM_CLIENT não inicializado. Impossível fazer chamada.")
        return None

    call_kwargs = {}
    if temperature is not None:
        call_kwargs['temperature'] = temperature

    try:
        response = LLM_CLIENT.invoke(messages, **call_kwargs)
        response_content = response.content

        try:
            # Limpa ```json ```
            content_cleaned = response_content.strip()
            if content_cleaned.startswith("```json"):
                content_cleaned = content_cleaned[7:]
                if content_cleaned.endswith("```"):
                    content_cleaned = content_cleaned[:-3]
            elif content_cleaned.startswith("```"):
                 content_cleaned = content_cleaned[3:]
                 if content_cleaned.endswith("```"):
                    content_cleaned = content_cleaned[:-3]

            data = json.loads(content_cleaned.strip())

            if expected_keys:
                if not all(key in data for key in expected_keys):
                    logging.warning(f"Resposta JSON sem chaves {expected_keys}. Recebido: {list(data.keys())}")
            return data

        except json.JSONDecodeError as e:
            logging.error(f"Erro JSON decode: {e}. Resp: {response_content[:100]}...")
            return None
        except Exception as e:
            logging.error(f"Erro parse JSON: {e}")
            return None
    except Exception as e:
        logging.error(f"Erro chamada LLM: {e}")
        return None


def _batch_generate_ucs(
    origins_batch: List[Dict[str, Any]],
    prompt_template: str
) -> List[Dict[str, Any]]:
    """Gera UCs para um batch de origens usando LLM_CLIENT.batch()."""
    if LLM_CLIENT is None:
        logging.error("LLM_CLIENT não inicializado para batch.")
        return []

    all_messages_list: List[List[BaseMessage]] = []
    successful_origins: List[Dict[str, Any]] = []

    for origin in origins_batch:
        origin_id = origin.get("origin_id")
        title = origin.get("title", "N/A")
        context = origin.get("context", "")
        try:
            formatted_prompt = prompt_template.replace("{{CONCEPT_TITLE}}", title)
            formatted_prompt = formatted_prompt.replace("{{CONTEXT}}", context if context else "N/A")
            messages: List[BaseMessage] = [
                SystemMessage(content="Você é um assistente expert em educação que SEMPRE responde em formato JSON válido, conforme instruído."),
                HumanMessage(content=formatted_prompt)
            ]
            all_messages_list.append(messages)
            successful_origins.append(origin)
        except Exception as e:
            logging.error(f"Erro formatando prompt para batch (origem {origin_id}): {e}")

    if not all_messages_list:
        return []

    generated_ucs_list: List[Dict[str, Any]] = []
    try:
        logging.info(f"Enviando batch de {len(all_messages_list)} prompts para LLM...")
        batch_results = LLM_CLIENT.batch(
            all_messages_list,
            config={'temperature': LLM_TEMPERATURE_GENERATION, 'max_concurrency': UC_GENERATION_MAX_CONCURRENCY}
        )
        logging.info(f"Recebidos {len(batch_results)} resultados do batch.")

        for i, result in enumerate(batch_results):
            origin = successful_origins[i]
            origin_id = origin.get("origin_id")
            if not isinstance(result, BaseMessage):
                logging.error(f"  Erro no resultado do batch para origem {origin_id}: {result}")
                continue

            response_content = result.content
            try:
                # Limpeza e parsing JSON
                content_cleaned = response_content.strip()
                if content_cleaned.startswith("```json"):
                    content_cleaned = content_cleaned[7:]
                    if content_cleaned.endswith("```"): content_cleaned = content_cleaned[:-3]
                elif content_cleaned.startswith("```"):
                    content_cleaned = content_cleaned[3:]
                    if content_cleaned.endswith("```"): content_cleaned = content_cleaned[:-3]

                data = json.loads(content_cleaned.strip())
                units = data.get("generated_units", [])

                if isinstance(units, list) and len(units) == 6:
                    logging.debug(f"  Batch Sucesso: 6 UCs para {origin_id}.")
                    for unit in units:
                        if isinstance(unit,dict) and "bloom_level" in unit and "uc_text" in unit:
                            unit["uc_id"] = str(uuid.uuid4())
                            unit["origin_id"] = origin_id
                            generated_ucs_list.append(unit)
                        else:
                            logging.warning(f"  Batch Formato UC inválido {origin_id}")
                else:
                    logging.warning(f"  Batch Resposta JSON != 6 UCs {origin_id}.")
            except json.JSONDecodeError as e:
                logging.error(f"  Batch Erro JSON decode {origin_id}: {e}. Resp: {response_content[:100]}...")
            except Exception as e:
                logging.error(f"  Batch Erro parse JSON {origin_id}: {e}")

    except Exception as e:
        logging.error(f"Erro durante a chamada LLM batch: {e}")

    return generated_ucs_list

def _prepare_expands_lookups(
    entities_df: Optional[pd.DataFrame],
    generated_ucs: List[Dict[str, Any]]
) -> Tuple[Dict[str, str], Dict[str, Dict[str, List[str]]]]:
    """Prepara os dicionários de lookup necessários para definir relações EXPANDS."""
    entity_name_to_id: Dict[str, str] = {}
    if entities_df is not None and 'title' in entities_df.columns and 'id' in entities_df.columns:
        entity_name_to_id = pd.Series(entities_df.id.values, index=entities_df.title).to_dict()
        logging.info(f"Criado mapa nome->ID ({len(entity_name_to_id)} entidades).")
    else:
        logging.warning("Não foi possível criar mapa nome->ID para EXPANDS.")

    ucs_by_origin_level: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for uc in generated_ucs:
        origin_id = uc.get("origin_id")
        bloom_level = uc.get("bloom_level")
        uc_id = uc.get("uc_id")
        if origin_id and bloom_level and uc_id and bloom_level in BLOOM_ORDER_MAP:
            ucs_by_origin_level[origin_id][bloom_level].append(uc_id)
    logging.info(f"Criado mapa UC por origem/nível ({len(ucs_by_origin_level)} origens).")

    return entity_name_to_id, ucs_by_origin_level

def _create_expands_links(
    relationships_df: pd.DataFrame,
    entity_name_to_id: Dict[str, str],
    ucs_by_origin_level: Dict[str, Dict[str, List[str]]]
) -> List[Dict[str, Any]]:
    """Cria as relações EXPANDS com base nas relações do GraphRAG."""
    new_expands_rels = []
    processed_graphrag_rels = 0
    skipped_missing_entity = 0
    LEVELS_TO_CONNECT = ["Lembrar", "Entender"]

    logging.info(f"Processando {len(relationships_df)} relações GraphRAG para EXPANDS (Níveis: {LEVELS_TO_CONNECT})...")
    if not ('source' in relationships_df.columns and 'target' in relationships_df.columns):
        logging.error("'source'/'target' faltando em relationships.parquet.")
        return []

    for row in relationships_df.itertuples(index=False):
        s_name = row.source
        t_name = row.target
        weight_val = getattr(row, 'weight', 1.0)
        rel_weight = float(weight_val) if pd.notna(weight_val) else 1.0
        rel_desc = getattr(row, 'description', None)
        desc_clean = rel_desc if pd.notna(rel_desc) else None

        s_id = entity_name_to_id.get(s_name)
        t_id = entity_name_to_id.get(t_name)

        if not s_id or not t_id:
            skipped_missing_entity += 1
            continue
        if s_id == t_id:
            continue

        # Verifica se UCs existem para ambas origens antes de iterar níveis
        if s_id in ucs_by_origin_level and t_id in ucs_by_origin_level:
            processed_graphrag_rels += 1
            for bloom_level in LEVELS_TO_CONNECT:
                s_ucs = ucs_by_origin_level[s_id].get(bloom_level, [])
                t_ucs = ucs_by_origin_level[t_id].get(bloom_level, [])
                if s_ucs and t_ucs:
                    for s_uc_id in s_ucs:
                        for t_uc_id in t_ucs:
                            rel_ab = {
                                "source": s_uc_id, "target": t_uc_id, "type": "EXPANDS",
                                "weight": rel_weight, "graphrag_rel_desc": desc_clean
                            }
                            new_expands_rels.append(rel_ab)
                            rel_ba = {
                                "source": t_uc_id, "target": s_uc_id, "type": "EXPANDS",
                                "weight": rel_weight, "graphrag_rel_desc": desc_clean
                            }
                            new_expands_rels.append(rel_ba)

    logging.info(f"Processadas {processed_graphrag_rels} relações GraphRAG com UCs.")
    if skipped_missing_entity > 0:
        logging.warning(f"{skipped_missing_entity} relações puladas (entidade não mapeada).")
    logging.info(f"Candidatas a {len(new_expands_rels)} novas relações EXPANDS.")
    return new_expands_rels

def _add_relationships_avoiding_duplicates(
    existing_rels: List[Dict[str, Any]],
    new_rels: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Adiciona novas relações a uma lista existente, evitando duplicatas."""
    if not new_rels:
        return existing_rels

    updated_rels = list(existing_rels)
    existing_rel_tuples = {(r.get("source"), r.get("target"), r.get("type")) for r in updated_rels}
    added_count = 0

    for rel in new_rels:
        rel_tuple = (rel.get("source"), rel.get("target"), rel.get("type"))
        if rel_tuple not in existing_rel_tuples:
            updated_rels.append(rel)
            existing_rel_tuples.add(rel_tuple)
            added_count += 1

    logging.info(f"{added_count} novas relações adicionadas ({len(new_rels) - added_count} duplicatas evitadas).")
    return updated_rels

def _select_difficulty_batch(
    level_uc_ids: List[str],
    uc_evaluation_count: Counter,
    batch_size: int,
    min_evals: int
) -> List[str]:
    """Seleciona o próximo batch de IDs de UC para avaliação."""
    batch_ids: List[str] = []
    ids_below_min = [uid for uid in level_uc_ids if uc_evaluation_count[uid] < min_evals]
    needed = batch_size

    if ids_below_min:
        take_from_below = min(len(ids_below_min), needed)
        batch_ids.extend(random.sample(ids_below_min, take_from_below))
        needed -= take_from_below

    if needed > 0:
        ids_at_or_above_min = [uid for uid in level_uc_ids if uc_evaluation_count[uid] >= min_evals]
        if ids_at_or_above_min:
            ids_at_or_above_min.sort(key=lambda uid: uc_evaluation_count[uid])
            take_from_above = min(len(ids_at_or_above_min), needed)
            batch_ids.extend(ids_at_or_above_min[:take_from_above])
            needed -= take_from_above

    if needed > 0 and ids_below_min:
        can_add_more = [uid for uid in ids_below_min if uid not in batch_ids]
        if not can_add_more and ids_below_min:
            can_add_more = ids_below_min
        if can_add_more:
             take_extra = min(len(can_add_more), needed)
             batch_ids.extend(random.sample(can_add_more, take_extra))

    return batch_ids

def _format_difficulty_prompt(
    batch_ucs_data: List[Dict[str, Any]],
    prompt_template: str
) -> str:
    """Formata o prompt de avaliação de dificuldade para um batch."""
    prompt_input_text = ""
    for uc_data in batch_ucs_data:
        uc_id = uc_data.get('uc_id', 'N/A')
        uc_text = uc_data.get('uc_text', 'N/A')
        prompt_input_text += f"- ID: {uc_id}\n  Texto: {uc_text}\n"
    return prompt_template.replace("{{BATCH_OF_UCS}}", prompt_input_text.strip())

def _call_llm_for_difficulty(
    formatted_prompt: str
) -> Optional[Dict[str, Any]]:
    """Chama o LLM global para avaliar a dificuldade."""
    messages: List[BaseMessage] = [
        SystemMessage(content="Você é um assistente expert em educação que SEMPRE responde em formato JSON válido, conforme instruído."),
        HumanMessage(content=formatted_prompt),
    ]
    return _call_llm_with_json_parsing(messages, expected_keys=["difficulty_assessments"])

def _update_evaluation_results(
    llm_response_data: Optional[Dict[str, Any]],
    batch_ids: List[str],
    uc_scores: Dict[str, List[int]],
    uc_justifications: Dict[str, List[str]],
    uc_evaluation_count: Counter
) -> None:
    """Atualiza os scores, justificativas e contagem com base na resposta do LLM."""
    if not llm_response_data:
        return

    assessments = llm_response_data.get("difficulty_assessments", [])
    if isinstance(assessments, list) and len(assessments) == len(batch_ids):
        logging.debug(f"    Recebidas {len(assessments)} avaliações do LLM para o batch.")
        assessment_map = {a.get("uc_id"): a for a in assessments}

        for uc_id in batch_ids:
            assessment = assessment_map.get(uc_id)
            if assessment:
                score = assessment.get("difficulty_score")
                justification = assessment.get("justification")
                if isinstance(score, int) and 0 <= score <= 100:
                    uc_scores[uc_id].append(score)
                    if justification:
                        uc_justifications[uc_id].append(justification)
                    uc_evaluation_count[uc_id] += 1
                else:
                    logging.warning(f"    Avaliação inválida recebida para {uc_id}: {assessment}")
            else:
                logging.warning(f"    Avaliação faltando para UC ID {uc_id} no batch.")
    else:
        logging.warning(f"    Resposta JSON não continha {len(batch_ids)} avaliações.")

def _calculate_final_difficulty(
    generated_ucs: List[Dict[str, Any]],
    uc_scores: Dict[str, List[int]],
    uc_justifications: Dict[str, List[str]],
    uc_evaluation_count: Counter
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Calcula o score final e atualiza os dicionários de UC."""
    logging.info("Calculando scores finais de dificuldade...")
    evaluated_count = 0
    min_evals_met_count = 0
    updated_ucs_list = []

    for original_uc in generated_ucs:
        uc = original_uc.copy()
        uc_id = uc.get("uc_id")
        scores = uc_scores.get(uc_id)
        eval_count = uc_evaluation_count.get(uc_id, 0)

        if scores:
            final_score = round(sum(scores) / len(scores))
            justification_text = " | ".join(uc_justifications.get(uc_id, ["N/A"]))
            uc["difficulty_score"] = final_score
            uc["difficulty_justification"] = justification_text
            uc["evaluation_count"] = eval_count
            evaluated_count += 1
            if eval_count >= MIN_EVALUATIONS_PER_UC:
                min_evals_met_count += 1
        else:
            uc["difficulty_score"] = None
            uc["difficulty_justification"] = "Não avaliado"
            uc["evaluation_count"] = 0

        updated_ucs_list.append(uc)

    logging.info(f"  {evaluated_count}/{len(generated_ucs)} UCs receberam score.")
    logging.info(f"  {min_evals_met_count}/{len(generated_ucs)} UCs atingiram {MIN_EVALUATIONS_PER_UC} avaliações.")
    return updated_ucs_list, evaluated_count, min_evals_met_count

# --- Definição das Etapas do Pipeline ---

def run_stage_1_prepare_origins(graphrag_dir: Path, output_dir: Path) -> bool:
    """Carrega dados do GraphRAG e prepara/salva as origens de UC."""
    logging.info("--- Iniciando Etapa 1: Preparar Origens ---")
    stage_ok = False
    try:
        entities_df = load_dataframe(graphrag_dir, "entities")
        reports_df = load_dataframe(graphrag_dir, "community_reports")

        if entities_df is None and reports_df is None:
            logging.error("Falha ao carregar entities e community_reports.")
            return False

        origins = prepare_uc_origins(entities_df, reports_df)
        if origins:
            origins_df = pd.DataFrame(origins)
            save_dataframe(origins_df, output_dir, "uc_origins")
            stage_ok = True
        else:
            logging.warning("Nenhuma origem de UC foi preparada.")
            stage_ok = True # OK se vazio
    except Exception as e:
        logging.exception("Erro inesperado na Etapa 1.")
        stage_ok = False
    logging.info(f"--- Etapa 1 Concluída (Sucesso: {stage_ok}) ---")
    return stage_ok

def run_stage_2_generate_ucs(input_dir: Path, output_dir: Path, graphrag_dir: Path) -> bool:
    """Carrega origens, gera UCs (batch/seq) e salva o resultado bruto."""
    logging.info("--- Iniciando Etapa 2: Gerar UCs ---")
    stage_ok = False
    if LLM_CLIENT is None:
        logging.error("LLM não inicializado. Abortando Etapa 2.")
        return False
    try:
        origins_df = load_dataframe(input_dir, "uc_origins")
        if origins_df is None:
            return False

        all_origins = origins_df.to_dict('records')
        if not all_origins:
            logging.warning("Nenhuma origem carregada.")
            return True

        origins_to_process = all_origins
        if MAX_ORIGINS_FOR_TESTING is not None and MAX_ORIGINS_FOR_TESTING > 0:
            # Passa graphrag_dir para a função de seleção poder carregar dados
            origins_to_process = _select_origins_for_testing(
                all_origins, graphrag_dir, MAX_ORIGINS_FOR_TESTING
            )

        logging.info(f"Origens a serem efetivamente processadas: {len(origins_to_process)}")
        if not origins_to_process:
            return True

        try:
            with open(PROMPT_UC_GENERATION_FILE, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except Exception as e:
            logging.error(f"Erro lendo prompt UC Gen: {e}")
            return False

        generated_ucs_list: List[Dict[str, Any]] = []
        supports_batch = hasattr(LLM_CLIENT, 'batch') and isinstance(LLM_CLIENT, ChatOpenAI)

        if supports_batch:
            logging.info(f"Gerando UCs via batch (Tamanho Máx: {UC_GENERATION_BATCH_SIZE})...")
            num_batches = math.ceil(len(origins_to_process) / UC_GENERATION_BATCH_SIZE)
            for i in range(0, len(origins_to_process), UC_GENERATION_BATCH_SIZE):
                batch_num = i // UC_GENERATION_BATCH_SIZE + 1
                batch_origins = origins_to_process[i:i + UC_GENERATION_BATCH_SIZE]
                logging.info(f"Processando batch {batch_num}/{num_batches}...")
                ucs_from_batch = _batch_generate_ucs(batch_origins, prompt_template)
                generated_ucs_list.extend(ucs_from_batch)
        else:
            # Lógica sequencial (usando _call_llm_with_json_parsing)
            logging.warning("LLM não suporta batch ou não é OpenAI. Processamento sequencial.")
            for i, origin in enumerate(origins_to_process):
                 # ... (código sequencial como na versão anterior) ...
                 logging.info(f"Processando origem {i+1}/{len(origins_to_process)} (seq)...")
                 try:
                    fmt_prompt = prompt_template.replace("{{CONCEPT_TITLE}}", origin.get("title", "N/A")).replace("{{CONTEXT}}", origin.get("context", "") or "N/A")
                    msgs: List[BaseMessage] = [ SystemMessage(content="..."), HumanMessage(content=fmt_prompt) ]
                    llm_resp = _call_llm_with_json_parsing(msgs, expected_keys=["generated_units"], temperature=LLM_TEMPERATURE_GENERATION)
                    if llm_resp:
                         units = llm_resp.get("generated_units", [])
                         if isinstance(units, list) and len(units) == 6:
                             for unit in units:
                                 if isinstance(unit,dict) and "bloom_level" in unit and "uc_text" in unit:
                                     unit["uc_id"]=str(uuid.uuid4()); unit["origin_id"]=origin.get("origin_id"); generated_ucs_list.append(unit)
                 except Exception as e: logging.error(f"Erro processando origem {origin.get('origin_id')} (seq): {e}")


        logging.info(f"Geração finalizada. Total UCs geradas: {len(generated_ucs_list)}")
        if generated_ucs_list:
            generated_ucs_df = pd.DataFrame(generated_ucs_list)
            save_dataframe(generated_ucs_df, output_dir, "generated_ucs_raw")
            stage_ok = True
        else:
             logging.warning("Nenhuma UC foi gerada com sucesso.")
             stage_ok = True

    except Exception as e:
        logging.exception("Erro inesperado na Etapa 2.")
        stage_ok = False
    logging.info(f"--- Etapa 2 Concluída (Sucesso: {stage_ok}) ---")
    return stage_ok

def run_stage_3_define_relationships(input_dir: Path, graphrag_dir: Path, output_dir: Path) -> bool:
    """Carrega UCs brutas e dados GraphRAG, define relações e salva."""
    logging.info("--- Iniciando Etapa 3: Definir Relações ---")
    stage_ok = False
    try:
        generated_ucs_df = load_dataframe(input_dir, "generated_ucs_raw")
        if generated_ucs_df is None: return False
        generated_ucs = generated_ucs_df.to_dict('records')
        if not generated_ucs: logging.warning("Nenhuma UC carregada."); return True

        # Define REQUIRES
        all_relationships: List[Dict[str, Any]] = []
        ucs_by_origin: Dict[str, List[Dict]] = defaultdict(list)
        for uc in generated_ucs:
            if uc.get("origin_id"):
                ucs_by_origin[uc.get("origin_id")].append(uc)

        new_requires_rels: List[Dict[str, Any]] = []
        for origin_id, ucs_in_group in ucs_by_origin.items():
            sorted_ucs = sorted(ucs_in_group, key=lambda uc: BLOOM_ORDER_MAP.get(uc.get("bloom_level"), 99))
            for i in range(len(sorted_ucs) - 1):
                s_uc = sorted_ucs[i]
                t_uc = sorted_ucs[i+1]
                s_idx = BLOOM_ORDER_MAP.get(s_uc.get("bloom_level"))
                t_idx = BLOOM_ORDER_MAP.get(t_uc.get("bloom_level"))
                if s_idx is not None and t_idx is not None and t_idx == s_idx + 1:
                    rel = {
                        "source": s_uc.get("uc_id"),
                        "target": t_uc.get("uc_id"),
                        "type": "REQUIRES",
                        "origin_id": origin_id
                    }
                    new_requires_rels.append(rel)
        all_relationships = _add_relationships_avoiding_duplicates(all_relationships, new_requires_rels)

        # Define EXPANDS
        relationships_df = load_dataframe(graphrag_dir, "relationships")
        entities_df = load_dataframe(graphrag_dir, "entities")
        if relationships_df is not None and entities_df is not None:
            entity_name_to_id, ucs_by_origin_level = _prepare_expands_lookups(entities_df, generated_ucs)
            if entity_name_to_id:
                new_expands_rels = _create_expands_links(relationships_df, entity_name_to_id, ucs_by_origin_level)
                all_relationships = _add_relationships_avoiding_duplicates(all_relationships, new_expands_rels)
            else:
                logging.warning("Pulando EXPANDS (mapa nome->ID falhou).")
        else:
            logging.warning("Pulando EXPANDS (relationships/entities não carregados).")

        if all_relationships:
            rels_df = pd.DataFrame(all_relationships)
            save_dataframe(rels_df, output_dir, "knowledge_relationships_intermediate")
            stage_ok = True
        else:
            logging.warning("Nenhuma relação foi definida.")
            stage_ok = True
    except Exception as e:
        logging.exception("Erro inesperado na Etapa 3.")
        stage_ok = False
    logging.info(f"--- Etapa 3 Concluída (Sucesso: {stage_ok}) ---")
    return stage_ok

def run_stage_4_evaluate_difficulty(input_dir: Path, output_dir: Path) -> bool:
    """Carrega UCs, avalia dificuldade (simplificado) e salva agregação."""
    logging.info("--- Iniciando Etapa 4: Avaliar Dificuldade (Simplificado) ---")
    stage_ok = False
    if LLM_CLIENT is None:
        logging.error("LLM não inicializado. Abortando Etapa 4."); return False
    try:
        generated_ucs_df = load_dataframe(input_dir, "generated_ucs_raw")
        if generated_ucs_df is None:
            return False
        generated_ucs = generated_ucs_df.to_dict('records')
        if not generated_ucs:
            logging.warning("Nenhuma UC para avaliar."); return True

        logging.warning("Executando avaliação de dificuldade SIMPLIFICADA (uma passada).")
        try:
            with open(PROMPT_UC_DIFFICULTY_FILE, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except Exception as e:
            logging.error(f"Erro lendo prompt diff: {e}"); return False

        ucs_by_bloom: Dict[str, List[Dict]] = defaultdict(list)
        uc_dict_by_id: Dict[str, Dict] = {}
        for uc in generated_ucs:
            level = uc.get("bloom_level")
            uc_id = uc.get("uc_id")
            if level in BLOOM_ORDER_MAP and uc_id:
                ucs_by_bloom[level].append(uc)
                uc_dict_by_id[uc_id] = uc

        uc_scores: Dict[str, List[int]] = defaultdict(list)
        uc_justifications: Dict[str, List[str]] = defaultdict(list)
        uc_evaluation_count: Counter = Counter()

        for bloom_level, ucs_in_level in ucs_by_bloom.items():
            logging.info(f"Avaliando nível '{bloom_level}' ({len(ucs_in_level)} UCs)...")
            indices = list(range(len(ucs_in_level)))
            random.shuffle(indices)

            for i in range(0, len(indices), DIFFICULTY_BATCH_SIZE):
                batch_indices = indices[i:i + DIFFICULTY_BATCH_SIZE]
                batch_ids = [ucs_in_level[idx]['uc_id'] for idx in batch_indices]
                logging.info(f"  Processando batch {i // DIFFICULTY_BATCH_SIZE + 1}...")

                batch_ucs_data = [ucs_in_level[idx] for idx in batch_indices]
                formatted_prompt = _format_difficulty_prompt(batch_ucs_data, prompt_template)
                llm_response_data = _call_llm_for_difficulty(formatted_prompt)
                _update_evaluation_results(llm_response_data, batch_ids, uc_scores, uc_justifications, uc_evaluation_count)

        aggregated_results = []
        for uc_id, scores in uc_scores.items():
             if scores:
                 avg_score = round(sum(scores) / len(scores))
                 justification = " | ".join(uc_justifications.get(uc_id, ["N/A"]))
                 eval_count = uc_evaluation_count.get(uc_id, 0)
                 aggregated_results.append({
                     "uc_id": uc_id, "final_difficulty_score": avg_score,
                     "final_difficulty_justification": justification, "final_evaluation_count": eval_count
                 })

        if aggregated_results:
            agg_df = pd.DataFrame(aggregated_results)
            save_dataframe(agg_df, output_dir, "uc_evaluations_aggregated")
            stage_ok = True
        else:
            logging.warning("Nenhuma avaliação de dificuldade bem-sucedida.")
            stage_ok = True

    except Exception as e:
        logging.exception("Erro inesperado na Etapa 4.")
        stage_ok = False
    logging.info(f"--- Etapa 4 Concluída (Sucesso: {stage_ok}) ---")
    return stage_ok

def run_stage_5_finalize_outputs(
    input_dir_ucs: Path,
    input_dir_rels: Path,
    input_dir_evals: Path,
    output_dir: Path
) -> bool:
    """Combina UCs brutas com avaliações e salva outputs finais."""
    logging.info("--- Iniciando Etapa 5: Finalizar Outputs ---")
    stage_ok = False
    try:
        ucs_raw_df = load_dataframe(input_dir_ucs, "generated_ucs_raw")
        rels_df = load_dataframe(input_dir_rels, "knowledge_relationships_intermediate")
        evals_df = load_dataframe(input_dir_evals, "uc_evaluations_aggregated")

        if ucs_raw_df is None:
            logging.error("UCs brutas não encontradas."); return False

        final_ucs_df = ucs_raw_df.copy()
        if evals_df is not None and not evals_df.empty:
            logging.info("Juntando UCs com avaliações...")
            final_ucs_df = pd.merge(final_ucs_df, evals_df, on="uc_id", how="left")
        else:
            logging.warning("Avaliações não encontradas/vazias. Adicionando colunas vazias.")
            final_ucs_df["final_difficulty_score"] = pd.NA
            final_ucs_df["final_difficulty_justification"] = "Não avaliado"
            final_ucs_df["final_evaluation_count"] = 0

        # Preencher NaNs e Renomear
        final_ucs_df["final_difficulty_justification"] = final_ucs_df["final_difficulty_justification"].fillna("Não avaliado")
        # Tenta converter scores/counts para Int64 que suporta NA
        for col in ["final_difficulty_score", "final_evaluation_count"]:
             if col in final_ucs_df.columns:
                  try: final_ucs_df[col] = final_ucs_df[col].astype('Int64')
                  except (TypeError, ValueError): final_ucs_df[col] = final_ucs_df[col].fillna(0).astype(int) # Fallback para int 0 se Int64 falhar
             else: final_ucs_df[col] = 0 ; final_ucs_df[col] = final_ucs_df[col].astype(int)


        final_ucs_df = final_ucs_df.rename(columns={
            "final_difficulty_score": "difficulty_score",
            "final_difficulty_justification": "difficulty_justification",
            "final_evaluation_count": "evaluation_count"
        })

        save_dataframe(final_ucs_df, output_dir, "final_knowledge_units")

        if rels_df is not None:
            save_dataframe(rels_df, output_dir, "final_knowledge_relationships")
        else:
            logging.warning("Relações intermediárias não encontradas.")

        stage_ok = True

    except Exception as e:
        logging.exception("Erro inesperado na Etapa 5.")
        stage_ok = False
    logging.info(f"--- Etapa 5 Concluída (Sucesso: {stage_ok}) ---")
    return stage_ok


# --- Orquestração do Pipeline ---
if __name__ == "__main__":
    logging.info(">>> INICIANDO PIPELINE DE GERAÇÃO DE GRAFO DE CONHECIMENTO <<<")

    # Define diretórios
    stage1_dir = PIPELINE_WORK_DIR / "1_origins"
    stage2_dir = PIPELINE_WORK_DIR / "2_generated_ucs"
    stage3_dir = PIPELINE_WORK_DIR / "3_relationships"
    stage4_dir = PIPELINE_WORK_DIR / "4_difficulty_evals"
    stage5_dir = PIPELINE_WORK_DIR / "5_final_outputs"

    # Executa etapas
    success = run_stage_1_prepare_origins(BASE_INPUT_DIR, stage1_dir)

    if success:
        success = run_stage_2_generate_ucs(stage1_dir, stage2_dir, BASE_INPUT_DIR) # Passa dir base para teste

    if success:
        success = run_stage_3_define_relationships(stage2_dir, BASE_INPUT_DIR, stage3_dir)

    if success:
        success = run_stage_4_evaluate_difficulty(stage2_dir, stage4_dir)

    if success:
        success = run_stage_5_finalize_outputs(stage2_dir, stage3_dir, stage4_dir, stage5_dir)

    if success:
        logging.info(">>> PIPELINE CONCLUÍDO COM SUCESSO <<<")
    else:
        logging.error(">>> PIPELINE FALHOU EM UMA DAS ETAPAS <<<")