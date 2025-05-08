# scripts/pipeline_tasks.py

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
from collections import defaultdict
import random
from dotenv import load_dotenv
import time
import datetime
import json
import numpy as np

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
import crud.knowledge_unit_origins as crud_knowledge_unit_origins
import crud.knowledge_relationships_intermediate as crud_rel_intermediate
import crud.final_knowledge_units as crud_final_ucs
import crud.final_knowledge_relationships as crud_final_rels
import crud.generated_ucs_raw as crud_generated_ucs_raw
import crud.knowledge_unit_evaluations_batch as crud_knowledge_unit_evaluations_batch
import crud.pipeline_run as crud_runs

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

def _build_community_maps(
        actual_communities_df: pd.DataFrame
) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, str]]:
    """
    Processa o DataFrame da ESTRUTURA das comunidades (ex: communities.parquet) para:
    1. Criar um mapa de human_readable_id (string) para UUID (string) da comunidade.
    2. Enriquecer os registros de ESTRUTURA da comunidade com 'parent_community_id' (UUID do pai).
    3. Criar um mapa de entity_id (string) para o UUID (string) da comunidade folha à qual pertence.
    """
    human_readable_to_uuid_map: Dict[str, str] = {}
    if 'human_readable_id' in actual_communities_df.columns and 'id' in actual_communities_df.columns:
        for hr_id_val, community_uuid_val in zip(actual_communities_df['human_readable_id'],
                                                 actual_communities_df['id']):
            if pd.notna(hr_id_val) and pd.notna(community_uuid_val):
                hr_id_str = str(int(hr_id_val)) if pd.api.types.is_number(hr_id_val) else str(hr_id_val)
                human_readable_to_uuid_map[hr_id_str] = str(community_uuid_val)
            elif pd.notna(community_uuid_val):
                logging.warning(
                    f"Comunidade com ID {community_uuid_val} não possui human_readable_id. Não será mapeável como pai por HR_ID.")
    else:
        logging.error(
            "'human_readable_id' ou 'id' não encontrados em communities.parquet. Mapas podem estar incompletos.")

    processed_community_structure_records: List[Dict[str, Any]] = []
    entity_to_community_map: Dict[str, str] = {}

    expected_db_cols_for_graphrag_community = {
        'id', 'human_readable_id', 'community', 'level', 'parent',
        'children', 'title', 'entity_ids', 'relationship_ids',
        'text_unit_ids', 'period', 'size'
    }

    for community_row_tuple in actual_communities_df.itertuples(index=False):
        community_db_rec: Dict[str, Any] = {}

        for col_name_from_df in actual_communities_df.columns:
            if col_name_from_df in expected_db_cols_for_graphrag_community:
                community_db_rec[col_name_from_df] = getattr(community_row_tuple, col_name_from_df)

        # Validações e defaults para colunas essenciais no DB record
        if not pd.notna(community_db_rec.get('id')):
            logging.error(f"Registro de comunidade em communities.parquet sem 'id' (UUID). Pulando.")
            continue

        community_uuid_str = str(community_db_rec.get('id'))
        community_title_val = community_db_rec.get('title', f'Comunidade Sem Título {community_uuid_str}')

        # Garantir que todas as colunas esperadas para a tabela estejam no dict, mesmo que com None
        for db_col in expected_db_cols_for_graphrag_community:
            if db_col not in community_db_rec:
                # Valores padrão para colunas JSON/list
                if db_col in ['children', 'entity_ids', 'relationship_ids', 'text_unit_ids']:
                    community_db_rec[db_col] = []  # ou None
                else:
                    community_db_rec[db_col] = None

        # Resolver parent_community_id (UUID do pai)
        parent_hr_id_val = community_db_rec.get(
            'parent')  # 'parent' é o human_readable_id do pai no communities.parquet
        parent_hr_id_str = None
        if pd.notna(parent_hr_id_val):
            parent_hr_id_str = str(int(parent_hr_id_val)) if pd.api.types.is_number(parent_hr_id_val) else str(
                parent_hr_id_val)

        if parent_hr_id_str and parent_hr_id_str in human_readable_to_uuid_map:
            community_db_rec['parent_community_id'] = human_readable_to_uuid_map[parent_hr_id_str]
        else:
            community_db_rec['parent_community_id'] = None
            if parent_hr_id_str and parent_hr_id_str != '-1' and parent_hr_id_str not in human_readable_to_uuid_map:
                logging.warning(f"Comunidade (estrutura) '{community_title_val}' (UUID: {community_uuid_str}) "
                                f"tem parent_hr_id '{parent_hr_id_str}' não mapeado para UUID.")

        # Popular entity_to_community_map
        entity_ids_data = community_db_rec.get('entity_ids')
        parsed_entity_ids_list = []
        if entity_ids_data is not None:
            if isinstance(entity_ids_data, str):
                try:
                    loaded_json = json.loads(entity_ids_data)
                    if isinstance(loaded_json, list):
                        parsed_entity_ids_list = loaded_json
                    else:
                        logging.error(
                            f"JSON 'entity_ids' para comunidade {community_uuid_str} não é lista: {loaded_json}.")
                except json.JSONDecodeError:
                    logging.error(
                        f"JSON 'entity_ids' malformado para comunidade {community_uuid_str}: {entity_ids_data}.")
            elif isinstance(entity_ids_data, list):
                parsed_entity_ids_list = entity_ids_data
            elif isinstance(entity_ids_data, np.ndarray):
                parsed_entity_ids_list = entity_ids_data.tolist()
            else:
                logging.warning(
                    f"'entity_ids' para comunidade {community_uuid_str} tem tipo inesperado: {type(entity_ids_data)}.")

            for ent_id_val in parsed_entity_ids_list:
                if pd.notna(ent_id_val):
                    entity_to_community_map[str(ent_id_val)] = community_uuid_str
                else:
                    logging.warning(f"entity_id nulo na lista de entidades da comunidade {community_uuid_str}.")

        processed_community_structure_records.append(community_db_rec)

    return human_readable_to_uuid_map, processed_community_structure_records, entity_to_community_map

def _enrich_entities_with_community_id(
        entities_df: pd.DataFrame,
        entity_to_community_map: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Adiciona 'parent_community_id' (UUID da comunidade folha) a cada registro de entidade.
    """
    processed_entity_records = []
    if entities_df is None or entities_df.empty:
        logging.warning("DataFrame de entidades vazio ou nulo em _enrich_entities_with_community_id.")
        return processed_entity_records

    for entity_row_tuple in entities_df.itertuples(index=False):
        entity_rec = entity_row_tuple._asdict()
        entity_uuid_val = entity_rec.get('id')
        entity_title_val = entity_rec.get('title', 'Título Desconhecido')

        if not pd.notna(entity_uuid_val):
            logging.error(f"Registro de entidade sem ID UUID encontrado: {entity_title_val}. Pulando este registro.")
            continue

        entity_uuid_str = str(entity_uuid_val)
        entity_rec['parent_community_id'] = entity_to_community_map.get(
            entity_uuid_str)
        processed_entity_records.append(entity_rec)
    return processed_entity_records


def task_prepare_origins(run_id: str, **context):
    """
    Tarefa 1: Prepara origens de UC e ingere outputs do GraphRAG (enriquecidos) no banco.
    Garante atomicidade e valida inputs/outputs do GraphRAG.
    """
    logging.info(f"--- TASK: prepare_origins (run_id={run_id}) ---")
    task_successful = False
    final_log_message = "FALHOU"

    with get_session() as db:
        try:
            base_input, _, _, _, _, _, _ = _get_dirs(run_id)
            graphrag_output_dir = base_input

            existing_origins_in_db = crud_knowledge_unit_origins.get_knowledge_unit_origins(db, run_id)
            existing_graphrag_entities_in_db = crud_graphrag_entities.get_entities(db, run_id)

            if existing_origins_in_db and existing_graphrag_entities_in_db:
                logging.info(
                    f"Dados de GraphRAG (entidades) e Origens de UC já existem no banco para run_id={run_id}. Pulando.")
                task_successful = True
                final_log_message = "CONCLUÍDA (Idempotente - dados já no banco)"
            else:
                logging.info("Carregando arquivos Parquet do GraphRAG...")
                actual_communities_df = load_dataframe(graphrag_output_dir, "communities")
                reports_df = load_dataframe(graphrag_output_dir, "community_reports")
                entities_df = load_dataframe(graphrag_output_dir, "entities")
                documents_df = load_dataframe(graphrag_output_dir, "documents")
                relationships_df = load_dataframe(graphrag_output_dir, "relationships")
                text_units_df = load_dataframe(graphrag_output_dir, "text_units")

                if entities_df is None or entities_df.empty:
                    raise ValueError(f"Erro crítico: entities.parquet não encontrado ou vazio.")
                if actual_communities_df is None or actual_communities_df.empty:
                    raise ValueError(f"Erro crítico: 'communities.parquet' (estrutura) não encontrado ou vazio.")
                if reports_df is None or reports_df.empty:
                    logging.warning(
                        f"'community_reports.parquet' não encontrado ou vazio. Origens de 'community_report' podem não ser geradas.")

                logging.info("Processando dados de ESTRUTURA de comunidades...")
                hr_id_to_uuid_map, processed_community_structure_records, entity_to_community_map = _build_community_maps(
                    actual_communities_df)

                logging.info("Enriquecendo dados de entidades com IDs de comunidade pai (folha)...")
                processed_entity_records = _enrich_entities_with_community_id(entities_df, entity_to_community_map)

                logging.info("Ingerindo dados processados do GraphRAG no banco de dados (pendente)...")
                if processed_community_structure_records:
                    crud_graphrag_communities.add_communities(db, run_id, processed_community_structure_records)
                if reports_df is not None and not reports_df.empty:
                    crud_graphrag_community_reports.add_community_reports(db, run_id, reports_df.to_dict('records'))
                if processed_entity_records:
                    crud_graphrag_entities.add_entities(db, run_id, processed_entity_records)

                if documents_df is not None and not documents_df.empty:
                    crud_graphrag_documents.add_documents(db, run_id, documents_df.to_dict('records'))
                if relationships_df is not None and not relationships_df.empty:
                    crud_graphrag_relationships.add_relationships(db, run_id, relationships_df.to_dict('records'))
                if text_units_df is not None and not text_units_df.empty:
                    crud_graphrag_text_units.add_text_units(db, run_id, text_units_df.to_dict('records'))
                logging.info("Dados do GraphRAG adicionados à sessão do banco.")

                # --- Preparar Knowledge Unit Origins ---
                origins_to_save = []
                if (processed_entity_records or (reports_df is not None and not reports_df.empty)):
                    logging.info("Preparando Knowledge Unit Origins...")
                    origins_to_save = prepare_uc_origins(
                        processed_entity_records if processed_entity_records else [],
                        reports_df.to_dict('records') if (reports_df is not None and not reports_df.empty) else [],
                        processed_community_structure_records if processed_community_structure_records else [],
                        hr_id_to_uuid_map
                    )
                else:
                    logging.warning("Nenhuma entidade ou relatório de comunidade para criar origens de UC.")

                if origins_to_save:
                    logging.info(f"Adicionando {len(origins_to_save)} origens de UC ao banco de dados (pendente)...")
                    crud_knowledge_unit_origins.add_knowledge_unit_origins(db, run_id, origins_to_save)
                else:
                    logging.warning("Nenhuma Knowledge Unit Origin foi preparada para salvar.")

                logging.info("Commitando todas as alterações da task...")
                db.commit()
                task_successful = True
                final_log_message = "CONCLUÍDA com sucesso"

        except ValueError as ve:
            logging.error(f"Erro de valor durante task_prepare_origins: {ve}", exc_info=True)
            db.rollback();
            final_log_message = "FALHOU (Erro de Valor)";
            raise
        except Exception as e:
            logging.error(f"Erro geral durante task_prepare_origins: {e}", exc_info=True)
            db.rollback();
            final_log_message = "FALHOU (Erro Geral)";
            raise

    log_level = logging.INFO if task_successful else logging.ERROR
    logging.log(log_level, f"--- TASK prepare_origins {final_log_message} (run_id={run_id}) ---")

def task_submit_uc_generation_batch(run_id: str, **context):
    """Tarefa 2: Prepara JSONL e submete batch de geração UC para um run_id."""
    logging.info(f"--- TASK: submit_uc_generation_batch (run_id={run_id}) ---")
    # Idempotência: se já houver UCs geradas, pular
    with get_session() as db:
        existing = crud_generated_ucs_raw.get_generated_ucs_raw(db, run_id)
        if existing:
            # Idempotência: já existem resultados, retorna um batch_id fictício para fluxo
            fake_id = 'fake_id_for_indepotency'
            logging.info(f"UCs geradas já existem para run_id={run_id}, pulando submissão de batch. Retornando fake batch_id={fake_id}")
            return fake_id
    llm = get_llm_strategy()
    try:
        with get_session() as db:
            records = crud_knowledge_unit_origins.get_knowledge_unit_origins(db, run_id)
        if not records:
            logging.warning("Nenhuma origem para gerar UCs. Pulando submissão.")
            return None
        all_origins = records
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

    # if not batch_id:
    #     logging.warning(f"Nenhum batch_id encontrado para {batch_id_key} via XCom. Pulando.")
    #     # Garante arquivo vazio para downstream
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     # Gera DataFrame vazio com colunas definidas em DEFAULT_OUTPUT_COLUMNS
    #     empty_cols = DEFAULT_OUTPUT_COLUMNS.get(output_filename, [])
    #     save_dataframe(pd.DataFrame(columns=empty_cols), output_dir, output_filename)
    #     return # Considera "sucesso" pois não havia nada a fazer

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
    """
    Tarefa: Define relações REQUIRES e EXPANDS usando Builders e salva
    na tabela intermediária, garantindo atomicidade.

    Levanta erro se os inputs necessários do banco (UCs geradas, dados GraphRAG)
    não forem encontrados ou se ocorrer um erro durante a construção/salvamento
    das relações.
    """
    logging.info(f"--- TASK: define_relationships (run_id={run_id}) ---")
    task_successful = False

    # Abrir a sessão UMA VEZ para toda a task
    with get_session() as db:
        try:
            # 1. Verificar idempotência para relações intermediárias
            existing_rels = crud_rel_intermediate.get_knowledge_relationships_intermediate(db, run_id)
            if existing_rels:
                logging.info(f"Relações intermediárias já existem para run_id={run_id}. Pulando definição.")
                task_successful = True

            else:
                 # 2. Carregar dados de entrada necessários do banco
                logging.info("Carregando dados necessários do banco para definir relações...")
                generated_records = crud_generated_ucs_raw.get_generated_ucs_raw(db, run_id)
                graphrag_rels_records = crud_graphrag_relationships.get_relationships(db, run_id)
                graphrag_ents_records = crud_graphrag_entities.get_entities(db, run_id)

                # Validar se os dados essenciais foram carregados
                if not generated_records:
                    raise ValueError(f"Erro crítico: Nenhum registro de UCs geradas (generated_ucs_raw) encontrado no banco para run_id={run_id}.")

                if not graphrag_rels_records:
                     logging.warning(f"Nenhum registro de relações GraphRAG (graphrag_relationships) encontrado para run_id={run_id}. Relações EXPANDS não serão geradas.")
                if not graphrag_ents_records:
                     logging.warning(f"Nenhum registro de entidades GraphRAG (graphrag_entities) encontrado para run_id={run_id}. Relações EXPANDS podem falhar ou ser incompletas.")


                # 3. Preparar dados para os builders
                generated_df = pd.DataFrame(generated_records)
                relationships_df = pd.DataFrame(graphrag_rels_records) # Pode ser vazio
                entities_df = pd.DataFrame(graphrag_ents_records) # Pode ser vazio

                generated_ucs = generated_df.to_dict('records')
                ctx = {
                    'generated_ucs': generated_ucs,
                    'relationships_df': relationships_df,
                    'entities_df': entities_df
                }

                # 4. Construir relações usando a cadeia de builders
                logging.info("Construindo relações REQUIRES e EXPANDS...")
                builder = RequiresBuilder()
                builder.set_next(ExpandsBuilder())

                all_rels = builder.build([], ctx) # Começa com lista vazia
                logging.info(f"Total de {len(all_rels)} relações intermediárias construídas.")

                # 5. Persistir resultados na tabela intermediária (pendente na transação)
                logging.info("Adicionando relações intermediárias ao banco de dados (pendente)...")
                crud_rel_intermediate.add_knowledge_relationships_intermediate(db, run_id, all_rels or [])

                # 6. Commitar a transação se tudo ocorreu bem
                db.commit()
                task_successful = True

        except Exception as e:
            # 7. Se qualquer erro ocorreu
            logging.error(f"Erro durante task_define_relationships para run_id={run_id}: {e}", exc_info=True)
            logging.info("Fazendo rollback das alterações da transação...")
            db.rollback()
            logging.info("Rollback concluído.")
            raise

    # Log final fora do bloco `with`
    if task_successful:
        log_message = "CONCLUÍDA com sucesso"
        if 'existing_rels' in locals() and existing_rels:
             log_message = "CONCLUÍDA (Idempotente - relações já existiam)"
        logging.info(f"--- TASK define_relationships {log_message} (run_id={run_id}) ---")
    else:
        logging.error(f"--- TASK define_relationships FALHOU (run_id={run_id}) ---")

def task_submit_difficulty_batch(run_id: str, **context):
    """Tarefa: Prepara e submete batch de avaliação de dificuldade para um run_id."""
    logging.info("--- TASK: submit_difficulty_batch ---")
    # Idempotência: se já houver avaliações, pular
    with get_session() as db:
        existing = crud_knowledge_unit_evaluations_batch.get_knowledge_unit_evaluations_batch(db, run_id)
        if existing:
            # Idempotência: já existem avaliações, retorna um batch_id fictício para fluxo
            fake_id = "fake_id_for_indepotency"
            logging.info(f"Avaliações já existem para run_id={run_id}, pulando submissão de batch. Retornando fake batch_id={fake_id}")
            return fake_id
    # Garante que o LLM strategy está configurado
    llm = get_llm_strategy()
    batch_job_id = None
    try:
        # Diretórios de saída e batch
        _, _, _, _, s4, _, batch_dir = _get_dirs(run_id)
        # Carrega UCs geradas via CRUD
        with get_session() as db:
            generated_ucs = crud_generated_ucs_raw.get_generated_ucs_raw(db, run_id)

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
    """
    Tarefa: Combina UCs geradas com avaliações de dificuldade,
    calcula scores finais, salva UCs e Relações finais no banco,
    e atualiza o status do PipelineRun para 'success', garantindo atomicidade.

    Levanta erro se inputs cruciais (UCs geradas) não forem encontrados no banco,
    ou se ocorrer um erro durante o processamento ou salvamento final.
    """
    logging.info(f"--- TASK: finalize_outputs (run_id={run_id}) ---")
    task_successful = False
    final_log_message = "FALHOU" # Default log message

    with get_session() as db:
        try:
            # 1. Verificar idempotência para outputs finais (checando UCs finais)
            # Também verificamos se o status já é 'success', pois não faria sentido rodar de novo.
            run_status = None
            existing_run = crud_runs.get_run(db, run_id)
            if existing_run:
                run_status = existing_run.status

            existing_final_ucs = crud_final_ucs.get_final_knowledge_units(db, run_id)

            if existing_final_ucs and run_status == 'success':
                logging.info(f"Outputs finais já existem e PipelineRun status é 'success' para run_id={run_id}. Pulando finalização.")
                task_successful = True
                final_log_message = "CONCLUÍDA (Idempotente - outputs e status já OK)"

            elif existing_final_ucs and run_status != 'success':
                 logging.warning(f"Outputs finais existem, mas status do PipelineRun ({run_status}) não é 'success' para run_id={run_id}. Tentando apenas atualizar status...")
                 try:
                     # Assumindo que update_run_status agora tem commit=True por padrão
                     crud_runs.update_run_status(db, run_id, status='success')
                     logging.info("Status do PipelineRun atualizado para 'success'.")
                     task_successful = True
                     final_log_message = "CONCLUÍDA (Outputs já existiam, status atualizado)"
                 except Exception as status_err:
                      logging.error(f"Falha ao tentar atualizar status para 'success' (outputs já existiam): {status_err}", exc_info=True)
                      final_log_message = "CONCLUÍDA (Outputs já existiam, FALHA ao atualizar status)"

            else:
                 logging.info("Outputs finais não encontrados ou run status não é 'success'. Procedendo com a finalização...")

                 # 2. Carregar dados de entrada necessários do banco
                 logging.info("Carregando dados intermediários do banco para finalização...")
                 generated_records = crud_generated_ucs_raw.get_generated_ucs_raw(db, run_id)
                 rels_intermed_records = crud_rel_intermediate.get_knowledge_relationships_intermediate(db, run_id)
                 evals_records = crud_knowledge_unit_evaluations_batch.get_knowledge_unit_evaluations_batch(db, run_id)

                 # Validar input essencial: UCs geradas brutas
                 if not generated_records:
                     raise ValueError(f"Erro crítico: Nenhum registro de UCs geradas (generated_ucs_raw) encontrado no banco para run_id={run_id}.")

                 # Logs informativos sobre inputs opcionais
                 if not rels_intermed_records:
                     logging.warning(f"Nenhum registro de relações intermediárias encontrado para run_id={run_id}. A tabela final_knowledge_relationships ficará vazia.")
                     rels_intermed_records = []
                 if not evals_records:
                     logging.warning(f"Nenhuma avaliação de dificuldade bruta encontrada para run_id={run_id}. UCs finais não terão scores.")
                     evals_records = []

                 # 3. Processar avaliações e calcular scores finais
                 final_ucs_list: List[Dict[str, Any]] = []
                 generated_ucs_list = generated_records

                 if evals_records:
                     logging.info("Calculando scores finais de dificuldade...")
                     final_ucs_list, evaluated_count, min_evals_met_count = _calculate_final_difficulty_from_raw(
                         generated_ucs_list, evals_records
                     )
                     logging.info(f"Cálculo de dificuldade concluído: {evaluated_count} UCs com score, {min_evals_met_count} atingiram o mínimo.")
                 else:
                     logging.info("Nenhuma avaliação encontrada, marcando UCs finais como não avaliadas.")
                     for uc in generated_ucs_list:
                         uc_copy = uc.copy()
                         uc_copy["difficulty_score"] = None
                         uc_copy["difficulty_justification"] = "Não avaliado"
                         uc_copy["evaluation_count"] = 0
                         final_ucs_list.append(uc_copy)

                 if not final_ucs_list:
                     raise ValueError("Erro inesperado: Lista final de UCs vazia após processamento.")

                 # 4. Adicionar UCs Finais ao banco (pendente)
                 logging.info(f"Adicionando {len(final_ucs_list)} UCs finais ao banco (pendente)...")
                 crud_final_ucs.add_final_knowledge_units(db, run_id, final_ucs_list)

                 # 5. Adicionar Relações Finais ao banco (pendente)
                 logging.info(f"Adicionando {len(rels_intermed_records)} relações finais ao banco (pendente)...")
                 crud_final_rels.add_final_knowledge_relationships(db, run_id, rels_intermed_records)

                 # 6. Atualizar status do PipelineRun para 'success' (pendente)
                 logging.info("Atualizando status do PipelineRun para 'success' (pendente)...")
                 # Chama a função refatorada com commit=False
                 crud_runs.update_run_status(db, run_id, status='success', commit=False)

                 # 7. Commitar TODAS as alterações (UCs finais, Rels finais, Status do Run)
                 logging.info("Commitando outputs finais e atualização de status do Run...")
                 db.commit()
                 logging.info("Commit atômico bem-sucedido.")
                 task_successful = True
                 final_log_message = "CONCLUÍDA com sucesso"

        except Exception as e:
            # 8. Se qualquer erro ocorreu ANTES do commit final
            logging.error(f"Erro durante task_finalize_outputs para run_id={run_id}: {e}", exc_info=True)
            logging.info("Fazendo rollback das alterações da transação (outputs finais, status)...")
            db.rollback()
            logging.info("Rollback concluído.")
            final_log_message = "FALHOU"
            raise

    # Log final (a mensagem é definida dentro do try/except/if)
    log_level = logging.INFO if task_successful else logging.ERROR
    logging.log(log_level, f"--- TASK finalize_outputs {final_log_message} (run_id={run_id}) ---")
    

def task_process_uc_generation_batch(run_id: str, batch_id: str, **context) -> bool:
    """
    Processa resultados de geração UC para um run_id e batch_id,
    salvando no banco de forma atômica.
    """
    logging.info(f"--- TASK: process_uc_generation_batch (run_id={run_id}, batch_id={batch_id}) ---")
    task_successful = False
    final_log_message = "FALHOU"

    # Se for o ID falso de idempotência, sucesso imediato
    if batch_id == "fake_id_for_indepotency":
         logging.info("Batch ID é 'fake_id_for_indepotency', indicando etapa anterior pulada. Sucesso idempotente.")
         task_successful = True
         final_log_message = "CONCLUÍDA (Idempotente - batch anterior pulado)"
         logging.info(f"--- TASK process_uc_generation_batch {final_log_message} (run_id={run_id}) ---")
         return True


    # Abrir sessão única para a task
    with get_session() as db:
        try:
            # 1. Checar status do batch
            logging.info(f"Verificando status final do batch {batch_id}...")
            status, output_file_id, error_file_id = check_batch_status(batch_id)

            if status != 'completed':
                raise ValueError(f"Batch {batch_id} não está 'completed' (status: {status}) ao iniciar processamento.")
            if not output_file_id:
                 raise ValueError(f"Batch {batch_id} está 'completed' mas não possui output_file_id.")

            logging.info(f"Batch {batch_id} confirmado como 'completed'. output_file_id: {output_file_id}")

            # 2. Chamar process_batch_results passando a sessão db
            logging.info("Iniciando processamento e adição ao banco (pendente)...")

            _, _, s2, _, _, _, _ = _get_dirs(run_id)
            
            processing_ok = process_batch_results(
                batch_id=batch_id,
                output_file_id=output_file_id,
                error_file_id=error_file_id,
                stage_output_dir=s2,
                output_filename=GENERATED_UCS_RAW,
                run_id=run_id,
                db=db
            )

            # 3. Verificar resultado e commitar/rollback
            if processing_ok:
                logging.info("Processamento do batch bem-sucedido. Commitando transação...")
                db.commit()
                task_successful = True
                final_log_message = "CONCLUÍDA com sucesso"
            else:
                logging.error("process_batch_results indicou falha. Fazendo rollback...")
                db.rollback()
                # Levantar erro para falhar a task do Airflow
                raise RuntimeError(f"Falha no processamento do arquivo de resultados do batch {batch_id}.")

        except Exception as e:
            # Captura erros do check_batch_status, process_batch_results ou commit
            logging.error(f"Erro durante task_process_uc_generation_batch para run_id={run_id}, batch_id={batch_id}: {e}", exc_info=True)
            logging.info("Fazendo rollback das alterações da transação (se houver)...")
            # Rollback é seguro mesmo que nada tenha sido adicionado
            db.rollback()
            final_log_message = "FALHOU"
            raise

    # Log final
    log_level = logging.INFO if task_successful else logging.ERROR
    logging.log(log_level, f"--- TASK process_uc_generation_batch {final_log_message} (run_id={run_id}) ---")
    return task_successful


def task_process_difficulty_batch(run_id: str, batch_id: str, **context) -> bool:
    """
    Processa resultados de avaliação de dificuldade para um run_id e batch_id,
    salvando no banco de forma atômica.
    """
    logging.info(f"--- TASK: process_difficulty_batch (run_id={run_id}, batch_id={batch_id}) ---")
    task_successful = False
    final_log_message = "FALHOU"

    # Se for o ID falso de idempotência, sucesso imediato
    if batch_id == "fake_id_for_indepotency":
         logging.info("Batch ID é 'fake_id_for_indepotency', indicando etapa anterior pulada. Sucesso idempotente.")
         task_successful = True
         final_log_message = "CONCLUÍDA (Idempotente - batch anterior pulado)"
         logging.info(f"--- TASK process_difficulty_batch {final_log_message} (run_id={run_id}) ---")
         return True

    with get_session() as db:
        try:
            # 1. Checar status do batch
            logging.info(f"Verificando status final do batch {batch_id}...")
            status, output_file_id, error_file_id = check_batch_status(batch_id)

            if status != 'completed':
                raise ValueError(f"Batch {batch_id} não está 'completed' (status: {status}) ao iniciar processamento.")
            if not output_file_id:
                 raise ValueError(f"Batch {batch_id} está 'completed' mas não possui output_file_id.")

            logging.info(f"Batch {batch_id} confirmado como 'completed'. output_file_id: {output_file_id}")

            # 2. Chamar process_batch_results passando a sessão db
            logging.info("Iniciando processamento e adição ao banco (pendente)...")
            # Precisa do diretório de stage e filename
            _, _, _, _, s4, _, _ = _get_dirs(run_id)
            processing_ok = process_batch_results(
                batch_id=batch_id,
                output_file_id=output_file_id,
                error_file_id=error_file_id,
                stage_output_dir=s4,
                output_filename=UC_EVALUATIONS_RAW,
                run_id=run_id,
                db=db
            )

            # 3. Verificar resultado e commitar/rollback
            if processing_ok:
                logging.info("Processamento do batch bem-sucedido. Commitando transação...")
                db.commit()
                task_successful = True
                final_log_message = "CONCLUÍDA com sucesso"
            else:
                logging.error("process_batch_results indicou falha. Fazendo rollback...")
                db.rollback()
                raise RuntimeError(f"Falha no processamento do arquivo de resultados do batch {batch_id}.")

        except Exception as e:
            logging.error(f"Erro durante task_process_difficulty_batch para run_id={run_id}, batch_id={batch_id}: {e}", exc_info=True)
            logging.info("Fazendo rollback das alterações da transação (se houver)...")
            db.rollback()
            final_log_message = "FALHOU"
            raise

    # Log final
    log_level = logging.INFO if task_successful else logging.ERROR
    logging.log(log_level, f"--- TASK process_difficulty_batch {final_log_message} (run_id={run_id}) ---")
    return task_successful
        