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
import uuid

from sqlalchemy.orm import Session 
from db import get_session
import models 

import crud.knowledge_unit_origins as crud_knowledge_unit_origins
import crud.generated_ucs_raw as crud_generated_ucs_raw
import crud.knowledge_unit_evaluations_batch as crud_knowledge_unit_evaluations_batch
import crud.knowledge_relationships_intermediate as crud_rel_intermediate
import crud.final_knowledge_units as crud_final_ucs
import crud.final_knowledge_relationships as crud_final_rels
import crud.pipeline_run as crud_runs
import crud.graphrag_communities as crud_graphrag_communities
import crud.graphrag_community_reports as crud_graphrag_community_reports
import crud.graphrag_documents as crud_graphrag_documents
import crud.graphrag_entities as crud_graphrag_entities
import crud.graphrag_relationships as crud_graphrag_relationships
import crud.graphrag_text_units as crud_graphrag_text_units

from scripts.llm_client import get_llm_strategy
from scripts.io_utils import save_dataframe, \
    load_dataframe
from scripts.data_lake import DataLake
from scripts.origins_utils import (
    prepare_uc_origins,
    _get_sort_key,
    _select_origins_for_testing,
    DefaultSelector,
    HubNeighborSelector,
)
from scripts.difficulty_utils import _format_difficulty_prompt, _calculate_final_difficulty_from_raw
from scripts.batch_utils import check_batch_status, \
    process_batch_results
from scripts.rel_utils import _add_relationships_avoiding_duplicates, _create_expands_links, _prepare_expands_lookups
from scripts.rel_builders import RequiresBuilder, ExpandsBuilder
from scripts.difficulty_scheduler import OriginDifficultyScheduler
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
    GENERATED_UCS_RAW,
    UC_EVALUATIONS_RAW,
    REL_TYPE_REQUIRES,
    REL_TYPE_EXPANDS,
    REL_INTERMEDIATE,
    FINAL_UC_FILE,
    FINAL_REL_FILE,
    AIRFLOW_DATA_DIR,
    BASE_INPUT_DIR,
)

load_dotenv()

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

def _build_community_maps(
        actual_communities_df: pd.DataFrame
) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, str]]:
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

        if not pd.notna(community_db_rec.get('id')):
            logging.error(f"Registro de comunidade em communities.parquet sem 'id' (UUID). Pulando.")
            continue

        community_uuid_str = str(community_db_rec.get('id'))
        community_title_val = community_db_rec.get('title', f'Comunidade Sem Título {community_uuid_str}')

        for db_col in expected_db_cols_for_graphrag_community:
            if db_col not in community_db_rec:
                if db_col in ['children', 'entity_ids', 'relationship_ids', 'text_unit_ids']:
                    community_db_rec[db_col] = []
                else:
                    community_db_rec[db_col] = None

        parent_hr_id_val = community_db_rec.get('parent')
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


def task_prepare_origins(run_id: str):
    """
    Task 1: Prepara origens de UC e ingere outputs do GraphRAG (enriquecidos) no banco.
    Esta função agora assume que será chamada pela API e fará suas operações de DB
    em uma sessão fornecida ou criada por ela, e comitará no final se bem-sucedida.
    A idempotência de alto nível (já executou para esta run?) é tratada pela API.
    """
    logging.info(f"--- LOGIC: prepare_origins (run_id={run_id}) ---")

    with get_session() as db:
        try:
            existing_graphrag_entities_in_db = crud_graphrag_entities.get_entities(db, run_id)
            if existing_graphrag_entities_in_db:
                logging.info(
                    f"Dados do GraphRAG já parecem ingeridos para run_id={run_id} (verificado por entidades). Pulando ingestão de Parquets.")

                existing_ku_origins = crud_knowledge_unit_origins.get_knowledge_unit_origins(db, run_id)
                if existing_ku_origins:
                    logging.info(f"Knowledge Unit Origins também já existem para run_id={run_id}. Nada a fazer.")
                    return
                else:
                    logging.info(
                        f"Dados GraphRAG ingeridos, mas Knowledge Unit Origins não. Tentando criá-las a partir dos dados do DB.")
                    pass

            base_input_for_graphrag, _, _, _, _, _, _, _ = _get_dirs(run_id)
            graphrag_output_dir = base_input_for_graphrag

            logging.info(f"Carregando arquivos Parquet do GraphRAG de {graphrag_output_dir}...")
            
            actual_communities_df = load_dataframe(graphrag_output_dir, "communities")
            reports_df = load_dataframe(graphrag_output_dir, "community_reports")
            entities_df_from_parquet = load_dataframe(graphrag_output_dir, "entities")
            documents_df = load_dataframe(graphrag_output_dir, "documents")
            relationships_df = load_dataframe(graphrag_output_dir, "relationships")
            text_units_df = load_dataframe(graphrag_output_dir, "text_units")

            if entities_df_from_parquet is None or entities_df_from_parquet.empty:
                raise ValueError(f"Erro crítico: entities.parquet não encontrado ou vazio em {graphrag_output_dir}.")
            if actual_communities_df is None or actual_communities_df.empty:
                raise ValueError(
                    f"Erro crítico: 'communities.parquet' (estrutura) não encontrado ou vazio em {graphrag_output_dir}.")

            logging.info("Processando dados de ESTRUTURA de comunidades...")
            hr_id_to_uuid_map, processed_community_structure_records, entity_to_community_map = _build_community_maps(
                actual_communities_df
            )

            logging.info("Enriquecendo dados de entidades com IDs de comunidade pai (folha)...")
            processed_entity_records = _enrich_entities_with_community_id(entities_df_from_parquet,
                                                                          entity_to_community_map)

            logging.info("Ingerindo dados processados do GraphRAG no banco de dados...")
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

            logging.info("Dados do GraphRAG adicionados à sessão do banco (pendente de commit).")

            origins_to_save = []
            if not crud_knowledge_unit_origins.get_knowledge_unit_origins(db, run_id):
                logging.info("Preparando Knowledge Unit Origins...")
                origins_to_save = prepare_uc_origins(
                    processed_entity_records if processed_entity_records else [],
                    reports_df.to_dict('records') if (reports_df is not None and not reports_df.empty) else [],
                    processed_community_structure_records if processed_community_structure_records else [],
                    hr_id_to_uuid_map
                )
                if origins_to_save:
                    logging.info(
                        f"Adicionando {len(origins_to_save)} Knowledge Unit Origins ao banco (pendente de commit)...")
                    crud_knowledge_unit_origins.add_knowledge_unit_origins(db, run_id, origins_to_save)
                else:
                    logging.warning("Nenhuma Knowledge Unit Origin foi preparada para salvar.")
            else:
                logging.info(f"Knowledge Unit Origins já existem para run_id={run_id}. Pulando criação.")

            logging.info("Commitando todas as alterações da task task_prepare_origins...")
            db.commit()
            logging.info(f"--- LOGIC: prepare_origins CONCLUÍDA com sucesso (run_id={run_id}) ---")

        except ValueError as ve:
            logging.error(f"Erro de valor durante task_prepare_origins: {ve}", exc_info=True)
            db.rollback()
            logging.error(f"--- LOGIC: prepare_origins FALHOU (Erro de Valor) (run_id={run_id}) ---")
            raise
        except Exception as e:
            logging.error(f"Erro geral durante task_prepare_origins: {e}", exc_info=True)
            db.rollback()
            logging.error(f"--- LOGIC: prepare_origins FALHOU (Erro Geral) (run_id={run_id}) ---")
            raise


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

def task_process_uc_generation_batch(run_id: str, llm_batch_id: str,
                                     db_session_from_api: Optional[Session] = None) -> bool:
    """
    Processa resultados de geração UC para um run_id e llm_batch_id.
    Usa db_session_from_api se fornecida, senão cria uma nova.
    NÃO faz commit/rollback se db_session_from_api for usada.
    Retorna True para sucesso, False para falha.
    """
    logging.info(f"--- LOGIC: process_uc_generation_batch (run_id={run_id}, llm_batch_id={llm_batch_id}) ---")

    def _core_logic(db: Session) -> bool:
        try:
            llm_status, output_file_id, error_file_id = check_batch_status(llm_batch_id)

            if llm_status != 'completed':
                logging.error(
                    f"LLM Batch {llm_batch_id} não está 'completed' (status atual: {llm_status}) ao tentar processar resultados.")
                return False
            if not output_file_id:
                logging.error(f"LLM Batch {llm_batch_id} está 'completed' mas não possui output_file_id.")
                return False

            logging.info(f"LLM Batch {llm_batch_id} confirmado como 'completed'. output_file_id: {output_file_id}")

            _, _, _, s1_dir, s2_dir_for_run, s3_dir, s4_dir, s5_dir = _get_dirs(run_id)

            processing_ok = process_batch_results(
                batch_id=llm_batch_id,
                output_file_id=output_file_id,
                error_file_id=error_file_id,
                stage_output_dir=s2_dir_for_run,
                output_filename=GENERATED_UCS_RAW,
                run_id=run_id,
                db=db
            )

            if processing_ok:
                logging.info(f"Processamento dos resultados do LLM Batch {llm_batch_id} (geração UC) bem-sucedido.")
                return True
            else:
                logging.error(
                    f"Falha no processamento interno dos resultados do LLM Batch {llm_batch_id} (geração UC).")
                return False
        except Exception as e:
            logging.exception(
                f"Erro crítico durante process_uc_generation_batch (run_id={run_id}, llm_batch_id={llm_batch_id})")
            return False

    if db_session_from_api:
        return _core_logic(db_session_from_api)
    else:
        with get_session() as db:
            success = _core_logic(db)
            if success:
                db.commit()
            else:
                db.rollback()
            return success


def task_define_relationships(run_id: str):
    """
    Define relações REQUIRES e EXPANDS e salva na tabela intermediária.
    A API que chama esta função gerencia a idempotência e a transação.
    """
    logging.info(f"--- LOGIC: define_relationships (run_id={run_id}) ---")

    with get_session() as db:
        try:
            existing_rels = crud_rel_intermediate.get_knowledge_relationships_intermediate(db, run_id)
            if existing_rels:
                logging.info(f"Relações intermediárias já existem no DB para run_id={run_id}. Pulando definição.")
                logging.info(f"--- LOGIC: define_relationships CONCLUÍDA (Idempotente) (run_id={run_id}) ---")
                return

            logging.info("Carregando dados necessários do banco para definir relações...")
            generated_ucs_records = crud_generated_ucs_raw.get_generated_ucs_raw(db, run_id)
            graphrag_rels_records = crud_graphrag_relationships.get_relationships(db, run_id)
            graphrag_ents_records = crud_graphrag_entities.get_entities(db, run_id)

            if not generated_ucs_records:
                raise ValueError(
                    f"Erro crítico: Nenhum registro de UCs geradas (generated_ucs_raw) encontrado no banco para run_id={run_id}.")

            if not graphrag_rels_records:
                logging.warning(
                    f"Nenhum registro de relações GraphRAG (graphrag_relationships) encontrado para run_id={run_id}. Relações EXPANDS podem não ser geradas ou ser limitadas.")
            if not graphrag_ents_records:
                logging.warning(
                    f"Nenhum registro de entidades GraphRAG (graphrag_entities) encontrado para run_id={run_id}. Relações EXPANDS podem ser afetadas.")

            generated_ucs_df = pd.DataFrame(generated_ucs_records)
            relationships_df_graphrag = pd.DataFrame(graphrag_rels_records if graphrag_rels_records else [])
            entities_df_graphrag = pd.DataFrame(graphrag_ents_records if graphrag_ents_records else [])

            context_for_builders = {
                'generated_ucs': generated_ucs_df.to_dict('records'),
                'relationships_df': relationships_df_graphrag,
                'entities_df': entities_df_graphrag
            }

            logging.info("Construindo relações REQUIRES e EXPANDS...")
            builder = RequiresBuilder()
            builder.set_next(ExpandsBuilder())

            all_intermediate_rels = builder.build([], context_for_builders)
            logging.info(f"Total de {len(all_intermediate_rels)} relações intermediárias construídas.")

            if all_intermediate_rels:
                crud_rel_intermediate.add_knowledge_relationships_intermediate(db, run_id, all_intermediate_rels)
                logging.info("Relações intermediárias adicionadas à sessão do banco (pendente de commit).")
            else:
                logging.warning("Nenhuma relação intermediária foi construída.")

            db.commit()
            logging.info(f"--- LOGIC: define_relationships CONCLUÍDA com sucesso (run_id={run_id}) ---")

        except ValueError as ve:
            logging.error(f"Erro de valor durante define_relationships: {ve}", exc_info=True)
            db.rollback()
            logging.error(f"--- LOGIC: define_relationships FALHOU (Erro de Valor) (run_id={run_id}) ---")
            raise
        except Exception as e:
            logging.error(f"Erro geral durante define_relationships: {e}", exc_info=True)
            db.rollback()
            logging.error(f"--- LOGIC: define_relationships FALHOU (Erro Geral) (run_id={run_id}) ---")
            raise


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
    _, work_dir_for_run, batch_files_dir_for_run, _, _, _, _, _ = _get_dirs(run_id)
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


def task_process_difficulty_batch(run_id: str, llm_batch_id: str,
                                  db_session_from_api: Optional[Session] = None) -> bool:
    """
    Processa resultados de avaliação de dificuldade para um run_id e llm_batch_id.
    Usa db_session_from_api se fornecida. NÃO faz commit/rollback se db_session_from_api for usada.
    Retorna True para sucesso, False para falha.
    """
    logging.info(f"--- LOGIC: process_difficulty_batch (run_id={run_id}, llm_batch_id={llm_batch_id}) ---")

    def _core_logic(db: Session) -> bool:
        try:
            llm_status, output_file_id, error_file_id = check_batch_status(llm_batch_id)
            if llm_status != 'completed':
                logging.error(f"LLM Batch {llm_batch_id} (dificuldade) não está 'completed' (status: {llm_status}).")
                return False
            if not output_file_id:
                logging.error(f"LLM Batch {llm_batch_id} (dificuldade) está 'completed' mas sem output_file_id.")
                return False

            logging.info(
                f"LLM Batch {llm_batch_id} (dificuldade) confirmado 'completed'. output_file_id: {output_file_id}")

            _, _, _, _, s4_dir_for_run, _, _, _ = _get_dirs(run_id)

            processing_ok = process_batch_results(
                batch_id=llm_batch_id,
                output_file_id=output_file_id,
                error_file_id=error_file_id,
                stage_output_dir=s4_dir_for_run,
                output_filename=UC_EVALUATIONS_RAW,
                run_id=run_id,
                db=db
            )

            if processing_ok:
                logging.info(f"Processamento dos resultados do LLM Batch {llm_batch_id} (dificuldade) bem-sucedido.")
                return True
            else:
                logging.error(
                    f"Falha no processamento interno dos resultados do LLM Batch {llm_batch_id} (dificuldade).")
                return False
        except Exception as e:
            logging.exception(
                f"Erro crítico durante process_difficulty_batch (run_id={run_id}, llm_batch_id={llm_batch_id})")
            return False

    if db_session_from_api:
        return _core_logic(db_session_from_api)
    else:
        with get_session() as db:
            success = _core_logic(db)
            if success:
                db.commit()
            else:
                db.rollback()
            return success


def task_finalize_outputs(run_id: str):
    """
    Combina UCs geradas com avaliações de dificuldade, calcula scores finais,
    salva UCs e Relações finais no banco, e atualiza o status do PipelineRun.
    A API que chama esta função gerencia a idempotência de alto nível e a transação principal.
    """
    logging.info(f"--- LOGIC: finalize_outputs (run_id={run_id}) ---")

    with get_session() as db:
        try:
            existing_final_ucs = crud_final_ucs.get_final_knowledge_units(db, run_id)
            existing_final_rels = crud_final_rels.get_final_knowledge_relationships(db, run_id)
            db_run = crud_runs.get_run(db, run_id)

            if not db_run:
                logging.error(f"PipelineRun {run_id} não encontrado no DB ao tentar finalizar outputs.")
                raise ValueError(f"PipelineRun {run_id} não existe.")

            if existing_final_ucs and existing_final_rels and db_run.status == 'success':
                logging.info(
                    f"Outputs finais já existem e PipelineRun status é 'success' para run_id={run_id}. Pulando finalização.")
                logging.info(f"--- LOGIC: finalize_outputs CONCLUÍDA (Idempotente) (run_id={run_id}) ---")
                return

            if existing_final_ucs and existing_final_rels and db_run.status != 'success':
                logging.warning(
                    f"Outputs finais existem para run_id={run_id}, mas run status é '{db_run.status}'. Tentando apenas atualizar status do run para 'success'.")
                crud_runs.update_run_status(db, run_id, status='success')
                logging.info(f"PipelineRun {run_id} status atualizado para 'success'.")
                logging.info(f"--- LOGIC: finalize_outputs CONCLUÍDA (Status do Run atualizado) (run_id={run_id}) ---")
                return

            logging.info("Carregando dados intermediários do banco para finalização...")
            generated_ucs_raw_list = crud_generated_ucs_raw.get_generated_ucs_raw(db, run_id)
            rels_intermed_list = crud_rel_intermediate.get_knowledge_relationships_intermediate(db, run_id)
            evals_raw_list = crud_knowledge_unit_evaluations_batch.get_knowledge_unit_evaluations_batch(db, run_id)

            if not generated_ucs_raw_list:
                raise ValueError(
                    f"Erro crítico: Nenhum registro de UCs geradas (generated_ucs_raw) encontrado no banco para run_id={run_id} ao finalizar.")

            if not rels_intermed_list:
                logging.warning(
                    f"Nenhum registro de relações intermediárias encontrado para run_id={run_id}. Tabela final_knowledge_relationships ficará vazia.")
                rels_intermed_list = []
            if not evals_raw_list:
                logging.warning(
                    f"Nenhuma avaliação de dificuldade bruta (batch) encontrada para run_id={run_id}. UCs finais não terão scores de dificuldade calculados nesta etapa.")
                evals_raw_list = []

            final_ucs_to_save: List[Dict[str, Any]] = []
            if evals_raw_list:
                logging.info("Calculando scores finais de dificuldade a partir das avaliações do batch...")
                final_ucs_to_save, evaluated_count, min_evals_met_count = _calculate_final_difficulty_from_raw(
                    generated_ucs_raw_list, evals_raw_list
                )
                logging.info(
                    f"Cálculo de dificuldade concluído: {evaluated_count} UCs com score, {min_evals_met_count} atingiram o mínimo de avaliações.")
            else:
                logging.info("Nenhuma avaliação de dificuldade encontrada. UCs finais não terão scores de dificuldade.")
                for uc_raw_dict in generated_ucs_raw_list:
                    uc_final_dict = uc_raw_dict.copy()
                    uc_final_dict["difficulty_score"] = None
                    uc_final_dict["difficulty_justification"] = "Não avaliado"
                    uc_final_dict["evaluation_count"] = 0
                    final_ucs_to_save.append(uc_final_dict)

            if not final_ucs_to_save and generated_ucs_raw_list:
                raise ValueError(
                    f"Erro inesperado: Lista final de UCs (final_ucs_to_save) vazia após processamento de dificuldade, mas UCs raw existiam para run_id={run_id}.")
            elif not final_ucs_to_save and not generated_ucs_raw_list:
                logging.warning(
                    f"Nenhuma UC gerada (raw) e, portanto, nenhuma UC final para salvar para run_id={run_id}.")

            if final_ucs_to_save:
                crud_final_ucs.add_final_knowledge_units(db, run_id, final_ucs_to_save)
                logging.info(f"{len(final_ucs_to_save)} UCs finais adicionadas à sessão do banco.")

            if rels_intermed_list:
                crud_final_rels.add_final_knowledge_relationships(db, run_id, rels_intermed_list)
                logging.info(
                    f"{len(rels_intermed_list)} relações finais (baseadas nas intermediárias) adicionadas à sessão do banco.")

            crud_runs.update_run_status(db, run_id, status='success')
            logging.info(f"Status do PipelineRun {run_id} definido para 'success' na sessão.")

            db.commit()
            logging.info("Commit atômico de outputs finais e status do Run bem-sucedido.")
            logging.info(f"--- LOGIC: finalize_outputs CONCLUÍDA com sucesso (run_id={run_id}) ---")

        except ValueError as ve:
            logging.error(f"Erro de valor durante finalize_outputs: {ve}", exc_info=True)
            db.rollback()
            logging.error(f"--- LOGIC: finalize_outputs FALHOU (Erro de Valor) (run_id={run_id}) ---")

            try:
                crud_runs.update_run_status(db, run_id, status='finalize_failed'); db.commit()
            except:
                logging.error(f"Falha ao tentar atualizar status do run {run_id} para 'finalize_failed'.")
            raise
        except Exception as e:
            logging.error(f"Erro geral durante finalize_outputs: {e}", exc_info=True)
            db.rollback()
            logging.error(f"--- LOGIC: finalize_outputs FALHOU (Erro Geral) (run_id={run_id}) ---")
            try:
                crud_runs.update_run_status(db, run_id, status='finalize_failed'); db.commit()
            except:
                logging.error(f"Falha ao tentar atualizar status do run {run_id} para 'finalize_failed'.")
            raise