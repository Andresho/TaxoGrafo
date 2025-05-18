import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import json
import numpy as np

from app.db import get_session

import app.crud.knowledge_unit_origins as crud_knowledge_unit_origins
import app.crud.graphrag_communities as crud_graphrag_communities
import app.crud.graphrag_community_reports as crud_graphrag_community_reports
import app.crud.graphrag_documents as crud_graphrag_documents
import app.crud.graphrag_entities as crud_graphrag_entities
import app.crud.graphrag_relationships as crud_graphrag_relationships
import app.crud.graphrag_text_units as crud_graphrag_text_units

from app.scripts.io_utils import load_dataframe
from app.scripts.origins_utils import prepare_uc_origins
from app.scripts.constants import AIRFLOW_DATA_DIR

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