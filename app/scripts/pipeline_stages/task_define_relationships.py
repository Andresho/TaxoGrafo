import pandas as pd

import app.crud.generated_ucs_raw as crud_generated_ucs_raw
import app.crud.knowledge_relationships_intermediate as crud_rel_intermediate
import app.crud.graphrag_entities as crud_graphrag_entities
import app.crud.graphrag_relationships as crud_graphrag_relationships

from app.scripts.rel_builders import RequiresBuilder, ExpandsBuilder

import logging
from app.db import get_session

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
