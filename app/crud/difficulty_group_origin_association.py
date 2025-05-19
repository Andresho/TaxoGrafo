import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
import app.models as models

def add_difficulty_group_origin_associations_raw(db: Session, association_data_list: List[Dict[str, Any]]) -> None:
    """
    Adiciona uma lista de associações entre DifficultyComparisonGroup e KnowledgeUnitOrigin.
    Usa INSERT ... ON CONFLICT DO NOTHING para evitar duplicatas.
    Cada dicionário em association_data_list deve corresponder às colunas da tabela de associação.
    """
    if not association_data_list:
        logging.debug("Nenhuma associação grupo-origem para adicionar.")
        return

    stmt = pg_insert(models.difficulty_group_origin_association).values(association_data_list)

    pk_columns_assoc = [
        models.difficulty_group_origin_association.c.pipeline_run_id.name,
        models.difficulty_group_origin_association.c.comparison_group_id.name,
        models.difficulty_group_origin_association.c.origin_id.name
    ]

    stmt = stmt.on_conflict_do_nothing(index_elements=pk_columns_assoc)

    try:
        db.execute(stmt)
        logging.info(
            f"Tentativa de inserção de {len(association_data_list)} registros em difficulty_group_origin_association (ON CONFLICT DO NOTHING).")
    except Exception as e:
        logging.error(f"Erro ao executar bulk insert para difficulty_group_origin_association: {e}", exc_info=True)
        raise


def get_origins_for_comparison_group(db: Session, pipeline_run_id: str, comparison_group_id: str) -> List[
    models.KnowledgeUnitOrigin]:
    """Retorna todas as KnowledgeUnitOrigins associadas a um DifficultyComparisonGroup."""
    return db.query(models.KnowledgeUnitOrigin) \
        .join(models.difficulty_group_origin_association,
              (models.KnowledgeUnitOrigin.pipeline_run_id == models.difficulty_group_origin_association.c.pipeline_run_id) &
              (models.KnowledgeUnitOrigin.origin_id == models.difficulty_group_origin_association.c.origin_id)
              ) \
        .filter(
        models.difficulty_group_origin_association.c.pipeline_run_id == pipeline_run_id,
        models.difficulty_group_origin_association.c.comparison_group_id == comparison_group_id
    ).all()
