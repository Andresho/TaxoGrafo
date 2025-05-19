import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
import app.models as models

def get_difficulty_comparison_group(
        db: Session, pipeline_run_id: str, comparison_group_id: str
) -> Optional[models.DifficultyComparisonGroup]:
    """Retorna um DifficultyComparisonGroup específico."""
    return db.query(models.DifficultyComparisonGroup).filter_by(
        pipeline_run_id=pipeline_run_id,
        comparison_group_id=comparison_group_id
    ).first()


def add_difficulty_comparison_groups_raw(db: Session, group_data_list: List[Dict[str, Any]]) -> None:
    """
    Adiciona uma lista de DifficultyComparisonGroups ao banco de dados.
    Usa INSERT ... ON CONFLICT DO NOTHING para evitar duplicatas baseadas na chave primária.
    Cada dicionário em group_data_list deve corresponder às colunas de DifficultyComparisonGroup.
    """
    if not group_data_list:
        logging.debug("Nenhum grupo de comparação para adicionar.")
        return

    stmt = pg_insert(models.DifficultyComparisonGroup.__table__).values(group_data_list)

    pk_columns = [
        models.DifficultyComparisonGroup.pipeline_run_id.name,
        models.DifficultyComparisonGroup.comparison_group_id.name
    ]

    stmt = stmt.on_conflict_do_nothing(index_elements=pk_columns)

    try:
        db.execute(stmt)
        logging.info(
            f"Tentativa de inserção de {len(group_data_list)} registros em DifficultyComparisonGroup (ON CONFLICT DO NOTHING).")
    except Exception as e:
        logging.error(f"Erro ao executar bulk insert para DifficultyComparisonGroup: {e}", exc_info=True)
        raise


def get_all_comparison_groups_for_run(db: Session, pipeline_run_id: str) -> List[models.DifficultyComparisonGroup]:
    """Retorna todos os DifficultyComparisonGroups para um dado pipeline_run_id."""
    return db.query(models.DifficultyComparisonGroup).filter_by(pipeline_run_id=pipeline_run_id).all()
