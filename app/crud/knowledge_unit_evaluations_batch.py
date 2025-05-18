"""CRUD operations for knowledge_unit_evaluations_aggregated_batch table."""
from sqlalchemy.orm import Session
import app.models as models
from app.crud.base import add_records

def add_knowledge_unit_evaluations_batch(db: Session, run_id: str, records: list) -> None:
    """Insert knowledge unit evaluation records for a given pipeline run."""
    add_records(db, models.KnowledgeUnitEvaluationsAggregatedBatch, run_id, records)

def get_knowledge_unit_evaluations_batch(db: Session, run_id: str) -> list[dict]:
    """Return list of knowledge unit evaluation records for given pipeline run."""
    rows = db.query(models.KnowledgeUnitEvaluationsAggregatedBatch).filter(
        models.KnowledgeUnitEvaluationsAggregatedBatch.pipeline_run_id == run_id
    ).all()

    results = []
    for row in rows:
        results.append({
            'knowledge_unit_id': row.knowledge_unit_id,
            'difficulty_score': row.difficulty_score,
            'justification': row.justification,
        })
    return results