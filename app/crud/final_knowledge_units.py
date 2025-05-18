"""CRUD operations for final_knowledge_units table."""
from sqlalchemy.orm import Session
import app.models
from app.crud.base import add_records
import logging

def add_final_knowledge_units(db: Session, run_id: str, records: list) -> None:
    """Insert final knowledge unit records for a given pipeline run."""
    add_records(db, models.FinalKnowledgeUnit, run_id, records)

def get_final_knowledge_units(db: Session, run_id: str) -> list[dict]:
    """Return list of final knowledge unit records for given pipeline run."""
    rows = db.query(models.FinalKnowledgeUnit).filter(
        models.FinalKnowledgeUnit.pipeline_run_id == run_id
    ).all()

    results = []
    for row in rows:
        results.append({
            'uc_id': row.uc_id,
            'origin_id': row.origin_id,
            'bloom_level': row.bloom_level,
            'uc_text': row.uc_text,
            'difficulty_score': row.difficulty_score,
            'evaluation_count': row.evaluation_count,
            'difficulty_justification': row.difficulty_justification,
        })
    return results