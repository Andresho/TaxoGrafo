"""CRUD operations for uc_origins table."""
from sqlalchemy.orm import Session
import app.models as models
from app.crud.base import add_records

def add_knowledge_unit_origins(db: Session, run_id: str, records: list) -> None:
    """Insert uc origin records for a given pipeline run."""
    add_records(db, models.KnowledgeUnitOrigin, run_id, records)

def get_knowledge_unit_origins(db: Session, run_id: str) -> list[dict]:
    """Return list of origin records for given pipeline run."""
    rows = db.query(models.KnowledgeUnitOrigin).filter(
        models.KnowledgeUnitOrigin.pipeline_run_id == run_id
    ).all()
    # Convert ORM objects to dicts
    results = []
    for row in rows:
        d = {
            'pipeline_run_id': row.pipeline_run_id,
            'origin_id': row.origin_id,
            'origin_type': row.origin_type,
            'title': row.title,
            'context': row.context,
            'frequency': row.frequency,
            'degree': row.degree,
            'entity_type': row.entity_type,
            'level': row.level,
            'parent_community_id_of_origin': row.parent_community_id_of_origin,
        }
        results.append(d)
    return results