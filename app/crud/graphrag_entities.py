"""CRUD operations for entities table."""
from sqlalchemy.orm import Session
import app.models as models
from app.crud.base import add_records

def add_entities(db: Session, run_id: str, records: list) -> None:
    """Insert entity records for a given pipeline run."""
    add_records(db, models.GraphragEntity, run_id, records)
    
def get_entities(db: Session, run_id: str) -> list[dict]:
    """Return list of entity records for given pipeline run."""
    rows = db.query(models.GraphragEntity).filter(
        models.GraphragEntity.pipeline_run_id == run_id
    ).all()

    results = []
    for row in rows:
        results.append({
            'id': row.id,
            'human_readable_id': row.human_readable_id,
            'title': row.title,
            'type': row.type,
            'description': row.description,
            'text_unit_ids': row.text_unit_ids,
            'frequency': row.frequency,
            'degree': row.degree,
            'x': row.x,
            'y': row.y,
            'parent_community_id': row.parent_community_id,
        })
    return results