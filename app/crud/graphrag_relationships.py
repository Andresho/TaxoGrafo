"""CRUD operations for relationships table."""
from sqlalchemy.orm import Session
import models
from crud.base import add_records

def add_relationships(db: Session, run_id: str, records: list) -> None:
    """Insert relationship records for a given pipeline run."""
    add_records(db, models.GraphragRelationship, run_id, records)
    
def get_relationships(db: Session, run_id: str) -> list[dict]:
    """Return list of relationship records for given pipeline run."""
    rows = db.query(models.GraphragRelationship).filter(
        models.GraphragRelationship.pipeline_run_id == run_id
    ).all()

    results = []
    for row in rows:
        results.append({
            'source': row.source,
            'target': row.target,
            'description': row.description,
            'weight': row.weight,
            'combined_degree': row.combined_degree,
            'text_unit_ids': row.text_unit_ids,
        })
    return results