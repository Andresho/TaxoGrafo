"""CRUD operations for final_knowledge_relationships table."""
from sqlalchemy.orm import Session
import app.models as models
from app.crud.base import add_records

def add_final_knowledge_relationships(db: Session, run_id: str, records: list) -> None:
    """Insert final knowledge relationship records for a given pipeline run."""
    add_records(db, models.FinalKnowledgeRelationship, run_id, records)

def get_final_knowledge_relationships(db: Session, run_id: str) -> list[dict]:
    """Return list of final knowledge relationship records for given pipeline run."""
    rows = db.query(models.FinalKnowledgeRelationship).filter(
        models.FinalKnowledgeRelationship.pipeline_run_id == run_id
    ).all()
    results = []

    for row in rows:
        results.append({
            'source': row.source,
            'target': row.target,
            'type': row.type,
            'origin_id': row.origin_id,
            'weight': row.weight,
            'graphrag_rel_desc': row.graphrag_rel_desc,
        })
    return results