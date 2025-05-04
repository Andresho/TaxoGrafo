"""CRUD operations for knowledge_relationships_intermediate table."""
from sqlalchemy.orm import Session
import models
from crud.base import add_records

def add_knowledge_relationships_intermediate(db: Session, run_id: str, records: list) -> None:
    """Insert intermediate knowledge relationships for a given pipeline run."""
    add_records(db, models.KnowledgeRelationshipIntermediate, run_id, records)

def get_knowledge_relationships_intermediate(db: Session, run_id: str) -> list[dict]:
    """Return list of intermediate knowledge relationships for given pipeline run."""
    rows = db.query(models.KnowledgeRelationshipIntermediate).filter(
        models.KnowledgeRelationshipIntermediate.pipeline_run_id == run_id
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