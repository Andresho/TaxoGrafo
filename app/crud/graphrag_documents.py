"""CRUD operations for documents table."""
from sqlalchemy.orm import Session
import models
from crud.base import add_records

def add_documents(db: Session, run_id: str, records: list) -> None:
    """Insert document records for a given pipeline run."""
    # Adjust reserved 'metadata' field: source data uses 'metadata', model uses 'doc_metadata'
    for rec in records:
        if 'metadata' in rec:
            rec['doc_metadata'] = rec.pop('metadata')
    add_records(db, models.GraphragDocument, run_id, records)
    
def get_documents(db: Session, run_id: str) -> list[dict]:
    """Return list of document records for given pipeline run."""
    rows = db.query(models.GraphragDocument).filter(
        models.GraphragDocument.pipeline_run_id == run_id
    ).all()
    
    results = []
    for row in rows:
        results.append({
            'id': row.id,
            'human_readable_id': row.human_readable_id,
            'title': row.title,
            'text': row.text,
            'text_unit_ids': row.text_unit_ids,
            'creation_date': row.creation_date,
            'doc_metadata': row.doc_metadata,
        })
    return results