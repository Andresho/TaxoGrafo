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