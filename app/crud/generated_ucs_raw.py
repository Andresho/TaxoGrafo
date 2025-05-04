"""CRUD operations for generated_ucs_raw table."""
from sqlalchemy.orm import Session
import models
from crud.base import add_records
import logging

def add_generated_ucs_raw(db: Session, run_id: str, records: list) -> None:
    """Insert generated UC raw records for a given pipeline run."""
    add_records(db, models.GeneratedUcsRaw, run_id, records)

def get_generated_ucs_raw(db: Session, run_id: str) -> list[dict]:
    """Return list of generated UC raw records for given pipeline run."""
    rows = db.query(models.GeneratedUcsRaw).filter(
        models.GeneratedUcsRaw.pipeline_run_id == run_id
    ).all()

    results = []
    for row in rows:
        results.append({
            'uc_id': row.uc_id,
            'origin_id': row.origin_id,
            'bloom_level': row.bloom_level,
            'uc_text': row.uc_text,
        })
    return results