"""CRUD operations for entities table."""
from sqlalchemy.orm import Session
import models
from crud.base import add_records

def add_entities(db: Session, run_id: str, records: list) -> None:
    """Insert entity records for a given pipeline run."""
    add_records(db, models.GraphragEntity, run_id, records)