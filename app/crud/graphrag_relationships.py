"""CRUD operations for relationships table."""
from sqlalchemy.orm import Session
import models
from crud.base import add_records

def add_relationships(db: Session, run_id: str, records: list) -> None:
    """Insert relationship records for a given pipeline run."""
    add_records(db, models.GraphragRelationship, run_id, records)