"""CRUD operations for text_units table."""
from sqlalchemy.orm import Session
import app.models as models
from app.crud.base import add_records

def add_text_units(db: Session, run_id: str, records: list) -> None:
    """Insert text unit records for a given pipeline run."""
    add_records(db, models.GraphragTextUnit, run_id, records)