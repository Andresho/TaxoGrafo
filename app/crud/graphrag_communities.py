"""CRUD operations for communities table."""
from sqlalchemy.orm import Session
import models
from crud.base import add_records

def add_communities(db: Session, run_id: str, records: list) -> None:
    """Insert community records for a given pipeline run."""
    add_records(db, models.GraphragCommunity, run_id, records)