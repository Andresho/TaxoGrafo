"""CRUD operations for community_reports table."""
from sqlalchemy.orm import Session
import models
from crud.base import add_records

def add_community_reports(db: Session, run_id: str, records: list) -> None:
    """Insert community report records for a given pipeline run."""
    add_records(db, models.GraphragCommunityReport, run_id, records)