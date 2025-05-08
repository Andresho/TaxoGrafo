"""CRUD operations for communities table."""
from sqlalchemy.orm import Session
import models
from crud.base import add_records
import logging

def add_communities(db: Session, run_id: str, records: list) -> None:
    """Insert community records for a given pipeline run."""
    add_records(db, models.GraphragCommunity, run_id, records)
    
def get_communities(db: Session, run_id: str) -> list[dict]:
    """Return list of community records for given pipeline run."""
    rows = db.query(models.GraphragCommunity).filter(
        models.GraphragCommunity.pipeline_run_id == run_id
    ).all()

    results = []
    for row in rows:
        results.append({
            'id': row.id,
            'human_readable_id': row.human_readable_id,
            'community': row.community,
            'level': row.level,
            'parent': row.parent,
            'children': row.children,
            'title': row.title,
            'entity_ids': row.entity_ids,
            'relationship_ids': row.relationship_ids,
            'text_unit_ids': row.text_unit_ids,
            'period': row.period,
            'size': row.size,
            'parent_community_id': row.parent_community_id,
        })
    return results