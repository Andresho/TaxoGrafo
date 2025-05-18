"""CRUD operations for community_reports table."""
from sqlalchemy.orm import Session
import app.models as models
from app.crud.base import add_records

def add_community_reports(db: Session, run_id: str, records: list) -> None:
    """Insert community report records for a given pipeline run."""
    add_records(db, models.GraphragCommunityReport, run_id, records)
    
def get_community_reports(db: Session, run_id: str) -> list[dict]:
    """Return list of community report records for given pipeline run."""
    rows = db.query(models.GraphragCommunityReport).filter(
        models.GraphragCommunityReport.pipeline_run_id == run_id
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
            'summary': row.summary,
            'full_content': row.full_content,
            'rank': row.rank,
            'rating_explanation': row.rating_explanation,
            'findings': row.findings,
            'full_content_json': row.full_content_json,
            'period': row.period,
            'size': row.size,
        })
    return results