"""CRUD operations for pipeline_runs table."""
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

import app.models as models


def get_run(db: Session, run_id: str) -> Optional[models.PipelineRun]:
    """Return the PipelineRun with given run_id, or None."""
    return db.query(models.PipelineRun).filter(models.PipelineRun.run_id == run_id).first()


def create_run(
    db: Session,
    run_id: str,
    trigger_source: Optional[str] = None,
    payload: Optional[dict] = None,
) -> models.PipelineRun:
    """Create a new PipelineRun or return existing one."""
    existing = get_run(db, run_id)
    if existing:
        return existing
    run = models.PipelineRun(
        run_id=run_id,
        trigger_source=trigger_source,
        payload=payload,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run

def update_run_status(db: Session, run_id: str, status: str, finished_at: Optional[datetime] = None, commit: bool = True) -> models.PipelineRun:
    run = get_run(db, run_id)
    if not run:
        raise ValueError(f"PipelineRun '{run_id}' not found")
    run.status = status
    run.finished_at = finished_at if finished_at is not None else datetime.utcnow()
    db.add(run) # Adiciona à sessão
    if commit:
        db.commit()
    return run