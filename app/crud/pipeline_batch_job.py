from sqlalchemy.orm import Session
from typing import Optional
import app.models as models
import logging

STATUS_PENDING_SUBMISSION = "PENDING_SUBMISSION"
STATUS_SUBMITTED = "SUBMITTED"
STATUS_SUBMISSION_FAILED = "SUBMISSION_FAILED"
STATUS_PENDING_PROCESSING = "PENDING_PROCESSING"
STATUS_PROCESSING = "PROCESSING"
STATUS_COMPLETED = "COMPLETED"
STATUS_PROCESSING_FAILED = "PROCESSING_FAILED"

VALID_STATUSES = {
    STATUS_PENDING_SUBMISSION, STATUS_SUBMITTED, STATUS_SUBMISSION_FAILED,
    STATUS_PENDING_PROCESSING, STATUS_PROCESSING, STATUS_COMPLETED, STATUS_PROCESSING_FAILED
}


def get_pipeline_batch_job(db: Session, pipeline_run_id: str, batch_type: str) -> Optional[models.PipelineBatchJob]:
    """
    Retrieves a specific pipeline batch job by pipeline_run_id and batch_type.
    """
    return db.query(models.PipelineBatchJob).filter(
        models.PipelineBatchJob.pipeline_run_id == pipeline_run_id,
        models.PipelineBatchJob.batch_type == batch_type
    ).first()


def create_or_get_pipeline_batch_job(
        db: Session,
        pipeline_run_id: str,
        batch_type: str,
        initial_status: str = STATUS_PENDING_SUBMISSION
) -> models.PipelineBatchJob:
    """
    Retrieves an existing pipeline batch job or creates a new one if it doesn't exist.
    If the job exists and its status allows for re-submission (e.g., FAILED states),
    it can be "reset" to PENDING_SUBMISSION.
    """
    if initial_status not in VALID_STATUSES:
        raise ValueError(f"Invalid initial_status: {initial_status}")

    job = get_pipeline_batch_job(db, pipeline_run_id, batch_type)

    if job:
        logging.info(f"Found existing job for run_id {pipeline_run_id}, type {batch_type} with status {job.status}")
        if job.status in [STATUS_SUBMISSION_FAILED, STATUS_PROCESSING_FAILED]:
            logging.info(f"Resetting job {job.id} from {job.status} to {initial_status} for re-submission attempt.")
            job.status = initial_status
            job.llm_batch_id = None
            job.last_error = None
        return job

    logging.info(f"Creating new job for run_id {pipeline_run_id}, type {batch_type} with status {initial_status}")
    new_job = models.PipelineBatchJob(
        pipeline_run_id=pipeline_run_id,
        batch_type=batch_type,
        status=initial_status
    )
    db.add(new_job)
    return new_job


def update_pipeline_batch_job(
        db: Session,
        job_id: int,
        status: Optional[str] = None,
        llm_batch_id: Optional[str] = None,
        last_error: Optional[str] = None
) -> Optional[models.PipelineBatchJob]:
    """
    Updates specified fields of a pipeline batch job.
    Only updates fields that are not None.
    """
    job = db.query(models.PipelineBatchJob).filter(models.PipelineBatchJob.id == job_id).first()
    if not job:
        return None

    updated = False
    if status is not None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status for update: {status}")
        job.status = status
        updated = True

    if llm_batch_id is not None:
        job.llm_batch_id = llm_batch_id if llm_batch_id else None
        updated = True

    if last_error is not None:
        job.last_error = last_error if last_error else None
        updated = True

    return job
