from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import models
import schemas
import uuid
from typing import Optional, List
from datetime import datetime

def get_resource(db: Session, resource_id: uuid.UUID) -> Optional[models.Resource]:
    return db.query(models.Resource).filter(models.Resource.resource_id == resource_id).first()

def create_resource(db: Session, resource_id: uuid.UUID, original_filename: str, original_mime_type: str, original_file_path: str) -> models.Resource:
    db_resource = models.Resource(
        resource_id=resource_id,
        original_filename=original_filename,
        original_mime_type=original_mime_type,
        original_file_path=original_file_path,
        status="uploaded"
    )
    db.add(db_resource)
    db.commit()
    db.refresh(db_resource)
    return db_resource

def update_resource_status(
    db: Session,
    resource_id: uuid.UUID,
    status: str,
    processed_txt_path: Optional[str] = None,
    error_message: Optional[str] = None
) -> Optional[models.Resource]:
    db_resource = get_resource(db, resource_id)
    if db_resource:
        db_resource.status = status
        if processed_txt_path:
            db_resource.processed_txt_path = processed_txt_path
        if error_message:
            db_resource.error_message = error_message
        if status in ["processed_txt_success", "processed_txt_error"]:
            db_resource.processed_at = datetime.utcnow()
        db.commit()
        db.refresh(db_resource)
    return db_resource

def get_resources_by_ids(db: Session, resource_ids: List[uuid.UUID]) -> List[models.Resource]:
    if not resource_ids:
        return []
    return db.query(models.Resource).filter(models.Resource.resource_id.in_(resource_ids)).all()
