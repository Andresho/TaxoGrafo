from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
import app.models as models
import uuid
from typing import List

def link_resources_to_run(db: Session, run_id: str, resource_ids: List[uuid.UUID]):
    if not resource_ids:
        return

    records_to_insert = [
        {"run_id": run_id, "resource_id": res_id} for res_id in resource_ids
    ]

    stmt = pg_insert(models.PipelineRunResource.__table__).values(records_to_insert)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=['run_id', 'resource_id']
    )
    db.execute(stmt)
    db.commit()

def get_resources_for_run(db: Session, run_id: str) -> List[models.Resource]:
    return db.query(models.Resource)\
             .join(models.PipelineRunResource, models.Resource.resource_id == models.PipelineRunResource.resource_id)\
             .filter(models.PipelineRunResource.run_id == run_id)\
             .all()
