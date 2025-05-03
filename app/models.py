from sqlalchemy import Column, String, TIMESTAMP, JSON
from sqlalchemy.sql import func

from db import Base


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_id = Column(String, primary_key=True, index=True)
    started_at = Column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at = Column(TIMESTAMP(timezone=True), nullable=True)
    status = Column(String, nullable=False, default="running")
    trigger_source = Column(String, nullable=True)
    payload = Column(JSON, nullable=True)