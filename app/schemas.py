from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid

# ------------------------------
# Pipeline Run Schemas
# ------------------------------

class PipelineRunBase(BaseModel):
    run_id: str
    status: str
    trigger_source: Optional[str] = None

class PipelineRunSummary(PipelineRunBase):
    started_at: datetime

    class Config:
        orm_mode = True

class PipelineRunDetail(PipelineRunSummary):
    finished_at: Optional[datetime] = None
    payload: Optional[dict] = None

    class Config:
        orm_mode = True


# ------------------------------
# Final Knowledge Unit Schemas
# ------------------------------

class FinalKnowledgeUnitBase(BaseModel):
    uc_id: str
    origin_id: Optional[str] = None
    bloom_level: Optional[str] = None
    uc_text: Optional[str] = None
    difficulty_score: Optional[int] = None
    evaluation_count: Optional[int] = None
    difficulty_justification: Optional[str] = None

class FinalKnowledgeUnitResponse(FinalKnowledgeUnitBase):
    pipeline_run_id: str

    class Config:
        orm_mode = True

# ------------------------------
# Final Knowledge Relationship Schemas
# ------------------------------

class FinalKnowledgeRelationshipBase(BaseModel):
    source_uc_id: str = Field(alias="source")
    target_uc_id: str = Field(alias="target")
    type: Optional[str] = None
    origin_id: Optional[str] = None
    weight: Optional[float] = None
    graphrag_rel_desc: Optional[str] = None


class FinalKnowledgeRelationshipResponse(FinalKnowledgeRelationshipBase):
    pipeline_run_id: str

    class Config:
        orm_mode = True
        allow_population_by_field_name = True


# ------------------------------
# Knowledge Unit Origin Schemas
# ------------------------------
class KnowledgeUnitOriginBase(BaseModel):
    origin_id: str
    origin_type: str
    title: str
    context: Optional[str] = None
    frequency: Optional[int] = None
    degree: Optional[int] = None
    entity_type: Optional[str] = None
    level: Optional[int] = None
    parent_community_id_of_origin: Optional[str] = None

class KnowledgeUnitOriginResponse(KnowledgeUnitOriginBase):
    pipeline_run_id: str

    class Config:
        orm_mode = True

# ------------------------------
# Graph Visualization Schemas
# ------------------------------

class GraphNode(BaseModel):
    id: str
    label: str
    group: Optional[str] = None 
    title: Optional[str] = None
    level: Optional[str] = None
    value: Optional[int] = None

class GraphEdge(BaseModel):
    source: str
    target: str
    label: Optional[str] = None

class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

# ------------------------------
# Resource Schemas
# ------------------------------

class ResourceBase(BaseModel):
    original_filename: str
    original_mime_type: str

class ResourceCreate(ResourceBase):
    pass

class ResourceUpdate(BaseModel):
    status: Optional[str] = None
    processed_txt_path: Optional[str] = None
    error_message: Optional[str] = None
    processed_at: Optional[datetime] = None

class ResourceResponse(ResourceBase):
    resource_id: uuid.UUID
    original_file_path: str
    processed_txt_path: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    uploaded_at: datetime
    processed_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class PipelineInitRequest(BaseModel):
    resource_ids: List[uuid.UUID]
    skip_graphrag: Optional[bool] = False
