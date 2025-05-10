from sqlalchemy import Column, String, TIMESTAMP, JSON, Integer, Float, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

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

# ------- Graphrag output tables -------
class GraphragCommunity(Base):
    __tablename__ = "graphrag_communities"
    id = Column(String, primary_key=True, index=True)
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.run_id'), nullable=False, index=True)
    human_readable_id = Column(Integer)
    community = Column(Integer)
    level = Column(Integer)
    parent = Column(Integer)
    children = Column(JSON)
    title = Column(Text)
    entity_ids = Column(JSON)
    relationship_ids = Column(JSON)
    text_unit_ids = Column(JSON)
    period = Column(String)
    size = Column(Integer)
    parent_community_id = Column(String)

class GraphragCommunityReport(Base):
    __tablename__ = "graphrag_community_reports"
    id = Column(String, primary_key=True, index=True)
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.run_id'), nullable=False, index=True)
    human_readable_id = Column(Integer)
    community = Column(Integer)
    level = Column(Integer)
    parent = Column(Integer)
    children = Column(JSON)
    title = Column(Text)
    summary = Column(Text)
    full_content = Column(Text)
    rank = Column(Float)
    rating_explanation = Column(Text)
    findings = Column(JSON)
    full_content_json = Column(JSON)
    period = Column(String)
    size = Column(Integer)

class GraphragDocument(Base):
    __tablename__ = "graphrag_documents"
    id = Column(String, primary_key=True, index=True)
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.run_id'), nullable=False, index=True)
    human_readable_id = Column(Integer)
    title = Column(Text)
    text = Column(Text)
    text_unit_ids = Column(JSON)
    creation_date = Column(String)
    doc_metadata = Column('doc_metadata', JSON, nullable=True)

class GraphragEntity(Base):
    __tablename__ = "graphrag_entities"
    id = Column(String, primary_key=True, index=True)
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.run_id'), nullable=False, index=True)
    human_readable_id = Column(Integer)
    title = Column(Text)
    type = Column(String)
    description = Column(Text)
    text_unit_ids = Column(JSON)
    frequency = Column(Integer)
    degree = Column(Integer)
    x = Column(Float)
    y = Column(Float)
    parent_community_id = Column(String)

class GraphragRelationship(Base):
    __tablename__ = "graphrag_relationships"
    id = Column(String, primary_key=True, index=True)
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.run_id'), nullable=False, index=True)
    human_readable_id = Column(Integer)
    source = Column(String)
    target = Column(String)
    description = Column(Text)
    weight = Column(Float)
    combined_degree = Column(Integer)
    text_unit_ids = Column(JSON)

class GraphragTextUnit(Base):
    __tablename__ = "graphrag_text_units"
    id = Column(String, primary_key=True, index=True)
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.run_id'), nullable=False, index=True)
    human_readable_id = Column(Integer)
    text = Column(Text)
    n_tokens = Column(Integer)
    document_ids = Column(JSON)
    entity_ids = Column(JSON)
    relationship_ids = Column(JSON)
    covariate_ids = Column(JSON)
    # ------------------------
    # UC origins table
    # ------------------------
class KnowledgeUnitOrigin(Base):
    __tablename__ = 'knowledge_unit_origins'
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True)
    origin_id = Column(String, primary_key=True)
    origin_type = Column(String, nullable=False)
    title = Column(Text, nullable=False)
    context = Column(Text)
    frequency = Column(Integer)
    degree = Column(Integer)
    entity_type = Column(String)
    level = Column(Integer)
    parent_community_id_of_origin = Column(String)

    # ------------------------
    # Generated UCs and relationships tables
    # ------------------------
class GeneratedUcsRaw(Base):
    __tablename__ = 'generated_ucs_raw'
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True)
    uc_id = Column(String, primary_key=True)
    origin_id = Column(String)
    bloom_level = Column(String)
    uc_text = Column(Text)

class KnowledgeRelationshipIntermediate(Base):
    __tablename__ = 'knowledge_relationships_intermediate'
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True)
    source = Column(String, primary_key=True)
    target = Column(String, primary_key=True)
    type = Column(String, primary_key=True)
    origin_id = Column(String)
    weight = Column(Float)
    graphrag_rel_desc = Column(Text)

class FinalKnowledgeUnit(Base):
    __tablename__ = 'final_knowledge_units'
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True)
    uc_id = Column(String, primary_key=True)
    origin_id = Column(String)
    bloom_level = Column(String)
    uc_text = Column(Text)
    difficulty_score = Column(Integer)
    evaluation_count = Column(Integer)
    difficulty_justification = Column(Text)

class FinalKnowledgeRelationship(Base):
    __tablename__ = 'final_knowledge_relationships'
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True)
    source = Column(String, primary_key=True)
    target = Column(String, primary_key=True)
    type = Column(String, primary_key=True)
    origin_id = Column(String)
    weight = Column(Float)
    graphrag_rel_desc = Column(Text)

# ------------------------
# UC evaluations raw table
# ------------------------
class KnowledgeUnitEvaluationsAggregatedBatch(Base):
    __tablename__ = 'knowledge_unit_evaluations_aggregated_batch'
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True)
    knowledge_unit_id = Column(String, primary_key=True)
    request_custom_id = Column(String, primary_key=True, nullable=False)
    difficulty_score = Column(Integer)
    justification = Column(Text)