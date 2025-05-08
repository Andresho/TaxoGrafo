"""
create output tables: communities, community_reports, documents, entities, relationships, text_units

Revision ID: 0002_create_output_tables
Revises: 0001_create_pipeline_runs
Create Date: 2025-05-03 00:30:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0002_create_output_tables'
down_revision = '0001_create_pipeline_runs'
branch_labels = None
depends_on = None

def upgrade():
    # communities
    op.create_table(
        'graphrag_communities',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id'), nullable=False),
        sa.Column('human_readable_id', sa.Integer()),
        sa.Column('community', sa.Integer()),
        sa.Column('level', sa.Integer()),
        sa.Column('parent', sa.Integer()),
        sa.Column('children', sa.JSON()),
        sa.Column('title', sa.Text()),
        sa.Column('entity_ids', sa.JSON()),
        sa.Column('relationship_ids', sa.JSON()),
        sa.Column('text_unit_ids', sa.JSON()),
        sa.Column('period', sa.String()),
        sa.Column('size', sa.Integer()),
        sa.Column('parent_community_id', sa.String()),
    )
    # community_reports
    op.create_table(
        'graphrag_community_reports',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id'), nullable=False),
        sa.Column('human_readable_id', sa.Integer()),
        sa.Column('community', sa.Integer()),
        sa.Column('level', sa.Integer()),
        sa.Column('parent', sa.Integer()),
        sa.Column('children', sa.JSON()),
        sa.Column('title', sa.Text()),
        sa.Column('summary', sa.Text()),
        sa.Column('full_content', sa.Text()),
        sa.Column('rank', sa.Float()),
        sa.Column('rating_explanation', sa.Text()),
        sa.Column('findings', sa.JSON()),
        sa.Column('full_content_json', sa.JSON()),
        sa.Column('period', sa.String()),
        sa.Column('size', sa.Integer()),
    )
    # documents
    op.create_table(
        'graphrag_documents',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id'), nullable=False),
        sa.Column('human_readable_id', sa.Integer()),
        sa.Column('title', sa.Text()),
        sa.Column('text', sa.Text()),
        sa.Column('text_unit_ids', sa.JSON()),
        sa.Column('creation_date', sa.String()),
        sa.Column('doc_metadata', sa.JSON()),
    )
    # entities
    op.create_table(
        'graphrag_entities',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id'), nullable=False),
        sa.Column('human_readable_id', sa.Integer()),
        sa.Column('title', sa.Text()),
        sa.Column('type', sa.String()),
        sa.Column('description', sa.Text()),
        sa.Column('text_unit_ids', sa.JSON()),
        sa.Column('frequency', sa.Integer()),
        sa.Column('degree', sa.Integer()),
        sa.Column('x', sa.Float()),
        sa.Column('y', sa.Float()),
        sa.Column('parent_community_id', sa.String()),
    )
    # relationships
    op.create_table(
        'graphrag_relationships',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id'), nullable=False),
        sa.Column('human_readable_id', sa.Integer()),
        sa.Column('source', sa.String()),
        sa.Column('target', sa.String()),
        sa.Column('description', sa.Text()),
        sa.Column('weight', sa.Float()),
        sa.Column('combined_degree', sa.Integer()),
        sa.Column('text_unit_ids', sa.JSON()),
    )
    # text_units
    op.create_table(
        'graphrag_text_units',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id'), nullable=False),
        sa.Column('human_readable_id', sa.Integer()),
        sa.Column('text', sa.Text()),
        sa.Column('n_tokens', sa.Integer()),
        sa.Column('document_ids', sa.JSON()),
        sa.Column('entity_ids', sa.JSON()),
        sa.Column('relationship_ids', sa.JSON()),
        sa.Column('covariate_ids', sa.JSON()),
    )

def downgrade():
    op.drop_table('graphrag_text_units')
    op.drop_table('graphrag_relationships')
    op.drop_table('graphrag_entities')
    op.drop_table('graphrag_documents')
    op.drop_table('graphrag_community_reports')
    op.drop_table('graphrag_communities')