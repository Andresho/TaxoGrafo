"""
create parquet output tables for generated UCs, intermediate and final relationships and final UCs

Revision ID: 0004_create_kg_unit_tables
Revises: 0003_knowledge_unit_origins
Create Date: 2025-05-03 13:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0004_create_kg_unit_tables'
down_revision = '0003_knowledge_unit_origins'
branch_labels = None
depends_on = None

def upgrade():
    # generated UCs raw
    op.create_table(
        'generated_ucs_raw',
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True),
        sa.Column('uc_id', sa.String(), primary_key=True),
        sa.Column('origin_id', sa.String(), nullable=True),
        sa.Column('bloom_level', sa.String(), nullable=True),
        sa.Column('uc_text', sa.Text(), nullable=True),
    )
    # intermediate relationships
    op.create_table(
        'knowledge_relationships_intermediate',
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True),
        sa.Column('source', sa.String(), primary_key=True),
        sa.Column('target', sa.String(), primary_key=True),
        sa.Column('type', sa.String(), primary_key=True),
        sa.Column('origin_id', sa.String(), nullable=True),
        sa.Column('weight', sa.Float(), nullable=True),
        sa.Column('graphrag_rel_desc', sa.Text(), nullable=True),
    )
    # final knowledge units
    op.create_table(
        'final_knowledge_units',
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True),
        sa.Column('uc_id', sa.String(), primary_key=True),
        sa.Column('origin_id', sa.String(), nullable=True),
        sa.Column('bloom_level', sa.String(), nullable=True),
        sa.Column('uc_text', sa.Text(), nullable=True),
        sa.Column('difficulty_score', sa.Integer(), nullable=True),
        sa.Column('evaluation_count', sa.Integer(), nullable=True),
        sa.Column('difficulty_justification', sa.Text(), nullable=True),
    )
    # final relationships
    op.create_table(
        'final_knowledge_relationships',
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True),
        sa.Column('source', sa.String(), primary_key=True),
        sa.Column('target', sa.String(), primary_key=True),
        sa.Column('type', sa.String(), primary_key=True),
        sa.Column('origin_id', sa.String(), nullable=True),
        sa.Column('weight', sa.Float(), nullable=True),
        sa.Column('graphrag_rel_desc', sa.Text(), nullable=True),
    )

def downgrade():
    op.drop_table('final_knowledge_relationships')
    op.drop_table('final_knowledge_units')
    op.drop_table('knowledge_relationships_intermediate')
    op.drop_table('generated_ucs_raw')