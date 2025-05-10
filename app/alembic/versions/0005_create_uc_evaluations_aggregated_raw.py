"""
create knowledge_unit_evaluations_aggregated_batch table

Revision ID: 0005_knowledge_eval_batch
Revises: 0004_create_kg_unit_tables
Create Date: 2025-05-03 14:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0005_knowledge_eval_batch'
down_revision = '0004_create_kg_unit_tables'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'knowledge_unit_evaluations_aggregated_batch',
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True),
        sa.Column('knowledge_unit_id', sa.String(), primary_key=True),
        sa.Column('request_custom_id', sa.String(), primary_key=True, nullable=False),
        sa.Column('difficulty_score', sa.Integer(), nullable=True),
        sa.Column('justification', sa.Text(), nullable=True),
    )

def downgrade():
    op.drop_table('knowledge_unit_evaluations_aggregated_batch')