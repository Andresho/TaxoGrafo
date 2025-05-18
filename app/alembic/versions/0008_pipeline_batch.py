"""create pipeline_batch_jobs table

Revision ID: 0008_pipeline_batch
Revises: 0007_upload_tables
Create Date: 2025-05-19 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import func

# revision identifiers, used by Alembic.
revision = '0008_pipeline_batch'
down_revision = '0007_upload_tables'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'pipeline_batch_jobs',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('batch_type', sa.String(length=50), nullable=False),
        sa.Column('llm_batch_id', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, index=True),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False),
        sa.UniqueConstraint('pipeline_run_id', 'batch_type', name='uq_pipeline_run_batch_type')
    )

def downgrade():
    op.drop_table('pipeline_batch_jobs')