"""
create pipeline_runs table

Revision ID: 0001_create_pipeline_runs
Revises: 
Create Date: 2025-05-03 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_create_pipeline_runs'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'pipeline_runs',
        sa.Column('run_id', sa.String(), primary_key=True),
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('finished_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('status', sa.String(), nullable=False, server_default=sa.text("'running'")),
        sa.Column('trigger_source', sa.String(), nullable=True),
        sa.Column('payload', sa.JSON(), nullable=True),
    )

def downgrade():
    op.drop_table('pipeline_runs')