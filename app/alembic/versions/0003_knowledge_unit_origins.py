"""
create knowledge_unit_origins table

Revision ID: 0003_knowledge_unit_origins
Revises: 0002_create_output_tables
Create Date: 2025-05-03 12:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0003_knowledge_unit_origins'
down_revision = '0002_create_output_tables'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'knowledge_unit_origins',
        sa.Column('pipeline_run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True),
        sa.Column('origin_id', sa.String(), primary_key=True),
        sa.Column('origin_type', sa.String(), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('context', sa.Text(), nullable=True),
        sa.Column('frequency', sa.Integer(), nullable=True),
        sa.Column('degree', sa.Integer(), nullable=True),
        sa.Column('entity_type', sa.String(), nullable=True),
        sa.Column('level', sa.Integer(), nullable=True),
        sa.Column('parent_community_id_of_origin', sa.String(), nullable=True),
    )

def downgrade():
    op.drop_table('knowledge_unit_origins')