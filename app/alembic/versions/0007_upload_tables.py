"""create_resources_and_run_resources_tables

Revision ID: 0007_upload_tables
Revises: 0006_create_diff_comp
Create Date: 2025-05-13 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0007_upload_tables'
down_revision = '0006_create_diff_comp'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'resources',
        sa.Column('resource_id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column('original_filename', sa.Text(), nullable=False),
        sa.Column('original_mime_type', sa.String(length=100), nullable=False),
        sa.Column('original_file_path', sa.Text(), nullable=False, unique=True),
        sa.Column('processed_txt_path', sa.Text(), nullable=True, unique=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='uploaded'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('uploaded_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('processed_at', sa.TIMESTAMP(timezone=True), nullable=True)
    )
    op.create_index(op.f('ix_resources_status'), 'resources', ['status'], unique=False)

    op.create_table(
        'pipeline_run_resources',
        sa.Column('run_id', sa.String(), sa.ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'), primary_key=True),
        sa.Column('resource_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('resources.resource_id', ondelete='CASCADE'), primary_key=True)
    )

def downgrade():
    op.drop_table('pipeline_run_resources')
    op.drop_index(op.f('ix_resources_status'), table_name='resources')
    op.drop_table('resources')
