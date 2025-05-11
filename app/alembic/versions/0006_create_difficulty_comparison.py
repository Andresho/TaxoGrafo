"""Create difficulty comparison groups and association tables, modify evaluations table

Revision ID: 0006_create_diff_comp
Revises: 0005_knowledge_eval_batch
Create Date: 2025-05-04 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0006_create_diff_comp'
down_revision = '0005_knowledge_eval_batch'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'difficulty_comparison_groups',
        sa.Column('pipeline_run_id', sa.String(255), sa.ForeignKey('pipeline_runs.run_id', ondelete='CASCADE'),
                  primary_key=True),
        sa.Column('comparison_group_id', sa.String(36), primary_key=True), 
        sa.Column('bloom_level', sa.String(50), nullable=False),
        sa.Column('coherence_level', sa.String(50), nullable=False), 
        sa.Column('llm_batch_request_custom_id', sa.String(100), nullable=False),
        
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Index('ix_difficulty_comparison_groups_llm_custom_id', 'llm_batch_request_custom_id')
    )

    op.create_table(
        'difficulty_group_origin_association',
        sa.Column('pipeline_run_id', sa.String(255), primary_key=True),
        sa.Column('comparison_group_id', sa.String(36), primary_key=True),
        sa.Column('origin_id', sa.String(36), primary_key=True),
        sa.Column('is_seed_origin', sa.Boolean(), default=False, nullable=False),
        sa.ForeignKeyConstraint(
            ['pipeline_run_id', 'comparison_group_id'],
            ['difficulty_comparison_groups.pipeline_run_id', 'difficulty_comparison_groups.comparison_group_id'],
            ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['pipeline_run_id', 'origin_id'],
            ['knowledge_unit_origins.pipeline_run_id', 'knowledge_unit_origins.origin_id'],
            ondelete='CASCADE'
        )
    )

    with op.batch_alter_table('knowledge_unit_evaluations_aggregated_batch', schema=None) as batch_op:
        batch_op.alter_column(
            'request_custom_id',
            new_column_name='comparison_group_id',
            type_=sa.String(36),
            existing_type=sa.String(),
            nullable=False
        )

        batch_op.create_foreign_key(
            'fk_eval_batch_to_comp_group',
            'difficulty_comparison_groups',
            ['pipeline_run_id', 'comparison_group_id'],
            ['pipeline_run_id', 'comparison_group_id'],
            ondelete='CASCADE'
        )


def downgrade():
    with op.batch_alter_table('knowledge_unit_evaluations_aggregated_batch', schema=None) as batch_op:
        batch_op.drop_constraint('fk_eval_batch_to_comp_group', type_='foreignkey')
        batch_op.alter_column(
            'comparison_group_id',
            new_column_name='request_custom_id',
            type_=sa.String(),
            existing_type=sa.String(36),
            nullable=False
        )

    op.drop_table('difficulty_group_origin_association')
    op.drop_table('difficulty_comparison_groups')