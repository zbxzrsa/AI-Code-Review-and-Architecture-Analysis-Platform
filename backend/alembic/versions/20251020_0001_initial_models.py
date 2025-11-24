"""initial models

Revision ID: 20251020_0001
Revises: 
Create Date: 2025-10-20 00:01:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20251020_0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # project
    op.create_table(
        'project',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('name', sa.String(length=255), nullable=False, unique=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )

    # analysis_session
    op.create_table(
        'analysis_session',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('project_id', sa.Integer(), sa.ForeignKey('project.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('label', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='completed'),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('summary', sa.JSON(), nullable=True),
    )

    # session_artifact
    op.create_table(
        'session_artifact',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('session_id', sa.Integer(), sa.ForeignKey('analysis_session.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('type', sa.String(length=100), nullable=False),
        sa.Column('path', sa.String(length=1024), nullable=True),
        sa.Column('size', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )

    # file_version
    op.create_table(
        'file_version',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('project_id', sa.Integer(), sa.ForeignKey('project.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('file_path', sa.String(length=1024), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_file_version_sha256', 'file_version', ['sha256'])

    # baseline
    op.create_table(
        'baseline',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('project_id', sa.Integer(), sa.ForeignKey('project.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.String(length=1024), nullable=True),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )

    # baseline_deviation
    op.create_table(
        'baseline_deviation',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('baseline_id', sa.Integer(), sa.ForeignKey('baseline.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('metric_name', sa.String(length=255), nullable=False),
        sa.Column('deviation_value', sa.Float(), nullable=False),
        sa.Column('severity', sa.String(length=50), nullable=False, server_default='medium'),
        sa.Column('detected_at', sa.DateTime(), nullable=False),
    )

    # alert_rule
    op.create_table(
        'alert_rule',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('project_id', sa.Integer(), sa.ForeignKey('project.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('rule_name', sa.String(length=255), nullable=False),
        sa.Column('condition', sa.JSON(), nullable=False),
        sa.Column('severity', sa.String(length=50), nullable=False, server_default='medium'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )

    # alert_event
    op.create_table(
        'alert_event',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('rule_id', sa.Integer(), sa.ForeignKey('alert_rule.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('session_id', sa.Integer(), sa.ForeignKey('analysis_session.id', ondelete='SET NULL'), nullable=True, index=True),
        sa.Column('triggered_at', sa.DateTime(), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='open'),
    )


def downgrade() -> None:
    op.drop_table('alert_event')
    op.drop_table('alert_rule')
    op.drop_table('baseline_deviation')
    op.drop_table('baseline')
    op.drop_index('ix_file_version_sha256', table_name='file_version')
    op.drop_table('file_version')
    op.drop_table('session_artifact')
    op.drop_table('analysis_session')
    op.drop_table('project')