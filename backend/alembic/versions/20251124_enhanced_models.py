"""Enhanced models migration

Revision ID: 20251124_enhanced_models
Revises: 20251120_add_tenant_repo
Create Date: 2025-11-24 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20251124_enhanced_models'
down_revision = '20251120_add_tenant_repo'
branch_labels = None
depends_on = None


def upgrade():
    # Create enhanced tables
    op.create_table('tenants',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('slug', sa.String(length=100), nullable=False),
        sa.Column('domain', sa.String(length=255), nullable=True),
        sa.Column('settings', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('slug'),
        sa.UniqueConstraint('domain')
    )
    op.create_index('idx_tenant_slug', 'tenants', ['slug'], unique=False)

    op.create_table('user_tenants',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('permissions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('invited_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('joined_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.UniqueConstraint('user_id', 'tenant_id')
    )

    op.create_table('repositories',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=500), nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=False),
        sa.Column('external_id', sa.Integer(), nullable=False),
        sa.Column('external_url', sa.String(length=500), nullable=True),
        sa.Column('clone_url', sa.String(length=500), nullable=True),
        sa.Column('default_branch', sa.String(length=100), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('last_synced_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('sync_status', sa.String(length=20), nullable=True),
        sa.Column('settings', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.UniqueConstraint('provider', 'external_id')
    )
    op.create_index('idx_repo_project_provider', 'repositories', ['project_id', 'provider'], unique=False)

    op.create_table('pull_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('repository_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('external_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('author', sa.String(length=255), nullable=False),
        sa.Column('source_branch', sa.String(length=100), nullable=False),
        sa.Column('target_branch', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('mergeable', sa.Boolean(), nullable=True),
        sa.Column('additions', sa.Integer(), nullable=True),
        sa.Column('deletions', sa.Integer(), nullable=True),
        sa.Column('changed_files', sa.Integer(), nullable=True),
        sa.Column('external_url', sa.String(length=500), nullable=True),
        sa.Column('external_updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['repository_id'], ['repositories.id'], ),
        sa.UniqueConstraint('repository_id', 'external_id')
    )
    op.create_index('idx_pr_repo_external', 'pull_requests', ['repository_id', 'external_id'], unique=False)

    op.create_table('findings',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rule_id', sa.String(length=100), nullable=False),
        sa.Column('rule_name', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('confidence', sa.String(length=20), nullable=False),
        sa.Column('file_path', sa.String(length=1000), nullable=False),
        sa.Column('line_number', sa.Integer(), nullable=True),
        sa.Column('end_line_number', sa.Integer(), nullable=True),
        sa.Column('column_number', sa.Integer(), nullable=True),
        sa.Column('end_column_number', sa.Integer(), nullable=True),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('recommendation', sa.Text(), nullable=True),
        sa.Column('code_snippet', sa.Text(), nullable=True),
        sa.Column('cwe_id', sa.String(length=10), nullable=True),
        sa.Column('owasp_category', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('is_suppressed', sa.Boolean(), nullable=True),
        sa.Column('suppression_reason', sa.Text(), nullable=True),
        sa.Column('suppressed_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('suppressed_by_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('ai_analysis', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('ai_suggestion', sa.Text(), nullable=True),
        sa.Column('ai_confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['session_id'], ['analysis_sessions.id'], ),
        sa.ForeignKeyConstraint(['suppressed_by_id'], ['users.id'], )
    )
    op.create_index('idx_finding_session_severity', 'findings', ['session_id', 'severity'], unique=False)
    op.create_index('idx_finding_file_line', 'findings', ['file_path', 'line_number'], unique=False)
    op.create_index('idx_finding_category_status', 'findings', ['category', 'status'], unique=False)

    op.create_table('policies',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('policy_type', sa.String(length=50), nullable=False),
        sa.Column('conditions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('actions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=True),
        sa.Column('created_by_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.ForeignKeyConstraint(['created_by_id'], ['users.id'], )
    )

    op.create_table('providers',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('provider_type', sa.String(length=50), nullable=False),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_default', sa.Boolean(), nullable=True),
        sa.Column('rate_limit', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name', 'provider_type')
    )

    op.create_table('audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('event_category', sa.String(length=50), nullable=False),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('resource_type', sa.String(length=50), nullable=True),
        sa.Column('resource_id', sa.String(length=100), nullable=True),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('old_values', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('new_values', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('request_id', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], )
    )
    op.create_index('idx_audit_tenant_user', 'audit_logs', ['tenant_id', 'user_id'], unique=False)
    op.create_index('idx_audit_event_time', 'audit_logs', ['event_type', 'created_at'], unique=False)

    op.create_table('saved_views',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('view_type', sa.String(length=50), nullable=False),
        sa.Column('filters', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('columns', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('sort_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_public', sa.Boolean(), nullable=True),
        sa.Column('is_default', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.UniqueConstraint('user_id', 'project_id', 'name')
    )
    op.create_index('idx_view_user_project', 'saved_views', ['user_id', 'project_id'], unique=False)

    # Add new columns to existing tables
    op.add_column('projects', sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('projects', sa.Column('visibility', sa.String(length=20), nullable=True))
    op.add_column('projects', sa.Column('status', sa.String(length=20), nullable=True))
    op.add_column('projects', sa.Column('settings', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    
    op.add_column('analysis_sessions', sa.Column('repository_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('analysis_sessions', sa.Column('pull_request_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('analysis_sessions', sa.Column('created_by_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('analysis_sessions', sa.Column('analysis_type', sa.String(length=50), nullable=True))
    op.add_column('analysis_sessions', sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('analysis_sessions', sa.Column('rules_enabled', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('analysis_sessions', sa.Column('progress', sa.Float(), nullable=True))
    op.add_column('analysis_sessions', sa.Column('started_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('analysis_sessions', sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('analysis_sessions', sa.Column('duration_seconds', sa.Integer(), nullable=True))
    op.add_column('analysis_sessions', sa.Column('total_findings', sa.Integer(), nullable=True))
    op.add_column('analysis_sessions', sa.Column('critical_findings', sa.Integer(), nullable=True))
    op.add_column('analysis_sessions', sa.Column('high_findings', sa.Integer(), nullable=True))
    op.add_column('analysis_sessions', sa.Column('medium_findings', sa.Integer(), nullable=True))
    op.add_column('analysis_sessions', sa.Column('low_findings', sa.Integer(), nullable=True))
    op.add_column('analysis_sessions', sa.Column('commit_sha', sa.String(length=40), nullable=True))
    op.add_column('analysis_sessions', sa.Column('branch', sa.String(length=100), nullable=True))

    # Add foreign key constraints
    op.create_foreign_key('fk_projects_tenant_id', 'projects', 'tenants', ['tenant_id'], ['id'])
    op.create_foreign_key('fk_sessions_repository_id', 'analysis_sessions', 'repositories', ['repository_id'], ['id'])
    op.create_foreign_key('fk_sessions_pull_request_id', 'analysis_sessions', 'pull_requests', ['pull_request_id'], ['id'])
    op.create_foreign_key('fk_sessions_created_by_id', 'analysis_sessions', 'users', ['created_by_id'], ['id'])


def downgrade():
    # Remove foreign key constraints
    op.drop_constraint('fk_projects_tenant_id', 'projects', type_='foreignkey')
    op.drop_constraint('fk_sessions_repository_id', 'analysis_sessions', type_='foreignkey')
    op.drop_constraint('fk_sessions_pull_request_id', 'analysis_sessions', type_='foreignkey')
    op.drop_constraint('fk_sessions_created_by_id', 'analysis_sessions', type_='foreignkey')

    # Remove new columns from existing tables
    op.drop_column('analysis_sessions', 'branch')
    op.drop_column('analysis_sessions', 'commit_sha')
    op.drop_column('analysis_sessions', 'low_findings')
    op.drop_column('analysis_sessions', 'medium_findings')
    op.drop_column('analysis_sessions', 'high_findings')
    op.drop_column('analysis_sessions', 'critical_findings')
    op.drop_column('analysis_sessions', 'total_findings')
    op.drop_column('analysis_sessions', 'duration_seconds')
    op.drop_column('analysis_sessions', 'completed_at')
    op.drop_column('analysis_sessions', 'started_at')
    op.drop_column('analysis_sessions', 'progress')
    op.drop_column('analysis_sessions', 'rules_enabled')
    op.drop_column('analysis_sessions', 'config')
    op.drop_column('analysis_sessions', 'analysis_type')
    op.drop_column('analysis_sessions', 'created_by_id')
    op.drop_column('analysis_sessions', 'pull_request_id')
    op.drop_column('analysis_sessions', 'repository_id')
    
    op.drop_column('projects', 'settings')
    op.drop_column('projects', 'status')
    op.drop_column('projects', 'visibility')
    op.drop_column('projects', 'tenant_id')

    # Drop new tables
    op.drop_table('saved_views')
    op.drop_table('audit_logs')
    op.drop_table('providers')
    op.drop_table('policies')
    op.drop_table('findings')
    op.drop_table('pull_requests')
    op.drop_table('repositories')
    op.drop_table('user_tenants')
    op.drop_table('tenants')