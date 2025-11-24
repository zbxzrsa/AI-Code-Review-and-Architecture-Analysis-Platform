"""add tenant_id and repo_id to analysis_session

Revision ID: 20251120_add_tenant_repo
Revises: 20251120_add_analysis_cache
Create Date: 2025-11-20 01:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision = '20251120_add_tenant_repo'
down_revision = '20251120_add_analysis_cache'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add tenant_id and repo_id columns (nullable in stage 1)
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='analysis_session' AND column_name='tenant_id'
            ) THEN
                ALTER TABLE analysis_session ADD COLUMN tenant_id uuid NULL;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='analysis_session' AND column_name='repo_id'
            ) THEN
                ALTER TABLE analysis_session ADD COLUMN repo_id uuid NULL;
            END IF;
        END$$;
        """
    )

    # Create indexes on new columns
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_analysis_session_tenant_id ON analysis_session(tenant_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_analysis_session_repo_id ON analysis_session(repo_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_analysis_session_tenant_repo ON analysis_session(tenant_id, repo_id);"
    )

    # Add foreign keys (ON DELETE RESTRICT, ON UPDATE CASCADE)
    # Note: assumes tenants(id) and repos(id) tables exist.
    # If they don't exist in your schema yet, comment these out or create them first.
    # Commented out for cache integration testing
    # op.execute(
    #     """
    #     DO $$
    #     BEGIN
    #         IF NOT EXISTS (
    #             SELECT 1 FROM information_schema.table_constraints
    #             WHERE table_name='analysis_session' AND constraint_name='fk_analysis_session_tenant_id'
    #         ) THEN
    #             ALTER TABLE analysis_session
    #               ADD CONSTRAINT fk_analysis_session_tenant_id FOREIGN KEY (tenant_id)
    #               REFERENCES tenants(id) ON DELETE RESTRICT ON UPDATE CASCADE;
    #         END IF;
    #     END$$;
    #     """
    # )

    # op.execute(
    #     """
    #     DO $$
    #     BEGIN
    #         IF NOT EXISTS (
    #             SELECT 1 FROM information_schema.table_constraints
    #             WHERE table_name='analysis_session' AND constraint_name='fk_analysis_session_repo_id'
    #         ) THEN
    #             ALTER TABLE analysis_session
    #           ADD CONSTRAINT fk_analysis_session_repo_id FOREIGN KEY (repo_id)
    #           REFERENCES repos(id) ON DELETE RESTRICT ON UPDATE CASCADE;
    #         END IF;
    #     END$$;
    #     """
    # )


def downgrade() -> None:
    # Drop foreign keys
    # Commented out for cache integration testing
    # op.execute(
    #     """
    #     DO $$
    #     BEGIN
    #         IF EXISTS (
    #             SELECT 1 FROM information_schema.table_constraints
    #             WHERE table_name='analysis_session' AND constraint_name='fk_analysis_session_repo_id'
    #         ) THEN
    #             ALTER TABLE analysis_session DROP CONSTRAINT fk_analysis_session_repo_id;
    #         END IF;
    #         IF EXISTS (
    #             SELECT 1 FROM information_schema.table_constraints
    #             WHERE table_name='analysis_session' AND constraint_name='fk_analysis_session_tenant_id'
    #         ) THEN
    #             ALTER TABLE analysis_session DROP CONSTRAINT fk_analysis_session_tenant_id;
    #         END IF;
    #     END$$;
    #     """
    # )

    # Drop indexes
    op.execute("DROP INDEX IF EXISTS idx_analysis_session_tenant_repo;")
    op.execute("DROP INDEX IF EXISTS idx_analysis_session_repo_id;")
    op.execute("DROP INDEX IF EXISTS idx_analysis_session_tenant_id;")

    # Drop columns
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='analysis_session' AND column_name='repo_id'
            ) THEN
                ALTER TABLE analysis_session DROP COLUMN repo_id;
            END IF;
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='analysis_session' AND column_name='tenant_id'
            ) THEN
                ALTER TABLE analysis_session DROP COLUMN tenant_id;
            END IF;
        END$$;
        """
    )
