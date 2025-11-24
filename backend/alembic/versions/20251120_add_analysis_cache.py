"""add analysis_cache, idempotency_key and findings columns

Revision ID: 20251120_add_analysis_cache
Revises: 20251020_0001
Create Date: 2025-11-20 00:00:00
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20251120_add_analysis_cache'
down_revision = '20251020_0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add idempotency_key to analysis_session if not exists
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='analysis_session' AND column_name='idempotency_key'
            ) THEN
                ALTER TABLE analysis_session ADD COLUMN idempotency_key varchar(128) NOT NULL DEFAULT '';
            END IF;
        END$$;
        """
    )

    # Add columns to findings (try plural then singular)
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='findings') THEN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='findings' AND column_name='source_version_id') THEN
                    ALTER TABLE findings ADD COLUMN source_version_id uuid NULL;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='findings' AND column_name='derived_from_cache') THEN
                    ALTER TABLE findings ADD COLUMN derived_from_cache boolean NOT NULL DEFAULT false;
                END IF;
            ELSIF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='finding') THEN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='finding' AND column_name='source_version_id') THEN
                    ALTER TABLE finding ADD COLUMN source_version_id uuid NULL;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='finding' AND column_name='derived_from_cache') THEN
                    ALTER TABLE finding ADD COLUMN derived_from_cache boolean NOT NULL DEFAULT false;
                END IF;
            END IF;
        END$$;
        """
    )

    # Create analysis_cache table if not exists
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_cache (
            tenant_id uuid NOT NULL,
            repo_id uuid NOT NULL,
            file_path text NOT NULL,
            file_hash varchar(64) NOT NULL,
            ast_fingerprint varchar(64) NOT NULL,
            rulepack_version varchar(32) NOT NULL,
            result_hash varchar(64) NOT NULL,
            payload_url text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            last_access_at timestamptz NOT NULL DEFAULT now(),
            expires_at timestamptz NOT NULL,
            PRIMARY KEY (tenant_id, repo_id, rulepack_version, file_path)
        );
        """
    )

    # Create indexes on analysis_cache
    op.execute("CREATE INDEX IF NOT EXISTS ix_cache_file_hash ON analysis_cache(file_hash);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_cache_ast_fp ON analysis_cache(ast_fingerprint);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_cache_expiry ON analysis_cache(expires_at);")

    # Create unique index on analysis_session if the necessary columns exist
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='analysis_session' AND column_name='tenant_id')
               AND EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='analysis_session' AND column_name='repo_id')
               AND EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='analysis_session' AND column_name='commit_sha')
               AND EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='analysis_session' AND column_name='rulepack_version') THEN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                    WHERE c.relname = 'uq_session_idem'
                ) THEN
                    CREATE UNIQUE INDEX uq_session_idem ON analysis_session(tenant_id, repo_id, commit_sha, rulepack_version);
                END IF;
            END IF;
        END$$;
        """
    )

    # Create audit table for analysis requests (queryable)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_audit (
            id serial PRIMARY KEY,
            session_id integer NULL,
            event_type varchar(100) NOT NULL,
            actor varchar(255) NULL,
            project_id integer NULL,
            commit_sha varchar(80) NULL,
            trace_id varchar(128) NULL,
            payload jsonb NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        );
        """
    )


def downgrade() -> None:
    # Drop analysis_audit
    op.execute("DROP TABLE IF EXISTS analysis_audit;")

    # Drop unique index if exists
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_class WHERE relname='uq_session_idem') THEN
                DROP INDEX uq_session_idem;
            END IF;
        END$$;
        """
    )

    # Drop indexes on analysis_cache
    op.execute("DROP INDEX IF EXISTS ix_cache_file_hash;")
    op.execute("DROP INDEX IF EXISTS ix_cache_ast_fp;")
    op.execute("DROP INDEX IF EXISTS ix_cache_expiry;")

    # Drop analysis_cache
    op.execute("DROP TABLE IF EXISTS analysis_cache;")

    # Remove columns from findings/finding if exist
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='findings') THEN
                IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='findings' AND column_name='source_version_id') THEN
                    ALTER TABLE findings DROP COLUMN source_version_id;
                END IF;
                IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='findings' AND column_name='derived_from_cache') THEN
                    ALTER TABLE findings DROP COLUMN derived_from_cache;
                END IF;
            ELSIF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='finding') THEN
                IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='finding' AND column_name='source_version_id') THEN
                    ALTER TABLE finding DROP COLUMN source_version_id;
                END IF;
                IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='finding' AND column_name='derived_from_cache') THEN
                    ALTER TABLE finding DROP COLUMN derived_from_cache;
                END IF;
            END IF;
        END$$;
        """
    )

    # Drop idempotency_key from analysis_session if exists
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='analysis_session' AND column_name='idempotency_key') THEN
                ALTER TABLE analysis_session DROP COLUMN idempotency_key;
            END IF;
        END$$;
        """
    )
