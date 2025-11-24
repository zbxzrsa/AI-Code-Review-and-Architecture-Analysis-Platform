"""initial stub revision

Revision ID: 0001_initial
Revises: 
Create Date: 2025-10-16 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # This is a stub migration; schemas are initialized via docker/postgres/init.sql
    pass


def downgrade():
    pass