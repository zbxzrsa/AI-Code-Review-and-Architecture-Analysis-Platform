from sqlalchemy import Column, String, Text, DateTime, func, Index
from sqlalchemy.dialects.postgresql import UUID
from .base import Base


class AnalysisCache(Base):
    __tablename__ = "analysis_cache"

    tenant_id = Column(UUID(as_uuid=True), nullable=False, primary_key=True)
    repo_id = Column(UUID(as_uuid=True), nullable=False, primary_key=True)
    file_path = Column(Text, nullable=False, primary_key=True)
    file_hash = Column(String(64), nullable=False, index=False)
    ast_fingerprint = Column(String(64), nullable=False, index=False)
    rulepack_version = Column(String(32), nullable=False, primary_key=False)
    result_hash = Column(String(64), nullable=False)
    payload_url = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_access_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)


# additional indexes
Index("ix_cache_file_hash", AnalysisCache.file_hash)
Index("ix_cache_ast_fp", AnalysisCache.ast_fingerprint)
Index("ix_cache_expiry", AnalysisCache.expires_at)
