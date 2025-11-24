from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .base import Base

class AnalysisSession(Base):
    __tablename__ = "analysis_session"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("project.id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="RESTRICT", onupdate="CASCADE"), nullable=True, index=True)
    repo_id = Column(UUID(as_uuid=True), ForeignKey("repos.id", ondelete="RESTRICT", onupdate="CASCADE"), nullable=True, index=True)
    label = Column(String(255), nullable=True)
    idempotency_key = Column(String(128), nullable=False, default="")
    status = Column(String(50), nullable=False, default="completed")  # e.g., pending/running/completed/failed
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    summary = Column(JSON, nullable=True)

    project = relationship("Project", back_populates="sessions")
    artifacts = relationship("SessionArtifact", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_analysis_session_tenant_repo', 'tenant_id', 'repo_id'),
    )

class SessionArtifact(Base):
    __tablename__ = "session_artifact"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("analysis_session.id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(String(100), nullable=False)  # e.g., "metrics","diff","report"
    path = Column(String(1024), nullable=True)
    size = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("AnalysisSession", back_populates="artifacts")
