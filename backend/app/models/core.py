"""
Core domain models for AI Code Review Platform
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime, 
    ForeignKey, JSON, Index, UniqueConstraint, func
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Tenant(Base):
    """Multi-tenant support"""
    __tablename__ = "tenants"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    domain = Column(String(255), unique=True, nullable=True)
    settings = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    projects = relationship("Project", back_populates="tenant")
    users = relationship("UserTenant", back_populates="tenant")


class User(Base):
    """User model with authentication"""
    __tablename__ = "users"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    avatar_url = Column(String(500))
    github_id = Column(Integer, unique=True, nullable=True)
    github_username = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    last_login = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    tenant_memberships = relationship("UserTenant", back_populates="user")
    created_projects = relationship("Project", back_populates="owner")
    sessions = relationship("AnalysisSession", back_populates="created_by")


class UserTenant(Base):
    """Many-to-many relationship between users and tenants"""
    __tablename__ = "user_tenants"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    tenant_id = Column(PG_UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    role = Column(String(50), nullable=False)  # admin, maintainer, developer, viewer
    permissions = Column(JSONB, default={})
    invited_at = Column(DateTime(timezone=True), server_default=func.now())
    joined_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="tenant_memberships")
    tenant = relationship("Tenant", back_populates="users")
    
    __table_args__ = (UniqueConstraint('user_id', 'tenant_id'),)


class Project(Base):
    """Project model for organizing repositories"""
    __tablename__ = "projects"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(PG_UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    owner_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), nullable=False)
    description = Column(Text)
    visibility = Column(String(20), default="private")  # public, private, internal
    status = Column(String(20), default="active")  # active, archived, suspended
    settings = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    tenant = relationship("Tenant", back_populates="projects")
    owner = relationship("User", back_populates="created_projects")
    repositories = relationship("Repository", back_populates="project")
    baselines = relationship("Baseline", back_populates="project")
    policies = relationship("Policy", back_populates="project")
    sessions = relationship("AnalysisSession", back_populates="project")
    
    __table_args__ = (
        Index('idx_project_tenant_slug', 'tenant_id', 'slug'),
        UniqueConstraint('tenant_id', 'slug'),
    )


class Repository(Base):
    """Repository model linked to external Git repositories"""
    __tablename__ = "repositories"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    full_name = Column(String(500), nullable=False)
    provider = Column(String(50), default="github")  # github, gitlab, bitbucket
    external_id = Column(Integer, nullable=False)
    external_url = Column(String(500))
    clone_url = Column(String(500))
    default_branch = Column(String(100), default="main")
    is_active = Column(Boolean, default=True)
    last_synced_at = Column(DateTime(timezone=True))
    sync_status = Column(String(20), default="pending")  # pending, syncing, success, error
    settings = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="repositories")
    pull_requests = relationship("PullRequest", back_populates="repository")
    sessions = relationship("AnalysisSession", back_populates="repository")
    
    __table_args__ = (
        Index('idx_repo_project_provider', 'project_id', 'provider'),
        UniqueConstraint('provider', 'external_id'),
    )


class PullRequest(Base):
    """Pull Request model for tracking PR analysis"""
    __tablename__ = "pull_requests"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    repository_id = Column(PG_UUID(as_uuid=True), ForeignKey("repositories.id"), nullable=False)
    external_id = Column(Integer, nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    author = Column(String(255), nullable=False)
    source_branch = Column(String(100), nullable=False)
    target_branch = Column(String(100), nullable=False)
    status = Column(String(20), default="open")  # open, closed, merged
    mergeable = Column(Boolean, nullable=True)
    additions = Column(Integer, default=0)
    deletions = Column(Integer, default=0)
    changed_files = Column(Integer, default=0)
    external_url = Column(String(500))
    external_updated_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    repository = relationship("Repository", back_populates="pull_requests")
    sessions = relationship("AnalysisSession", back_populates="pull_request")
    
    __table_args__ = (
        Index('idx_pr_repo_external', 'repository_id', 'external_id'),
        UniqueConstraint('repository_id', 'external_id'),
    )


class AnalysisSession(Base):
    """Analysis session for tracking analysis runs"""
    __tablename__ = "analysis_sessions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    repository_id = Column(PG_UUID(as_uuid=True), ForeignKey("repositories.id"), nullable=True)
    pull_request_id = Column(PG_UUID(as_uuid=True), ForeignKey("pull_requests.id"), nullable=True)
    created_by_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Analysis configuration
    analysis_type = Column(String(50), nullable=False)  # full, pr, diff, custom
    config = Column(JSONB, default={})
    rules_enabled = Column(JSONB, default=[])
    
    # Status and timing
    status = Column(String(20), default="pending")  # pending, running, completed, failed, cancelled
    progress = Column(Float, default=0.0)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer)
    
    # Results summary
    total_findings = Column(Integer, default=0)
    critical_findings = Column(Integer, default=0)
    high_findings = Column(Integer, default=0)
    medium_findings = Column(Integer, default=0)
    low_findings = Column(Integer, default=0)
    
    # Metadata
    commit_sha = Column(String(40))
    branch = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="sessions")
    repository = relationship("Repository", back_populates="sessions")
    pull_request = relationship("PullRequest", back_populates="sessions")
    created_by = relationship("User", back_populates="sessions")
    findings = relationship("Finding", back_populates="session")
    artifacts = relationship("SessionArtifact", back_populates="session")
    
    __table_args__ = (
        Index('idx_session_project_status', 'project_id', 'status'),
        Index('idx_session_repo_pr', 'repository_id', 'pull_request_id'),
    )


class Finding(Base):
    """Finding model for individual analysis results"""
    __tablename__ = "findings"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey("analysis_sessions.id"), nullable=False)
    
    # Finding details
    rule_id = Column(String(100), nullable=False)
    rule_name = Column(String(255), nullable=False)
    category = Column(String(50), nullable=False)  # security, quality, performance, architecture
    severity = Column(String(20), nullable=False)  # critical, high, medium, low, info
    confidence = Column(String(20), nullable=False)  # high, medium, low
    
    # Location information
    file_path = Column(String(1000), nullable=False)
    line_number = Column(Integer)
    end_line_number = Column(Integer)
    column_number = Column(Integer)
    end_column_number = Column(Integer)
    
    # Content
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    recommendation = Column(Text)
    code_snippet = Column(Text)
    cwe_id = Column(String(10))  # CWE identifier for security issues
    owasp_category = Column(String(100))
    
    # Status and metadata
    status = Column(String(20), default="open")  # open, acknowledged, resolved, false_positive, suppressed
    is_suppressed = Column(Boolean, default=False)
    suppression_reason = Column(Text)
    suppressed_until = Column(DateTime(timezone=True))
    suppressed_by_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    
    # AI-generated content
    ai_analysis = Column(JSONB)
    ai_suggestion = Column(Text)
    ai_confidence = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    session = relationship("AnalysisSession", back_populates="findings")
    suppressed_by = relationship("User")
    
    __table_args__ = (
        Index('idx_finding_session_severity', 'session_id', 'severity'),
        Index('idx_finding_file_line', 'file_path', 'line_number'),
        Index('idx_finding_category_status', 'category', 'status'),
    )


class SessionArtifact(Base):
    """Artifacts generated during analysis sessions"""
    __tablename__ = "session_artifacts"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey("analysis_sessions.id"), nullable=False)
    
    # Artifact details
    artifact_type = Column(String(50), nullable=False)  # report, graph, log, screenshot, patch
    name = Column(String(255), nullable=False)
    description = Column(Text)
    file_path = Column(String(1000))
    file_size = Column(Integer)
    mime_type = Column(String(100))
    checksum = Column(String(64))  # SHA-256
    
    # Storage information
    storage_type = Column(String(20), default="local")  # local, s3, gcs
    storage_path = Column(String(1000))
    storage_metadata = Column(JSONB)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("AnalysisSession", back_populates="artifacts")


class Baseline(Base):
    """Baseline model for quality governance"""
    __tablename__ = "baselines"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Baseline configuration
    thresholds = Column(JSONB, nullable=False)  # Quality thresholds
    rules = Column(JSONB, default=[])  # Enabled rules
    exceptions = Column(JSONB, default=[])  # Rule exceptions
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    version = Column(Integer, default=1)
    created_by_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    applied_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="baselines")
    created_by = relationship("User")
    
    __table_args__ = (
        Index('idx_baseline_project_active', 'project_id', 'is_active'),
        UniqueConstraint('project_id', 'name'),
    )


class Policy(Base):
    """Policy model for governance rules"""
    __tablename__ = "policies"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Policy configuration
    policy_type = Column(String(50), nullable=False)  # quality_gate, security_policy, approval_policy
    conditions = Column(JSONB, nullable=False)  # Policy conditions
    actions = Column(JSONB, nullable=False)  # Policy actions
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=0)
    created_by_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="policies")
    created_by = relationship("User")


class Provider(Base):
    """AI/Analysis provider configuration"""
    __tablename__ = "providers"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False)
    provider_type = Column(String(50), nullable=False)  # ai, static_analysis, graph_analysis
    config = Column(JSONB, nullable=False)  # Provider-specific configuration
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    rate_limit = Column(JSONB)  # Rate limiting configuration
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (UniqueConstraint('name', 'provider_type'),)


class AuditLog(Base):
    """Audit log for compliance and tracking"""
    __tablename__ = "audit_logs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(PG_UUID(as_uuid=True), ForeignKey("tenants.id"))
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Event details
    event_type = Column(String(100), nullable=False)
    event_category = Column(String(50), nullable=False)  # auth, project, analysis, policy
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(100))
    
    # Event data
    details = Column(JSONB)
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    
    # Request context
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    request_id = Column(String(100))
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    tenant = relationship("Tenant")
    user = relationship("User")
    
    __table_args__ = (
        Index('idx_audit_tenant_user', 'tenant_id', 'user_id'),
        Index('idx_audit_event_time', 'event_type', 'created_at'),
    )


class SavedView(Base):
    """Saved search views and filters"""
    __tablename__ = "saved_views"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey("projects.id"))
    
    # View details
    name = Column(String(255), nullable=False)
    description = Column(Text)
    view_type = Column(String(50), nullable=False)  # findings, sessions, prs, reports
    
    # View configuration
    filters = Column(JSONB, nullable=False)
    columns = Column(JSONB)
    sort_config = Column(JSONB)
    
    # Sharing and metadata
    is_public = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User")
    project = relationship("Project")
    
    __table_args__ = (
        Index('idx_view_user_project', 'user_id', 'project_id'),
        UniqueConstraint('user_id', 'project_id', 'name'),
    )