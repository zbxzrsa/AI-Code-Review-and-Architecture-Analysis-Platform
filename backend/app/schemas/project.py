"""
Pydantic schemas for Project API
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, validator


class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    visibility: str = Field("private", regex="^(public|private|internal)$")
    settings: Optional[Dict[str, Any]] = None


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    visibility: Optional[str] = Field(None, regex="^(public|private|internal)$")
    settings: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(None, regex="^(active|archived|suspended)$")


class RepositoryInfo(BaseModel):
    id: str
    name: str
    full_name: str
    provider: str
    is_active: bool
    last_synced_at: Optional[datetime] = None


class ProjectResponse(ProjectBase):
    id: str
    owner_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    repositories: List[RepositoryInfo] = []
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_orm(cls, obj):
        """Create response from ORM object"""
        return cls(
            id=str(obj.id),
            owner_id=str(obj.owner_id),
            name=obj.name,
            slug=obj.slug,
            description=obj.description,
            visibility=obj.visibility,
            status=obj.status,
            settings=obj.settings,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            repositories=[
                RepositoryInfo(
                    id=str(repo.id),
                    name=repo.name,
                    full_name=repo.full_name,
                    provider=repo.provider,
                    is_active=repo.is_active,
                    last_synced_at=repo.last_synced_at
                )
                for repo in getattr(obj, 'repositories', [])
            ]
        )


class ProjectListResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: Optional[str]
    visibility: str
    status: str
    created_at: datetime
    updated_at: datetime
    repository_count: int
    last_analysis: Optional[datetime] = None


class ProjectStatistics(BaseModel):
    project_id: str
    total_sessions: int
    total_findings: int
    critical_findings: int
    repository_count: int
    last_analysis: Optional[datetime] = None