"""
FastAPI v1 API Router for Projects
"""
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Project, User, AnalysisSession
from app.schemas.project import (
    ProjectCreate, ProjectUpdate, ProjectResponse, ProjectListResponse
)
from app.api.deps import get_current_user, get_current_active_user

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


@router.get("/", response_model=List[ProjectListResponse])
async def list_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    visibility: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List projects with filtering and pagination"""
    query = db.query(Project).filter(Project.owner_id == current_user.id)
    
    if search:
        query = query.filter(
            Project.name.ilike(f"%{search}%") | 
            Project.description.ilike(f"%{search}%")
        )
    
    if status:
        query = query.filter(Project.status == status)
    
    if visibility:
        query = query.filter(Project.visibility == visibility)
    
    projects = query.offset(skip).limit(limit).all()
    return [
        ProjectListResponse(
            id=str(project.id),
            name=project.name,
            slug=project.slug,
            description=project.description,
            visibility=project.visibility,
            status=project.status,
            created_at=project.created_at,
            updated_at=project.updated_at,
            repository_count=len(project.repositories),
            last_analysis=max(
                (s.created_at for s in project.sessions), default=None
            )
        )
        for project in projects
    ]


@router.post("/", response_model=ProjectResponse)
async def create_project(
    project_data: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new project"""
    # Check if project slug already exists for this user
    existing = db.query(Project).filter(
        Project.owner_id == current_user.id,
        Project.slug == project_data.slug
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Project with this slug already exists"
        )
    
    project = Project(
        owner_id=current_user.id,
        name=project_data.name,
        slug=project_data.slug,
        description=project_data.description,
        visibility=project_data.visibility,
        settings=project_data.settings or {}
    )
    
    db.add(project)
    db.commit()
    db.refresh(project)
    
    return ProjectResponse.from_orm(project)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get project by ID"""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return ProjectResponse.from_orm(project)


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: UUID,
    project_data: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update project"""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Update fields
    update_data = project_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)
    
    db.commit()
    db.refresh(project)
    
    return ProjectResponse.from_orm(project)


@router.delete("/{project_id}")
async def delete_project(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete project"""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(project)
    db.commit()
    
    return {"message": "Project deleted successfully"}


@router.get("/{project_id}/statistics")
async def get_project_statistics(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get project statistics"""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get statistics
    total_sessions = db.query(AnalysisSession).filter(
        AnalysisSession.project_id == project_id
    ).count()
    
    total_findings = db.query(AnalysisSession).filter(
        AnalysisSession.project_id == project_id
    ).with_entities(
        db.func.sum(AnalysisSession.total_findings)
    ).scalar() or 0
    
    critical_findings = db.query(AnalysisSession).filter(
        AnalysisSession.project_id == project_id
    ).with_entities(
        db.func.sum(AnalysisSession.critical_findings)
    ).scalar() or 0
    
    return {
        "project_id": str(project_id),
        "total_sessions": total_sessions,
        "total_findings": total_findings,
        "critical_findings": critical_findings,
        "repository_count": len(project.repositories),
        "last_analysis": max(
            (s.created_at for s in project.sessions), default=None
        )
    }