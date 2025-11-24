"""
Analysis Sessions API endpoints
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


class SessionArtifact(BaseModel):
    id: int
    type: str
    path: str
    size: int
    created_at: datetime


class AnalysisSession(BaseModel):
    id: int
    project_id: int
    label: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    summary: Optional[str]
    artifacts: List[SessionArtifact] = []


class SessionCreate(BaseModel):
    project_id: int
    label: str


class SessionUpdate(BaseModel):
    label: Optional[str] = None
    status: Optional[str] = None
    summary: Optional[str] = None


# Mock data for development
MOCK_SESSIONS = [
    AnalysisSession(
        id=1,
        project_id=1,
        label="Initial Analysis",
        status="completed",
        started_at=datetime(2024, 1, 15, 10, 0),
        completed_at=datetime(2024, 1, 15, 10, 30),
        summary="Found 12 issues, 3 security vulnerabilities",
        artifacts=[
            SessionArtifact(id=1, type="report", path="/reports/session_1.pdf", size=2048, created_at=datetime(2024, 1, 15, 10, 30)),
            SessionArtifact(id=2, type="log", path="/logs/session_1.log", size=1024, created_at=datetime(2024, 1, 15, 10, 30))
        ]
    ),
    AnalysisSession(
        id=2,
        project_id=1,
        label="Security Scan",
        status="running",
        started_at=datetime(2024, 1, 20, 14, 0),
        completed_at=None,
        summary=None,
        artifacts=[]
    ),
    AnalysisSession(
        id=3,
        project_id=2,
        label="Code Quality Check",
        status="completed",
        started_at=datetime(2024, 1, 18, 9, 0),
        completed_at=datetime(2024, 1, 18, 9, 45),
        summary="Code quality score: 8.5/10",
        artifacts=[
            SessionArtifact(id=3, type="report", path="/reports/session_3.pdf", size=3072, created_at=datetime(2024, 1, 18, 9, 45))
        ]
    )
]


@router.get("/", response_model=List[AnalysisSession])
async def list_sessions(
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Number of sessions to return")
):
    """Get list of analysis sessions with optional filtering"""
    sessions = MOCK_SESSIONS.copy()
    
    if project_id:
        sessions = [s for s in sessions if s.project_id == project_id]
    
    if status:
        sessions = [s for s in sessions if s.status == status]
    
    return sessions[:limit]


@router.get("/{session_id}", response_model=AnalysisSession)
async def get_session(session_id: int):
    """Get specific analysis session by ID"""
    session = next((s for s in MOCK_SESSIONS if s.id == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/", response_model=AnalysisSession)
async def create_session(session_data: SessionCreate):
    """Create new analysis session"""
    new_session = AnalysisSession(
        id=len(MOCK_SESSIONS) + 1,
        project_id=session_data.project_id,
        label=session_data.label,
        status="pending",
        started_at=datetime.now(),
        completed_at=None,
        summary=None,
        artifacts=[]
    )
    MOCK_SESSIONS.append(new_session)
    return new_session


@router.put("/{session_id}", response_model=AnalysisSession)
async def update_session(session_id: int, session_data: SessionUpdate):
    """Update analysis session"""
    session = next((s for s in MOCK_SESSIONS if s.id == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session_data.label is not None:
        session.label = session_data.label
    if session_data.status is not None:
        session.status = session_data.status
    if session_data.summary is not None:
        session.summary = session_data.summary
    
    return session


@router.delete("/{session_id}")
async def delete_session(session_id: int):
    """Delete analysis session"""
    global MOCK_SESSIONS
    session = next((s for s in MOCK_SESSIONS if s.id == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    MOCK_SESSIONS = [s for s in MOCK_SESSIONS if s.id != session_id]
    return {"message": "Session deleted successfully"}


@router.get("/{session_id}/artifacts", response_model=List[SessionArtifact])
async def get_session_artifacts(session_id: int):
    """Get artifacts for specific session"""
    session = next((s for s in MOCK_SESSIONS if s.id == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.artifacts