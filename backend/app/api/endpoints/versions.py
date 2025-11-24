"""
File Versions API endpoints
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/versions", tags=["versions"])


class FileVersion(BaseModel):
    id: int
    project_id: int
    file_path: str
    sha256: str
    created_at: datetime


class FileVersionCreate(BaseModel):
    project_id: int
    file_path: str
    sha256: str


class VersionComparison(BaseModel):
    file_path: str
    old_version: Optional[FileVersion]
    new_version: Optional[FileVersion]
    status: str  # "added", "modified", "deleted", "unchanged"


# Mock data for development
MOCK_VERSIONS = [
    FileVersion(
        id=1,
        project_id=1,
        file_path="src/main.py",
        sha256="a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
        created_at=datetime(2024, 1, 15, 10, 0)
    ),
    FileVersion(
        id=2,
        project_id=1,
        file_path="src/utils.py",
        sha256="b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567",
        created_at=datetime(2024, 1, 15, 10, 5)
    ),
    FileVersion(
        id=3,
        project_id=1,
        file_path="src/main.py",
        sha256="c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345678",
        created_at=datetime(2024, 1, 20, 14, 0)
    ),
    FileVersion(
        id=4,
        project_id=2,
        file_path="app/models.py",
        sha256="d4e5f6789012345678901234567890abcdef1234567890abcdef123456789",
        created_at=datetime(2024, 1, 18, 9, 0)
    )
]


@router.get("/", response_model=List[FileVersion])
async def list_versions(
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    file_path: Optional[str] = Query(None, description="Filter by file path"),
    limit: int = Query(50, ge=1, le=100, description="Number of versions to return")
):
    """Get list of file versions with optional filtering"""
    versions = MOCK_VERSIONS.copy()
    
    if project_id:
        versions = [v for v in versions if v.project_id == project_id]
    
    if file_path:
        versions = [v for v in versions if file_path in v.file_path]
    
    # Sort by created_at descending (newest first)
    versions.sort(key=lambda x: x.created_at, reverse=True)
    
    return versions[:limit]


@router.get("/{version_id}", response_model=FileVersion)
async def get_version(version_id: int):
    """Get specific file version by ID"""
    version = next((v for v in MOCK_VERSIONS if v.id == version_id), None)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    return version


@router.post("/", response_model=FileVersion)
async def create_version(version_data: FileVersionCreate):
    """Create new file version"""
    new_version = FileVersion(
        id=len(MOCK_VERSIONS) + 1,
        project_id=version_data.project_id,
        file_path=version_data.file_path,
        sha256=version_data.sha256,
        created_at=datetime.now()
    )
    MOCK_VERSIONS.append(new_version)
    return new_version


@router.get("/project/{project_id}/files", response_model=List[str])
async def get_project_files(project_id: int):
    """Get list of unique file paths for a project"""
    project_versions = [v for v in MOCK_VERSIONS if v.project_id == project_id]
    unique_files = list(set(v.file_path for v in project_versions))
    return sorted(unique_files)


@router.get("/project/{project_id}/file/{file_path:path}/history", response_model=List[FileVersion])
async def get_file_history(project_id: int, file_path: str):
    """Get version history for a specific file"""
    file_versions = [
        v for v in MOCK_VERSIONS 
        if v.project_id == project_id and v.file_path == file_path
    ]
    # Sort by created_at descending (newest first)
    file_versions.sort(key=lambda x: x.created_at, reverse=True)
    return file_versions


@router.post("/compare", response_model=List[VersionComparison])
async def compare_versions(
    project_id: int,
    old_timestamp: Optional[datetime] = None,
    new_timestamp: Optional[datetime] = None
):
    """Compare file versions between two timestamps"""
    if not old_timestamp:
        old_timestamp = datetime.min
    if not new_timestamp:
        new_timestamp = datetime.now()
    
    project_versions = [v for v in MOCK_VERSIONS if v.project_id == project_id]
    
    # Get versions at old timestamp
    old_versions = {}
    for version in project_versions:
        if version.created_at <= old_timestamp:
            if version.file_path not in old_versions or version.created_at > old_versions[version.file_path].created_at:
                old_versions[version.file_path] = version
    
    # Get versions at new timestamp
    new_versions = {}
    for version in project_versions:
        if version.created_at <= new_timestamp:
            if version.file_path not in new_versions or version.created_at > new_versions[version.file_path].created_at:
                new_versions[version.file_path] = version
    
    # Compare versions
    all_files = set(old_versions.keys()) | set(new_versions.keys())
    comparisons = []
    
    for file_path in all_files:
        old_version = old_versions.get(file_path)
        new_version = new_versions.get(file_path)
        
        if not old_version and new_version:
            status = "added"
        elif old_version and not new_version:
            status = "deleted"
        elif old_version and new_version:
            if old_version.sha256 == new_version.sha256:
                status = "unchanged"
            else:
                status = "modified"
        else:
            continue
        
        comparisons.append(VersionComparison(
            file_path=file_path,
            old_version=old_version,
            new_version=new_version,
            status=status
        ))
    
    return sorted(comparisons, key=lambda x: x.file_path)