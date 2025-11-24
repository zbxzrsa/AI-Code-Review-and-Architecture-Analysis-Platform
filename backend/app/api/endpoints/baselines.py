"""
Baselines API endpoints
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/baselines", tags=["baselines"])


class BaselineDeviation(BaseModel):
    id: int
    baseline_id: int
    metric_name: str
    deviation_value: float
    severity: str
    detected_at: datetime


class Baseline(BaseModel):
    id: int
    project_id: int
    name: str
    description: str
    config: Dict[str, Any]
    created_at: datetime
    deviations: List[BaselineDeviation] = []


class BaselineCreate(BaseModel):
    project_id: int
    name: str
    description: str
    config: Dict[str, Any]


class BaselineUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class DeviationCreate(BaseModel):
    baseline_id: int
    metric_name: str
    deviation_value: float
    severity: str


# Mock data for development
MOCK_DEVIATIONS = [
    BaselineDeviation(
        id=1,
        baseline_id=1,
        metric_name="code_coverage",
        deviation_value=-5.2,
        severity="medium",
        detected_at=datetime(2024, 1, 16, 10, 0)
    ),
    BaselineDeviation(
        id=2,
        baseline_id=1,
        metric_name="cyclomatic_complexity",
        deviation_value=3.8,
        severity="high",
        detected_at=datetime(2024, 1, 17, 14, 30)
    ),
    BaselineDeviation(
        id=3,
        baseline_id=2,
        metric_name="security_score",
        deviation_value=-12.5,
        severity="critical",
        detected_at=datetime(2024, 1, 19, 9, 15)
    )
]

MOCK_BASELINES = [
    Baseline(
        id=1,
        project_id=1,
        name="Quality Baseline v1.0",
        description="Initial quality metrics baseline for e-commerce platform",
        config={
            "metrics": {
                "code_coverage": {"min": 80.0, "target": 90.0},
                "cyclomatic_complexity": {"max": 10.0, "target": 5.0},
                "duplication_ratio": {"max": 5.0, "target": 2.0}
            },
            "thresholds": {
                "critical": 20.0,
                "high": 10.0,
                "medium": 5.0
            }
        },
        created_at=datetime(2024, 1, 15, 9, 0),
        deviations=[d for d in MOCK_DEVIATIONS if d.baseline_id == 1]
    ),
    Baseline(
        id=2,
        project_id=1,
        name="Security Baseline v1.0",
        description="Security metrics baseline with vulnerability thresholds",
        config={
            "metrics": {
                "security_score": {"min": 85.0, "target": 95.0},
                "vulnerability_count": {"max": 0, "target": 0},
                "dependency_vulnerabilities": {"max": 2, "target": 0}
            },
            "scan_frequency": "daily",
            "auto_alerts": True
        },
        created_at=datetime(2024, 1, 15, 9, 30),
        deviations=[d for d in MOCK_DEVIATIONS if d.baseline_id == 2]
    ),
    Baseline(
        id=3,
        project_id=2,
        name="Performance Baseline v1.0",
        description="Performance metrics baseline for dashboard application",
        config={
            "metrics": {
                "response_time": {"max": 200.0, "target": 100.0},
                "memory_usage": {"max": 512.0, "target": 256.0},
                "cpu_usage": {"max": 70.0, "target": 50.0}
            },
            "monitoring": {
                "interval": "5m",
                "retention": "30d"
            }
        },
        created_at=datetime(2024, 1, 18, 10, 0),
        deviations=[]
    )
]


@router.get("/", response_model=List[Baseline])
async def list_baselines(
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    limit: int = Query(50, ge=1, le=100, description="Number of baselines to return")
):
    """Get list of baselines with optional filtering"""
    baselines = MOCK_BASELINES.copy()
    
    if project_id:
        baselines = [b for b in baselines if b.project_id == project_id]
    
    return baselines[:limit]


@router.get("/{baseline_id}", response_model=Baseline)
async def get_baseline(baseline_id: int):
    """Get specific baseline by ID"""
    baseline = next((b for b in MOCK_BASELINES if b.id == baseline_id), None)
    if not baseline:
        raise HTTPException(status_code=404, detail="Baseline not found")
    return baseline


@router.post("/", response_model=Baseline)
async def create_baseline(baseline_data: BaselineCreate):
    """Create new baseline"""
    new_baseline = Baseline(
        id=len(MOCK_BASELINES) + 1,
        project_id=baseline_data.project_id,
        name=baseline_data.name,
        description=baseline_data.description,
        config=baseline_data.config,
        created_at=datetime.now(),
        deviations=[]
    )
    MOCK_BASELINES.append(new_baseline)
    return new_baseline


@router.put("/{baseline_id}", response_model=Baseline)
async def update_baseline(baseline_id: int, baseline_data: BaselineUpdate):
    """Update baseline"""
    baseline = next((b for b in MOCK_BASELINES if b.id == baseline_id), None)
    if not baseline:
        raise HTTPException(status_code=404, detail="Baseline not found")
    
    if baseline_data.name is not None:
        baseline.name = baseline_data.name
    if baseline_data.description is not None:
        baseline.description = baseline_data.description
    if baseline_data.config is not None:
        baseline.config = baseline_data.config
    
    return baseline


@router.delete("/{baseline_id}")
async def delete_baseline(baseline_id: int):
    """Delete baseline"""
    global MOCK_BASELINES
    baseline = next((b for b in MOCK_BASELINES if b.id == baseline_id), None)
    if not baseline:
        raise HTTPException(status_code=404, detail="Baseline not found")
    
    MOCK_BASELINES = [b for b in MOCK_BASELINES if b.id != baseline_id]
    return {"message": "Baseline deleted successfully"}


@router.get("/{baseline_id}/deviations", response_model=List[BaselineDeviation])
async def get_baseline_deviations(
    baseline_id: int,
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=100, description="Number of deviations to return")
):
    """Get deviations for specific baseline"""
    baseline = next((b for b in MOCK_BASELINES if b.id == baseline_id), None)
    if not baseline:
        raise HTTPException(status_code=404, detail="Baseline not found")
    
    deviations = baseline.deviations.copy()
    
    if severity:
        deviations = [d for d in deviations if d.severity == severity]
    
    # Sort by detected_at descending (newest first)
    deviations.sort(key=lambda x: x.detected_at, reverse=True)
    
    return deviations[:limit]


@router.post("/{baseline_id}/deviations", response_model=BaselineDeviation)
async def create_deviation(baseline_id: int, deviation_data: DeviationCreate):
    """Create new baseline deviation"""
    baseline = next((b for b in MOCK_BASELINES if b.id == baseline_id), None)
    if not baseline:
        raise HTTPException(status_code=404, detail="Baseline not found")
    
    new_deviation = BaselineDeviation(
        id=len(MOCK_DEVIATIONS) + 1,
        baseline_id=baseline_id,
        metric_name=deviation_data.metric_name,
        deviation_value=deviation_data.deviation_value,
        severity=deviation_data.severity,
        detected_at=datetime.now()
    )
    
    MOCK_DEVIATIONS.append(new_deviation)
    baseline.deviations.append(new_deviation)
    
    return new_deviation


@router.get("/{baseline_id}/status")
async def get_baseline_status(baseline_id: int):
    """Get current status and health of baseline"""
    baseline = next((b for b in MOCK_BASELINES if b.id == baseline_id), None)
    if not baseline:
        raise HTTPException(status_code=404, detail="Baseline not found")
    
    # Calculate status based on deviations
    critical_count = len([d for d in baseline.deviations if d.severity == "critical"])
    high_count = len([d for d in baseline.deviations if d.severity == "high"])
    medium_count = len([d for d in baseline.deviations if d.severity == "medium"])
    low_count = len([d for d in baseline.deviations if d.severity == "low"])
    
    if critical_count > 0:
        status = "critical"
    elif high_count > 0:
        status = "warning"
    elif medium_count > 0:
        status = "attention"
    else:
        status = "healthy"
    
    return {
        "baseline_id": baseline_id,
        "status": status,
        "total_deviations": len(baseline.deviations),
        "deviation_counts": {
            "critical": critical_count,
            "high": high_count,
            "medium": medium_count,
            "low": low_count
        },
        "last_check": max([d.detected_at for d in baseline.deviations], default=baseline.created_at)
    }