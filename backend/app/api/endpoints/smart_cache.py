"""
智能预热API端点 - 简化版本
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from app.services.smart_cache_service import get_smart_cache_service

router = APIRouter(prefix="/api/v1/smart-cache", tags=["Smart Cache"])


class WarmupRequest(BaseModel):
    tenant_id: str = Field(..., description="租户ID")
    repo_id: str = Field(..., description="仓库ID")
    strategy: str = Field(default="pattern", description="预热策略: pattern, recent_changes")
    pattern: str = Field(default="src/**", description="文件路径模式")
    days: int = Field(default=7, description="天数（用于recent_changes策略）")


class WarmupResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    tenant_id: str = Field(..., description="租户ID")
    repo_id: str = Field(..., description="仓库ID")
    strategy: str = Field(..., description="使用的策略")
    pattern: str = Field(..., description="文件路径模式")
    files_count: int = Field(..., description="文件总数")
    successful_count: int = Field(..., description="成功预热文件数")
    message: str = Field(..., description="状态消息")
    results: list = Field(default=[], description="预热结果列表")


class StatsResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    stats: Dict[str, Any] = Field(..., description="统计信息")


@router.post("/warmup", response_model=WarmupResponse)
async def trigger_warmup(
    request: WarmupRequest,
    background_tasks: BackgroundTasks
) -> WarmupResponse:
    """
    触发智能预热任务
    """
    try:
        service = get_smart_cache_service()
        
        if request.strategy == "pattern":
            result = await service.warmup_by_pattern(
                request.tenant_id,
                request.repo_id,
                request.pattern
            )
        elif request.strategy == "recent_changes":
            result = await service.warmup_recent_changes(
                request.tenant_id,
                request.repo_id,
                request.days
            )
        else:
            result = await service.warmup_by_pattern(
                request.tenant_id,
                request.repo_id,
                request.pattern
            )
        
        return WarmupResponse(
            success=True,
            tenant_id=request.tenant_id,
            repo_id=request.repo_id,
            strategy=request.strategy,
            pattern=request.pattern,
            files_count=result.get("files_count", 0),
            successful_count=result.get("successful_count", 0),
            message=result.get("message", "Warmup completed"),
            results=result.get("results", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_warmup_stats() -> StatsResponse:
    """
    获取预热统计信息
    """
    try:
        service = get_smart_cache_service()
        stats = service.get_stats()
        
        return StatsResponse(
            success=True,
            stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    健康检查端点
    """
    return {
        "status": "healthy",
        "service": "smart-cache",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }