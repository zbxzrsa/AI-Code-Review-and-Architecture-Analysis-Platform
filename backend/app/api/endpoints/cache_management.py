"""
智能预热API端点
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

from app.services.cache.smart_warmer import smart_warmup_task
from app.core.dependencies import rate_limiter
from app.core.metrics import get_metrics

router = APIRouter(prefix="/api/v1/cache", tags=["Cache Management"])


class WarmupRequest(BaseModel):
    tenant_id: str = Field(..., description="租户ID")
    repo_id: str = Field(..., description="仓库ID")
    strategy: str = Field(default="pattern", description="预热策略: pattern, recent_changes")
    pattern: Optional[str] = Field(None, description="文件路径模式")
    days: Optional[int] = Field(None, description="天数（用于recent_changes策略）")


class WarmupResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    files_count: int = Field(..., description="文件总数")
    preloaded_count: int = Field(..., description="预加载文件数")
    duration_seconds: float = Field(..., description="执行时长")
    message: str = Field(..., description="状态消息")


@router.post("/warmup", response_model=WarmupResponse)
async def trigger_warmup(
    request: WarmupRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(rate_limiter(max_requests=5, window_seconds=60))
) -> WarmupResponse:
    """
    触发智能预热任务
    """
    try:
        # 触发异步预热任务
        task = smart_warmup_task.delay(
            tenant_id=request.tenant_id,
            repo_id=request.repo_id,
            strategy=request.strategy,
            pattern=request.pattern,
            days=request.days
        )
        
        return WarmupResponse(
            success=True,
            task_id=task.id,
            status="queued",
            files_count=0,
            preloaded_count=0,
            duration_seconds=0.0,
            message=f"Warmup task queued for {request.tenant_id}/{request.repo_id}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/warmup/{task_id}", response_model=WarmupResponse)
async def get_warmup_status(
    task_id: str,
    background_tasks: BackgroundTasks
) -> WarmupResponse:
    """
    获取预热任务状态
    """
    try:
        # 获取任务状态
        result = AsyncResult(task.id)
        
        return WarmupResponse(
            success=True,
            task_id=task_id,
            status=result.state,
            files_count=result.result.get("files_count", 0),
            preloaded_count=result.result.get("preloaded_count", 0),
            duration_seconds=result.result.get("duration_seconds", 0.0),
            message=f"Warmup task {task_id} status: {result.state}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Dict[str, Any])
async def get_cache_stats() -> Dict[str, Any]:
    """
    获取缓存统计信息
    """
    try:
        from app.core.cache.cache_manager import get_cache_manager, get_metrics
        
        cache_manager = get_cache_manager()
        metrics = get_metrics()
        
        # 获取缓存统计
        stats = await cache_manager.stats()
        
        return {
            "success": True,
            "cache_stats": stats,
            "metrics": {
                "cache_hits_total": metrics.CACHE_HITS_TOTAL._value.get(),
                "cache_misses_total": metrics.CACHE_MISS_TOTAL._value.get(),
                "warmup_jobs_total": metrics.WARMUP_JOBS_TOTAL._value.get(),
                "cache_hit_ratio": metrics.get_hit_ratio(),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.post("/invalidate", response_model=Dict[str, Any])
async def invalidate_cache(
    strategy: str = Field(..., description="失效策略"),
    context: Dict[str, Any] = Field(..., description="失效上下文")
) -> Dict[str, Any]:
    """
    按策略失效缓存
    """
    try:
        from app.core.cache.cache_manager import get_cache_manager
        
        cache_manager = get_cache_manager()
        count = await cache_manager.invalidate_by_strategy(strategy, context)
        
        return {
            "success": True,
            "invalidated_count": count,
            "strategy": strategy,
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# 注册缓存预热任务
from app.worker import celery_app
from celery.result import AsyncResult

def AsyncResult(task_id: str) -> Any:
    """获取Celery任务结果"""
    return celery_app.AsyncResult(task_id)