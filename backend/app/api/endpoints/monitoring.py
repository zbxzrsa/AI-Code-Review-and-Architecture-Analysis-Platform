from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from app.core.health import HealthCheckService, SystemHealth
from app.core.metrics import metrics_service
from app.core.logger import logger

router = APIRouter()

class PerformanceMetricsResponse(BaseModel):
    """性能指标响应模型"""
    metrics: Dict[str, List[Dict[str, Any]]]
    timestamp: float

class HealthStatusResponse(BaseModel):
    """健康状态响应模型"""
    status: str
    version: str
    uptime_seconds: float
    dependencies: List[Dict[str, Any]]
    timestamp: float

@router.get("/health", response_model=HealthStatusResponse)
async def get_health_status():
    """获取系统健康状态"""
    try:
        # 获取健康检查服务实例
        health_service = HealthCheckService(app_version="1.0.0")
        
        # 执行健康检查
        health_status = await health_service.check_all()
        
        # 转换为响应模型
        response = HealthStatusResponse(
            status=health_status.status,
            version=health_status.version,
            uptime_seconds=health_status.uptime_seconds,
            dependencies=[dep.dict() for dep in health_status.dependencies],
            timestamp=health_status.timestamp
        )
        
        return response
    except Exception as e:
        logger.error(f"获取健康状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取健康状态失败: {str(e)}")

@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    metric_types: Optional[List[str]] = Query(None),
    time_range: str = Query("1h"),
    limit: int = Query(100, ge=1, le=1000)
):
    """获取性能指标数据
    
    参数:
    - metric_types: 指标类型列表，如果为空则返回所有类型
    - time_range: 时间范围，如 "1h", "6h", "24h", "7d"
    - limit: 每个指标返回的最大数据点数量
    """
    try:
        # 获取所有指标
        all_metrics = metrics_service.get_all_metrics()
        
        # 过滤指标类型
        filtered_metrics = {}
        for name, metric in all_metrics.items():
            if metric_types is None or metric.type in metric_types:
                # 转换指标值为字典列表
                values = [v.dict() for v in metric.values[-limit:]]
                
                # 添加聚合数据
                metric_data = {
                    "type": metric.type,
                    "description": metric.description,
                    "values": values
                }
                
                # 添加直方图和计时器的聚合数据
                if metric.type in ["histogram", "timer"]:
                    metric_data.update({
                        "min_value": metric.min_value,
                        "max_value": metric.max_value,
                        "avg_value": metric.avg_value,
                        "median_value": metric.median_value,
                        "percentile_95": metric.percentile_95,
                        "percentile_99": metric.percentile_99,
                        "count": metric.count
                    })
                
                filtered_metrics[name] = metric_data
        
        # 创建响应
        response = PerformanceMetricsResponse(
            metrics=filtered_metrics,
            timestamp=time.time()
        )
        
        return response
    except Exception as e:
        logger.error(f"获取性能指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")

@router.post("/metrics/client")
async def record_client_metrics(metrics: List[Dict[str, Any]]):
    """记录客户端上报的性能指标"""
    try:
        for metric in metrics:
            metric_type = metric.get("type")
            name = metric.get("name")
            value = metric.get("value")
            tags = metric.get("tags", {})
            
            if not all([metric_type, name, value is not None]):
                continue
                
            # 根据指标类型记录
            if metric_type == "counter":
                metrics_service.increment_counter(name, value, tags)
            elif metric_type == "gauge":
                metrics_service.set_gauge(name, value, tags)
            elif metric_type == "histogram":
                metrics_service.record_histogram(name, value, tags)
            elif metric_type == "timer":
                metrics_service.record_timer(name, value, tags)
        
        return {"status": "success", "message": f"Recorded {len(metrics)} metrics"}
    except Exception as e:
        logger.error(f"记录客户端指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"记录客户端指标失败: {str(e)}")

import time

@router.get("/metrics/simulate")
async def simulate_metrics():
    """生成模拟指标数据（仅用于演示）"""
    try:
        # 模拟API响应时间
        metrics_service.record_timer(
            name="api.response_time",
            value=random.randint(50, 500),
            tags={"endpoint": "/api/items", "method": "GET"}
        )
        
        # 模拟数据库查询时间
        metrics_service.record_timer(
            name="db.query_time",
            value=random.randint(10, 200),
            tags={"query": "select", "table": "items"}
        )
        
        # 模拟活跃用户数
        metrics_service.set_gauge(
            name="users.active",
            value=random.randint(10, 100)
        )
        
        # 模拟请求计数
        metrics_service.increment_counter(
            name="api.requests",
            value=1,
            tags={"endpoint": "/api/items", "method": "GET"}
        )
        
        # 模拟错误计数
        if random.random() < 0.1:  # 10%的概率产生错误
            metrics_service.increment_counter(
                name="api.errors",
                value=1,
                tags={"endpoint": "/api/items", "method": "GET", "code": "500"}
            )
        
        return {"status": "success", "message": "Generated simulation metrics"}
    except Exception as e:
        logger.error(f"生成模拟指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成模拟指标失败: {str(e)}")

import random

@router.get("/errors/simulate")
async def simulate_error(error_type: str = Query("validation")):
    """模拟产生错误（用于测试错误处理）"""
    try:
        if error_type == "validation":
            raise ValueError("模拟的输入验证错误")
        elif error_type == "not_found":
            raise HTTPException(status_code=404, detail="模拟的资源未找到错误")
        elif error_type == "server":
            raise Exception("模拟的服务器内部错误")
        elif error_type == "timeout":
            # 模拟超时
            await asyncio.sleep(10)
            return {"status": "success"}
        else:
            return {"status": "success", "message": "No error simulated"}
    except ValueError as e:
        logger.warning(f"模拟验证错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模拟服务器错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

import asyncio