"""
代码执行沙箱API端点
提供安全的代码执行环境
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field

from app.services.code_sandbox import (
    code_sandbox,
    SandboxConfig,
    SandboxStatus,
    ExecutionResult
)
from app.core.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sandbox", tags=["sandbox"])


class ExecuteRequest(BaseModel):
    """代码执行请求"""
    code: str = Field(..., description="要执行的代码")
    language: str = Field("python", description="编程语言")
    files: Optional[Dict[str, str]] = Field(None, description="额外文件")
    environment: Optional[Dict[str, str]] = Field(None, description="环境变量")
    timeout: Optional[int] = Field(30, description="超时时间（秒）")
    memory_limit: Optional[str] = Field("512m", description="内存限制")
    read_only: bool = Field(False, description="只读模式")


class ExecuteResponse(BaseModel):
    """代码执行响应"""
    execution_id: str
    status: str
    exit_code: Optional[int] = None
    stdout: str
    stderr: str
    error_message: Optional[str] = None
    execution_time: float
    resource_usage: Dict[str, Any]
    created_files: List[str]
    deleted_files: List[str]


class SandboxStatusResponse(BaseModel):
    """沙箱状态响应"""
    status: str
    active_executions: List[str]
    total_executions: int
    memory_usage: Dict[str, Any]
    cpu_usage: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    image_name: Optional[str] = Field(None, description="镜像名称")
    memory_limit: Optional[str] = Field(None, description="内存限制")
    cpu_limit: Optional[str] = Field(None, description="CPU限制")
    timeout_seconds: Optional[int] = Field(None, description="超时时间")
    network_enabled: bool = Field(False, description="启用网络访问")
    allowed_commands: Optional[List[str]] = Field(None, description="允许的命令")


@router.post("/execute", response_model=ExecuteResponse)
async def execute_code(
    request: ExecuteRequest,
    current_user: Dict = Depends(get_current_user)
):
    """执行代码"""
    try:
        # 创建沙箱配置
        config = SandboxConfig(
            memory_limit=request.memory_limit or "512m",
            cpu_limit=request.cpu_limit or "1.0",
            timeout_seconds=request.timeout or 30,
            network_enabled=request.network_enabled,
            read_only=request.read_only
            allowed_commands=request.allowed_commands,
            environment_variables=request.environment
        )
        
        # 初始化沙箱
        await code_sandbox.initialize(config)
        
        # 执行代码
        result = await code_sandbox.execute_code(
            code=request.code,
            language=request.language,
            files=request.files,
            environment=request.environment,
            timeout=request.timeout
        )
        
        return ExecuteResponse(
            execution_id=result.execution_id,
            status=result.status.value,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            error_message=result.error_message,
            execution_time=result.execution_time,
            resource_usage=result.resource_usage,
            created_files=result.created_files,
            deleted_files=result.deleted_files
        )
        
    except Exception as e:
        logger.error(f"Error executing code: {e}")
        raise HTTPException(status_code=500, detail=f"代码执行失败: {str(e)}")


@router.get("/status", response_model=SandboxStatusResponse)
async def get_sandbox_status(
    current_user: Dict = Depends(get_current_user)
    """获取沙箱状态"""
    try:
        status = code_sandbox.get_status()
        
        return SandboxStatusResponse(
            status=status.status.value,
            active_executions=status.active_executions,
            total_executions=status.total_executions,
            memory_usage=status.memory_usage,
            cpu_usage=status.cpu_usage
        )
        
    except Exception as e:
        logger.error(f"Error getting sandbox status: {e}")
        raise HTTPException(status_code=500, detail=f"获取沙箱状态失败: {str(e)}")


@router.post("/config/update", response_model=Dict[str, Any])
async def update_sandbox_config(
    request: ConfigUpdateRequest,
    current_user: Dict = Depends(get_current_user)
):
    """更新沙箱配置"""
    try:
        # 更新配置
        if request.image_name:
            code_sandbox.config.image_name = request.image_name
        if request.memory_limit:
            code_sandbox.config.memory_limit = request.memory_limit
        if request.cpu_limit:
            code_sandbox.config.cpu_limit = request.cpu_limit
        if request.timeout_seconds:
            code_sandbox.config.timeout_seconds = request.timeout_seconds
        if request.network_enabled is not None:
            code_sandbox.config.network_enabled = request.network_enabled
        if request.allowed_commands:
            code_sandbox.config.allowed_commands = request.allowed_commands
        if request.environment_variables:
            code_sandbox.config.environment_variables.update(request.environment_variables)
        
        # 重新初始化沙箱
        await code_sandbox.initialize(code_sandbox.config)
        
        return {
            "message": "Sandbox configuration updated successfully",
            "config": {
                "image_name": code_sandbox.config.image_name,
                "memory_limit": code_sandbox.config.memory_limit,
                "cpu_limit": code_sandbox.config.cpu_limit,
                "timeout_seconds": code_sandbox.config.timeout_seconds,
                "network_enabled": code_sandbox.config.network_enabled,
                "allowed_commands": code_sandbox.config.allowed_commands,
                "environment_variables": code_sandbox.config.environment_variables
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating sandbox config: {e}")
        raise HTTPException(status_code=500, detail=f"更新沙箱配置失败: {str(e)}")


@router.post("/stop/{execution_id}", response_model=Dict[str, Any])
async def stop_execution(
    execution_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """停止代码执行"""
    try:
        success = await code_sandbox.stop_execution(execution_id)
        
        if success:
            return {"message": f"Execution {execution_id} stopped successfully"}
        else:
            return {"error": f"Failed to stop execution {execution_id}"}
        
    except Exception as e:
        logger.error(f"Error stopping execution {execution_id}: {e}")
        raise HTTPException(status_code=500, detail=f"停止执行失败: {str(e)}")


@router.get("/history", response_model=List[Dict[str, Any]])
async def get_execution_history(
    current_user: Dict = Depends(get_current_user)
):
    """获取执行历史"""
    try:
        history = code_sandbox.get_execution_history()
        
        return history
        
    except Exception as e:
        logger.error(f"Error getting execution history: {e}")
        raise HTTPException(status_code=500, detail=f"获取执行历史失败: {str(e)}")


@router.get("/execution/{execution_id}", response_model=Dict[str, Any])
async def get_execution_details(
    execution_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """获取执行详情"""
    try:
        result = code_sandbox.get_execution_result(execution_id)
        
        if result:
            return {
                "execution_id": result.execution_id,
                "status": result.status.value,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error_message": result.error_message,
                "execution_time": result.execution_time,
                "resource_usage": result.resource_usage,
                "created_files": result.created_files,
                "deleted_files": result.deleted_files
            }
        else:
            return {"error": f"Execution {execution_id} not found"}
        
    except Exception as e:
        logger.error(f"Error getting execution details: {e}")
        raise HTTPException(status_code=500, detail=f"获取执行详情失败: {str(e)}")


@router.delete("/cache", response_model=Dict[str, Any])
async def clear_sandbox_cache(
    current_user: Dict = Depends(get_current_user)
):
    """清空沙箱缓存"""
    try:
        code_sandbox.clear_cache()
        
        return {"message": "Sandbox cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing sandbox cache: {e}")
        raise HTTPException(status_code=500, detail=f"清空沙箱缓存失败: {str(e)}")


@router.get("/stats", response_model=Dict[str, Any])
async def get_sandbox_stats(
    current_user: Dict = Depends(get_current_user)
):
    """获取沙箱统计"""
    try:
        stats = code_sandbox.get_sandbox_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting sandbox stats: {e}")
        raise HTTPException(status_code=500, detail=f"获取沙箱统计失败: {str(e)}")


@router.post("/cleanup", response_model=Dict[str, Any])
async def cleanup_sandbox(
    current_user: Dict = Depends(get_current_user)
):
    """清理沙箱环境"""
    try:
            # 清理所有活跃容器
            await code_sandbox.cleanup()
            
            return {"message": "Sandbox cleaned up successfully"}
        
        except Exception as e:
            logger.error(f"Error cleaning up sandbox: {e}")
            raise HTTPException(status_code=500, detail=f"清理沙箱失败: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def sandbox_health_check(
    current_user: Dict = Depends(get_current_user)
):
    """沙箱健康检查"""
    try:
        # 检查Docker连接
        health_status = await code_sandbox.health_check()
        
        return {
            "status": "healthy" if health_status else "unhealthy",
            "docker_connected": health_status,
            "cache_status": "healthy" if code_sandbox.get_cache_stats()["cache_size"] > 0 else "empty"
        }
        
    except Exception as e:
            logger.error(f"Sandbox health check failed: {e}")
            raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")