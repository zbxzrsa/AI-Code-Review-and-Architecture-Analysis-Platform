from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import traceback
from typing import Callable, Dict, Any, Optional, Union

from app.core.exceptions import AppException
from app.core.logger import logger


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    全局错误处理中间件
    捕获所有未处理的异常，并返回统一的错误响应格式
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except AppException as exc:
            # 处理应用自定义异常
            return self._handle_app_exception(exc)
        except Exception as exc:
            # 处理未预期的异常
            return self._handle_unexpected_exception(exc, request)
    
    def _handle_app_exception(self, exc: AppException) -> JSONResponse:
        """处理应用自定义异常"""
        logger.error(f"应用异常: {exc.detail}", extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "error_details": [detail.dict() for detail in exc.error_details] if exc.error_details else None
        })
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "code": exc.error_code,
                "message": exc.detail,
                "details": [detail.dict() for detail in exc.error_details] if exc.error_details else None
            },
            headers=exc.headers
        )
    
    def _handle_unexpected_exception(self, exc: Exception, request: Request) -> JSONResponse:
        """处理未预期的异常"""
        # 获取完整的异常堆栈
        stack_trace = traceback.format_exc()
        
        # 记录详细错误日志
        logger.error(
            f"未捕获的异常: {str(exc)}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client": request.client.host if request.client else None,
                "stack_trace": stack_trace
            }
        )
        
        # 返回通用错误响应，不暴露内部错误详情
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "code": "INTERNAL_SERVER_ERROR",
                "message": "服务器内部错误，请稍后重试",
                "request_id": request.state.request_id if hasattr(request.state, "request_id") else None
            }
        )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    记录所有请求的处理时间和结果
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 生成请求ID
        request_id = self._generate_request_id()
        request.state.request_id = request_id
        
        # 记录请求开始
        start_time = time.time()
        path = request.url.path
        method = request.method
        
        # 记录请求信息
        logger.info(
            f"开始处理请求: {method} {path}",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent", "")
            }
        )
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录响应信息
        logger.info(
            f"请求处理完成: {method} {path}",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2)
            }
        )
        
        # 添加请求ID到响应头
        response.headers["X-Request-ID"] = request_id
        
        # 添加处理时间到响应头
        response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
        
        return response
    
    def _generate_request_id(self) -> str:
        """生成唯一请求ID"""
        import uuid
        return str(uuid.uuid4())


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    请求限流中间件
    基于IP地址或API密钥进行请求限流
    """
    
    def __init__(
        self, 
        app: ASGIApp, 
        rate_limit_per_minute: int = 60,
        rate_limit_window_seconds: int = 60,
        exclude_paths: list = None
    ):
        super().__init__(app)
        self.rate_limit_per_minute = rate_limit_per_minute
        self.rate_limit_window_seconds = rate_limit_window_seconds
        self.exclude_paths = exclude_paths or []
        self.request_counts: Dict[str, Dict[str, Any]] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 检查是否排除当前路径
        if self._should_exclude(request.url.path):
            return await call_next(request)
        
        # 获取客户端标识符（IP或API密钥）
        client_id = self._get_client_identifier(request)
        
        # 检查是否超过限流阈值
        if self._is_rate_limited(client_id):
            logger.warning(f"请求被限流: {client_id}", extra={
                "path": request.url.path,
                "method": request.method,
                "client_id": client_id
            })
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "请求频率过高，请稍后重试"
                }
            )
        
        # 更新请求计数
        self._update_request_count(client_id)
        
        # 处理请求
        return await call_next(request)
    
    def _should_exclude(self, path: str) -> bool:
        """检查是否应该排除当前路径"""
        for exclude_path in self.exclude_paths:
            if path.startswith(exclude_path):
                return True
        return False
    
    def _get_client_identifier(self, request: Request) -> str:
        """获取客户端标识符"""
        # 优先使用API密钥
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        # 使用客户端IP
        return f"ip:{request.client.host}" if request.client else "ip:unknown"
    
    def _is_rate_limited(self, client_id: str) -> bool:
        """检查是否超过限流阈值"""
        now = time.time()
        
        if client_id not in self.request_counts:
            return False
        
        client_data = self.request_counts[client_id]
        
        # 清理过期的计数
        if now - client_data["window_start"] > self.rate_limit_window_seconds:
            client_data["count"] = 0
            client_data["window_start"] = now
            return False
        
        # 检查是否超过限制
        return client_data["count"] >= self.rate_limit_per_minute
    
    def _update_request_count(self, client_id: str) -> None:
        """更新请求计数"""
        now = time.time()
        
        if client_id not in self.request_counts:
            self.request_counts[client_id] = {
                "count": 1,
                "window_start": now
            }
            return
        
        client_data = self.request_counts[client_id]
        
        # 如果窗口已过期，重置计数
        if now - client_data["window_start"] > self.rate_limit_window_seconds:
            client_data["count"] = 1
            client_data["window_start"] = now
        else:
            client_data["count"] += 1


def setup_middlewares(app: FastAPI) -> None:
    """设置所有中间件"""
    # 添加请求日志中间件
    app.add_middleware(RequestLoggingMiddleware)
    
    # 添加限流中间件
    app.add_middleware(
        RateLimitMiddleware,
        rate_limit_per_minute=100,
        exclude_paths=["/docs", "/redoc", "/openapi.json", "/health"]
    )
    
    # 添加错误处理中间件（最后添加，以便捕获所有异常）
    app.add_middleware(ErrorHandlerMiddleware)