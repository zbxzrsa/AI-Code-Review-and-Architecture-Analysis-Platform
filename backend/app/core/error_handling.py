from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Type, Union
import time
import traceback
import logging
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 配置日志
logger = logging.getLogger(__name__)

# 错误类型枚举
class ErrorType(str, Enum):
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    RESOURCE_ERROR = "resource_error"
    DEPENDENCY_ERROR = "dependency_error"
    DATABASE_ERROR = "database_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"
    EXTERNAL_API_ERROR = "external_api_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"

# 错误严重性枚举
class ErrorSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# 错误详情模型
class ErrorDetail(BaseModel):
    type: ErrorType
    code: str
    message: str
    severity: ErrorSeverity
    timestamp: float
    request_id: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    user_message: Optional[str] = None
    suggestion: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "type": "database_error",
                "code": "DB_CONNECTION_ERROR",
                "message": "无法连接到数据库",
                "severity": "high",
                "timestamp": 1623456789.123,
                "request_id": "req-123456",
                "path": "/api/users",
                "method": "GET",
                "details": {"host": "db.example.com", "port": 5432},
                "user_message": "系统暂时无法访问数据，请稍后再试",
                "suggestion": "检查数据库连接配置和网络状态"
            }
        }

# 自定义异常基类
class AppException(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        type: ErrorType = ErrorType.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
        status_code: int = 500,
    ):
        self.code = code
        self.message = message
        self.type = type
        self.severity = severity
        self.details = details or {}
        self.user_message = user_message or "操作处理过程中发生错误，请稍后再试"
        self.suggestion = suggestion
        self.status_code = status_code
        self.timestamp = time.time()
        super().__init__(self.message)

# 具体异常类型
class ValidationException(AppException):
    def __init__(
        self,
        code: str = "VALIDATION_ERROR",
        message: str = "输入数据验证失败",
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        super().__init__(
            code=code,
            message=message,
            type=ErrorType.VALIDATION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            user_message=user_message or "请检查输入数据格式是否正确",
            suggestion=suggestion or "参考API文档修正请求格式",
            status_code=400,
        )

class AuthenticationException(AppException):
    def __init__(
        self,
        code: str = "AUTHENTICATION_ERROR",
        message: str = "身份验证失败",
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        super().__init__(
            code=code,
            message=message,
            type=ErrorType.AUTHENTICATION_ERROR,
            severity=ErrorSeverity.HIGH,
            details=details,
            user_message=user_message or "请重新登录后再试",
            suggestion=suggestion or "检查认证凭据是否有效",
            status_code=401,
        )

class AuthorizationException(AppException):
    def __init__(
        self,
        code: str = "AUTHORIZATION_ERROR",
        message: str = "权限不足",
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        super().__init__(
            code=code,
            message=message,
            type=ErrorType.AUTHORIZATION_ERROR,
            severity=ErrorSeverity.HIGH,
            details=details,
            user_message=user_message or "您没有执行此操作的权限",
            suggestion=suggestion or "联系管理员获取必要权限",
            status_code=403,
        )

class ResourceException(AppException):
    def __init__(
        self,
        code: str = "RESOURCE_NOT_FOUND",
        message: str = "资源不存在",
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        super().__init__(
            code=code,
            message=message,
            type=ErrorType.RESOURCE_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            user_message=user_message or "请求的资源不存在或已被删除",
            suggestion=suggestion or "检查资源标识符是否正确",
            status_code=404,
        )

class DependencyException(AppException):
    def __init__(
        self,
        code: str = "DEPENDENCY_ERROR",
        message: str = "依赖服务异常",
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        super().__init__(
            code=code,
            message=message,
            type=ErrorType.DEPENDENCY_ERROR,
            severity=ErrorSeverity.HIGH,
            details=details,
            user_message=user_message or "系统依赖服务暂时不可用",
            suggestion=suggestion or "检查依赖服务状态",
            status_code=503,
        )

class DatabaseException(AppException):
    def __init__(
        self,
        code: str = "DATABASE_ERROR",
        message: str = "数据库操作失败",
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        super().__init__(
            code=code,
            message=message,
            type=ErrorType.DATABASE_ERROR,
            severity=ErrorSeverity.HIGH,
            details=details,
            user_message=user_message or "数据操作失败，请稍后再试",
            suggestion=suggestion or "检查数据库连接和查询语句",
            status_code=500,
        )

class BusinessLogicException(AppException):
    def __init__(
        self,
        code: str = "BUSINESS_LOGIC_ERROR",
        message: str = "业务逻辑错误",
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
        status_code: int = 400,
    ):
        super().__init__(
            code=code,
            message=message,
            type=ErrorType.BUSINESS_LOGIC_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            user_message=user_message or "请求无法完成，不符合业务规则",
            suggestion=suggestion or "检查业务参数是否符合要求",
            status_code=status_code,
        )

class ExternalAPIException(AppException):
    def __init__(
        self,
        code: str = "EXTERNAL_API_ERROR",
        message: str = "外部API调用失败",
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        super().__init__(
            code=code,
            message=message,
            type=ErrorType.EXTERNAL_API_ERROR,
            severity=ErrorSeverity.HIGH,
            details=details,
            user_message=user_message or "外部服务暂时不可用",
            suggestion=suggestion or "检查外部API状态和请求参数",
            status_code=502,
        )

class SystemException(AppException):
    def __init__(
        self,
        code: str = "SYSTEM_ERROR",
        message: str = "系统内部错误",
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        super().__init__(
            code=code,
            message=message,
            type=ErrorType.SYSTEM_ERROR,
            severity=ErrorSeverity.CRITICAL,
            details=details,
            user_message=user_message or "系统发生内部错误，请稍后再试",
            suggestion=suggestion or "检查系统日志和服务状态",
            status_code=500,
        )

# 错误处理器类
class ErrorHandler:
    def __init__(self):
        self.exception_handlers: Dict[Type[Exception], Callable] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        # 注册默认异常处理器
        self.register_exception_handler(AppException, self._handle_app_exception)
        self.register_exception_handler(Exception, self._handle_generic_exception)
    
    def register_exception_handler(self, exc_class: Type[Exception], handler: Callable):
        """注册异常处理器"""
        self.exception_handlers[exc_class] = handler
    
    async def _handle_app_exception(self, request: Request, exc: AppException) -> JSONResponse:
        """处理应用自定义异常"""
        error_detail = ErrorDetail(
            type=exc.type,
            code=exc.code,
            message=exc.message,
            severity=exc.severity,
            timestamp=exc.timestamp,
            request_id=request.headers.get("X-Request-ID"),
            path=request.url.path,
            method=request.method,
            details=exc.details,
            user_message=exc.user_message,
            suggestion=exc.suggestion
        )
        
        # 记录错误日志
        log_message = f"[{error_detail.severity}] {error_detail.type}.{error_detail.code}: {error_detail.message}"
        if error_detail.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            logger.error(log_message, exc_info=True, extra={"error_detail": error_detail.dict()})
        else:
            logger.warning(log_message, extra={"error_detail": error_detail.dict()})
        
        # 返回JSON响应
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": error_detail.dict(exclude={"details"} if error_detail.severity != ErrorSeverity.LOW else None)
            }
        )
    
    async def _handle_generic_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """处理通用异常"""
        error_detail = ErrorDetail(
            type=ErrorType.UNKNOWN_ERROR,
            code="UNHANDLED_EXCEPTION",
            message=str(exc),
            severity=ErrorSeverity.HIGH,
            timestamp=time.time(),
            request_id=request.headers.get("X-Request-ID"),
            path=request.url.path,
            method=request.method,
            details={"traceback": traceback.format_exc()},
            user_message="系统发生未预期的错误，请稍后再试",
            suggestion="检查系统日志获取详细信息"
        )
        
        # 记录错误日志
        logger.error(
            f"未处理异常: {exc.__class__.__name__}: {str(exc)}",
            exc_info=True,
            extra={"error_detail": error_detail.dict()}
        )
        
        # 返回JSON响应
        return JSONResponse(
            status_code=500,
            content={
                "error": error_detail.dict(exclude={"details"})
            }
        )
    
    async def handle_exception(self, request: Request, exc: Exception) -> Response:
        """处理异常的入口方法"""
        # 查找最匹配的异常处理器
        for exc_class, handler in self.exception_handlers.items():
            if isinstance(exc, exc_class):
                return await handler(request, exc)
        
        # 如果没有找到匹配的处理器，使用通用异常处理器
        return await self._handle_generic_exception(request, exc)

# 创建全局错误处理器实例
error_handler = ErrorHandler()

# 重试装饰器
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
          exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception):
    """
    重试装饰器，用于自动重试可能失败的操作
    
    参数:
        max_attempts: 最大尝试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟时间的增长因子
        exceptions: 需要捕获的异常类型
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    last_exception = e
                    
                    # 记录重试信息
                    logger.warning(
                        f"操作失败，正在重试 ({attempt}/{max_attempts}): {func.__name__}",
                        exc_info=True,
                        extra={"retry_info": {"attempt": attempt, "max_attempts": max_attempts}}
                    )
                    
                    # 如果已达到最大尝试次数，则抛出最后一个异常
                    if attempt >= max_attempts:
                        break
                    
                    # 等待一段时间后重试
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            # 重试失败，抛出异常
            if isinstance(last_exception, AppException):
                raise last_exception
            else:
                raise SystemException(
                    code="RETRY_EXHAUSTED",
                    message=f"操作重试{max_attempts}次后仍然失败: {str(last_exception)}",
                    details={"original_exception": str(last_exception), "function": func.__name__}
                )
        
        # 同步函数版本
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    last_exception = e
                    
                    # 记录重试信息
                    logger.warning(
                        f"操作失败，正在重试 ({attempt}/{max_attempts}): {func.__name__}",
                        exc_info=True,
                        extra={"retry_info": {"attempt": attempt, "max_attempts": max_attempts}}
                    )
                    
                    # 如果已达到最大尝试次数，则抛出最后一个异常
                    if attempt >= max_attempts:
                        break
                    
                    # 等待一段时间后重试
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # 重试失败，抛出异常
            if isinstance(last_exception, AppException):
                raise last_exception
            else:
                raise SystemException(
                    code="RETRY_EXHAUSTED",
                    message=f"操作重试{max_attempts}次后仍然失败: {str(last_exception)}",
                    details={"original_exception": str(last_exception), "function": func.__name__}
                )
        
        # 根据函数是否为协程函数选择适当的包装器
        import asyncio
        import inspect
        if inspect.iscoroutinefunction(func):
            return wrapper
        else:
            return sync_wrapper
    
    return decorator

# 熔断器状态
class CircuitState(str, Enum):
    CLOSED = "closed"  # 正常状态，允许请求通过
    OPEN = "open"      # 熔断状态，拒绝所有请求
    HALF_OPEN = "half_open"  # 半开状态，允许有限请求通过以测试服务是否恢复

# 熔断器类
class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.exceptions = exceptions
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
    
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            self._check_state_transition()
            
            if self.state == CircuitState.OPEN:
                raise DependencyException(
                    code="CIRCUIT_OPEN",
                    message="服务暂时不可用，熔断器已打开",
                    details={"circuit_state": self.state, "recovery_timeout": self.recovery_timeout}
                )
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise DependencyException(
                        code="CIRCUIT_HALF_OPEN_MAX_CALLS",
                        message="服务正在恢复中，请稍后再试",
                        details={"circuit_state": self.state, "half_open_calls": self.half_open_calls}
                    )
                self.half_open_calls += 1
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.exceptions as e:
                self._on_failure()
                raise e
        
        # 同步函数版本
        def sync_wrapper(*args, **kwargs):
            self._check_state_transition()
            
            if self.state == CircuitState.OPEN:
                raise DependencyException(
                    code="CIRCUIT_OPEN",
                    message="服务暂时不可用，熔断器已打开",
                    details={"circuit_state": self.state, "recovery_timeout": self.recovery_timeout}
                )
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise DependencyException(
                        code="CIRCUIT_HALF_OPEN_MAX_CALLS",
                        message="服务正在恢复中，请稍后再试",
                        details={"circuit_state": self.state, "half_open_calls": self.half_open_calls}
                    )
                self.half_open_calls += 1
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.exceptions as e:
                self._on_failure()
                raise e
        
        # 根据函数是否为协程函数选择适当的包装器
        import inspect
        if inspect.iscoroutinefunction(func):
            return wrapper
        else:
            return sync_wrapper
    
    def _check_state_transition(self):
        """检查是否需要状态转换"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"熔断器状态从 OPEN 转换为 HALF_OPEN")
    
    def _on_success(self):
        """成功调用后的处理"""
        if self.state == CircuitState.HALF_OPEN:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            logger.info(f"熔断器状态从 HALF_OPEN 转换为 CLOSED")
    
    def _on_failure(self):
        """失败调用后的处理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"熔断器状态从 CLOSED 转换为 OPEN，失败次数: {self.failure_count}")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"熔断器状态从 HALF_OPEN 转换为 OPEN，恢复测试失败")
    
    def get_state(self) -> CircuitState:
        """获取当前熔断器状态"""
        self._check_state_transition()
        return self.state
    
    def reset(self):
        """重置熔断器状态"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        logger.info("熔断器状态已重置为 CLOSED")