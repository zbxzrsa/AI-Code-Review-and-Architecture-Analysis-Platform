import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import aiohttp
from pydantic import BaseModel
from app.core.logger import logger

class HealthStatus(str, Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class DependencyType(str, Enum):
    """依赖类型枚举"""
    DATABASE = "database"
    CACHE = "cache"
    API = "api"
    STORAGE = "storage"
    QUEUE = "queue"
    OTHER = "other"

class HealthCheckResult(BaseModel):
    """健康检查结果模型"""
    name: str
    status: HealthStatus
    type: DependencyType
    response_time_ms: float
    message: Optional[str] = None
    last_check_time: float
    details: Optional[Dict[str, Any]] = None

class SystemHealth(BaseModel):
    """系统健康状态模型"""
    status: HealthStatus
    version: str
    uptime_seconds: float
    dependencies: List[HealthCheckResult]
    timestamp: float

class HealthCheck:
    """健康检查基类"""
    def __init__(self, name: str, dependency_type: DependencyType):
        self.name = name
        self.type = dependency_type
        self.last_result: Optional[HealthCheckResult] = None
    
    async def check(self) -> HealthCheckResult:
        """执行健康检查"""
        start_time = time.time()
        status = HealthStatus.UNKNOWN
        message = None
        details = None
        
        try:
            status, message, details = await self._perform_check()
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"健康检查异常: {str(e)}"
            logger.error(f"健康检查 {self.name} 失败", extra={
                "exception": str(e),
                "check_name": self.name,
                "check_type": self.type
            })
        
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        result = HealthCheckResult(
            name=self.name,
            status=status,
            type=self.type,
            response_time_ms=response_time_ms,
            message=message,
            last_check_time=end_time,
            details=details
        )
        
        self.last_result = result
        return result
    
    async def _perform_check(self) -> tuple[HealthStatus, Optional[str], Optional[Dict[str, Any]]]:
        """执行具体的健康检查逻辑，由子类实现"""
        raise NotImplementedError("子类必须实现此方法")

class HttpHealthCheck(HealthCheck):
    """HTTP服务健康检查"""
    def __init__(
        self,
        name: str,
        url: str,
        dependency_type: DependencyType = DependencyType.API,
        method: str = "GET",
        timeout: float = 5.0,
        headers: Optional[Dict[str, str]] = None,
        expected_status: int = 200,
        expected_content: Optional[str] = None
    ):
        super().__init__(name, dependency_type)
        self.url = url
        self.method = method
        self.timeout = timeout
        self.headers = headers or {}
        self.expected_status = expected_status
        self.expected_content = expected_content
    
    async def _perform_check(self) -> tuple[HealthStatus, Optional[str], Optional[Dict[str, Any]]]:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method=self.method,
                    url=self.url,
                    headers=self.headers,
                    timeout=self.timeout
                ) as response:
                    content = await response.text()
                    
                    if response.status != self.expected_status:
                        return (
                            HealthStatus.UNHEALTHY,
                            f"HTTP状态码不匹配: 期望 {self.expected_status}, 实际 {response.status}",
                            {"status_code": response.status, "content": content[:200]}
                        )
                    
                    if self.expected_content and self.expected_content not in content:
                        return (
                            HealthStatus.DEGRADED,
                            f"响应内容不匹配",
                            {"content": content[:200]}
                        )
                    
                    return (
                        HealthStatus.HEALTHY,
                        "HTTP服务正常",
                        {"status_code": response.status}
                    )
            except asyncio.TimeoutError:
                return (
                    HealthStatus.UNHEALTHY,
                    f"请求超时 (>{self.timeout}秒)",
                    {"url": self.url}
                )
            except Exception as e:
                return (
                    HealthStatus.UNHEALTHY,
                    f"HTTP请求异常: {str(e)}",
                    {"url": self.url, "exception": str(e)}
                )

class DatabaseHealthCheck(HealthCheck):
    """数据库健康检查"""
    def __init__(
        self,
        name: str,
        db_session,
        query: str = "SELECT 1",
        timeout: float = 5.0
    ):
        super().__init__(name, DependencyType.DATABASE)
        self.db_session = db_session
        self.query = query
        self.timeout = timeout
    
    async def _perform_check(self) -> tuple[HealthStatus, Optional[str], Optional[Dict[str, Any]]]:
        try:
            # 执行简单查询检查数据库连接
            result = await asyncio.wait_for(
                self._execute_query(),
                timeout=self.timeout
            )
            
            return (
                HealthStatus.HEALTHY,
                "数据库连接正常",
                {"query": self.query, "result": str(result)}
            )
        except asyncio.TimeoutError:
            return (
                HealthStatus.UNHEALTHY,
                f"数据库查询超时 (>{self.timeout}秒)",
                {"query": self.query}
            )
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"数据库查询异常: {str(e)}",
                {"query": self.query, "exception": str(e)}
            )
    
    async def _execute_query(self):
        """执行数据库查询"""
        # 这里的实现取决于您使用的ORM/数据库库
        # 以下是使用SQLAlchemy异步会话的示例
        async with self.db_session() as session:
            result = await session.execute(self.query)
            return result.scalar()

class HealthCheckService:
    """健康检查服务"""
    def __init__(self, app_version: str, check_interval: float = 60.0):
        self.app_version = app_version
        self.check_interval = check_interval
        self.start_time = time.time()
        self.health_checks: List[HealthCheck] = []
        self.last_system_health: Optional[SystemHealth] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def register_check(self, health_check: HealthCheck) -> None:
        """注册健康检查"""
        self.health_checks.append(health_check)
    
    async def check_all(self) -> SystemHealth:
        """执行所有健康检查"""
        check_results = await asyncio.gather(
            *[check.check() for check in self.health_checks],
            return_exceptions=True
        )
        
        # 处理结果，将异常转换为失败的健康检查结果
        processed_results: List[HealthCheckResult] = []
        for i, result in enumerate(check_results):
            if isinstance(result, Exception):
                check = self.health_checks[i]
                processed_results.append(
                    HealthCheckResult(
                        name=check.name,
                        status=HealthStatus.UNHEALTHY,
                        type=check.type,
                        response_time_ms=0.0,
                        message=f"检查执行异常: {str(result)}",
                        last_check_time=time.time(),
                        details={"exception": str(result)}
                    )
                )
            else:
                processed_results.append(result)
        
        # 确定整体系统健康状态
        if any(r.status == HealthStatus.UNHEALTHY for r in processed_results):
            overall_status = HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.DEGRADED for r in processed_results):
            overall_status = HealthStatus.DEGRADED
        elif all(r.status == HealthStatus.HEALTHY for r in processed_results):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        # 创建系统健康状态
        system_health = SystemHealth(
            status=overall_status,
            version=self.app_version,
            uptime_seconds=time.time() - self.start_time,
            dependencies=processed_results,
            timestamp=time.time()
        )
        
        self.last_system_health = system_health
        
        # 记录健康状态
        log_level = "error" if overall_status == HealthStatus.UNHEALTHY else \
                   "warning" if overall_status == HealthStatus.DEGRADED else \
                   "info"
        
        getattr(logger, log_level)(
            f"系统健康状态: {overall_status}",
            extra={
                "health_status": overall_status,
                "unhealthy_services": [
                    r.name for r in processed_results if r.status == HealthStatus.UNHEALTHY
                ],
                "degraded_services": [
                    r.name for r in processed_results if r.status == HealthStatus.DEGRADED
                ]
            }
        )
        
        return system_health
    
    async def start_monitoring(self) -> None:
        """启动定期健康检查监控"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"健康检查监控已启动，检查间隔: {self.check_interval}秒")
    
    async def stop_monitoring(self) -> None:
        """停止健康检查监控"""
        if not self._running or not self._task:
            return
        
        self._running = False
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        
        logger.info("健康检查监控已停止")
    
    async def _monitoring_loop(self) -> None:
        """健康检查监控循环"""
        while self._running:
            try:
                await self.check_all()
            except Exception as e:
                logger.error(f"健康检查监控异常: {str(e)}")
            
            await asyncio.sleep(self.check_interval)

# 创建健康检查服务实例
health_service = HealthCheckService(app_version="1.0.0")