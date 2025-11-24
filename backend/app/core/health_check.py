import asyncio
import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable
import aiohttp
import psutil
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# 健康状态枚举
class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

# 健康检查结果模型
class HealthCheckResult(BaseModel):
    name: str
    status: HealthStatus
    response_time_ms: float
    details: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    timestamp: float = time.time()

# 系统资源信息模型
class SystemResources(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, Any]
    process_count: int
    uptime: float

# 健康检查管理器
class HealthCheckManager:
    def __init__(self):
        self.checks: Dict[str, Callable[[], Awaitable[HealthCheckResult]]] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.check_interval: int = 60  # 默认检查间隔(秒)
        self._running = False
        self._task = None
    
    def register_check(self, name: str, check_func: Callable[[], Awaitable[HealthCheckResult]]):
        """注册健康检查函数"""
        self.checks[name] = check_func
        logger.info(f"已注册健康检查: {name}")
    
    async def run_checks(self) -> Dict[str, HealthCheckResult]:
        """运行所有注册的健康检查"""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[name] = result
                self.last_results[name] = result
                
                if result.status != HealthStatus.HEALTHY:
                    logger.warning(f"健康检查 {name} 状态: {result.status}, 消息: {result.message}")
            except Exception as e:
                logger.error(f"健康检查 {name} 执行失败: {str(e)}", exc_info=True)
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=-1,
                    message=f"检查执行失败: {str(e)}"
                )
                self.last_results[name] = results[name]
        
        return results
    
    def start_background_checks(self, interval: int = None):
        """启动后台健康检查任务"""
        if interval is not None:
            self.check_interval = interval
        
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._background_check_task())
        logger.info(f"已启动后台健康检查，间隔: {self.check_interval}秒")
    
    def stop_background_checks(self):
        """停止后台健康检查任务"""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("已停止后台健康检查")
    
    async def _background_check_task(self):
        """后台健康检查任务"""
        while self._running:
            try:
                await self.run_checks()
            except Exception as e:
                logger.error(f"后台健康检查任务异常: {str(e)}", exc_info=True)
            
            await asyncio.sleep(self.check_interval)
    
    def get_last_results(self) -> Dict[str, HealthCheckResult]:
        """获取最近一次的健康检查结果"""
        return self.last_results
    
    def get_overall_status(self) -> HealthStatus:
        """获取整体健康状态"""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.last_results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

# 创建健康检查管理器实例
health_check_manager = HealthCheckManager()

# 数据库健康检查
async def check_database_health(db_session_factory) -> HealthCheckResult:
    """检查数据库连接健康状态"""
    start_time = time.time()
    
    try:
        async with db_session_factory() as session:
            session: AsyncSession
            # 执行简单查询测试连接
            result = await session.execute(text("SELECT 1"))
            await session.commit()
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                details={"query": "SELECT 1", "result": "success"},
                message="数据库连接正常"
            )
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="database",
            status=HealthStatus.UNHEALTHY,
            response_time_ms=response_time,
            details={"error": str(e)},
            message=f"数据库连接失败: {str(e)}"
        )

# Redis健康检查
async def check_redis_health(redis_client) -> HealthCheckResult:
    """检查Redis连接健康状态"""
    start_time = time.time()
    
    try:
        # 执行PING命令测试连接
        result = await redis_client.ping()
        
        response_time = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="redis",
            status=HealthStatus.HEALTHY if result else HealthStatus.DEGRADED,
            response_time_ms=response_time,
            details={"ping": "success" if result else "failed"},
            message="Redis连接正常" if result else "Redis响应异常"
        )
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            response_time_ms=response_time,
            details={"error": str(e)},
            message=f"Redis连接失败: {str(e)}"
        )

# 外部API健康检查
async def check_external_api_health(api_url: str, timeout: float = 5.0) -> HealthCheckResult:
    """检查外部API健康状态"""
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, timeout=timeout) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status < 400:
                    status = HealthStatus.HEALTHY
                    message = f"API响应正常: {response.status}"
                elif response.status < 500:
                    status = HealthStatus.DEGRADED
                    message = f"API响应异常: {response.status}"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = f"API服务器错误: {response.status}"
                
                return HealthCheckResult(
                    name=f"api_{api_url.replace('://', '_').replace('/', '_').replace('.', '_')}",
                    status=status,
                    response_time_ms=response_time,
                    details={"status_code": response.status, "url": api_url},
                    message=message
                )
    except asyncio.TimeoutError:
        response_time = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name=f"api_{api_url.replace('://', '_').replace('/', '_').replace('.', '_')}",
            status=HealthStatus.UNHEALTHY,
            response_time_ms=response_time,
            details={"error": "timeout", "url": api_url},
            message=f"API请求超时: {api_url}"
        )
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name=f"api_{api_url.replace('://', '_').replace('/', '_').replace('.', '_')}",
            status=HealthStatus.UNHEALTHY,
            response_time_ms=response_time,
            details={"error": str(e), "url": api_url},
            message=f"API请求失败: {str(e)}"
        )

# 系统资源健康检查
async def check_system_resources() -> HealthCheckResult:
    """检查系统资源使用情况"""
    start_time = time.time()
    
    try:
        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.5)
        
        # 获取内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 获取磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # 获取网络IO
        net_io = psutil.net_io_counters()
        
        # 获取进程数
        process_count = len(psutil.pids())
        
        # 获取系统启动时间
        uptime = time.time() - psutil.boot_time()
        
        # 计算系统资源状态
        if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
            status = HealthStatus.UNHEALTHY
            message = "系统资源严重不足"
        elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 80:
            status = HealthStatus.DEGRADED
            message = "系统资源紧张"
        else:
            status = HealthStatus.HEALTHY
            message = "系统资源充足"
        
        response_time = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="system_resources",
            status=status,
            response_time_ms=response_time,
            details={
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "network": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv
                },
                "process_count": process_count,
                "uptime_seconds": uptime
            },
            message=message
        )
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="system_resources",
            status=HealthStatus.UNKNOWN,
            response_time_ms=response_time,
            details={"error": str(e)},
            message=f"系统资源检查失败: {str(e)}"
        )

# 依赖服务检查
async def check_dependency_services(services: List[Dict[str, Any]]) -> Dict[str, HealthCheckResult]:
    """检查多个依赖服务的健康状态"""
    results = {}
    
    for service in services:
        service_type = service.get("type", "unknown")
        service_name = service.get("name", f"{service_type}_{len(results)}")
        
        if service_type == "http":
            result = await check_external_api_health(
                service.get("url", ""),
                timeout=service.get("timeout", 5.0)
            )
            results[service_name] = result
        elif service_type == "redis":
            # 假设redis_client已经在其他地方定义
            if "client" in service:
                result = await check_redis_health(service["client"])
                results[service_name] = result
        elif service_type == "database":
            # 假设db_session_factory已经在其他地方定义
            if "session_factory" in service:
                result = await check_database_health(service["session_factory"])
                results[service_name] = result
    
    return results

# 初始化健康检查
def init_health_checks(app=None, db_session_factory=None, redis_client=None, external_apis=None):
    """初始化并注册所有健康检查"""
    # 注册系统资源检查
    health_check_manager.register_check("system", check_system_resources)
    
    # 注册数据库健康检查
    if db_session_factory:
        health_check_manager.register_check(
            "database",
            lambda: check_database_health(db_session_factory)
        )
    
    # 注册Redis健康检查
    if redis_client:
        health_check_manager.register_check(
            "redis",
            lambda: check_redis_health(redis_client)
        )
    
    # 注册外部API健康检查
    if external_apis:
        for i, api_url in enumerate(external_apis):
            health_check_manager.register_check(
                f"external_api_{i}",
                lambda url=api_url: check_external_api_health(url)
            )
    
    # 启动后台健康检查
    health_check_manager.start_background_checks()
    
    # 如果提供了FastAPI应用实例，添加健康检查路由
    if app:
        from fastapi import APIRouter
        
        health_router = APIRouter()
        
        @health_router.get("/health")
        async def health_check():
            """健康检查接口"""
            results = await health_check_manager.run_checks()
            overall_status = health_check_manager.get_overall_status()
            
            return {
                "status": overall_status,
                "timestamp": time.time(),
                "checks": {name: result.dict() for name, result in results.items()}
            }
        
        @health_router.get("/health/{check_name}")
        async def specific_health_check(check_name: str):
            """特定健康检查接口"""
            if check_name not in health_check_manager.checks:
                return {"error": f"未找到健康检查: {check_name}"}
            
            result = await health_check_manager.checks[check_name]()
            return result.dict()
        
        # 注册健康检查路由
        app.include_router(health_router, tags=["health"])
        
        logger.info("已注册健康检查路由: /health 和 /health/{check_name}")
    
    return health_check_manager