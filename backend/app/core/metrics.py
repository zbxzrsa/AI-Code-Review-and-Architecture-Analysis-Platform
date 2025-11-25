import time
import asyncio
import functools
import statistics
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union, cast
from pydantic import BaseModel
from app.core.logger import logger

# 类型变量定义
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

class MetricType(str, Enum):
    """指标类型枚举"""
    COUNTER = "counter"  # 计数器，只增不减
    GAUGE = "gauge"      # 仪表盘，可增可减
    HISTOGRAM = "histogram"  # 直方图，记录数值分布
    TIMER = "timer"      # 计时器，特殊的直方图

class MetricValue(BaseModel):
    """指标值模型"""
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = {}

class Metric(BaseModel):
    """指标模型"""
    name: str
    type: MetricType
    description: str
    values: List[MetricValue] = []
    
    # 直方图和计时器特有字段
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    sum_value: Optional[float] = None
    count: Optional[int] = None
    avg_value: Optional[float] = None
    median_value: Optional[float] = None
    percentile_95: Optional[float] = None
    percentile_99: Optional[float] = None

class MetricsService:
    """指标服务"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._metrics: Dict[str, Metric] = {}
        self._reporting_interval = 60  # 默认60秒上报一次
        self._reporting_task = None
        self._reporting_endpoint = None
        self._max_values_per_metric = 1000  # 每个指标最多保存的值数量
        self._initialized = True
    
    def configure(
        self,
        reporting_interval: int = 60,
        reporting_endpoint: Optional[str] = None,
        max_values_per_metric: int = 1000
    ) -> None:
        """配置指标服务"""
        self._reporting_interval = reporting_interval
        self._reporting_endpoint = reporting_endpoint
        self._max_values_per_metric = max_values_per_metric
    
    async def start_reporting(self) -> None:
        """启动定期上报"""
        if self._reporting_task is not None:
            return
            
        self._reporting_task = asyncio.create_task(self._reporting_loop())
        logger.info(f"指标上报已启动，间隔: {self._reporting_interval}秒")
    
    async def stop_reporting(self) -> None:
        """停止定期上报"""
        if self._reporting_task is None:
            return
            
        self._reporting_task.cancel()
        try:
            await self._reporting_task
        except asyncio.CancelledError:
            pass
            
        self._reporting_task = None
        logger.info("指标上报已停止")
    
    async def _reporting_loop(self) -> None:
        """上报循环"""
        while True:
            try:
                # 计算聚合指标
                self._calculate_aggregates()
                
                # 上报指标
                if self._reporting_endpoint:
                    await self._report_metrics()
                
                # 清理旧数据
                self._clean_old_values()
                
            except Exception as e:
                logger.error(f"指标上报异常: {str(e)}")
                
            await asyncio.sleep(self._reporting_interval)
    
    def _calculate_aggregates(self) -> None:
        """计算聚合指标"""
        for metric_name, metric in self._metrics.items():
            if metric.type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                values = [v.value for v in metric.values]
                if values:
                    metric.min_value = min(values)
                    metric.max_value = max(values)
                    metric.sum_value = sum(values)
                    metric.count = len(values)
                    metric.avg_value = metric.sum_value / metric.count
                    
                    # 计算中位数和百分位数
                    sorted_values = sorted(values)
                    metric.median_value = statistics.median(sorted_values)
                    
                    # 95th 百分位数
                    idx_95 = int(len(sorted_values) * 0.95)
                    metric.percentile_95 = sorted_values[idx_95]
                    
                    # 99th 百分位数
                    idx_99 = int(len(sorted_values) * 0.99)
                    metric.percentile_99 = sorted_values[idx_99]
    
    async def _report_metrics(self) -> None:
        """上报指标到外部系统"""
        if not self._reporting_endpoint:
            return
            
        # 这里可以实现将指标上报到监控系统的逻辑
        # 例如 Prometheus, Datadog, InfluxDB 等
        logger.info(f"上报 {len(self._metrics)} 个指标到 {self._reporting_endpoint}")
    
    def _clean_old_values(self) -> None:
        """清理旧的指标值"""
        for metric_name, metric in self._metrics.items():
            if len(metric.values) > self._max_values_per_metric:
                # 保留最新的值
                metric.values = metric.values[-self._max_values_per_metric:]
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str
    ) -> Metric:
        """注册新指标"""
        if name in self._metrics:
            return self._metrics[name]
            
        metric = Metric(
            name=name,
            type=metric_type,
            description=description
        )
        
        self._metrics[name] = metric
        return metric
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """获取指标"""
        return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """获取所有指标"""
        return self._metrics
    
    def increment_counter(
        self,
        name: str,
        value: int = 1,
        tags: Dict[str, str] = None,
        description: str = ""
    ) -> None:
        """增加计数器值"""
        if name not in self._metrics:
            self.register_metric(
                name=name,
                metric_type=MetricType.COUNTER,
                description=description or f"Counter metric: {name}"
            )
            
        metric = self._metrics[name]
        if metric.type != MetricType.COUNTER:
            logger.warning(f"尝试增加非计数器指标: {name}")
            return
            
        metric_value = MetricValue(
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        metric.values.append(metric_value)
    
    def set_gauge(
        self,
        name: str,
        value: Union[int, float],
        tags: Dict[str, str] = None,
        description: str = ""
    ) -> None:
        """设置仪表盘值"""
        if name not in self._metrics:
            self.register_metric(
                name=name,
                metric_type=MetricType.GAUGE,
                description=description or f"Gauge metric: {name}"
            )
            
        metric = self._metrics[name]
        if metric.type != MetricType.GAUGE:
            logger.warning(f"尝试设置非仪表盘指标: {name}")
            return
            
        metric_value = MetricValue(
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        metric.values.append(metric_value)
    
    def record_histogram(
        self,
        name: str,
        value: Union[int, float],
        tags: Dict[str, str] = None,
        description: str = ""
    ) -> None:
        """记录直方图值"""
        if name not in self._metrics:
            self.register_metric(
                name=name,
                metric_type=MetricType.HISTOGRAM,
                description=description or f"Histogram metric: {name}"
            )
            
        metric = self._metrics[name]
        if metric.type != MetricType.HISTOGRAM:
            logger.warning(f"尝试记录非直方图指标: {name}")
            return
            
        metric_value = MetricValue(
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        metric.values.append(metric_value)
    
    def record_timer(
        self,
        name: str,
        value: float,
        tags: Dict[str, str] = None,
        description: str = ""
    ) -> None:
        """记录计时器值"""
        if name not in self._metrics:
            self.register_metric(
                name=name,
                metric_type=MetricType.TIMER,
                description=description or f"Timer metric: {name}"
            )
            
        metric = self._metrics[name]
        if metric.type != MetricType.TIMER:
            logger.warning(f"尝试记录非计时器指标: {name}")
            return
            
        metric_value = MetricValue(
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        metric.values.append(metric_value)
    
    def time_function(
        self,
        name: str,
        tags: Dict[str, str] = None,
        description: str = ""
    ) -> Callable[[F], F]:
        """函数执行时间装饰器"""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    self.record_timer(
                        name=name,
                        value=duration_ms,
                        tags=tags,
                        description=description
                    )
            return cast(F, wrapper)
        return decorator
    
    def time_async_function(
        self,
        name: str,
        tags: Dict[str, str] = None,
        description: str = ""
    ) -> Callable[[AsyncF], AsyncF]:
        """异步函数执行时间装饰器"""
        def decorator(func: AsyncF) -> AsyncF:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    self.record_timer(
                        name=name,
                        value=duration_ms,
                        tags=tags,
                        description=description
                    )
            return cast(AsyncF, wrapper)
        return decorator

# 创建单例实例
metrics_service = MetricsService()

# 便捷函数
def increment_counter(name: str, value: int = 1, tags: Dict[str, str] = None, description: str = "") -> None:
    """增加计数器值"""
    metrics_service.increment_counter(name, value, tags, description)

def set_gauge(name: str, value: Union[int, float], tags: Dict[str, str] = None, description: str = "") -> None:
    """设置仪表盘值"""
    metrics_service.set_gauge(name, value, tags, description)

def record_histogram(name: str, value: Union[int, float], tags: Dict[str, str] = None, description: str = "") -> None:
    """记录直方图值"""
    metrics_service.record_histogram(name, value, tags, description)

def record_timer(name: str, value: float, tags: Dict[str, str] = None, description: str = "") -> None:
    """记录计时器值"""
    metrics_service.record_timer(name, value, tags, description)

def time_function(name: str, tags: Dict[str, str] = None, description: str = "") -> Callable[[F], F]:
    """函数执行时间装饰器"""
    return metrics_service.time_function(name, tags, description)

def time_async_function(name: str, tags: Dict[str, str] = None, description: str = "") -> Callable[[AsyncF], AsyncF]:
    """异步函数执行时间装饰器"""
    return metrics_service.time_async_function(name, tags, description)

# 上下文管理器用于测量代码块执行时间
class TimerContext:
    """计时器上下文管理器"""
    def __init__(self, name: str, tags: Dict[str, str] = None, description: str = ""):
        self.name = name
        self.tags = tags or {}
        self.description = description
        self.start_time = 0.0
    
    def __enter__(self) -> 'TimerContext':
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        end_time = time.time()
        duration_ms = (end_time - self.start_time) * 1000
        
        # 如果发生异常，添加异常信息到标签
        if exc_type is not None:
            self.tags['exception'] = str(exc_type.__name__)
            self.tags['exception_message'] = str(exc_val)
        
        metrics_service.record_timer(
            name=self.name,
            value=duration_ms,
            tags=self.tags,
            description=self.description
        )

# 异步上下文管理器
class AsyncTimerContext:
    """异步计时器上下文管理器"""
    def __init__(self, name: str, tags: Dict[str, str] = None, description: str = ""):
        self.name = name
        self.tags = tags or {}
        self.description = description
        self.start_time = 0.0
    
    async def __aenter__(self) -> 'AsyncTimerContext':
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        end_time = time.time()
        duration_ms = (end_time - self.start_time) * 1000
        
        # 如果发生异常，添加异常信息到标签
        if exc_type is not None:
            self.tags['exception'] = str(exc_type.__name__)
            self.tags['exception_message'] = str(exc_val)
        
        metrics_service.record_timer(
            name=self.name,
            value=duration_ms,
            tags=self.tags,
            description=self.description
        )

# 便捷函数
def timer(name: str, tags: Dict[str, str] = None, description: str = "") -> TimerContext:
    """创建计时器上下文管理器"""
    return TimerContext(name, tags, description)

def async_timer(name: str, tags: Dict[str, str] = None, description: str = "") -> AsyncTimerContext:
    """创建异步计时器上下文管理器"""
    return AsyncTimerContext(name, tags, description)

# 资源使用情况监控
class ResourceMonitor:
    """资源使用情况监控"""
    def __init__(self, interval: int = 60):
        self.interval = interval
        self._task = None
        self._running = False
    
    async def start(self) -> None:
        """启动资源监控"""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"资源监控已启动，间隔: {self.interval}秒")
    
    async def stop(self) -> None:
        """停止资源监控"""
        if not self._running or not self._task:
            return
            
        self._running = False
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
            
        logger.info("资源监控已停止")
    
    async def _monitoring_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                # 收集内存使用情况
                self._collect_memory_metrics()
                
                # 收集CPU使用情况
                self._collect_cpu_metrics()
                
                # 收集磁盘使用情况
                self._collect_disk_metrics()
                
            except Exception as e:
                logger.error(f"资源监控异常: {str(e)}")
                
            await asyncio.sleep(self.interval)
    
    def _collect_memory_metrics(self) -> None:
        """收集内存指标"""
        try:
            import psutil
            
            # 获取内存信息
            memory = psutil.virtual_memory()
            
            # 记录内存使用率
            set_gauge(
                name="system.memory.usage_percent",
                value=memory.percent,
                description="系统内存使用率百分比"
            )
            
            # 记录已用内存
            set_gauge(
                name="system.memory.used_bytes",
                value=memory.used,
                description="系统已用内存字节数"
            )
            
            # 记录可用内存
            set_gauge(
                name="system.memory.available_bytes",
                value=memory.available,
                description="系统可用内存字节数"
            )
            
            # 记录进程内存使用
            process = psutil.Process()
            process_memory = process.memory_info()
            
            set_gauge(
                name="process.memory.rss_bytes",
                value=process_memory.rss,
                description="进程常驻内存集大小"
            )
            
            set_gauge(
                name="process.memory.vms_bytes",
                value=process_memory.vms,
                description="进程虚拟内存大小"
            )
            
        except ImportError:
            logger.warning("无法导入psutil模块，跳过内存指标收集")
        except Exception as e:
            logger.error(f"收集内存指标异常: {str(e)}")
    
    def _collect_cpu_metrics(self) -> None:
        """收集CPU指标"""
        try:
            import psutil
            
            # 记录系统CPU使用率
            set_gauge(
                name="system.cpu.usage_percent",
                value=psutil.cpu_percent(interval=1),
                description="系统CPU使用率百分比"
            )
            
            # 记录进程CPU使用率
            process = psutil.Process()
            set_gauge(
                name="process.cpu.usage_percent",
                value=process.cpu_percent(interval=1) / psutil.cpu_count(),
                description="进程CPU使用率百分比"
            )
            
        except ImportError:
            logger.warning("无法导入psutil模块，跳过CPU指标收集")
        except Exception as e:
            logger.error(f"收集CPU指标异常: {str(e)}")
    
    def _collect_disk_metrics(self) -> None:
        """收集磁盘指标"""
        try:
            import psutil
            
            # 获取磁盘使用情况
            disk = psutil.disk_usage('/')
            
            # 记录磁盘使用率
            set_gauge(
                name="system.disk.usage_percent",
                value=disk.percent,
                description="系统磁盘使用率百分比"
            )
            
            # 记录已用磁盘空间
            set_gauge(
                name="system.disk.used_bytes",
                value=disk.used,
                description="系统已用磁盘空间字节数"
            )
            
            # 记录可用磁盘空间
            set_gauge(
                name="system.disk.free_bytes",
                value=disk.free,
                description="系统可用磁盘空间字节数"
            )
            
        except ImportError:
            logger.warning("无法导入psutil模块，跳过磁盘指标收集")
        except Exception as e:
            logger.error(f"收集磁盘指标异常: {str(e)}")

# 创建资源监控实例
resource_monitor = ResourceMonitor()

# 向后兼容的度量常量（用于现有代码）
ANALYSIS_STARTED = "analysis_started"
ANALYSIS_COMPLETED = "analysis_completed"
CACHED_FILES_SKIPPED = "cached_files_skipped"
ANALYSIS_DURATION = "analysis_duration"
INCREMENTAL_HIT = "incremental_hit"