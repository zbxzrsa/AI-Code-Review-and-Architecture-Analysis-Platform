"""
综合监控和日志系统
支持性能监控、健康检查、指标收集和分布式追踪
"""
import asyncio
import json
import logging
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import psutil
import threading
from collections import defaultdict, deque
from functools import wraps
import hashlib

# 尝试导入可选依赖
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MetricType(Enum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"

@dataclass
class MetricValue:
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class LogEntry:
    timestamp: datetime
    level: LogLevel
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    exception: Optional[str] = None

class PerformanceTracker:
    """性能追踪器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """记录指标"""
        with self.lock:
            key = f"{name}:{hash(str(labels))}" if labels else name
            self.metrics[key].append({
                'value': value,
                'timestamp': time.time(),
                'labels': labels or {}
            })
    
    def get_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """获取统计信息"""
        key = f"{name}:{hash(str(labels))}" if labels else name
        values = [m['value'] for m in self.metrics[key]]
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1] if values else 0
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        result = {}
        for key in self.metrics:
            if ':' not in key:
                result[key] = self.get_stats(key)
        return result

class DistributedTracer:
    """分布式追踪器"""
    
    def __init__(self):
        self.active_spans: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def start_span(self, operation_name: str, parent_span_id: str = None) -> str:
        """开始新的span"""
        span_id = hashlib.md5(f"{operation_name}{time.time()}".encode()).hexdigest()[:16]
        
        with self.lock:
            self.active_spans[span_id] = {
                'operation_name': operation_name,
                'start_time': time.time(),
                'parent_span_id': parent_span_id,
                'tags': {},
                'logs': []
            }
        
        return span_id
    
    def finish_span(self, span_id: str, error: Exception = None) -> None:
        """结束span"""
        with self.lock:
            if span_id in self.active_spans:
                span = self.active_spans[span_id]
                span['end_time'] = time.time()
                span['duration'] = span['end_time'] - span['start_time']
                span['error'] = str(error) if error else None
                
                # 移除活跃span
                del self.active_spans[span_id]
    
    def add_tag(self, span_id: str, key: str, value: Any) -> None:
        """添加标签"""
        with self.lock:
            if span_id in self.active_spans:
                self.active_spans[span_id]['tags'][key] = value
    
    def add_log(self, span_id: str, level: str, message: str) -> None:
        """添加日志"""
        with self.lock:
            if span_id in self.active_spans:
                self.active_spans[span_id]['logs'].append({
                    'timestamp': time.time(),
                    'level': level,
                    'message': message
                })

class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """注册健康检查"""
        self.checks[name] = check_func
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """运行所有健康检查"""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                duration = time.time() - start_time
                
                results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'duration': duration,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'details': result if isinstance(result, dict) else {}
                }
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
        
        with self.lock:
            self.results = results
        
        # 计算整体状态
        overall_status = 'healthy' if all(
            r['status'] == 'healthy' for r in results.values()
        ) else 'unhealthy'
        
        return {
            'status': overall_status,
            'checks': results,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

class MonitoringSystem:
    """综合监控系统"""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.tracer = DistributedTracer()
        self.health_checker = HealthChecker()
        self.logger = self._setup_logger()
        self.metrics: Dict[str, Any] = {}
        
        # Prometheus指标（如果可用）
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
        
        # 注册默认健康检查
        self._register_default_health_checks()
    
    def _setup_logger(self) -> logging.Logger:
        """设置结构化日志"""
        if STRUCTLOG_AVAILABLE:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            return structlog.get_logger()
        else:
            logger = logging.getLogger("monitoring")
            logger.setLevel(logging.INFO)
            
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            
            return logger
    
    def _setup_prometheus_metrics(self) -> None:
        """设置Prometheus指标"""
        self.prometheus_metrics = {
            'http_requests_total': Counter(
                'http_requests_total',
                'Total HTTP requests',
                ['method', 'endpoint', 'status']
            ),
            'http_request_duration': Histogram(
                'http_request_duration_seconds',
                'HTTP request duration',
                ['method', 'endpoint']
            ),
            'active_connections': Gauge(
                'active_connections',
                'Number of active connections'
            ),
            'system_memory_usage': Gauge(
                'system_memory_usage_bytes',
                'System memory usage in bytes'
            )
        }
    
    def _register_default_health_checks(self) -> None:
        """注册默认健康检查"""
        self.health_checker.register_check('memory', self._check_memory)
        self.health_checker.register_check('disk', self._check_disk)
        self.health_checker.register_check('cpu', self._check_cpu)
    
    def _check_memory(self) -> Dict[str, Any]:
        """检查内存使用"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used,
            'status': 'healthy' if memory.percent < 90 else 'warning'
        }
    
    def _check_disk(self) -> Dict[str, Any]:
        """检查磁盘使用"""
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': (disk.used / disk.total) * 100,
            'status': 'healthy' if (disk.used / disk.total) < 0.9 else 'warning'
        }
    
    def _check_cpu(self) -> Dict[str, Any]:
        """检查CPU使用"""
        cpu_percent = psutil.cpu_percent(interval=1)
        return {
            'percent': cpu_percent,
            'count': psutil.cpu_count(),
            'status': 'healthy' if cpu_percent < 80 else 'warning'
        }
    
    def log(self, level: LogLevel, message: str, context: Dict[str, Any] = None) -> None:
        """记录日志"""
        log_entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            message=message,
            context=context or {}
        )
        
        if STRUCTLOG_AVAILABLE:
            getattr(self.logger, level.value.lower())(message, **(context or {}))
        else:
            getattr(self.logger, level.value.lower())(
                f"{message} - Context: {context or {}}"
            )
    
    def record_metric(self, name: str, value: Union[int, float], 
                    metric_type: MetricType = MetricType.HISTOGRAM,
                    labels: Dict[str, str] = None) -> None:
        """记录指标"""
        self.performance_tracker.record_metric(name, float(value), labels)
        
        # 更新Prometheus指标
        if PROMETHEUS_AVAILABLE and name in self.prometheus_metrics:
            metric = self.prometheus_metrics[name]
            if hasattr(metric, 'labels') and labels:
                metric.labels(**labels).observe(value)
            elif hasattr(metric, 'set'):
                metric.set(value)
            elif hasattr(metric, 'inc'):
                metric.inc()
    
    @asynccontextmanager
    async def trace(self, operation_name: str, labels: Dict[str, str] = None):
        """分布式追踪上下文管理器"""
        span_id = self.tracer.start_span(operation_name)
        
        try:
            start_time = time.time()
            self.log(LogLevel.DEBUG, f"Starting operation: {operation_name}", 
                    {'span_id': span_id, 'labels': labels})
            
            yield span_id
            
            duration = time.time() - start_time
            self.record_metric(f"{operation_name}_duration", duration, 
                            MetricType.HISTOGRAM, labels)
            self.log(LogLevel.DEBUG, f"Completed operation: {operation_name}", 
                    {'span_id': span_id, 'duration': duration})
            
        except Exception as e:
            self.log(LogLevel.ERROR, f"Failed operation: {operation_name}", 
                    {'span_id': span_id, 'error': str(e)})
            raise
        finally:
            self.tracer.finish_span(span_id)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        return {
            'performance': self.performance_tracker.get_all_metrics(),
            'system': {
                'memory': self._check_memory(),
                'disk': self._check_disk(),
                'cpu': self._check_cpu()
            },
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'structlog_available': STRUCTLOG_AVAILABLE
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return await self.health_checker.run_all_checks()
    
    def get_prometheus_metrics(self) -> Optional[str]:
        """获取Prometheus格式的指标"""
        if PROMETHEUS_AVAILABLE:
            return generate_latest()
        return None

# 全局监控系统实例
monitoring_system = MonitoringSystem()

# 装饰器
def monitor_performance(operation_name: str = None):
    """性能监控装饰器"""
    def decorator(func):
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with monitoring_system.trace(name):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    monitoring_system.record_metric(f"{name}_success", 1, MetricType.COUNTER)
                    return result
                except Exception as e:
                    monitoring_system.record_metric(f"{name}_error", 1, MetricType.COUNTER)
                    raise
                finally:
                    duration = time.time() - start_time
                    monitoring_system.record_metric(f"{name}_duration", duration, MetricType.HISTOGRAM)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                monitoring_system.record_metric(f"{name}_success", 1, MetricType.COUNTER)
                return result
            except Exception as e:
                monitoring_system.record_metric(f"{name}_error", 1, MetricType.COUNTER)
                raise
            finally:
                duration = time.time() - start_time
                monitoring_system.record_metric(f"{name}_duration", duration, MetricType.HISTOGRAM)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def log_errors(operation_name: str = None):
    """错误日志装饰器"""
    def decorator(func):
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                monitoring_system.log(
                    LogLevel.ERROR,
                    f"Error in {name}: {str(e)}",
                    {
                        'operation': name,
                        'error_type': type(e).__name__,
                        'traceback': traceback.format_exc()
                    }
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                monitoring_system.log(
                    LogLevel.ERROR,
                    f"Error in {name}: {str(e)}",
                    {
                        'operation': name,
                        'error_type': type(e).__name__,
                        'traceback': traceback.format_exc()
                    }
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# 便捷函数
def get_monitoring_system() -> MonitoringSystem:
    """获取全局监控系统实例"""
    return monitoring_system

def log_info(message: str, context: Dict[str, Any] = None) -> None:
    """记录信息日志"""
    monitoring_system.log(LogLevel.INFO, message, context)

def log_error(message: str, context: Dict[str, Any] = None) -> None:
    """记录错误日志"""
    monitoring_system.log(LogLevel.ERROR, message, context)

def log_warning(message: str, context: Dict[str, Any] = None) -> None:
    """记录警告日志"""
    monitoring_system.log(LogLevel.WARNING, message, context)

def record_metric(name: str, value: Union[int, float], 
                labels: Dict[str, str] = None) -> None:
    """记录指标"""
    monitoring_system.record_metric(name, value, MetricType.HISTOGRAM, labels)