import time
import functools
import asyncio
import logging
import threading
import statistics
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Awaitable
from datetime import datetime
from contextlib import contextmanager
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# 性能指标类型枚举
class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

# 性能指标模型
class PerformanceMetric(BaseModel):
    name: str
    type: MetricType
    value: Union[float, int]
    timestamp: float
    tags: Dict[str, str] = {}
    details: Optional[Dict[str, Any]] = None

# 性能计时结果模型
class TimingResult(BaseModel):
    name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    tags: Dict[str, str] = {}
    details: Optional[Dict[str, Any]] = None

# 性能监控管理器
class PerformanceMonitor:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PerformanceMonitor, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._metrics: Dict[str, List[PerformanceMetric]] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._timers: Dict[str, List[TimingResult]] = {}
        self._retention_limit = 1000  # 每个指标保留的最大记录数
        self._reporting_interval = 60  # 默认报告间隔(秒)
        self._reporting_task = None
        self._running = False
        self._initialized = True
    
    # 计数器操作
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """增加计数器值"""
        if name not in self._counters:
            self._counters[name] = 0
        
        self._counters[name] += value
        
        self._record_metric(
            name=name,
            type=MetricType.COUNTER,
            value=self._counters[name],
            tags=tags or {}
        )
    
    def decrement_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """减少计数器值"""
        if name not in self._counters:
            self._counters[name] = 0
        
        self._counters[name] -= value
        
        self._record_metric(
            name=name,
            type=MetricType.COUNTER,
            value=self._counters[name],
            tags=tags or {}
        )
    
    def reset_counter(self, name: str, tags: Dict[str, str] = None):
        """重置计数器"""
        if name in self._counters:
            self._counters[name] = 0
            
            self._record_metric(
                name=name,
                type=MetricType.COUNTER,
                value=0,
                tags=tags or {}
            )
    
    # 仪表盘操作
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """设置仪表盘值"""
        self._gauges[name] = value
        
        self._record_metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            tags=tags or {}
        )
    
    # 直方图操作
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """记录直方图值"""
        if name not in self._histograms:
            self._histograms[name] = []
        
        self._histograms[name].append(value)
        
        # 保持直方图数据在限制范围内
        if len(self._histograms[name]) > self._retention_limit:
            self._histograms[name] = self._histograms[name][-self._retention_limit:]
        
        self._record_metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            tags=tags or {}
        )
    
    # 计时器操作
    @contextmanager
    def timer(self, name: str, tags: Dict[str, str] = None, details: Dict[str, Any] = None):
        """计时上下文管理器"""
        start_time = time.time()
        success = True
        
        try:
            yield
        except Exception as e:
            success = False
            if details is None:
                details = {}
            details["error"] = str(e)
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            self._record_timing(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=success,
                tags=tags or {},
                details=details
            )
    
    def time_function(self, name: str = None, tags: Dict[str, str] = None):
        """函数计时装饰器"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                metric_name = name or f"{func.__module__}.{func.__name__}"
                with self.timer(metric_name, tags):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def time_async_function(self, name: str = None, tags: Dict[str, str] = None):
        """异步函数计时装饰器"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                metric_name = name or f"{func.__module__}.{func.__name__}"
                start_time = time.time()
                success = True
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    details = {"error": str(e)}
                    raise
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    self._record_timing(
                        name=metric_name,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        success=success,
                        tags=tags or {}
                    )
            return wrapper
        return decorator
    
    # 指标查询
    def get_metrics(self, name: str = None, metric_type: MetricType = None, 
                   tags: Dict[str, str] = None, limit: int = 100) -> List[PerformanceMetric]:
        """获取指标数据"""
        result = []
        
        for metric_name, metrics in self._metrics.items():
            if name and metric_name != name:
                continue
            
            for metric in metrics:
                if metric_type and metric.type != metric_type:
                    continue
                
                if tags:
                    match = True
                    for tag_key, tag_value in tags.items():
                        if tag_key not in metric.tags or metric.tags[tag_key] != tag_value:
                            match = False
                            break
                    
                    if not match:
                        continue
                
                result.append(metric)
        
        # 按时间戳排序并限制结果数量
        result.sort(key=lambda x: x.timestamp, reverse=True)
        return result[:limit]
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """获取直方图统计信息"""
        if name not in self._histograms or not self._histograms[name]:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "p95": 0,
                "p99": 0
            }
        
        values = self._histograms[name]
        sorted_values = sorted(values)
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted_values[int(len(sorted_values) * 0.95)],
            "p99": sorted_values[int(len(sorted_values) * 0.99)]
        }
    
    def get_timer_stats(self, name: str) -> Dict[str, Any]:
        """获取计时器统计信息"""
        if name not in self._timers or not self._timers[name]:
            return {
                "count": 0,
                "success_count": 0,
                "error_count": 0,
                "min_duration": 0,
                "max_duration": 0,
                "mean_duration": 0,
                "p95_duration": 0,
                "p99_duration": 0
            }
        
        timers = self._timers[name]
        durations = [t.duration for t in timers]
        sorted_durations = sorted(durations)
        success_count = sum(1 for t in timers if t.success)
        
        return {
            "count": len(timers),
            "success_count": success_count,
            "error_count": len(timers) - success_count,
            "success_rate": success_count / len(timers) if timers else 0,
            "min_duration": min(durations),
            "max_duration": max(durations),
            "mean_duration": statistics.mean(durations),
            "median_duration": statistics.median(durations),
            "p95_duration": sorted_durations[int(len(sorted_durations) * 0.95)],
            "p99_duration": sorted_durations[int(len(sorted_durations) * 0.99)]
        }
    
    # 报告和管理
    def start_reporting(self, interval: int = None):
        """启动定期报告任务"""
        if interval is not None:
            self._reporting_interval = interval
        
        if self._running:
            return
        
        self._running = True
        self._reporting_task = asyncio.create_task(self._reporting_loop())
        logger.info(f"性能监控报告已启动，间隔: {self._reporting_interval}秒")
    
    def stop_reporting(self):
        """停止定期报告任务"""
        if not self._running:
            return
        
        self._running = False
        if self._reporting_task:
            self._reporting_task.cancel()
            self._reporting_task = None
        logger.info("性能监控报告已停止")
    
    def clear_metrics(self, older_than: float = None):
        """清除指标数据"""
        if older_than is None:
            self._metrics = {}
            self._histograms = {}
            self._timers = {}
            return
        
        # 清除旧于指定时间的指标
        now = time.time()
        for name in list(self._metrics.keys()):
            self._metrics[name] = [
                m for m in self._metrics[name] 
                if now - m.timestamp < older_than
            ]
        
        # 不清除直方图和计数器的原始数据，因为它们用于计算统计信息
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            "timestamp": time.time(),
            "counters": {},
            "gauges": {},
            "histograms": {},
            "timers": {}
        }
        
        # 收集计数器
        for name in self._counters:
            summary["counters"][name] = self._counters[name]
        
        # 收集仪表盘
        for name in self._gauges:
            summary["gauges"][name] = self._gauges[name]
        
        # 收集直方图统计
        for name in self._histograms:
            summary["histograms"][name] = self.get_histogram_stats(name)
        
        # 收集计时器统计
        for name in self._timers:
            summary["timers"][name] = self.get_timer_stats(name)
        
        return summary
    
    # 内部方法
    def _record_metric(self, name: str, type: MetricType, value: Union[float, int], 
                      tags: Dict[str, str] = None, details: Dict[str, Any] = None):
        """记录指标"""
        if name not in self._metrics:
            self._metrics[name] = []
        
        metric = PerformanceMetric(
            name=name,
            type=type,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            details=details
        )
        
        self._metrics[name].append(metric)
        
        # 保持指标数据在限制范围内
        if len(self._metrics[name]) > self._retention_limit:
            self._metrics[name] = self._metrics[name][-self._retention_limit:]
    
    def _record_timing(self, name: str, start_time: float, end_time: float, 
                      duration: float, success: bool, tags: Dict[str, str] = None,
                      details: Dict[str, Any] = None):
        """记录计时结果"""
        if name not in self._timers:
            self._timers[name] = []
        
        result = TimingResult(
            name=name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success=success,
            tags=tags or {},
            details=details
        )
        
        self._timers[name].append(result)
        
        # 同时记录为直方图
        self.record_histogram(f"{name}_duration", duration, tags)
        
        # 保持计时器数据在限制范围内
        if len(self._timers[name]) > self._retention_limit:
            self._timers[name] = self._timers[name][-self._retention_limit:]
    
    async def _reporting_loop(self):
        """定期报告循环"""
        while self._running:
            try:
                summary = self.get_performance_summary()
                logger.info(f"性能监控摘要: {len(summary['counters'])} 计数器, "
                           f"{len(summary['gauges'])} 仪表盘, "
                           f"{len(summary['histograms'])} 直方图, "
                           f"{len(summary['timers'])} 计时器")
                
                # 记录一些重要的性能指标
                for timer_name, stats in summary["timers"].items():
                    if stats["count"] > 0:
                        logger.info(f"计时器 {timer_name}: "
                                   f"平均 {stats['mean_duration']:.2f}ms, "
                                   f"最大 {stats['max_duration']:.2f}ms, "
                                   f"P95 {stats['p95_duration']:.2f}ms, "
                                   f"成功率 {stats['success_rate']*100:.1f}%")
            except Exception as e:
                logger.error(f"性能报告生成失败: {str(e)}", exc_info=True)
            
            await asyncio.sleep(self._reporting_interval)

# 创建性能监控实例
performance_monitor = PerformanceMonitor()

# 便捷函数
def timer(name: str, tags: Dict[str, str] = None, details: Dict[str, Any] = None):
    """计时上下文管理器"""
    return performance_monitor.timer(name, tags, details)

def time_function(name: str = None, tags: Dict[str, str] = None):
    """函数计时装饰器"""
    return performance_monitor.time_function(name, tags)

def time_async_function(name: str = None, tags: Dict[str, str] = None):
    """异步函数计时装饰器"""
    return performance_monitor.time_async_function(name, tags)

def increment_counter(name: str, value: int = 1, tags: Dict[str, str] = None):
    """增加计数器"""
    performance_monitor.increment_counter(name, value, tags)

def set_gauge(name: str, value: float, tags: Dict[str, str] = None):
    """设置仪表盘值"""
    performance_monitor.set_gauge(name, value, tags)

def record_histogram(name: str, value: float, tags: Dict[str, str] = None):
    """记录直方图值"""
    performance_monitor.record_histogram(name, value, tags)

# 初始化性能监控
def init_performance_monitoring(app=None, reporting_interval: int = 60):
    """初始化性能监控"""
    performance_monitor.start_reporting(reporting_interval)
    
    # 如果提供了FastAPI应用实例，添加性能监控路由
    if app:
        from fastapi import APIRouter, Depends, Query
        from typing import Optional
        
        perf_router = APIRouter()
        
        @perf_router.get("/metrics/summary")
        async def get_metrics_summary():
            """获取性能指标摘要"""
            return performance_monitor.get_performance_summary()
        
        @perf_router.get("/metrics")
        async def get_metrics(
            name: Optional[str] = Query(None, description="指标名称"),
            type: Optional[str] = Query(None, description="指标类型"),
            tag_key: Optional[str] = Query(None, description="标签键"),
            tag_value: Optional[str] = Query(None, description="标签值"),
            limit: int = Query(100, description="结果限制")
        ):
            """获取性能指标"""
            tags = {tag_key: tag_value} if tag_key and tag_value else None
            metric_type = MetricType(type) if type else None
            
            metrics = performance_monitor.get_metrics(
                name=name,
                metric_type=metric_type,
                tags=tags,
                limit=limit
            )
            
            return [m.dict() for m in metrics]
        
        @perf_router.get("/metrics/histogram/{name}")
        async def get_histogram_stats(name: str):
            """获取直方图统计信息"""
            return performance_monitor.get_histogram_stats(name)
        
        @perf_router.get("/metrics/timer/{name}")
        async def get_timer_stats(name: str):
            """获取计时器统计信息"""
            return performance_monitor.get_timer_stats(name)
        
        # 注册性能监控路由
        app.include_router(perf_router, prefix="/performance", tags=["performance"])
        
        logger.info("已注册性能监控路由: /performance/metrics/summary, /performance/metrics, "
                   "/performance/metrics/histogram/{name}, /performance/metrics/timer/{name}")
    
    return performance_monitor