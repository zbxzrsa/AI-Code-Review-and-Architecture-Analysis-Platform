"""
监控系统 - 指标收集器
支持性能指标、质量指标、业务指标的收集和聚合
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import statistics
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    # 性能指标
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    CACHE_HIT_RATE = "cache_hit_rate"
    
    # 质量指标
    TRANSLATION_ACCURACY = "translation_accuracy"
    USER_SATISFACTION = "user_satisfaction"
    ERROR_RATE = "error_rate"
    QUALITY_SCORE = "quality_score"
    
    # 业务指标
    TRANSLATION_COUNT = "translation_count"
    LANGUAGE_PAIR_USAGE = "language_pair_usage"
    USER_ACTIVITY = "user_activity"
    USER_RETENTION = "user_retention"
    FEATURE_USAGE = "feature_usage"
    
    # 自定义指标
    CUSTOM = "custom"


class AggregationType(Enum):
    """聚合类型"""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_50 = "p50"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    RATE = "rate"


@dataclass
class MetricPoint:
    """指标数据点"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.CUSTOM
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags,
            'metric_type': self.metric_type.value,
            'unit': self.unit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricPoint':
        return cls(
            name=data['name'],
            value=data['value'],
            timestamp=data.get('timestamp', time.time()),
            tags=data.get('tags', {}),
            metric_type=MetricType(data.get('metric_type', 'custom')),
            unit=data.get('unit', '')
        )


@dataclass
class AggregatedMetric:
    """聚合指标"""
    name: str
    aggregation_type: AggregationType
    value: float
    count: int
    start_time: float
    end_time: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricBuffer:
    """指标缓冲区"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, metric: MetricPoint):
        """添加指标"""
        with self.lock:
            self.buffer.append(metric)
    
    def get_metrics(self, start_time: Optional[float] = None, 
                   end_time: Optional[float] = None,
                   metric_names: Optional[Set[str]] = None) -> List[MetricPoint]:
        """获取指标"""
        with self.lock:
            metrics = list(self.buffer)
        
        # 时间过滤
        if start_time is not None:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        
        if end_time is not None:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        
        # 名称过滤
        if metric_names is not None:
            metrics = [m for m in metrics if m.name in metric_names]
        
        return metrics
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """获取缓冲区大小"""
        return len(self.buffer)


class MetricAggregator:
    """指标聚合器"""
    
    def __init__(self):
        self.aggregation_functions = {
            AggregationType.SUM: self._sum,
            AggregationType.AVERAGE: self._average,
            AggregationType.MIN: self._min,
            AggregationType.MAX: self._max,
            AggregationType.COUNT: self._count,
            AggregationType.PERCENTILE_50: lambda values: self._percentile(values, 50),
            AggregationType.PERCENTILE_95: lambda values: self._percentile(values, 95),
            AggregationType.PERCENTILE_99: lambda values: self._percentile(values, 99),
            AggregationType.RATE: self._rate
        }
    
    def aggregate(self, metrics: List[MetricPoint], 
                 aggregation_type: AggregationType,
                 time_window: Optional[float] = None) -> Optional[AggregatedMetric]:
        """聚合指标"""
        if not metrics:
            return None
        
        values = [m.value for m in metrics]
        start_time = min(m.timestamp for m in metrics)
        end_time = max(m.timestamp for m in metrics)
        
        # 合并标签
        common_tags = {}
        if metrics:
            first_tags = metrics[0].tags
            for key, value in first_tags.items():
                if all(m.tags.get(key) == value for m in metrics):
                    common_tags[key] = value
        
        # 执行聚合
        agg_func = self.aggregation_functions.get(aggregation_type)
        if not agg_func:
            return None
        
        if aggregation_type == AggregationType.RATE and time_window:
            agg_value = agg_func(values, time_window)
        else:
            agg_value = agg_func(values)
        
        return AggregatedMetric(
            name=metrics[0].name,
            aggregation_type=aggregation_type,
            value=agg_value,
            count=len(metrics),
            start_time=start_time,
            end_time=end_time,
            tags=common_tags
        )
    
    def _sum(self, values: List[float]) -> float:
        return sum(values)
    
    def _average(self, values: List[float]) -> float:
        return statistics.mean(values) if values else 0
    
    def _min(self, values: List[float]) -> float:
        return min(values) if values else 0
    
    def _max(self, values: List[float]) -> float:
        return max(values) if values else 0
    
    def _count(self, values: List[float]) -> float:
        return len(values)
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        if not values:
            return 0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = k - f
        
        if f + 1 < len(sorted_values):
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
        else:
            return sorted_values[f]
    
    def _rate(self, values: List[float], time_window: float) -> float:
        """计算速率（每秒）"""
        if not values or time_window <= 0:
            return 0
        
        return sum(values) / time_window


class MetricCollector:
    """指标收集器"""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer = MetricBuffer(buffer_size)
        self.aggregator = MetricAggregator()
        self.collectors: Dict[str, Callable] = {}
        self.is_running = False
        self.collection_task = None
        self.collection_interval = 10  # 收集间隔（秒）
        self.subscribers: List[Callable] = []
    
    def register_collector(self, name: str, collector_func: Callable):
        """注册指标收集器"""
        self.collectors[name] = collector_func
        logger.info(f"Registered metric collector: {name}")
    
    def unregister_collector(self, name: str):
        """注销指标收集器"""
        if name in self.collectors:
            del self.collectors[name]
            logger.info(f"Unregistered metric collector: {name}")
    
    def add_subscriber(self, subscriber: Callable):
        """添加订阅者"""
        self.subscribers.append(subscriber)
    
    def remove_subscriber(self, subscriber: Callable):
        """移除订阅者"""
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)
    
    async def start(self):
        """启动指标收集"""
        if self.is_running:
            return
        
        self.is_running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metric collector started")
    
    async def stop(self):
        """停止指标收集"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metric collector stopped")
    
    async def _collection_loop(self):
        """收集循环"""
        while self.is_running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metric collection loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_all_metrics(self):
        """收集所有指标"""
        for name, collector_func in self.collectors.items():
            try:
                if asyncio.iscoroutinefunction(collector_func):
                    metrics = await collector_func()
                else:
                    metrics = collector_func()
                
                if isinstance(metrics, MetricPoint):
                    metrics = [metrics]
                
                for metric in metrics:
                    self.add_metric(metric)
            
            except Exception as e:
                logger.error(f"Error collecting metrics from {name}: {e}")
    
    def add_metric(self, metric: MetricPoint):
        """添加指标"""
        self.buffer.add(metric)
        
        # 通知订阅者
        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    asyncio.create_task(subscriber(metric))
                else:
                    subscriber(metric)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def get_metrics(self, metric_names: Optional[List[str]] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   tags: Optional[Dict[str, str]] = None) -> List[MetricPoint]:
        """获取指标"""
        metric_name_set = set(metric_names) if metric_names else None
        metrics = self.buffer.get_metrics(start_time, end_time, metric_name_set)
        
        # 标签过滤
        if tags:
            filtered_metrics = []
            for metric in metrics:
                if all(metric.tags.get(k) == v for k, v in tags.items()):
                    filtered_metrics.append(metric)
            metrics = filtered_metrics
        
        return metrics
    
    def get_aggregated_metrics(self, metric_name: str,
                             aggregation_type: AggregationType,
                             start_time: Optional[float] = None,
                             end_time: Optional[float] = None,
                             tags: Optional[Dict[str, str]] = None,
                             time_window: Optional[float] = None) -> Optional[AggregatedMetric]:
        """获取聚合指标"""
        metrics = self.get_metrics([metric_name], start_time, end_time, tags)
        
        if not metrics:
            return None
        
        return self.aggregator.aggregate(metrics, aggregation_type, time_window)
    
    def get_time_series(self, metric_name: str,
                       start_time: float,
                       end_time: float,
                       interval: float,
                       aggregation_type: AggregationType = AggregationType.AVERAGE,
                       tags: Optional[Dict[str, str]] = None) -> List[AggregatedMetric]:
        """获取时间序列数据"""
        time_series = []
        current_time = start_time
        
        while current_time < end_time:
            window_end = min(current_time + interval, end_time)
            
            window_metrics = self.get_metrics(
                [metric_name], current_time, window_end, tags
            )
            
            if window_metrics:
                agg_metric = self.aggregator.aggregate(
                    window_metrics, aggregation_type, interval
                )
                if agg_metric:
                    time_series.append(agg_metric)
            
            current_time = window_end
        
        return time_series
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计"""
        return {
            'buffer_size': self.buffer.size(),
            'max_size': self.buffer.max_size,
            'collectors_count': len(self.collectors),
            'subscribers_count': len(self.subscribers),
            'is_running': self.is_running
        }


class PerformanceMetricCollector:
    """性能指标收集器"""
    
    def __init__(self):
        self.request_times: Dict[str, float] = {}
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
    
    def start_request(self, request_id: str):
        """开始请求计时"""
        self.request_times[request_id] = time.time()
    
    def end_request(self, request_id: str, service_name: str = "default", 
                   success: bool = True) -> Optional[MetricPoint]:
        """结束请求计时"""
        if request_id not in self.request_times:
            return None
        
        start_time = self.request_times.pop(request_id)
        response_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        self.request_counts[service_name] += 1
        if not success:
            self.error_counts[service_name] += 1
        
        return MetricPoint(
            name="response_time",
            value=response_time,
            metric_type=MetricType.RESPONSE_TIME,
            unit="ms",
            tags={"service": service_name, "success": str(success)}
        )
    
    async def collect_system_metrics(self) -> List[MetricPoint]:
        """收集系统指标"""
        metrics = []
        
        try:
            # 模拟系统指标收集
            import random
            
            # CPU使用率
            cpu_usage = random.uniform(10, 90)
            metrics.append(MetricPoint(
                name="cpu_usage",
                value=cpu_usage,
                metric_type=MetricType.CPU_USAGE,
                unit="percent"
            ))
            
            # 内存使用率
            memory_usage = random.uniform(20, 80)
            metrics.append(MetricPoint(
                name="memory_usage",
                value=memory_usage,
                metric_type=MetricType.MEMORY_USAGE,
                unit="percent"
            ))
            
            # 缓存命中率
            cache_hit_rate = random.uniform(70, 95)
            metrics.append(MetricPoint(
                name="cache_hit_rate",
                value=cache_hit_rate,
                metric_type=MetricType.CACHE_HIT_RATE,
                unit="percent"
            ))
        
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def get_throughput_metrics(self) -> List[MetricPoint]:
        """获取吞吐量指标"""
        metrics = []
        
        for service_name, count in self.request_counts.items():
            metrics.append(MetricPoint(
                name="throughput",
                value=count,
                metric_type=MetricType.THROUGHPUT,
                unit="requests",
                tags={"service": service_name}
            ))
        
        return metrics
    
    def get_error_rate_metrics(self) -> List[MetricPoint]:
        """获取错误率指标"""
        metrics = []
        
        for service_name in self.request_counts:
            total_requests = self.request_counts[service_name]
            error_requests = self.error_counts[service_name]
            
            error_rate = (error_requests / total_requests * 100) if total_requests > 0 else 0
            
            metrics.append(MetricPoint(
                name="error_rate",
                value=error_rate,
                metric_type=MetricType.ERROR_RATE,
                unit="percent",
                tags={"service": service_name}
            ))
        
        return metrics


class QualityMetricCollector:
    """质量指标收集器"""
    
    def __init__(self):
        self.translation_scores: List[float] = []
        self.user_ratings: List[float] = []
    
    def record_translation_quality(self, accuracy_score: float, 
                                 source_lang: str, target_lang: str):
        """记录翻译质量"""
        self.translation_scores.append(accuracy_score)
        
        return MetricPoint(
            name="translation_accuracy",
            value=accuracy_score,
            metric_type=MetricType.TRANSLATION_ACCURACY,
            unit="score",
            tags={"source_lang": source_lang, "target_lang": target_lang}
        )
    
    def record_user_satisfaction(self, rating: float, user_id: str):
        """记录用户满意度"""
        self.user_ratings.append(rating)
        
        return MetricPoint(
            name="user_satisfaction",
            value=rating,
            metric_type=MetricType.USER_SATISFACTION,
            unit="rating",
            tags={"user_id": user_id}
        )
    
    async def collect_quality_metrics(self) -> List[MetricPoint]:
        """收集质量指标"""
        metrics = []
        
        if self.translation_scores:
            avg_accuracy = statistics.mean(self.translation_scores)
            metrics.append(MetricPoint(
                name="avg_translation_accuracy",
                value=avg_accuracy,
                metric_type=MetricType.TRANSLATION_ACCURACY,
                unit="score"
            ))
        
        if self.user_ratings:
            avg_satisfaction = statistics.mean(self.user_ratings)
            metrics.append(MetricPoint(
                name="avg_user_satisfaction",
                value=avg_satisfaction,
                metric_type=MetricType.USER_SATISFACTION,
                unit="rating"
            ))
        
        return metrics


class BusinessMetricCollector:
    """业务指标收集器"""
    
    def __init__(self):
        self.translation_counts: Dict[str, int] = defaultdict(int)
        self.language_pair_usage: Dict[str, int] = defaultdict(int)
        self.user_activities: Dict[str, int] = defaultdict(int)
    
    def record_translation(self, source_lang: str, target_lang: str, user_id: str):
        """记录翻译"""
        language_pair = f"{source_lang}-{target_lang}"
        
        self.translation_counts["total"] += 1
        self.language_pair_usage[language_pair] += 1
        self.user_activities[user_id] += 1
        
        return [
            MetricPoint(
                name="translation_count",
                value=1,
                metric_type=MetricType.TRANSLATION_COUNT,
                unit="count",
                tags={"language_pair": language_pair}
            ),
            MetricPoint(
                name="user_activity",
                value=1,
                metric_type=MetricType.USER_ACTIVITY,
                unit="count",
                tags={"user_id": user_id}
            )
        ]
    
    async def collect_business_metrics(self) -> List[MetricPoint]:
        """收集业务指标"""
        metrics = []
        
        # 总翻译数量
        total_translations = self.translation_counts.get("total", 0)
        metrics.append(MetricPoint(
            name="total_translations",
            value=total_translations,
            metric_type=MetricType.TRANSLATION_COUNT,
            unit="count"
        ))
        
        # 语言对使用统计
        for language_pair, count in self.language_pair_usage.items():
            metrics.append(MetricPoint(
                name="language_pair_usage",
                value=count,
                metric_type=MetricType.LANGUAGE_PAIR_USAGE,
                unit="count",
                tags={"language_pair": language_pair}
            ))
        
        # 活跃用户数
        active_users = len(self.user_activities)
        metrics.append(MetricPoint(
            name="active_users",
            value=active_users,
            metric_type=MetricType.USER_ACTIVITY,
            unit="count"
        ))
        
        return metrics


# 全局实例
_metric_collector = None
_performance_collector = None
_quality_collector = None
_business_collector = None


def get_metric_collector() -> MetricCollector:
    """获取指标收集器实例"""
    global _metric_collector
    if _metric_collector is None:
        _metric_collector = MetricCollector()
    return _metric_collector


def get_performance_collector() -> PerformanceMetricCollector:
    """获取性能指标收集器实例"""
    global _performance_collector
    if _performance_collector is None:
        _performance_collector = PerformanceMetricCollector()
    return _performance_collector


def get_quality_collector() -> QualityMetricCollector:
    """获取质量指标收集器实例"""
    global _quality_collector
    if _quality_collector is None:
        _quality_collector = QualityMetricCollector()
    return _quality_collector


def get_business_collector() -> BusinessMetricCollector:
    """获取业务指标收集器实例"""
    global _business_collector
    if _business_collector is None:
        _business_collector = BusinessMetricCollector()
    return _business_collector


async def initialize_monitoring_system():
    """初始化监控系统"""
    collector = get_metric_collector()
    perf_collector = get_performance_collector()
    quality_collector = get_quality_collector()
    business_collector = get_business_collector()
    
    # 注册收集器
    collector.register_collector("system_metrics", perf_collector.collect_system_metrics)
    collector.register_collector("quality_metrics", quality_collector.collect_quality_metrics)
    collector.register_collector("business_metrics", business_collector.collect_business_metrics)
    
    # 启动收集器
    await collector.start()
    
    logger.info("Monitoring system initialized")
    return collector