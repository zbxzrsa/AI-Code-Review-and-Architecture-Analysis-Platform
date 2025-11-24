"""
缓存监控指标 - Prometheus 集成

收集缓存性能、使用情况和失效事件
"""

from prometheus_client import Counter, Gauge, Histogram
from typing import Dict, Any


class CacheMetrics:
    """缓存性能指标"""

    # ============ Counter (累计计数) ============

    # 缓存命中次数
    cache_hits_total = Counter(
        'cache_hits_total',
        'Total cache hits',
        ['cache_level', 'cache_type']
    )

    # 缓存未命中次数
    cache_misses_total = Counter(
        'cache_misses_total',
        'Total cache misses',
        ['cache_type']
    )

    # 缓存失效事件
    cache_invalidation_events_total = Counter(
        'cache_invalidation_events_total',
        'Cache invalidation events',
        ['reason', 'cache_type']
    )

    # ============ Gauge (仪表盘) ============

    # 缓存使用率
    cache_usage_ratio = Gauge(
        'cache_usage_ratio',
        'Cache usage ratio',
        ['cache_level']
    )

    # 缓存条目数
    cache_entries_count = Gauge(
        'cache_entries_count',
        'Total cache entries',
        ['cache_level', 'cache_type']
    )

    # 增量分析缓存命中率
    incremental_hit_ratio = Gauge(
        'incremental_hit_ratio',
        'Incremental analysis cache hit ratio',
        ['repo_id']
    )

    # Redis 内存使用
    cache_redis_memory_bytes = Gauge(
        'cache_redis_memory_bytes',
        'Redis memory usage in bytes'
    )

    # 数据库缓存大小
    cache_database_size_bytes = Gauge(
        'cache_database_size_bytes',
        'Database cache size in bytes',
        ['cache_type']
    )

    # ============ Histogram (分布) ============

    # 缓存访问延迟
    cache_access_latency = Histogram(
        'cache_access_latency_ms',
        'Cache access latency in milliseconds',
        ['cache_level'],
        buckets=[1, 5, 10, 50, 100, 500, 1000, 5000]
    )

    # 缓存值大小
    cache_value_size = Histogram(
        'cache_value_size_bytes',
        'Cache value size in bytes',
        ['cache_type'],
        buckets=[100, 1000, 10000, 100000, 1000000, 10000000]
    )

    @staticmethod
    def record_cache_hit(cache_level: str, cache_type: str = "generic") -> None:
        """记录缓存命中"""
        CacheMetrics.cache_hits_total.labels(
            cache_level=cache_level,
            cache_type=cache_type
        ).inc()

    @staticmethod
    def record_cache_miss(cache_type: str = "generic") -> None:
        """记录缓存未命中"""
        CacheMetrics.cache_misses_total.labels(
            cache_type=cache_type
        ).inc()

    @staticmethod
    def record_invalidation(reason: str, cache_type: str = "generic") -> None:
        """记录缓存失效事件"""
        CacheMetrics.cache_invalidation_events_total.labels(
            reason=reason,
            cache_type=cache_type
        ).inc()

    @staticmethod
    def set_usage_ratio(cache_level: str, ratio: float) -> None:
        """设置缓存使用率"""
        CacheMetrics.cache_usage_ratio.labels(
            cache_level=cache_level
        ).set(ratio)

    @staticmethod
    def set_entries_count(cache_level: str, cache_type: str, count: int) -> None:
        """设置缓存条目数"""
        CacheMetrics.cache_entries_count.labels(
            cache_level=cache_level,
            cache_type=cache_type
        ).set(count)

    @staticmethod
    def set_incremental_hit_ratio(repo_id: str, ratio: float) -> None:
        """设置增量分析缓存命中率"""
        CacheMetrics.incremental_hit_ratio.labels(
            repo_id=repo_id
        ).set(ratio)

    @staticmethod
    def observe_access_latency(cache_level: str, latency_ms: float) -> None:
        """观测缓存访问延迟"""
        CacheMetrics.cache_access_latency.labels(
            cache_level=cache_level
        ).observe(latency_ms)

    @staticmethod
    def observe_value_size(cache_type: str, size_bytes: int) -> None:
        """观测缓存值大小"""
        CacheMetrics.cache_value_size.labels(
            cache_type=cache_type
        ).observe(size_bytes)

    @staticmethod
    def set_redis_memory(memory_bytes: int) -> None:
        """设置 Redis 内存使用"""
        CacheMetrics.cache_redis_memory_bytes.set(memory_bytes)

    @staticmethod
    def set_database_size(cache_type: str, size_bytes: int) -> None:
        """设置数据库缓存大小"""
        CacheMetrics.cache_database_size_bytes.labels(
            cache_type=cache_type
        ).set(size_bytes)


# 监控仪表盘定义

MONITORING_CACHE_DASHBOARD = {
    "dashboard": {
        "title": "缓存监控仪表盘",
        "description": "多层缓存性能、使用情况和失效事件监控",
        "tags": ["cache", "performance"],
        "timezone": "UTC",
        "panels": [
            {
                "id": 1,
                "title": "缓存全局命中率",
                "type": "gauge",
                "targets": [
                    {
                        "expr": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))",
                        "legendFormat": "Hit Ratio"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "min": 0,
                        "max": 1,
                        "unit": "percentUnit",
                    }
                }
            },
            {
                "id": 2,
                "title": "增量分析缓存命中率",
                "type": "timeseries",
                "targets": [
                    {
                        "expr": "incremental_hit_ratio",
                        "legendFormat": "{{repo_id}}"
                    }
                ]
            },
            {
                "id": 3,
                "title": "分层缓存命中分布",
                "type": "barchart",
                "targets": [
                    {
                        "expr": "rate(cache_hits_total[5m])",
                        "legendFormat": "{{cache_level}}"
                    }
                ]
            },
            {
                "id": 4,
                "title": "缓存使用率",
                "type": "gauge",
                "targets": [
                    {
                        "expr": "cache_usage_ratio",
                        "legendFormat": "{{cache_level}}"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "min": 0,
                        "max": 1,
                        "unit": "percentUnit",
                    }
                }
            },
            {
                "id": 5,
                "title": "缓存访问延迟 P95",
                "type": "timeseries",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, cache_access_latency)",
                        "legendFormat": "{{cache_level}}"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "ms"
                    }
                }
            },
            {
                "id": 6,
                "title": "缓存失效原因分布",
                "type": "piechart",
                "targets": [
                    {
                        "expr": "increase(cache_invalidation_events_total[1h])",
                        "legendFormat": "{{reason}}"
                    }
                ]
            },
            {
                "id": 7,
                "title": "Redis 内存使用",
                "type": "timeseries",
                "targets": [
                    {
                        "expr": "cache_redis_memory_bytes",
                        "legendFormat": "Memory"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "bytes"
                    }
                }
            },
            {
                "id": 8,
                "title": "缓存条目数",
                "type": "stacked_area",
                "targets": [
                    {
                        "expr": "cache_entries_count",
                        "legendFormat": "{{cache_level}}-{{cache_type}}"
                    }
                ]
            }
        ]
    }
}


# Prometheus 告警规则

PROMETHEUS_ALERT_RULES = """
groups:
  - name: cache_alerts
    interval: 30s
    rules:
      # 缓存命中率过低
      - alert: LowCacheHitRatio
        expr: |
          (rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))) < 0.3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "缓存命中率过低 (< 30%)"
          description: "全局缓存命中率低于 30%，可能影响性能"

      # 增量分析缓存命中率低
      - alert: LowIncrementalCacheHitRatio
        expr: incremental_hit_ratio < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "增量分析缓存命中率过低 (< 50%)"
          description: "仓库 {{$labels.repo_id}} 增量分析缓存命中率低于 50%"

      # Redis 内存占用过高
      - alert: HighRedisMemoryUsage
        expr: cache_redis_memory_bytes / (8 * 1024 * 1024 * 1024) > 0.8
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Redis 内存占用过高 (> 80%)"
          description: "Redis 内存使用率超过 80%，可能导致缓存驱逐"

      # 缓存访问延迟过高
      - alert: HighCacheLatency
        expr: |
          histogram_quantile(0.95, cache_access_latency) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "缓存访问延迟过高"
          description: "{{$labels.cache_level}} 缓存访问 P95 延迟 > 100ms"

      # 频繁缓存失效
      - alert: FrequentCacheInvalidation
        expr: |
          rate(cache_invalidation_events_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "频繁缓存失效"
          description: "缓存失效速率过高 (> 10/s)，原因: {{$labels.reason}}"
"""
