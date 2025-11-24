"""
缓存管理器 - 协调三层缓存，支持智能预热
"""

import asyncio
import logging
import time
from typing import Any, Optional, Dict, List, Callable, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """缓存层级"""
    L1_LOCAL = "l1_local"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"


class InvalidationStrategy(ABC):
    """缓存失效策略基类"""

    @abstractmethod
    async def should_invalidate(self, key: str, context: Dict) -> bool:
        """判断是否应该失效"""
        pass

    @abstractmethod
    async def get_affected_keys(self, context: Dict) -> List[str]:
        """获取受影响的缓存键"""
        pass


class RulePackVersionInvalidationStrategy(InvalidationStrategy):
    """规则包版本变更失效策略"""

    async def should_invalidate(self, key: str, context: Dict) -> bool:
        """
        规则结果缓存被规则包版本变更失效

        key 格式: RULE_RESULT:repo_id:commit_sha:rule_pack_version:file_path
        """
        if not key.startswith("RULE_RESULT:"):
            return False

        parts = key.split(":")
        if len(parts) < 5:
            return False

        current_rule_pack_version = parts[3]
        new_rule_pack_version = context.get("new_rule_pack_version")

        return current_rule_pack_version != new_rule_pack_version

    async def get_affected_keys(self, context: Dict) -> List[str]:
        """返回受影响的键的模式"""
        old_version = context.get("old_rule_pack_version")
        return [f"RULE_RESULT:*:*:{old_version}:*"]


class ConfigurationInvalidationStrategy(InvalidationStrategy):
    """配置变更失效策略"""

    async def should_invalidate(self, key: str, context: Dict) -> bool:
        """配置变更失效相关缓存"""
        config_scope = context.get("config_scope")  # 'global', 'tenant', 'rule'

        if config_scope == "global":
            return True  # 全局配置变更失效所有缓存

        if config_scope == "tenant":
            tenant_id = context.get("tenant_id")
            # 检查键中是否包含该租户 ID
            return f":{tenant_id}:" in key

        if config_scope == "rule":
            rule_id = context.get("rule_id")
            return f":{rule_id}:" in key

        return False

    async def get_affected_keys(self, context: Dict) -> List[str]:
        """返回受影响的键的模式"""
        config_scope = context.get("config_scope")

        if config_scope == "global":
            return ["*"]
        elif config_scope == "tenant":
            tenant_id = context.get("tenant_id")
            return [f"*:{tenant_id}:*"]
        elif config_scope == "rule":
            rule_id = context.get("rule_id")
            return [f"*:{rule_id}:*"]

        return []


class CacheManager:
    """
    多层缓存管理器

    协调 L1 (本地) -> L2 (Redis) -> L3 (数据库) 的缓存访问
    """

    def __init__(self,
                 l1_cache,  # LocalLRUCache
                 l2_cache,  # RedisCache
                 l3_cache):  # DatabaseCache
        """
        初始化缓存管理器

        Args:
            l1_cache: L1 本地缓存
            l2_cache: L2 Redis 缓存
            l3_cache: L3 数据库缓存
        """
        self.l1 = l1_cache
        self.l2 = l2_cache
        self.l3 = l3_cache

        # 监控指标
        self.metrics = {
            'hits_by_level': {
                'l1': 0,
                'l2': 0,
                'l3': 0,
            },
            'misses': 0,
            'invalidations': 0,
            'total_access_time_ms': 0,
        }

        # 失效策略
        self.invalidation_strategies: List[InvalidationStrategy] = [
            RulePackVersionInvalidationStrategy(),
            ConfigurationInvalidationStrategy(),
        ]

    async def get(self, key: str,
                  fallback: Optional[Callable] = None,
                  cache_levels: Tuple[CacheLevel, ...] = (CacheLevel.L1_LOCAL, CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE)) -> Optional[Any]:
        """
        获取缓存值，使用三层缓存查询策略

        查询链: L1 -> L2 -> L3 -> fallback 函数 -> 回写 L1/L2/L3

        Args:
            key: 缓存键
            fallback: 缓存未命中时的回调函数（用于计算值）
            cache_levels: 要查询的缓存层级

        Returns:
            缓存值，或 None 如果不存在
        """
        start_time = time.time()

        # L1 查询
        if CacheLevel.L1_LOCAL in cache_levels:
            value = self.l1.get(key)
            if value is not None:
                self.metrics['hits_by_level']['l1'] += 1
                self._record_access_time(start_time)
                logger.debug(f"Cache hit (L1): {key}")
                return value

        # L2 查询
        if CacheLevel.L2_REDIS in cache_levels:
            value = self.l2.get(key)
            if value is not None:
                self.metrics['hits_by_level']['l2'] += 1
                # 回写 L1
                self.l1.put(key, value)
                self._record_access_time(start_time)
                logger.debug(f"Cache hit (L2): {key}")
                return value

        # L3 查询
        if CacheLevel.L3_DATABASE in cache_levels:
            value = await self.l3.get(key)
            if value is not None:
                self.metrics['hits_by_level']['l3'] += 1
                # 回写 L1/L2
                self.l1.put(key, value)
                self.l2.put(key, value)
                self._record_access_time(start_time)
                logger.debug(f"Cache hit (L3): {key}")
                return value

        # 缓存未命中
        self.metrics['misses'] += 1

        if fallback is None:
            self._record_access_time(start_time)
            return None

        # 执行回调函数计算值
        logger.debug(f"Cache miss, computing value: {key}")

        if asyncio.iscoroutinefunction(fallback):
            value = await fallback()
        else:
            value = fallback()

        if value is not None:
            # 回写所有缓存层
            self.l1.put(key, value)
            self.l2.put(key, value)
            await self.l3.put(key, value, cache_type="computed", repo_id="", commit_sha="")
            logger.debug(f"Cache miss computed and stored: {key}")

        self._record_access_time(start_time)
        return value

    async def put(self, key: str, value: Any,
                  cache_levels: Tuple[CacheLevel, ...] = (CacheLevel.L1_LOCAL, CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE),
                  ttl_seconds: Optional[int] = None,
                  cache_type: str = "computed",
                  repo_id: str = "",
                  commit_sha: str = "") -> bool:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            cache_levels: 要写入的缓存层级
            ttl_seconds: TTL（秒）
            cache_type: 缓存类型
            repo_id: 仓库 ID
            commit_sha: 提交 SHA

        Returns:
            是否成功设置
        """
        success = True

        if CacheLevel.L1_LOCAL in cache_levels:
            self.l1.put(key, value, ttl_seconds)

        if CacheLevel.L2_REDIS in cache_levels:
            self.l2.put(key, value, ttl_seconds)

        if CacheLevel.L3_DATABASE in cache_levels:
            expires_at = None
            if ttl_seconds:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

            success &= await self.l3.put(key, value, cache_type, repo_id, commit_sha, expires_at)

        logger.debug(f"Cache put: {key} (levels: {cache_levels})")
        return success

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        self.l1.delete(key)
        self.l2.delete(key)
        await self.l3.delete(key)
        logger.debug(f"Cache delete: {key}")
        return True

    async def invalidate_by_strategy(self, strategy_name: str, context: Dict) -> int:
        """
        按失效策略失效缓存

        Args:
            strategy_name: 失效策略名称
            context: 失效上下文

        Returns:
            失效的缓存条目数
        """
        count = 0

        for strategy in self.invalidation_strategies:
            if strategy.__class__.__name__ != f"{strategy_name}Strategy":
                continue

            affected_patterns = await strategy.get_affected_keys(context)

            for pattern in affected_patterns:
                # 删除 L2 中匹配的键
                count += self.l2.delete_by_pattern(pattern)
                # 删除 L3 中匹配的记录
                count += await self.l3.delete_by_pattern(pattern)

            self.metrics['invalidations'] += count
            logger.info(f"Cache invalidated by {strategy_name}: {count} items")

        return count

    async def clear(self) -> None:
        """清空所有缓存"""
        self.l1.clear()
        self.l2.clear()
        await self.l3.delete_by_pattern("*")
        logger.warning("All caches cleared")

    def get_hit_ratio(self, window_minutes: Optional[int] = None) -> float:
        """
        获取缓存命中率

        Args:
            window_minutes: 时间窗口（分钟），暂未实现

        Returns:
            命中率 (0.0 - 1.0)
        """
        total_hits = sum(self.metrics['hits_by_level'].values())
        total_accesses = total_hits + self.metrics['misses']

        if total_accesses == 0:
            return 0.0

        return total_hits / total_accesses

    async def stats(self) -> Dict[str, Any]:
        """获取详细的缓存统计信息"""
        return {
            'hit_ratio': self.get_hit_ratio(),
            'hits_by_level': self.metrics['hits_by_level'],
            'misses': self.metrics['misses'],
            'invalidations': self.metrics['invalidations'],
            'avg_access_time_ms': self.metrics['total_access_time_ms'] / max(sum(self.metrics['hits_by_level'].values()) + self.metrics['misses'], 1),
            'l1_stats': self.l1.stats(),
            'l2_stats': self.l2.stats(),
            'l3_stats': await self.l3.stats(),
        }

    def _record_access_time(self, start_time: float) -> None:
        """记录缓存访问时间"""
        elapsed_ms = (time.time() - start_time) * 1000
        self.metrics['total_access_time_ms'] += elapsed_ms
