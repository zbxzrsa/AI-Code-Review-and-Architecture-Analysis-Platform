"""
高性能缓存管理器
支持多级缓存、智能预热和自适应策略
"""
import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    L1_MEMORY = "L1_MEMORY"
    L2_REDIS = "L2_REDIS"
    L3_DATABASE = "L3_DATABASE"

@dataclass
class CacheConfig:
    max_size: int = 1000
    ttl_seconds: int = 3600
    enable_compression: bool = True
    enable_metrics: bool = True
    prefetch_threshold: float = 0.8

class PerformanceCache:
    """高性能LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.access_counts = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # 检查TTL
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                self._remove(key)
                self.misses += 1
                return None
            
            # 移动到末尾（LRU）
            self.cache.move_to_end(key)
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.hits += 1
            return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # 移除最少使用的项
                    oldest_key = next(iter(self.cache))
                    self._remove(oldest_key)
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.access_counts[key] = 0
    
    def _remove(self, key: str) -> None:
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.access_counts.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size
        }

class MultiLevelCacheManager:
    """多级缓存管理器"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.l1_cache = PerformanceCache(
            max_size=self.config.max_size,
            ttl_seconds=self.config.ttl_seconds
        )
        self.l2_cache = None  # Redis缓存（可选）
        self.l3_cache = None  # 数据库缓存（可选）
        self.metrics = {
            "total_requests": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "cache_misses": 0
        }
        self.prefetch_queue = asyncio.Queue()
        self.background_tasks = set()
    
    async def get(self, key: str, fetch_func: Callable = None) -> Optional[Any]:
        """获取缓存值，支持多级回退"""
        self.metrics["total_requests"] += 1
        
        # L1缓存检查
        value = self.l1_cache.get(key)
        if value is not None:
            self.metrics["l1_hits"] += 1
            await self._maybe_prefetch(key)
            return value
        
        # L2缓存检查（Redis）
        if self.l2_cache:
            try:
                value = await self.l2_cache.get(key)
                if value is not None:
                    self.metrics["l2_hits"] += 1
                    # 回填L1缓存
                    self.l1_cache.set(key, value)
                    return value
            except Exception as e:
                logger.warning(f"L2 cache error: {e}")
        
        # L3缓存检查（数据库）
        if self.l3_cache:
            try:
                value = await self.l3_cache.get(key)
                if value is not None:
                    self.metrics["l3_hits"] += 1
                    # 回填上级缓存
                    self.l1_cache.set(key, value)
                    if self.l2_cache:
                        await self.l2_cache.set(key, value)
                    return value
            except Exception as e:
                logger.warning(f"L3 cache error: {e}")
        
        # 缓存未命中，执行获取函数
        self.metrics["cache_misses"] += 1
        if fetch_func:
            try:
                value = await fetch_func()
                await self.set(key, value)
                return value
            except Exception as e:
                logger.error(f"Fetch function error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        # 设置L1缓存
        self.l1_cache.set(key, value)
        
        # 设置L2缓存
        if self.l2_cache:
            try:
                await self.l2_cache.set(key, value)
            except Exception as e:
                logger.warning(f"L2 cache set error: {e}")
        
        # 设置L3缓存
        if self.l3_cache:
            try:
                await self.l3_cache.set(key, value)
            except Exception as e:
                logger.warning(f"L3 cache set error: {e}")
    
    async def _maybe_prefetch(self, key: str) -> None:
        """智能预取相关数据"""
        if not hasattr(self, '_prefetch_enabled'):
            self._prefetch_enabled = True
            # 启动后台预取任务
            task = asyncio.create_task(self._prefetch_worker())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        
        # 将预取任务加入队列
        await self.prefetch_queue.put(key)
    
    async def _prefetch_worker(self) -> None:
        """后台预取工作线程"""
        while True:
            try:
                key = await asyncio.wait_for(
                    self.prefetch_queue.get(), timeout=1.0
                )
                # 预取相关键的逻辑
                related_keys = self._get_related_keys(key)
                for related_key in related_keys:
                    if self.l1_cache.get(related_key) is None:
                        # 异步预取
                        asyncio.create_task(self._prefetch_key(related_key))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
    
    def _get_related_keys(self, key: str) -> List[str]:
        """获取相关键（示例实现）"""
        # 简单的相关键生成逻辑
        parts = key.split(':')
        if len(parts) > 1:
            base = ':'.join(parts[:-1])
            return [f"{base}:*", f"{base}:meta"]
        return []
    
    async def _prefetch_key(self, key: str) -> None:
        """预取单个键"""
        # 实际实现中会调用相应的获取函数
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取缓存性能指标"""
        total = self.metrics["total_requests"]
        l1_rate = self.metrics["l1_hits"] / total if total > 0 else 0
        l2_rate = self.metrics["l2_hits"] / total if total > 0 else 0
        l3_rate = self.metrics["l3_hits"] / total if total > 0 else 0
        miss_rate = self.metrics["cache_misses"] / total if total > 0 else 0
        
        return {
            **self.metrics,
            "l1_hit_rate": l1_rate,
            "l2_hit_rate": l2_rate,
            "l3_hit_rate": l3_rate,
            "overall_hit_rate": 1 - miss_rate,
            "l1_stats": self.l1_cache.get_stats()
        }
    
    async def clear(self, pattern: str = None) -> None:
        """清理缓存"""
        if pattern is None:
            # 清理所有缓存
            self.l1_cache.cache.clear()
            self.l1_cache.timestamps.clear()
            self.l1_cache.access_counts.clear()
            
            if self.l2_cache:
                await self.l2_cache.clear()
            if self.l3_cache:
                await self.l3_cache.clear()
        else:
            # 按模式清理
            keys_to_remove = [
                key for key in self.l1_cache.cache.keys()
                if pattern in key
            ]
            for key in keys_to_remove:
                self.l1_cache._remove(key)

# 全局缓存管理器实例
cache_manager = MultiLevelCacheManager()

def get_cache_manager() -> MultiLevelCacheManager:
    """获取全局缓存管理器实例"""
    return cache_manager