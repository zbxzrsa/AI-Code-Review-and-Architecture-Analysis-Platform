"""
本地 LRU 缓存实现 (L1)

特点:
- 内存驻留，访问延迟极低
- 自动 TTL 过期
- LRU 淘汰策略
- 线程安全
"""

import time
import logging
from typing import Any, Optional, Dict
from collections import OrderedDict
from threading import RLock
from datetime import datetime

logger = logging.getLogger(__name__)


class LocalLRUCache:
    """本地 LRU 缓存"""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        """
        初始化本地缓存

        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 默认 TTL（秒）
        """
        self.cache: OrderedDict = OrderedDict()
        self.ttl: Dict[str, float] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.lock = RLock()
        self.created_at = datetime.now()

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，或 None 如果不存在
        """
        with self.lock:
            self._cleanup_expired()

            if key in self.cache:
                # 缓存命中：移到最后（最近使用）
                self.cache.move_to_end(key)
                self.hits += 1
                logger.debug(f"Cache hit: {key}")
                return self.cache[key]

            self.misses += 1
            logger.debug(f"Cache miss: {key}")
            return None

    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl_seconds: TTL（秒），默认使用全局 TTL
        """
        with self.lock:
            ttl = ttl_seconds or self.ttl_seconds

            # 如果键已存在，移到最后
            if key in self.cache:
                self.cache.move_to_end(key)
            # 如果缓存满，删除最老的条目
            elif len(self.cache) >= self.max_size:
                removed_key, _ = self.cache.popitem(last=False)
                self.ttl.pop(removed_key, None)
                logger.debug(f"Evicted LRU item: {removed_key}")

            self.cache[key] = value
            self.ttl[key] = time.time() + ttl
            logger.debug(f"Cache put: {key} (TTL: {ttl}s)")

    def delete(self, key: str) -> bool:
        """
        删除缓存值

        Args:
            key: 缓存键

        Returns:
            是否成功删除
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.ttl.pop(key, None)
                logger.debug(f"Cache delete: {key}")
                return True
            return False

    def clear(self) -> None:
        """清空所有缓存"""
        with self.lock:
            self.cache.clear()
            self.ttl.clear()
            logger.info("Cache cleared")

    def _cleanup_expired(self) -> int:
        """
        清理已过期的缓存条目

        Returns:
            清理的条目数
        """
        now = time.time()
        expired_keys = [k for k, t in self.ttl.items() if t < now]

        for k in expired_keys:
            self.cache.pop(k, None)
            self.ttl.pop(k, None)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

        return len(expired_keys)

    def hit_ratio(self) -> float:
        """获取缓存命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            self._cleanup_expired()
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'usage_ratio': len(self.cache) / self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_ratio': self.hit_ratio(),
                'created_at': self.created_at.isoformat(),
                'uptime_seconds': (datetime.now() - self.created_at).total_seconds(),
            }

    def size(self) -> int:
        """获取缓存条目数"""
        with self.lock:
            return len(self.cache)

    def usage_ratio(self) -> float:
        """获取缓存使用率"""
        with self.lock:
            return len(self.cache) / self.max_size
