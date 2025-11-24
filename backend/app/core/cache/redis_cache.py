"""
Redis 分布式缓存实现 (L2)

特点:
- 分布式缓存，支持多进程/多机
- 支持 Redis 集群
- JSON 序列化
- TTL 支持
"""

import json
import logging
from typing import Any, Optional, Dict
from datetime import datetime
import redis
from redis import Redis, RedisCluster

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis 分布式缓存"""

    def __init__(self, redis_client: Redis, prefix: str = "cache:", default_ttl_hours: int = 24):
        """
        初始化 Redis 缓存

        Args:
            redis_client: Redis 客户端
            prefix: 缓存键前缀
            default_ttl_hours: 默认 TTL（小时）
        """
        self.redis = redis_client
        self.prefix = prefix
        self.default_ttl_hours = default_ttl_hours
        self.default_ttl_seconds = default_ttl_hours * 3600
        self.created_at = datetime.now()

    def _make_key(self, key: str) -> str:
        """生成 Redis 键"""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，或 None 如果不存在
        """
        try:
            redis_key = self._make_key(key)
            value = self.redis.get(redis_key)

            if value is None:
                logger.debug(f"Redis cache miss: {key}")
                return None

            result = json.loads(value)
            logger.debug(f"Redis cache hit: {key}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize Redis cache: {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Redis get error: {key}: {e}")
            return None

    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl_seconds: TTL（秒）

        Returns:
            是否成功设置
        """
        try:
            redis_key = self._make_key(key)
            ttl = ttl_seconds or self.default_ttl_seconds

            serialized = json.dumps(value)
            self.redis.setex(redis_key, ttl, serialized)
            logger.debug(f"Redis cache put: {key} (TTL: {ttl}s)")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Failed to serialize Redis cache: {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Redis put error: {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        删除缓存值

        Args:
            key: 缓存键

        Returns:
            是否成功删除
        """
        try:
            redis_key = self._make_key(key)
            result = self.redis.delete(redis_key)
            if result > 0:
                logger.debug(f"Redis cache delete: {key}")
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {key}: {e}")
            return False

    def delete_by_pattern(self, pattern: str) -> int:
        """
        删除匹配的所有键

        Args:
            pattern: 键的模式（支持通配符 * ）

        Returns:
            删除的键数
        """
        try:
            full_pattern = self._make_key(pattern)
            pipe = self.redis.pipeline()

            for key in self.redis.scan_iter(match=full_pattern, count=1000):
                pipe.delete(key)

            results = pipe.execute()
            count = sum(results)
            logger.info(f"Redis delete by pattern: {pattern} ({count} keys)")
            return count

        except Exception as e:
            logger.error(f"Redis delete by pattern error: {pattern}: {e}")
            return 0

    def clear(self) -> bool:
        """清空所有缓存（谨慎使用）"""
        try:
            pipe = self.redis.pipeline()
            for key in self.redis.scan_iter(match=f"{self.prefix}*", count=1000):
                pipe.delete(key)
            pipe.execute()
            logger.warning("Redis cache cleared")
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            info = self.redis.info()
            return {
                'used_memory_human': info.get('used_memory_human', 'N/A'),
                'used_memory': info.get('used_memory', 0),
                'max_memory': info.get('maxmemory', 0),
                'evicted_keys': info.get('evicted_keys', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'created_at': self.created_at.isoformat(),
                'uptime_seconds': info.get('uptime_in_seconds', 0),
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {}

    def health_check(self) -> bool:
        """检查 Redis 连接健康状态"""
        try:
            self.redis.ping()
            logger.debug("Redis health check OK")
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def get_ttl(self, key: str) -> int:
        """
        获取键的剩余 TTL

        Returns:
            TTL（秒），-1 表示无 TTL，-2 表示键不存在
        """
        try:
            redis_key = self._make_key(key)
            return self.redis.ttl(redis_key)
        except Exception as e:
            logger.error(f"Redis ttl error: {key}: {e}")
            return -2
