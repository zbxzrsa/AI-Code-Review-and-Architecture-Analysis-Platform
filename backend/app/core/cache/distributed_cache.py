"""
分布式缓存Redis Cluster配置
支持Redis Cluster模式，提供高可用性和可扩展性
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import pickle
import time
from abc import ABC, abstractmethod

try:
    import redis
    from redis.cluster import RedisCluster
    from redis.exceptions import RedisError, ConnectionError, ClusterDownError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    RedisCluster = None
    RedisError = Exception
    ConnectionError = Exception
    ClusterDownError = Exception

logger = logging.getLogger(__name__)


class CacheMode(Enum):
    """缓存模式"""
    SINGLE = "single"          # 单节点Redis
    CLUSTER = "cluster"        # Redis Cluster
    SENTINEL = "sentinel"      # Redis Sentinel


class ConsistencyLevel(Enum):
    """一致性级别"""
    EVENTUAL = "eventual"      # 最终一致性
    STRONG = "strong"          # 强一致性
    QUORUM = "quorum"          # 法定人数一致性


@dataclass
class RedisClusterConfig:
    """Redis Cluster配置"""
    startup_nodes: List[Dict[str, Union[str, int]]]
    password: Optional[str] = None
    ssl: bool = False
    ssl_ca_certs: Optional[str] = None
    ssl_cert_reqs: Optional[str] = None
    max_connections: int = 32
    retry_on_timeout: bool = True
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    health_check_interval: int = 30
    skip_full_coverage_check: bool = False
    decode_responses: bool = False


@dataclass
class RedisSingleConfig:
    """单节点Redis配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    ssl_ca_certs: Optional[str] = None
    max_connections: int = 32
    retry_on_timeout: bool = True
    socket_timeout: float = 5.0
    decode_responses: bool = False


@dataclass
class CacheMetrics:
    """缓存指标"""
    hits: int = 0
    misses: int = 0
    errors: int = 0
    total_operations: int = 0
    hit_rate: float = 0.0
    avg_response_time: float = 0.0
    last_error: Optional[str] = None
    cluster_status: Dict[str, Any] = None
    
    def update_hit_rate(self):
        """更新命中率"""
        if self.total_operations > 0:
            self.hit_rate = self.hits / self.total_operations
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class CacheBackend(ABC):
    """缓存后端接口"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass
    
    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        pass
    
    @abstractmethod
    async def ttl(self, key: str) -> int:
        """获取剩余过期时间"""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str) -> List[str]:
        """获取匹配的键"""
        pass
    
    @abstractmethod
    async def flushdb(self) -> bool:
        """清空数据库"""
        pass
    
    @abstractmethod
    async def ping(self) -> bool:
        """检查连接状态"""
        pass
    
    @abstractmethod
    async def get_cluster_info(self) -> Dict[str, Any]:
        """获取集群信息"""
        pass


class RedisClusterBackend(CacheBackend):
    """Redis Cluster后端"""
    
    def __init__(self, config: RedisClusterConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for Redis Cluster support")
        
        self.config = config
        self.client: Optional[RedisCluster] = None
        self.metrics = CacheMetrics()
        self._connection_pool = None
        
    async def connect(self):
        """连接到Redis Cluster"""
        try:
            self.client = RedisCluster(
                startup_nodes=self.config.startup_nodes,
                password=self.config.password,
                ssl=self.config.ssl,
                ssl_ca_certs=self.config.ssl_ca_certs,
                ssl_cert_reqs=self.config.ssl_cert_reqs,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                health_check_interval=self.config.health_check_interval,
                skip_full_coverage_check=self.config.skip_full_coverage_check,
                decode_responses=self.config.decode_responses
            )
            
            # 测试连接
            await self.ping()
            
            logger.info("Connected to Redis Cluster successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis Cluster: {e}")
            raise
    
    async def disconnect(self):
        """断开连接"""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Disconnected from Redis Cluster")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self.client:
            raise ConnectionError("Not connected to Redis Cluster")
        
        start_time = time.time()
        
        try:
            value = await self.client.get(key)
            
            if value is not None:
                # 反序列化
                try:
                    if isinstance(value, bytes):
                        value = pickle.loads(value)
                    self.metrics.hits += 1
                except (pickle.PickleError, TypeError):
                    # 如果反序列化失败，返回原始值
                    self.metrics.hits += 1
            else:
                self.metrics.misses += 1
            
            self.metrics.total_operations += 1
            self.metrics.update_hit_rate()
            
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)
            
            return value
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """设置缓存值"""
        if not self.client:
            raise ConnectionError("Not connected to Redis Cluster")
        
        start_time = time.time()
        
        try:
            # 序列化值
            if not isinstance(value, (str, bytes)):
                serialized_value = pickle.dumps(value)
            else:
                serialized_value = value
            
            if ttl:
                result = await self.client.setex(key, ttl, serialized_value)
            else:
                result = await self.client.set(key, serialized_value)
            
            self.metrics.total_operations += 1
            
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)
            
            return bool(result)
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        if not self.client:
            raise ConnectionError("Not connected to Redis Cluster")
        
        try:
            result = await self.client.delete(key)
            self.metrics.total_operations += 1
            return bool(result)
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self.client:
            raise ConnectionError("Not connected to Redis Cluster")
        
        try:
            result = await self.client.exists(key)
            self.metrics.total_operations += 1
            return bool(result)
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        if not self.client:
            raise ConnectionError("Not connected to Redis Cluster")
        
        try:
            result = await self.client.expire(key, ttl)
            self.metrics.total_operations += 1
            return bool(result)
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            logger.error(f"Redis expire error for key {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """获取剩余过期时间"""
        if not self.client:
            raise ConnectionError("Not connected to Redis Cluster")
        
        try:
            result = await self.client.ttl(key)
            self.metrics.total_operations += 1
            return result
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            logger.error(f"Redis ttl error for key {key}: {e}")
            return -1
    
    async def keys(self, pattern: str) -> List[str]:
        """获取匹配的键"""
        if not self.client:
            raise ConnectionError("Not connected to Redis Cluster")
        
        try:
            # 在集群模式下，需要在每个节点上执行KEYS命令
            keys = []
            for node in self.client.get_primaries():
                node_keys = await node.keys(pattern)
                keys.extend(node_keys)
            
            self.metrics.total_operations += 1
            return keys
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            logger.error(f"Redis keys error for pattern {pattern}: {e}")
            return []
    
    async def flushdb(self) -> bool:
        """清空数据库"""
        if not self.client:
            raise ConnectionError("Not connected to Redis Cluster")
        
        try:
            # 在集群模式下，需要在每个节点上执行FLUSHDB
            for node in self.client.get_primaries():
                await node.flushdb()
            
            self.metrics.total_operations += 1
            return True
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            logger.error(f"Redis flushdb error: {e}")
            return False
    
    async def ping(self) -> bool:
        """检查连接状态"""
        if not self.client:
            return False
        
        try:
            result = await self.client.ping()
            return result
            
        except Exception as e:
            logger.error(f"Redis ping error: {e}")
            return False
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """获取集群信息"""
        if not self.client:
            return {"status": "disconnected"}
        
        try:
            cluster_info = await self.client.cluster_info()
            cluster_nodes = await self.client.cluster_nodes()
            
            # 解析节点信息
            nodes = []
            for node_info in cluster_nodes.split('\n'):
                if not node_info.strip():
                    continue
                
                parts = node_info.split()
                if len(parts) >= 8:
                    nodes.append({
                        "id": parts[0],
                        "address": parts[1],
                        "flags": parts[2],
                        "master": parts[3],
                        "ping_sent": parts[4],
                        "pong_recv": parts[5],
                        "config_epoch": parts[6],
                        "link_state": parts[7],
                        "slots": parts[8] if len(parts) > 8 else ""
                    })
            
            return {
                "status": "connected",
                "cluster_info": cluster_info,
                "nodes": nodes,
                "node_count": len(nodes)
            }
            
        except Exception as e:
            logger.error(f"Error getting cluster info: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _update_avg_response_time(self, response_time: float):
        """更新平均响应时间"""
        if self.metrics.total_operations == 1:
            self.metrics.avg_response_time = response_time
        else:
            # 使用指数移动平均
            alpha = 0.1
            self.metrics.avg_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.metrics.avg_response_time
            )
    
    def get_metrics(self) -> CacheMetrics:
        """获取指标"""
        # 更新集群状态
        try:
            self.metrics.cluster_status = asyncio.create_task(
                self.get_cluster_info()
            ).result()
        except Exception:
            self.metrics.cluster_status = {"status": "error"}
        
        return self.metrics


class RedisSingleBackend(CacheBackend):
    """单节点Redis后端"""
    
    def __init__(self, config: RedisSingleConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for Redis support")
        
        self.config = config
        self.client: Optional[redis.Redis] = None
        self.metrics = CacheMetrics()
    
    async def connect(self):
        """连接到Redis"""
        try:
            self.client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                ssl=self.config.ssl,
                ssl_ca_certs=self.config.ssl_ca_certs,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_timeout=self.config.socket_timeout,
                decode_responses=self.config.decode_responses
            )
            
            # 测试连接
            await self.ping()
            
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """断开连接"""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Disconnected from Redis")
    
    # 实现其他方法...
    async def get(self, key: str) -> Optional[Any]:
        if not self.client:
            raise ConnectionError("Not connected to Redis")
        
        try:
            value = await self.client.get(key)
            if value is not None:
                if isinstance(value, bytes):
                    value = pickle.loads(value)
                self.metrics.hits += 1
            else:
                self.metrics.misses += 1
            
            self.metrics.total_operations += 1
            self.metrics.update_hit_rate()
            return value
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self.client:
            raise ConnectionError("Not connected to Redis")
        
        try:
            if not isinstance(value, (str, bytes)):
                value = pickle.dumps(value)
            
            if ttl:
                result = await self.client.setex(key, ttl, value)
            else:
                result = await self.client.set(key, value)
            
            self.metrics.total_operations += 1
            return bool(result)
            
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            return False
    
    async def delete(self, key: str) -> bool:
        if not self.client:
            raise ConnectionError("Not connected to Redis")
        
        try:
            result = await self.client.delete(key)
            self.metrics.total_operations += 1
            return bool(result)
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            return False
    
    async def exists(self, key: str) -> bool:
        if not self.client:
            raise ConnectionError("Not connected to Redis")
        
        try:
            result = await self.client.exists(key)
            self.metrics.total_operations += 1
            return bool(result)
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        if not self.client:
            raise ConnectionError("Not connected to Redis")
        
        try:
            result = await self.client.expire(key, ttl)
            self.metrics.total_operations += 1
            return bool(result)
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            return False
    
    async def ttl(self, key: str) -> int:
        if not self.client:
            raise ConnectionError("Not connected to Redis")
        
        try:
            result = await self.client.ttl(key)
            self.metrics.total_operations += 1
            return result
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            return -1
    
    async def keys(self, pattern: str) -> List[str]:
        if not self.client:
            raise ConnectionError("Not connected to Redis")
        
        try:
            keys = await self.client.keys(pattern)
            self.metrics.total_operations += 1
            return keys
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            return []
    
    async def flushdb(self) -> bool:
        if not self.client:
            raise ConnectionError("Not connected to Redis")
        
        try:
            result = await self.client.flushdb()
            self.metrics.total_operations += 1
            return result
        except Exception as e:
            self.metrics.errors += 1
            self.metrics.total_operations += 1
            self.metrics.last_error = str(e)
            return False
    
    async def ping(self) -> bool:
        if not self.client:
            return False
        
        try:
            result = await self.client.ping()
            return result
        except Exception as e:
            return False
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        if not self.client:
            return {"status": "disconnected"}
        
        try:
            info = await self.client.info()
            return {
                "status": "connected",
                "mode": "single",
                "info": info
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_metrics(self) -> CacheMetrics:
        return self.metrics


class DistributedCacheManager:
    """分布式缓存管理器"""
    
    def __init__(self, backend: CacheBackend):
        self.backend = backend
        self.consistency_level = ConsistencyLevel.EVENTUAL
        self.key_prefix = "analysis_cache:"
        self.default_ttl = 3600  # 1小时
    
    async def connect(self):
        """连接到缓存后端"""
        await self.backend.connect()
    
    async def disconnect(self):
        """断开连接"""
        await self.backend.disconnect()
    
    def _make_key(self, *parts: str) -> str:
        """构建缓存键"""
        key = ":".join(str(part) for part in parts)
        return f"{self.key_prefix}{key}"
    
    async def get(
        self, 
        tenant_id: str, 
        repo_id: str, 
        rulepack_version: str, 
        file_path: str
    ) -> Optional[Any]:
        """获取缓存值"""
        key = self._make_key(tenant_id, repo_id, rulepack_version, file_path)
        return await self.backend.get(key)
    
    async def set(
        self, 
        tenant_id: str, 
        repo_id: str, 
        rulepack_version: str, 
        file_path: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """设置缓存值"""
        key = self._make_key(tenant_id, repo_id, rulepack_version, file_path)
        return await self.backend.set(key, value, ttl or self.default_ttl)
    
    async def delete(
        self, 
        tenant_id: str, 
        repo_id: str, 
        rulepack_version: str, 
        file_path: str
    ) -> bool:
        """删除缓存值"""
        key = self._make_key(tenant_id, repo_id, rulepack_version, file_path)
        return await self.backend.delete(key)
    
    async def exists(
        self, 
        tenant_id: str, 
        repo_id: str, 
        rulepack_version: str, 
        file_path: str
    ) -> bool:
        """检查缓存是否存在"""
        key = self._make_key(tenant_id, repo_id, rulepack_version, file_path)
        return await self.backend.exists(key)
    
    async def get_tenant_cache_keys(self, tenant_id: str) -> List[str]:
        """获取租户的所有缓存键"""
        pattern = self._make_key(tenant_id, "*", "*", "*")
        return await self.backend.keys(pattern)
    
    async def clear_tenant_cache(self, tenant_id: str) -> int:
        """清空租户缓存"""
        keys = await self.get_tenant_cache_keys(tenant_id)
        deleted_count = 0
        
        for key in keys:
            if await self.backend.delete(key):
                deleted_count += 1
        
        return deleted_count
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取缓存指标"""
        backend_metrics = self.backend.get_metrics()
        return backend_metrics.to_dict()
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        is_healthy = await self.backend.ping()
        cluster_info = await self.backend.get_cluster_info()
        
        return {
            "healthy": is_healthy,
            "cluster_info": cluster_info,
            "metrics": await self.get_metrics()
        }


# 工厂函数
def create_cache_manager(
    mode: CacheMode,
    config: Union[RedisClusterConfig, RedisSingleConfig]
) -> DistributedCacheManager:
    """创建缓存管理器"""
    
    if mode == CacheMode.CLUSTER:
        if not isinstance(config, RedisClusterConfig):
            raise ValueError("RedisClusterConfig required for cluster mode")
        backend = RedisClusterBackend(config)
    elif mode == CacheMode.SINGLE:
        if not isinstance(config, RedisSingleConfig):
            raise ValueError("RedisSingleConfig required for single mode")
        backend = RedisSingleBackend(config)
    else:
        raise ValueError(f"Unsupported cache mode: {mode}")
    
    return DistributedCacheManager(backend)


# 示例配置
EXAMPLE_CLUSTER_CONFIG = RedisClusterConfig(
    startup_nodes=[
        {"host": "redis-node-1", "port": 6379},
        {"host": "redis-node-2", "port": 6379},
        {"host": "redis-node-3", "port": 6379},
        {"host": "redis-node-4", "port": 6379},
        {"host": "redis-node-5", "port": 6379},
        {"host": "redis-node-6", "port": 6379},
    ],
    password="your-redis-password",
    ssl=True,
    max_connections=64,
    socket_timeout=5.0,
    health_check_interval=30
)

EXAMPLE_SINGLE_CONFIG = RedisSingleConfig(
    host="localhost",
    port=6379,
    db=0,
    password=None,
    max_connections=32,
    socket_timeout=5.0
)