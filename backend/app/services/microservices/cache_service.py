"""
缓存微服务 - 分布式缓存服务接口定义
支持多级缓存、分片和一致性哈希
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """缓存级别"""
    L1_MEMORY = "l1_memory"      # 内存缓存
    L2_LOCAL = "l2_local"        # 本地存储
    L3_DISTRIBUTED = "l3_distributed"  # 分布式缓存
    L4_PERSISTENT = "l4_persistent"    # 持久化存储


class CacheStrategy(Enum):
    """缓存策略"""
    LRU = "lru"                  # 最近最少使用
    LFU = "lfu"                  # 最少使用频率
    TTL = "ttl"                  # 基于时间
    ADAPTIVE = "adaptive"        # 自适应策略


@dataclass
class CacheItem:
    """缓存项数据结构"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[int] = None  # 生存时间（秒）
    size: int = 0  # 数据大小（字节）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    def update_access(self):
        """更新访问信息"""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """缓存统计信息"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    memory_usage: int = 0  # 字节
    item_count: int = 0
    avg_response_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """缓存未命中率"""
        return 1.0 - self.hit_rate


class CacheServiceInterface(ABC):
    """缓存服务接口"""
    
    @abstractmethod
    async def get(self, key: str, level: Optional[CacheLevel] = None) -> Optional[Any]:
        """获取缓存项"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  level: Optional[CacheLevel] = None) -> bool:
        """设置缓存项"""
        pass
    
    @abstractmethod
    async def delete(self, key: str, level: Optional[CacheLevel] = None) -> bool:
        """删除缓存项"""
        pass
    
    @abstractmethod
    async def exists(self, key: str, level: Optional[CacheLevel] = None) -> bool:
        """检查缓存项是否存在"""
        pass
    
    @abstractmethod
    async def clear(self, level: Optional[CacheLevel] = None) -> bool:
        """清空缓存"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        pass


class MultiLevelCacheService(CacheServiceInterface):
    """多级缓存服务实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.caches: Dict[CacheLevel, Dict[str, CacheItem]] = {
            CacheLevel.L1_MEMORY: {},
            CacheLevel.L2_LOCAL: {},
            CacheLevel.L3_DISTRIBUTED: {},
            CacheLevel.L4_PERSISTENT: {}
        }
        self.stats = CacheStats()
        self.locks: Dict[str, asyncio.Lock] = {}
        
        # 配置各级缓存的容量限制
        self.capacity_limits = {
            CacheLevel.L1_MEMORY: config.get('l1_capacity', 1000),
            CacheLevel.L2_LOCAL: config.get('l2_capacity', 10000),
            CacheLevel.L3_DISTRIBUTED: config.get('l3_capacity', 100000),
            CacheLevel.L4_PERSISTENT: config.get('l4_capacity', 1000000)
        }
        
        # 启动后台清理任务
        asyncio.create_task(self._cleanup_expired_items())
    
    async def get(self, key: str, level: Optional[CacheLevel] = None) -> Optional[Any]:
        """获取缓存项 - 支持多级缓存穿透"""
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # 如果指定了级别，只在该级别查找
            if level:
                return await self._get_from_level(key, level)
            
            # 从高级别到低级别依次查找
            levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_LOCAL, 
                     CacheLevel.L3_DISTRIBUTED, CacheLevel.L4_PERSISTENT]
            
            for cache_level in levels:
                value = await self._get_from_level(key, cache_level)
                if value is not None:
                    # 缓存命中，回填到更高级别的缓存
                    await self._backfill_cache(key, value, cache_level)
                    self.stats.cache_hits += 1
                    return value
            
            # 所有级别都未命中
            self.stats.cache_misses += 1
            return None
            
        finally:
            # 更新平均响应时间
            response_time = time.time() - start_time
            self.stats.avg_response_time = (
                (self.stats.avg_response_time * (self.stats.total_requests - 1) + response_time) 
                / self.stats.total_requests
            )
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  level: Optional[CacheLevel] = None) -> bool:
        """设置缓存项"""
        try:
            # 计算数据大小
            size = len(json.dumps(value, default=str).encode('utf-8'))
            
            # 创建缓存项
            cache_item = CacheItem(
                key=key,
                value=value,
                ttl=ttl,
                size=size
            )
            
            # 如果指定了级别，只在该级别设置
            if level:
                return await self._set_to_level(key, cache_item, level)
            
            # 根据数据大小和TTL决定存储级别
            target_levels = self._determine_storage_levels(cache_item)
            
            success = True
            for cache_level in target_levels:
                result = await self._set_to_level(key, cache_item, cache_level)
                success = success and result
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to set cache item {key}: {e}")
            return False
    
    async def delete(self, key: str, level: Optional[CacheLevel] = None) -> bool:
        """删除缓存项"""
        try:
            if level:
                return await self._delete_from_level(key, level)
            
            # 从所有级别删除
            success = True
            for cache_level in CacheLevel:
                result = await self._delete_from_level(key, cache_level)
                success = success and result
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete cache item {key}: {e}")
            return False
    
    async def exists(self, key: str, level: Optional[CacheLevel] = None) -> bool:
        """检查缓存项是否存在"""
        if level:
            return key in self.caches[level] and not self.caches[level][key].is_expired()
        
        # 检查所有级别
        for cache_level in CacheLevel:
            if key in self.caches[cache_level] and not self.caches[cache_level][key].is_expired():
                return True
        
        return False
    
    async def clear(self, level: Optional[CacheLevel] = None) -> bool:
        """清空缓存"""
        try:
            if level:
                self.caches[level].clear()
            else:
                for cache_level in CacheLevel:
                    self.caches[cache_level].clear()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    async def get_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        # 更新当前统计信息
        total_memory = 0
        total_items = 0
        
        for cache_level in CacheLevel:
            cache = self.caches[cache_level]
            for item in cache.values():
                if not item.is_expired():
                    total_memory += item.size
                    total_items += 1
        
        self.stats.memory_usage = total_memory
        self.stats.item_count = total_items
        
        return self.stats
    
    async def _get_from_level(self, key: str, level: CacheLevel) -> Optional[Any]:
        """从指定级别获取缓存项"""
        cache = self.caches[level]
        
        if key not in cache:
            return None
        
        item = cache[key]
        
        # 检查是否过期
        if item.is_expired():
            del cache[key]
            return None
        
        # 更新访问信息
        item.update_access()
        return item.value
    
    async def _set_to_level(self, key: str, item: CacheItem, level: CacheLevel) -> bool:
        """设置缓存项到指定级别"""
        cache = self.caches[level]
        
        # 检查容量限制
        if len(cache) >= self.capacity_limits[level]:
            await self._evict_items(level)
        
        cache[key] = item
        return True
    
    async def _delete_from_level(self, key: str, level: CacheLevel) -> bool:
        """从指定级别删除缓存项"""
        cache = self.caches[level]
        if key in cache:
            del cache[key]
            return True
        return False
    
    async def _backfill_cache(self, key: str, value: Any, source_level: CacheLevel):
        """回填缓存到更高级别"""
        levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_LOCAL, 
                 CacheLevel.L3_DISTRIBUTED, CacheLevel.L4_PERSISTENT]
        
        source_index = levels.index(source_level)
        
        # 回填到更高级别
        for i in range(source_index):
            target_level = levels[i]
            cache_item = CacheItem(key=key, value=value)
            await self._set_to_level(key, cache_item, target_level)
    
    def _determine_storage_levels(self, item: CacheItem) -> List[CacheLevel]:
        """根据数据特征确定存储级别"""
        levels = []
        
        # 小数据存储到内存缓存
        if item.size < 1024:  # 1KB
            levels.append(CacheLevel.L1_MEMORY)
        
        # 中等数据存储到本地缓存
        if item.size < 10240:  # 10KB
            levels.append(CacheLevel.L2_LOCAL)
        
        # 大数据或长期数据存储到分布式缓存
        if item.ttl is None or item.ttl > 3600:  # 1小时
            levels.append(CacheLevel.L3_DISTRIBUTED)
        
        # 持久化数据
        if item.ttl is None or item.ttl > 86400:  # 1天
            levels.append(CacheLevel.L4_PERSISTENT)
        
        return levels if levels else [CacheLevel.L2_LOCAL]
    
    async def _evict_items(self, level: CacheLevel):
        """淘汰缓存项"""
        cache = self.caches[level]
        capacity = self.capacity_limits[level]
        
        if len(cache) < capacity:
            return
        
        # 使用LRU策略淘汰
        items = list(cache.items())
        items.sort(key=lambda x: x[1].accessed_at)
        
        # 淘汰最旧的20%
        evict_count = max(1, len(items) // 5)
        
        for i in range(evict_count):
            key, _ = items[i]
            del cache[key]
            self.stats.evictions += 1
    
    async def _cleanup_expired_items(self):
        """定期清理过期项"""
        while True:
            try:
                for level in CacheLevel:
                    cache = self.caches[level]
                    expired_keys = [
                        key for key, item in cache.items() 
                        if item.is_expired()
                    ]
                    
                    for key in expired_keys:
                        del cache[key]
                
                # 每分钟清理一次
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")
                await asyncio.sleep(60)


class DistributedCacheService(CacheServiceInterface):
    """分布式缓存服务 - 支持一致性哈希和分片"""
    
    def __init__(self, nodes: List[str], config: Dict[str, Any]):
        self.nodes = nodes
        self.config = config
        self.hash_ring = self._build_hash_ring()
        self.local_cache = MultiLevelCacheService(config)
    
    def _build_hash_ring(self) -> Dict[int, str]:
        """构建一致性哈希环"""
        hash_ring = {}
        virtual_nodes = self.config.get('virtual_nodes', 150)
        
        for node in self.nodes:
            for i in range(virtual_nodes):
                virtual_key = f"{node}:{i}"
                hash_value = int(hashlib.md5(virtual_key.encode()).hexdigest(), 16)
                hash_ring[hash_value] = node
        
        return dict(sorted(hash_ring.items()))
    
    def _get_node(self, key: str) -> str:
        """根据键获取对应的节点"""
        key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        for hash_value, node in self.hash_ring.items():
            if key_hash <= hash_value:
                return node
        
        # 如果没有找到，返回第一个节点
        return list(self.hash_ring.values())[0]
    
    async def get(self, key: str, level: Optional[CacheLevel] = None) -> Optional[Any]:
        """分布式获取"""
        node = self._get_node(key)
        
        if node == "local":
            return await self.local_cache.get(key, level)
        else:
            # 这里应该实现远程节点访问
            # 为了演示，我们使用本地缓存
            return await self.local_cache.get(key, level)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  level: Optional[CacheLevel] = None) -> bool:
        """分布式设置"""
        node = self._get_node(key)
        
        if node == "local":
            return await self.local_cache.set(key, value, ttl, level)
        else:
            # 这里应该实现远程节点访问
            # 为了演示，我们使用本地缓存
            return await self.local_cache.set(key, value, ttl, level)
    
    async def delete(self, key: str, level: Optional[CacheLevel] = None) -> bool:
        """分布式删除"""
        node = self._get_node(key)
        
        if node == "local":
            return await self.local_cache.delete(key, level)
        else:
            # 这里应该实现远程节点访问
            return await self.local_cache.delete(key, level)
    
    async def exists(self, key: str, level: Optional[CacheLevel] = None) -> bool:
        """分布式存在检查"""
        node = self._get_node(key)
        
        if node == "local":
            return await self.local_cache.exists(key, level)
        else:
            # 这里应该实现远程节点访问
            return await self.local_cache.exists(key, level)
    
    async def clear(self, level: Optional[CacheLevel] = None) -> bool:
        """分布式清空"""
        # 清空所有节点
        return await self.local_cache.clear(level)
    
    async def get_stats(self) -> CacheStats:
        """获取分布式缓存统计"""
        return await self.local_cache.get_stats()


# 全局缓存服务实例
cache_service: Optional[CacheServiceInterface] = None


def initialize_cache_service(config: Dict[str, Any]) -> CacheServiceInterface:
    """初始化缓存服务"""
    global cache_service
    
    if config.get('distributed', False):
        nodes = config.get('nodes', ['local'])
        cache_service = DistributedCacheService(nodes, config)
    else:
        cache_service = MultiLevelCacheService(config)
    
    return cache_service


def get_cache_service() -> CacheServiceInterface:
    """获取缓存服务实例"""
    if cache_service is None:
        raise RuntimeError("Cache service not initialized")
    return cache_service