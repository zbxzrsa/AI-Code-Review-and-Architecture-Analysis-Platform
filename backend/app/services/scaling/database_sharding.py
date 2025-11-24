"""
数据库分片系统 - 支持水平分片和数据路由
"""

import asyncio
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """分片策略"""
    HASH = "hash"
    RANGE = "range"
    DIRECTORY = "directory"
    CONSISTENT_HASH = "consistent_hash"
    COMPOSITE = "composite"


class ShardStatus(Enum):
    """分片状态"""
    ACTIVE = "active"
    READONLY = "readonly"
    MAINTENANCE = "maintenance"
    MIGRATING = "migrating"
    OFFLINE = "offline"


@dataclass
class ShardConfig:
    """分片配置"""
    shard_id: str
    database_url: str
    weight: int = 1
    status: ShardStatus = ShardStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 范围分片配置
    range_start: Optional[Any] = None
    range_end: Optional[Any] = None
    
    # 统计信息
    total_records: int = 0
    storage_size: int = 0  # bytes
    last_updated: float = field(default_factory=time.time)
    
    def is_available(self) -> bool:
        """检查分片是否可用"""
        return self.status in [ShardStatus.ACTIVE, ShardStatus.READONLY]
    
    def is_writable(self) -> bool:
        """检查分片是否可写"""
        return self.status == ShardStatus.ACTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'shard_id': self.shard_id,
            'database_url': self.database_url,
            'weight': self.weight,
            'status': self.status.value,
            'range_start': self.range_start,
            'range_end': self.range_end,
            'total_records': self.total_records,
            'storage_size': self.storage_size,
            'last_updated': self.last_updated,
            'metadata': self.metadata
        }


class ShardingRule:
    """分片规则"""
    
    def __init__(self, table_name: str, shard_key: str, 
                 strategy: ShardingStrategy, **kwargs):
        self.table_name = table_name
        self.shard_key = shard_key
        self.strategy = strategy
        self.config = kwargs
        
        # 一致性哈希配置
        self.virtual_nodes = kwargs.get('virtual_nodes', 150)
        self.hash_ring = {}
        
        # 范围分片配置
        self.ranges = kwargs.get('ranges', [])
        
        # 目录分片配置
        self.directory = kwargs.get('directory', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'table_name': self.table_name,
            'shard_key': self.shard_key,
            'strategy': self.strategy.value,
            'config': self.config
        }


class ShardRouter:
    """分片路由器"""
    
    def __init__(self):
        self.shards: Dict[str, ShardConfig] = {}
        self.rules: Dict[str, ShardingRule] = {}
        self.hash_rings: Dict[str, List[tuple]] = {}
        self.lock = asyncio.Lock()
    
    async def add_shard(self, shard: ShardConfig):
        """添加分片"""
        async with self.lock:
            self.shards[shard.shard_id] = shard
            
            # 重建哈希环
            await self._rebuild_hash_rings()
            
            logger.info(f"Added shard: {shard.shard_id}")
    
    async def remove_shard(self, shard_id: str):
        """移除分片"""
        async with self.lock:
            if shard_id in self.shards:
                del self.shards[shard_id]
                
                # 重建哈希环
                await self._rebuild_hash_rings()
                
                logger.info(f"Removed shard: {shard_id}")
    
    async def update_shard_status(self, shard_id: str, status: ShardStatus):
        """更新分片状态"""
        async with self.lock:
            if shard_id in self.shards:
                old_status = self.shards[shard_id].status
                self.shards[shard_id].status = status
                
                # 如果状态变化影响可用性，重建哈希环
                if old_status.value != status.value:
                    await self._rebuild_hash_rings()
                
                logger.info(f"Updated shard status: {shard_id} "
                          f"{old_status.value} -> {status.value}")
    
    def add_sharding_rule(self, rule: ShardingRule):
        """添加分片规则"""
        self.rules[rule.table_name] = rule
        logger.info(f"Added sharding rule for table: {rule.table_name}")
    
    async def route_query(self, table_name: str, shard_key_value: Any, 
                         operation: str = 'read') -> List[str]:
        """路由查询到分片"""
        rule = self.rules.get(table_name)
        if not rule:
            # 没有分片规则，返回所有可用分片
            return self._get_available_shards(operation)
        
        return await self._route_by_strategy(rule, shard_key_value, operation)
    
    async def _route_by_strategy(self, rule: ShardingRule, shard_key_value: Any,
                               operation: str) -> List[str]:
        """根据策略路由"""
        if rule.strategy == ShardingStrategy.HASH:
            return await self._hash_route(rule, shard_key_value, operation)
        elif rule.strategy == ShardingStrategy.RANGE:
            return await self._range_route(rule, shard_key_value, operation)
        elif rule.strategy == ShardingStrategy.DIRECTORY:
            return await self._directory_route(rule, shard_key_value, operation)
        elif rule.strategy == ShardingStrategy.CONSISTENT_HASH:
            return await self._consistent_hash_route(rule, shard_key_value, operation)
        elif rule.strategy == ShardingStrategy.COMPOSITE:
            return await self._composite_route(rule, shard_key_value, operation)
        else:
            return self._get_available_shards(operation)
    
    async def _hash_route(self, rule: ShardingRule, shard_key_value: Any,
                         operation: str) -> List[str]:
        """哈希路由"""
        available_shards = self._get_available_shards(operation)
        if not available_shards:
            return []
        
        hash_value = self._hash(str(shard_key_value))
        shard_index = hash_value % len(available_shards)
        
        return [available_shards[shard_index]]
    
    async def _range_route(self, rule: ShardingRule, shard_key_value: Any,
                          operation: str) -> List[str]:
        """范围路由"""
        target_shards = []
        
        for shard_id, shard in self.shards.items():
            if not self._is_shard_available(shard, operation):
                continue
            
            # 检查值是否在分片范围内
            if (shard.range_start is None or shard_key_value >= shard.range_start) and \
               (shard.range_end is None or shard_key_value < shard.range_end):
                target_shards.append(shard_id)
        
        return target_shards
    
    async def _directory_route(self, rule: ShardingRule, shard_key_value: Any,
                              operation: str) -> List[str]:
        """目录路由"""
        directory = rule.directory
        shard_id = directory.get(str(shard_key_value))
        
        if shard_id and shard_id in self.shards:
            shard = self.shards[shard_id]
            if self._is_shard_available(shard, operation):
                return [shard_id]
        
        # 如果没有找到映射，使用默认分片
        default_shard = directory.get('default')
        if default_shard and default_shard in self.shards:
            shard = self.shards[default_shard]
            if self._is_shard_available(shard, operation):
                return [default_shard]
        
        return []
    
    async def _consistent_hash_route(self, rule: ShardingRule, shard_key_value: Any,
                                   operation: str) -> List[str]:
        """一致性哈希路由"""
        ring_key = rule.table_name
        
        if ring_key not in self.hash_rings:
            await self._build_hash_ring(rule)
        
        ring = self.hash_rings[ring_key]
        if not ring:
            return []
        
        key_hash = self._hash(str(shard_key_value))
        
        # 在哈希环中查找
        for hash_value, shard_id in sorted(ring):
            if hash_value >= key_hash:
                shard = self.shards.get(shard_id)
                if shard and self._is_shard_available(shard, operation):
                    return [shard_id]
        
        # 如果没找到，返回第一个节点（环形）
        if ring:
            _, shard_id = sorted(ring)[0]
            shard = self.shards.get(shard_id)
            if shard and self._is_shard_available(shard, operation):
                return [shard_id]
        
        return []
    
    async def _composite_route(self, rule: ShardingRule, shard_key_value: Any,
                              operation: str) -> List[str]:
        """复合路由"""
        # 复合策略可以结合多种路由方式
        # 这里实现一个简单的示例：先按范围，再按哈希
        
        # 首先尝试范围路由
        range_shards = await self._range_route(rule, shard_key_value, operation)
        
        if range_shards:
            # 如果范围路由找到分片，在这些分片中进行哈希
            hash_value = self._hash(str(shard_key_value))
            shard_index = hash_value % len(range_shards)
            return [range_shards[shard_index]]
        else:
            # 否则使用普通哈希路由
            return await self._hash_route(rule, shard_key_value, operation)
    
    async def _build_hash_ring(self, rule: ShardingRule):
        """构建一致性哈希环"""
        ring = []
        virtual_nodes = rule.virtual_nodes
        
        for shard_id, shard in self.shards.items():
            if shard.is_available():
                for i in range(virtual_nodes):
                    virtual_key = f"{shard_id}:{i}"
                    hash_value = self._hash(virtual_key)
                    ring.append((hash_value, shard_id))
        
        self.hash_rings[rule.table_name] = ring
    
    async def _rebuild_hash_rings(self):
        """重建所有哈希环"""
        for table_name, rule in self.rules.items():
            if rule.strategy == ShardingStrategy.CONSISTENT_HASH:
                await self._build_hash_ring(rule)
    
    def _get_available_shards(self, operation: str) -> List[str]:
        """获取可用分片"""
        available = []
        
        for shard_id, shard in self.shards.items():
            if self._is_shard_available(shard, operation):
                available.append(shard_id)
        
        return available
    
    def _is_shard_available(self, shard: ShardConfig, operation: str) -> bool:
        """检查分片是否可用于指定操作"""
        if operation == 'write':
            return shard.is_writable()
        else:
            return shard.is_available()
    
    def _hash(self, key: str) -> int:
        """计算哈希值"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def get_shard_info(self, shard_id: str) -> Optional[Dict[str, Any]]:
        """获取分片信息"""
        shard = self.shards.get(shard_id)
        return shard.to_dict() if shard else None
    
    def get_all_shards(self) -> Dict[str, Dict[str, Any]]:
        """获取所有分片信息"""
        return {shard_id: shard.to_dict() 
                for shard_id, shard in self.shards.items()}
    
    def get_sharding_rules(self) -> Dict[str, Dict[str, Any]]:
        """获取分片规则"""
        return {table_name: rule.to_dict() 
                for table_name, rule in self.rules.items()}


class ShardManager:
    """分片管理器"""
    
    def __init__(self, router: ShardRouter):
        self.router = router
        self.migration_tasks: Dict[str, asyncio.Task] = {}
        self.rebalance_tasks: Dict[str, asyncio.Task] = {}
    
    async def create_shard(self, shard_config: Dict[str, Any]) -> bool:
        """创建新分片"""
        try:
            shard = ShardConfig(
                shard_id=shard_config['shard_id'],
                database_url=shard_config['database_url'],
                weight=shard_config.get('weight', 1),
                metadata=shard_config.get('metadata', {})
            )
            
            # 初始化分片数据库
            await self._initialize_shard_database(shard)
            
            # 添加到路由器
            await self.router.add_shard(shard)
            
            logger.info(f"Created shard: {shard.shard_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create shard: {e}")
            return False
    
    async def migrate_shard(self, source_shard_id: str, target_shard_id: str,
                           migration_config: Dict[str, Any] = None) -> bool:
        """迁移分片数据"""
        migration_key = f"{source_shard_id}->{target_shard_id}"
        
        if migration_key in self.migration_tasks:
            logger.warning(f"Migration already in progress: {migration_key}")
            return False
        
        try:
            # 创建迁移任务
            task = asyncio.create_task(
                self._perform_migration(source_shard_id, target_shard_id, migration_config)
            )
            self.migration_tasks[migration_key] = task
            
            # 等待迁移完成
            result = await task
            
            # 清理任务
            del self.migration_tasks[migration_key]
            
            return result
        
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            if migration_key in self.migration_tasks:
                del self.migration_tasks[migration_key]
            return False
    
    async def _perform_migration(self, source_shard_id: str, target_shard_id: str,
                               migration_config: Dict[str, Any] = None) -> bool:
        """执行数据迁移"""
        try:
            # 设置源分片为只读
            await self.router.update_shard_status(source_shard_id, ShardStatus.READONLY)
            
            # 设置目标分片为迁移状态
            await self.router.update_shard_status(target_shard_id, ShardStatus.MIGRATING)
            
            # 这里应该实现实际的数据迁移逻辑
            # 1. 复制数据
            # 2. 验证数据一致性
            # 3. 更新路由规则
            
            # 模拟迁移过程
            await asyncio.sleep(1)
            
            # 迁移完成，更新状态
            await self.router.update_shard_status(target_shard_id, ShardStatus.ACTIVE)
            await self.router.update_shard_status(source_shard_id, ShardStatus.OFFLINE)
            
            logger.info(f"Migration completed: {source_shard_id} -> {target_shard_id}")
            return True
        
        except Exception as e:
            logger.error(f"Migration error: {e}")
            
            # 恢复状态
            await self.router.update_shard_status(source_shard_id, ShardStatus.ACTIVE)
            await self.router.update_shard_status(target_shard_id, ShardStatus.OFFLINE)
            
            return False
    
    async def rebalance_shards(self, table_name: str) -> bool:
        """重新平衡分片"""
        if table_name in self.rebalance_tasks:
            logger.warning(f"Rebalance already in progress for table: {table_name}")
            return False
        
        try:
            task = asyncio.create_task(self._perform_rebalance(table_name))
            self.rebalance_tasks[table_name] = task
            
            result = await task
            
            del self.rebalance_tasks[table_name]
            return result
        
        except Exception as e:
            logger.error(f"Rebalance failed for table {table_name}: {e}")
            if table_name in self.rebalance_tasks:
                del self.rebalance_tasks[table_name]
            return False
    
    async def _perform_rebalance(self, table_name: str) -> bool:
        """执行分片重新平衡"""
        try:
            # 分析当前分片负载
            shard_stats = await self._analyze_shard_load(table_name)
            
            # 计算重新平衡策略
            rebalance_plan = await self._calculate_rebalance_plan(table_name, shard_stats)
            
            # 执行重新平衡
            for migration in rebalance_plan:
                await self.migrate_shard(
                    migration['source'], 
                    migration['target'], 
                    migration.get('config')
                )
            
            logger.info(f"Rebalance completed for table: {table_name}")
            return True
        
        except Exception as e:
            logger.error(f"Rebalance error for table {table_name}: {e}")
            return False
    
    async def _analyze_shard_load(self, table_name: str) -> Dict[str, Any]:
        """分析分片负载"""
        # 这里应该实现实际的负载分析逻辑
        # 收集各分片的统计信息：记录数、存储大小、查询频率等
        
        stats = {}
        for shard_id, shard in self.router.shards.items():
            stats[shard_id] = {
                'records': shard.total_records,
                'storage_size': shard.storage_size,
                'weight': shard.weight,
                'status': shard.status.value
            }
        
        return stats
    
    async def _calculate_rebalance_plan(self, table_name: str, 
                                      shard_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """计算重新平衡计划"""
        # 这里应该实现智能的重新平衡算法
        # 考虑因素：负载均衡、数据分布、迁移成本等
        
        # 简单示例：如果某个分片负载过高，迁移部分数据到负载较低的分片
        plan = []
        
        # 计算平均负载
        total_records = sum(stats['records'] for stats in shard_stats.values())
        active_shards = len([s for s in shard_stats.values() 
                           if s['status'] == 'active'])
        
        if active_shards == 0:
            return plan
        
        avg_records = total_records / active_shards
        threshold = avg_records * 1.5  # 超过平均值50%认为负载过高
        
        overloaded_shards = []
        underloaded_shards = []
        
        for shard_id, stats in shard_stats.items():
            if stats['status'] != 'active':
                continue
            
            if stats['records'] > threshold:
                overloaded_shards.append((shard_id, stats['records']))
            elif stats['records'] < avg_records * 0.7:
                underloaded_shards.append((shard_id, stats['records']))
        
        # 生成迁移计划
        for overloaded_shard, _ in overloaded_shards:
            if underloaded_shards:
                target_shard, _ = underloaded_shards.pop(0)
                plan.append({
                    'source': overloaded_shard,
                    'target': target_shard,
                    'config': {'partial_migration': True}
                })
        
        return plan
    
    async def _initialize_shard_database(self, shard: ShardConfig):
        """初始化分片数据库"""
        # 这里应该实现数据库初始化逻辑
        # 创建表结构、索引等
        logger.info(f"Initializing database for shard: {shard.shard_id}")
        
        # 模拟初始化过程
        await asyncio.sleep(0.1)
    
    def get_migration_status(self) -> Dict[str, str]:
        """获取迁移状态"""
        status = {}
        for migration_key, task in self.migration_tasks.items():
            if task.done():
                status[migration_key] = "completed"
            else:
                status[migration_key] = "in_progress"
        
        return status
    
    def get_rebalance_status(self) -> Dict[str, str]:
        """获取重新平衡状态"""
        status = {}
        for table_name, task in self.rebalance_tasks.items():
            if task.done():
                status[table_name] = "completed"
            else:
                status[table_name] = "in_progress"
        
        return status


# 全局实例
_shard_router = None
_shard_manager = None


def get_shard_router() -> ShardRouter:
    """获取分片路由器实例"""
    global _shard_router
    if _shard_router is None:
        _shard_router = ShardRouter()
    return _shard_router


def get_shard_manager() -> ShardManager:
    """获取分片管理器实例"""
    global _shard_manager
    if _shard_manager is None:
        router = get_shard_router()
        _shard_manager = ShardManager(router)
    return _shard_manager


async def initialize_sharding(shards_config: List[Dict[str, Any]] = None,
                            rules_config: List[Dict[str, Any]] = None):
    """初始化分片系统"""
    router = get_shard_router()
    manager = get_shard_manager()
    
    # 添加分片
    if shards_config:
        for shard_config in shards_config:
            shard = ShardConfig(
                shard_id=shard_config['shard_id'],
                database_url=shard_config['database_url'],
                weight=shard_config.get('weight', 1),
                range_start=shard_config.get('range_start'),
                range_end=shard_config.get('range_end'),
                metadata=shard_config.get('metadata', {})
            )
            await router.add_shard(shard)
    
    # 添加分片规则
    if rules_config:
        for rule_config in rules_config:
            rule = ShardingRule(
                table_name=rule_config['table_name'],
                shard_key=rule_config['shard_key'],
                strategy=ShardingStrategy(rule_config['strategy']),
                **rule_config.get('config', {})
            )
            router.add_sharding_rule(rule)
    
    logger.info("Database sharding system initialized")
    return router, manager