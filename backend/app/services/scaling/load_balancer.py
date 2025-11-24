"""
负载均衡器 - 支持多种负载均衡策略和服务发现
"""

import asyncio
import random
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import json

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    CONSISTENT_HASH = "consistent_hash"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"


class ServiceStatus(Enum):
    """服务状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"


@dataclass
class ServiceInstance:
    """服务实例"""
    id: str
    host: str
    port: int
    weight: int = 1
    status: ServiceStatus = ServiceStatus.HEALTHY
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 运行时统计
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_response_time: float = 0.0
    average_response_time: float = 0.0
    last_health_check: float = 0.0
    
    def __post_init__(self):
        self.endpoint = f"{self.host}:{self.port}"
    
    def is_available(self) -> bool:
        """检查服务是否可用"""
        return self.status in [ServiceStatus.HEALTHY, ServiceStatus.DRAINING]
    
    def update_response_time(self, response_time: float):
        """更新响应时间统计"""
        self.last_response_time = response_time
        
        # 计算移动平均响应时间
        if self.average_response_time == 0:
            self.average_response_time = response_time
        else:
            # 使用指数移动平均
            alpha = 0.1
            self.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.average_response_time
            )
    
    def record_request(self, success: bool = True):
        """记录请求"""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'host': self.host,
            'port': self.port,
            'weight': self.weight,
            'status': self.status.value,
            'endpoint': self.endpoint,
            'active_connections': self.active_connections,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'error_rate': self.get_error_rate(),
            'average_response_time': self.average_response_time,
            'metadata': self.metadata
        }


class ServiceRegistry:
    """服务注册中心"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = defaultdict(list)
        self.service_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def register_service(self, service_name: str, instance: ServiceInstance):
        """注册服务实例"""
        async with self.lock:
            instances = self.services[service_name]
            
            # 检查是否已存在
            existing = next((i for i in instances if i.id == instance.id), None)
            if existing:
                # 更新现有实例
                existing.host = instance.host
                existing.port = instance.port
                existing.weight = instance.weight
                existing.metadata = instance.metadata
                existing.status = instance.status
            else:
                # 添加新实例
                instances.append(instance)
            
            logger.info(f"Registered service instance: {service_name}/{instance.id}")
            
            # 通知监听器
            await self._notify_listeners(service_name, 'register', instance)
    
    async def deregister_service(self, service_name: str, instance_id: str):
        """注销服务实例"""
        async with self.lock:
            instances = self.services[service_name]
            instance = next((i for i in instances if i.id == instance_id), None)
            
            if instance:
                instances.remove(instance)
                logger.info(f"Deregistered service instance: {service_name}/{instance_id}")
                
                # 通知监听器
                await self._notify_listeners(service_name, 'deregister', instance)
    
    async def update_service_status(self, service_name: str, instance_id: str, 
                                  status: ServiceStatus):
        """更新服务状态"""
        async with self.lock:
            instances = self.services[service_name]
            instance = next((i for i in instances if i.id == instance_id), None)
            
            if instance:
                old_status = instance.status
                instance.status = status
                logger.info(f"Updated service status: {service_name}/{instance_id} "
                          f"{old_status.value} -> {status.value}")
                
                # 通知监听器
                await self._notify_listeners(service_name, 'status_change', instance)
    
    def get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """获取服务实例列表"""
        return self.services.get(service_name, []).copy()
    
    def get_healthy_instances(self, service_name: str) -> List[ServiceInstance]:
        """获取健康的服务实例"""
        instances = self.services.get(service_name, [])
        return [i for i in instances if i.is_available()]
    
    def add_service_listener(self, service_name: str, listener: Callable):
        """添加服务变更监听器"""
        self.service_listeners[service_name].append(listener)
    
    async def _notify_listeners(self, service_name: str, event_type: str, 
                              instance: ServiceInstance):
        """通知监听器"""
        listeners = self.service_listeners.get(service_name, [])
        
        for listener in listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event_type, instance)
                else:
                    listener(event_type, instance)
            except Exception as e:
                logger.error(f"Error notifying listener: {e}")


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, service_registry: ServiceRegistry, 
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.service_registry = service_registry
        self.strategy = strategy
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        self.consistent_hash_ring: Dict[str, List[tuple]] = {}
        self.lock = asyncio.Lock()
    
    async def select_instance(self, service_name: str, 
                            client_info: Dict[str, Any] = None) -> Optional[ServiceInstance]:
        """选择服务实例"""
        instances = self.service_registry.get_healthy_instances(service_name)
        
        if not instances:
            logger.warning(f"No healthy instances available for service: {service_name}")
            return None
        
        if len(instances) == 1:
            return instances[0]
        
        # 根据策略选择实例
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return await self._round_robin_select(service_name, instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return await self._weighted_round_robin_select(service_name, instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_LEAST_CONNECTIONS:
            return self._weighted_least_connections_select(instances)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            return self._weighted_random_select(instances)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return await self._consistent_hash_select(service_name, instances, client_info)
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            return self._ip_hash_select(instances, client_info)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(instances)
        else:
            return random.choice(instances)
    
    async def _round_robin_select(self, service_name: str, 
                                instances: List[ServiceInstance]) -> ServiceInstance:
        """轮询选择"""
        async with self.lock:
            counter = self.round_robin_counters[service_name]
            selected = instances[counter % len(instances)]
            self.round_robin_counters[service_name] = counter + 1
            return selected
    
    async def _weighted_round_robin_select(self, service_name: str, 
                                         instances: List[ServiceInstance]) -> ServiceInstance:
        """加权轮询选择"""
        # 构建加权列表
        weighted_instances = []
        for instance in instances:
            weighted_instances.extend([instance] * instance.weight)
        
        if not weighted_instances:
            return random.choice(instances)
        
        async with self.lock:
            counter = self.round_robin_counters[service_name]
            selected = weighted_instances[counter % len(weighted_instances)]
            self.round_robin_counters[service_name] = counter + 1
            return selected
    
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最少连接选择"""
        return min(instances, key=lambda x: x.active_connections)
    
    def _weighted_least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """加权最少连接选择"""
        def score(instance):
            if instance.weight == 0:
                return float('inf')
            return instance.active_connections / instance.weight
        
        return min(instances, key=score)
    
    def _weighted_random_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """加权随机选择"""
        total_weight = sum(instance.weight for instance in instances)
        if total_weight == 0:
            return random.choice(instances)
        
        rand_weight = random.randint(1, total_weight)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.weight
            if current_weight >= rand_weight:
                return instance
        
        return instances[-1]
    
    async def _consistent_hash_select(self, service_name: str, 
                                    instances: List[ServiceInstance],
                                    client_info: Dict[str, Any] = None) -> ServiceInstance:
        """一致性哈希选择"""
        # 构建哈希环
        if service_name not in self.consistent_hash_ring:
            await self._build_hash_ring(service_name, instances)
        
        # 获取客户端标识
        client_key = self._get_client_key(client_info)
        client_hash = self._hash(client_key)
        
        # 在哈希环中查找
        ring = self.consistent_hash_ring[service_name]
        if not ring:
            return random.choice(instances)
        
        # 找到第一个大于客户端哈希值的节点
        for hash_value, instance in sorted(ring):
            if hash_value >= client_hash:
                return instance
        
        # 如果没找到，返回第一个节点（环形）
        return sorted(ring)[0][1]
    
    async def _build_hash_ring(self, service_name: str, instances: List[ServiceInstance]):
        """构建一致性哈希环"""
        ring = []
        virtual_nodes = 150  # 每个实例的虚拟节点数
        
        for instance in instances:
            for i in range(virtual_nodes):
                virtual_key = f"{instance.id}:{i}"
                hash_value = self._hash(virtual_key)
                ring.append((hash_value, instance))
        
        self.consistent_hash_ring[service_name] = ring
    
    def _ip_hash_select(self, instances: List[ServiceInstance], 
                       client_info: Dict[str, Any] = None) -> ServiceInstance:
        """IP哈希选择"""
        client_ip = self._get_client_ip(client_info)
        hash_value = self._hash(client_ip)
        index = hash_value % len(instances)
        return instances[index]
    
    def _least_response_time_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最少响应时间选择"""
        # 结合响应时间和连接数
        def score(instance):
            response_time = instance.average_response_time or 0.1
            connections = instance.active_connections + 1
            return response_time * connections
        
        return min(instances, key=score)
    
    def _get_client_key(self, client_info: Dict[str, Any] = None) -> str:
        """获取客户端标识"""
        if not client_info:
            return "default"
        
        # 优先使用用户ID，其次使用IP
        if 'user_id' in client_info:
            return str(client_info['user_id'])
        elif 'client_ip' in client_info:
            return client_info['client_ip']
        else:
            return "default"
    
    def _get_client_ip(self, client_info: Dict[str, Any] = None) -> str:
        """获取客户端IP"""
        if client_info and 'client_ip' in client_info:
            return client_info['client_ip']
        return "127.0.0.1"
    
    def _hash(self, key: str) -> int:
        """计算哈希值"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    async def record_request_start(self, instance: ServiceInstance):
        """记录请求开始"""
        instance.active_connections += 1
    
    async def record_request_end(self, instance: ServiceInstance, 
                               response_time: float, success: bool = True):
        """记录请求结束"""
        instance.active_connections = max(0, instance.active_connections - 1)
        instance.update_response_time(response_time)
        instance.record_request(success)


class HealthChecker:
    """健康检查器"""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.check_interval = 30  # 秒
        self.timeout = 5  # 秒
        self.running = False
        self.check_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_health_checks(self, service_name: str):
        """开始健康检查"""
        if service_name in self.check_tasks:
            return
        
        task = asyncio.create_task(self._health_check_loop(service_name))
        self.check_tasks[service_name] = task
        logger.info(f"Started health checks for service: {service_name}")
    
    async def stop_health_checks(self, service_name: str):
        """停止健康检查"""
        if service_name in self.check_tasks:
            task = self.check_tasks[service_name]
            task.cancel()
            del self.check_tasks[service_name]
            logger.info(f"Stopped health checks for service: {service_name}")
    
    async def _health_check_loop(self, service_name: str):
        """健康检查循环"""
        while True:
            try:
                instances = self.service_registry.get_service_instances(service_name)
                
                for instance in instances:
                    await self._check_instance_health(service_name, instance)
                
                await asyncio.sleep(self.check_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop for {service_name}: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_instance_health(self, service_name: str, instance: ServiceInstance):
        """检查实例健康状态"""
        try:
            # 这里应该实现实际的健康检查逻辑
            # 例如HTTP健康检查、TCP连接检查等
            
            # 模拟健康检查
            start_time = time.time()
            
            # 简单的连接测试（实际应该根据服务类型实现）
            is_healthy = await self._perform_health_check(instance)
            
            check_time = time.time() - start_time
            instance.last_health_check = time.time()
            
            # 更新状态
            if is_healthy:
                if instance.status == ServiceStatus.UNHEALTHY:
                    await self.service_registry.update_service_status(
                        service_name, instance.id, ServiceStatus.HEALTHY
                    )
            else:
                if instance.status == ServiceStatus.HEALTHY:
                    await self.service_registry.update_service_status(
                        service_name, instance.id, ServiceStatus.UNHEALTHY
                    )
        
        except Exception as e:
            logger.error(f"Health check failed for {service_name}/{instance.id}: {e}")
            
            if instance.status == ServiceStatus.HEALTHY:
                await self.service_registry.update_service_status(
                    service_name, instance.id, ServiceStatus.UNHEALTHY
                )
    
    async def _perform_health_check(self, instance: ServiceInstance) -> bool:
        """执行健康检查"""
        try:
            # 这里应该根据服务类型实现具体的健康检查
            # 例如：HTTP GET /health, TCP连接测试等
            
            # 模拟健康检查（实际实现应该连接到服务）
            await asyncio.sleep(0.1)  # 模拟网络延迟
            
            # 简单的错误率检查
            error_rate = instance.get_error_rate()
            if error_rate > 0.5:  # 错误率超过50%认为不健康
                return False
            
            return True
        
        except Exception:
            return False


# 全局实例
_service_registry = None
_load_balancer = None
_health_checker = None


def get_service_registry() -> ServiceRegistry:
    """获取服务注册中心实例"""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry


def get_load_balancer(strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN) -> LoadBalancer:
    """获取负载均衡器实例"""
    global _load_balancer
    if _load_balancer is None:
        registry = get_service_registry()
        _load_balancer = LoadBalancer(registry, strategy)
    return _load_balancer


def get_health_checker() -> HealthChecker:
    """获取健康检查器实例"""
    global _health_checker
    if _health_checker is None:
        registry = get_service_registry()
        _health_checker = HealthChecker(registry)
    return _health_checker


async def initialize_load_balancing(services: Dict[str, List[Dict[str, Any]]] = None,
                                  strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
    """初始化负载均衡系统"""
    registry = get_service_registry()
    load_balancer = get_load_balancer(strategy)
    health_checker = get_health_checker()
    
    # 注册服务实例
    if services:
        for service_name, instances_config in services.items():
            for instance_config in instances_config:
                instance = ServiceInstance(
                    id=instance_config['id'],
                    host=instance_config['host'],
                    port=instance_config['port'],
                    weight=instance_config.get('weight', 1),
                    metadata=instance_config.get('metadata', {})
                )
                
                await registry.register_service(service_name, instance)
            
            # 启动健康检查
            await health_checker.start_health_checks(service_name)
    
    logger.info("Load balancing system initialized")
    return registry, load_balancer, health_checker