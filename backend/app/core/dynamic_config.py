"""
动态配置更新系统
支持配置热更新、远程配置推送和实时配置同步
"""

from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
import logging
from abc import ABC, abstractmethod
import aiohttp
import websockets
from pathlib import Path
import hashlib
import time

from .config_manager import ConfigManager, ConfigChange, get_config_manager

logger = logging.getLogger(__name__)

class UpdateStrategy(Enum):
    """配置更新策略"""
    IMMEDIATE = "immediate"  # 立即更新
    SCHEDULED = "scheduled"  # 定时更新
    MANUAL = "manual"  # 手动更新
    GRADUAL = "gradual"  # 渐进式更新

class UpdateStatus(Enum):
    """更新状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class ConfigScope(Enum):
    """配置作用域"""
    GLOBAL = "global"  # 全局配置
    SERVICE = "service"  # 服务级配置
    INSTANCE = "instance"  # 实例级配置
    USER = "user"  # 用户级配置

@dataclass
class ConfigUpdate:
    """配置更新"""
    id: str
    config_key: str
    updates: Dict[str, Any]
    strategy: UpdateStrategy
    scope: ConfigScope
    version: str
    timestamp: datetime
    source: str
    target_instances: Optional[List[str]] = None
    rollback_config: Optional[Dict[str, Any]] = None
    status: UpdateStatus = UpdateStatus.PENDING
    error_message: Optional[str] = None
    applied_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "config_key": self.config_key,
            "updates": self.updates,
            "strategy": self.strategy.value,
            "scope": self.scope.value,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "target_instances": self.target_instances,
            "rollback_config": self.rollback_config,
            "status": self.status.value,
            "error_message": self.error_message,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None
        }

@dataclass
class ConfigSubscription:
    """配置订阅"""
    config_key: str
    callback: Callable[[Dict[str, Any]], None]
    scope: ConfigScope
    filters: Dict[str, Any] = field(default_factory=dict)
    last_update: Optional[datetime] = None

class ConfigUpdateHandler(ABC):
    """配置更新处理器抽象基类"""
    
    @abstractmethod
    async def can_handle(self, update: ConfigUpdate) -> bool:
        """检查是否可以处理此更新"""
        pass
    
    @abstractmethod
    async def apply_update(self, update: ConfigUpdate) -> bool:
        """应用配置更新"""
        pass
    
    @abstractmethod
    async def rollback_update(self, update: ConfigUpdate) -> bool:
        """回滚配置更新"""
        pass

class ServiceConfigHandler(ConfigUpdateHandler):
    """服务配置更新处理器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.supported_configs = {
            "translation", "monitoring", "security", "database", "redis"
        }
    
    async def can_handle(self, update: ConfigUpdate) -> bool:
        """检查是否可以处理此更新"""
        return (
            update.scope in [ConfigScope.GLOBAL, ConfigScope.SERVICE] and
            update.config_key in self.supported_configs
        )
    
    async def apply_update(self, update: ConfigUpdate) -> bool:
        """应用服务配置更新"""
        try:
            # 获取当前配置作为回滚备份
            current_config = await self.config_manager.load_config(update.config_key)
            update.rollback_config = current_config.copy()
            
            # 应用更新
            success = await self.config_manager.update_config(
                update.config_key, 
                update.updates
            )
            
            if success:
                update.status = UpdateStatus.COMPLETED
                update.applied_at = datetime.now()
                logger.info(f"Applied config update {update.id} for {update.config_key}")
            else:
                update.status = UpdateStatus.FAILED
                update.error_message = "Failed to update configuration"
            
            return success
            
        except Exception as e:
            update.status = UpdateStatus.FAILED
            update.error_message = str(e)
            logger.error(f"Failed to apply config update {update.id}: {e}")
            return False
    
    async def rollback_update(self, update: ConfigUpdate) -> bool:
        """回滚服务配置更新"""
        try:
            if not update.rollback_config:
                logger.error(f"No rollback config available for update {update.id}")
                return False
            
            success = await self.config_manager.save_config(
                update.config_key,
                update.rollback_config
            )
            
            if success:
                update.status = UpdateStatus.ROLLED_BACK
                logger.info(f"Rolled back config update {update.id} for {update.config_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rollback config update {update.id}: {e}")
            return False

class RemoteConfigClient:
    """远程配置客户端"""
    
    def __init__(self, server_url: str, instance_id: str, auth_token: str):
        self.server_url = server_url
        self.instance_id = instance_id
        self.auth_token = auth_token
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.reconnect_interval = 30
        self.heartbeat_interval = 60
        self.last_heartbeat = datetime.now()
    
    async def connect(self) -> bool:
        """连接到远程配置服务器"""
        try:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.auth_token}"}
            )
            
            # 建立WebSocket连接
            ws_url = self.server_url.replace("http", "ws") + f"/ws/{self.instance_id}"
            self.websocket = await websockets.connect(
                ws_url,
                extra_headers={"Authorization": f"Bearer {self.auth_token}"}
            )
            
            self.connected = True
            logger.info(f"Connected to remote config server: {self.server_url}")
            
            # 启动心跳和消息处理
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._message_handler())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to remote config server: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Disconnected from remote config server")
    
    async def fetch_config(self, config_key: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取远程配置"""
        try:
            if not self.session:
                return None
            
            url = f"{self.server_url}/api/config/{config_key}"
            params = {"instance_id": self.instance_id}
            if version:
                params["version"] = version
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch config {config_key}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching config {config_key}: {e}")
            return None
    
    async def push_config(self, config_key: str, config: Dict[str, Any]) -> bool:
        """推送配置到远程服务器"""
        try:
            if not self.session:
                return False
            
            url = f"{self.server_url}/api/config/{config_key}"
            data = {
                "instance_id": self.instance_id,
                "config": config,
                "timestamp": datetime.now().isoformat()
            }
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    logger.info(f"Successfully pushed config {config_key}")
                    return True
                else:
                    logger.error(f"Failed to push config {config_key}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error pushing config {config_key}: {e}")
            return False
    
    async def subscribe_updates(self, config_keys: List[str]) -> bool:
        """订阅配置更新"""
        try:
            if not self.websocket:
                return False
            
            message = {
                "type": "subscribe",
                "config_keys": config_keys,
                "instance_id": self.instance_id
            }
            
            await self.websocket.send(json.dumps(message))
            logger.info(f"Subscribed to config updates: {config_keys}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to config updates: {e}")
            return False
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self.connected:
            try:
                if self.websocket:
                    await self.websocket.send(json.dumps({
                        "type": "heartbeat",
                        "instance_id": self.instance_id,
                        "timestamp": datetime.now().isoformat()
                    }))
                    self.last_heartbeat = datetime.now()
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                self.connected = False
                break
    
    async def _message_handler(self):
        """消息处理循环"""
        while self.connected:
            try:
                if not self.websocket:
                    break
                
                message = await self.websocket.recv()
                data = json.loads(message)
                
                await self._handle_message(data)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.connected = False
                break
            except Exception as e:
                logger.error(f"Message handler error: {e}")
                continue
    
    async def _handle_message(self, data: Dict[str, Any]):
        """处理接收到的消息"""
        message_type = data.get("type")
        
        if message_type == "config_update":
            # 处理配置更新消息
            update_data = data.get("update")
            if update_data:
                update = ConfigUpdate(
                    id=update_data["id"],
                    config_key=update_data["config_key"],
                    updates=update_data["updates"],
                    strategy=UpdateStrategy(update_data["strategy"]),
                    scope=ConfigScope(update_data["scope"]),
                    version=update_data["version"],
                    timestamp=datetime.fromisoformat(update_data["timestamp"]),
                    source=update_data["source"]
                )
                
                # 通知动态配置管理器
                dynamic_manager = get_dynamic_config_manager()
                await dynamic_manager.handle_remote_update(update)
        
        elif message_type == "heartbeat_response":
            # 处理心跳响应
            self.last_heartbeat = datetime.now()
        
        elif message_type == "error":
            # 处理错误消息
            logger.error(f"Remote config server error: {data.get('message')}")

class DynamicConfigManager:
    """动态配置管理器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.update_handlers: List[ConfigUpdateHandler] = []
        self.subscriptions: Dict[str, List[ConfigSubscription]] = {}
        self.pending_updates: Dict[str, ConfigUpdate] = {}
        self.update_history: List[ConfigUpdate] = []
        self.remote_client: Optional[RemoteConfigClient] = None
        self.running = False
        self.update_queue = asyncio.Queue()
        
        # 注册默认处理器
        self.register_handler(ServiceConfigHandler(config_manager))
    
    def register_handler(self, handler: ConfigUpdateHandler):
        """注册配置更新处理器"""
        self.update_handlers.append(handler)
        logger.info(f"Registered config update handler: {handler.__class__.__name__}")
    
    def subscribe(self, config_key: str, callback: Callable[[Dict[str, Any]], None], 
                 scope: ConfigScope = ConfigScope.GLOBAL, filters: Dict[str, Any] = None):
        """订阅配置变化"""
        subscription = ConfigSubscription(
            config_key=config_key,
            callback=callback,
            scope=scope,
            filters=filters or {}
        )
        
        if config_key not in self.subscriptions:
            self.subscriptions[config_key] = []
        
        self.subscriptions[config_key].append(subscription)
        logger.info(f"Added subscription for config {config_key}")
    
    def unsubscribe(self, config_key: str, callback: Callable[[Dict[str, Any]], None]):
        """取消订阅配置变化"""
        if config_key in self.subscriptions:
            self.subscriptions[config_key] = [
                sub for sub in self.subscriptions[config_key]
                if sub.callback != callback
            ]
            
            if not self.subscriptions[config_key]:
                del self.subscriptions[config_key]
        
        logger.info(f"Removed subscription for config {config_key}")
    
    async def setup_remote_client(self, server_url: str, instance_id: str, auth_token: str):
        """设置远程配置客户端"""
        self.remote_client = RemoteConfigClient(server_url, instance_id, auth_token)
        
        # 尝试连接
        connected = await self.remote_client.connect()
        if connected:
            # 订阅所有已订阅的配置
            config_keys = list(self.subscriptions.keys())
            if config_keys:
                await self.remote_client.subscribe_updates(config_keys)
        
        return connected
    
    async def start(self):
        """启动动态配置管理器"""
        if self.running:
            return
        
        self.running = True
        
        # 启动更新处理循环
        asyncio.create_task(self._update_processor())
        
        logger.info("Dynamic config manager started")
    
    async def stop(self):
        """停止动态配置管理器"""
        self.running = False
        
        if self.remote_client:
            await self.remote_client.disconnect()
        
        logger.info("Dynamic config manager stopped")
    
    async def apply_update(self, update: ConfigUpdate) -> bool:
        """应用配置更新"""
        try:
            # 找到合适的处理器
            handler = None
            for h in self.update_handlers:
                if await h.can_handle(update):
                    handler = h
                    break
            
            if not handler:
                update.status = UpdateStatus.FAILED
                update.error_message = "No suitable handler found"
                logger.error(f"No handler found for update {update.id}")
                return False
            
            # 根据策略应用更新
            if update.strategy == UpdateStrategy.IMMEDIATE:
                success = await handler.apply_update(update)
            elif update.strategy == UpdateStrategy.SCHEDULED:
                # 添加到待处理队列
                self.pending_updates[update.id] = update
                await self.update_queue.put(update)
                success = True
            else:
                # 手动更新，仅记录
                self.pending_updates[update.id] = update
                success = True
            
            # 记录历史
            self.update_history.append(update)
            
            # 通知订阅者
            if success and update.status == UpdateStatus.COMPLETED:
                await self._notify_subscribers(update.config_key, update.updates)
            
            return success
            
        except Exception as e:
            update.status = UpdateStatus.FAILED
            update.error_message = str(e)
            logger.error(f"Failed to apply update {update.id}: {e}")
            return False
    
    async def rollback_update(self, update_id: str) -> bool:
        """回滚配置更新"""
        try:
            # 从历史中查找更新
            update = None
            for u in self.update_history:
                if u.id == update_id:
                    update = u
                    break
            
            if not update:
                logger.error(f"Update {update_id} not found in history")
                return False
            
            # 找到合适的处理器
            handler = None
            for h in self.update_handlers:
                if await h.can_handle(update):
                    handler = h
                    break
            
            if not handler:
                logger.error(f"No handler found for rollback {update_id}")
                return False
            
            # 执行回滚
            success = await handler.rollback_update(update)
            
            if success:
                # 通知订阅者
                if update.rollback_config:
                    await self._notify_subscribers(update.config_key, update.rollback_config)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rollback update {update_id}: {e}")
            return False
    
    async def handle_remote_update(self, update: ConfigUpdate):
        """处理远程配置更新"""
        logger.info(f"Received remote config update {update.id} for {update.config_key}")
        await self.apply_update(update)
    
    async def get_pending_updates(self) -> List[ConfigUpdate]:
        """获取待处理的更新"""
        return list(self.pending_updates.values())
    
    async def get_update_history(self, limit: int = 100) -> List[ConfigUpdate]:
        """获取更新历史"""
        return self.update_history[-limit:]
    
    async def _update_processor(self):
        """更新处理循环"""
        while self.running:
            try:
                # 从队列获取更新
                update = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)
                
                # 找到合适的处理器并应用更新
                for handler in self.update_handlers:
                    if await handler.can_handle(update):
                        update.status = UpdateStatus.IN_PROGRESS
                        success = await handler.apply_update(update)
                        
                        if success:
                            # 从待处理列表中移除
                            if update.id in self.pending_updates:
                                del self.pending_updates[update.id]
                        
                        break
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in update processor: {e}")
                continue
    
    async def _notify_subscribers(self, config_key: str, config: Dict[str, Any]):
        """通知配置订阅者"""
        if config_key in self.subscriptions:
            for subscription in self.subscriptions[config_key]:
                try:
                    # 应用过滤器
                    if self._matches_filters(config, subscription.filters):
                        if asyncio.iscoroutinefunction(subscription.callback):
                            await subscription.callback(config)
                        else:
                            subscription.callback(config)
                        
                        subscription.last_update = datetime.now()
                
                except Exception as e:
                    logger.error(f"Error in subscription callback: {e}")
    
    def _matches_filters(self, config: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """检查配置是否匹配过滤器"""
        if not filters:
            return True
        
        for key, expected_value in filters.items():
            config_value = config.get(key)
            if config_value != expected_value:
                return False
        
        return True

# 全局动态配置管理器实例
_dynamic_config_manager: Optional[DynamicConfigManager] = None

def get_dynamic_config_manager() -> DynamicConfigManager:
    """获取动态配置管理器实例"""
    global _dynamic_config_manager
    if _dynamic_config_manager is None:
        config_manager = get_config_manager()
        _dynamic_config_manager = DynamicConfigManager(config_manager)
    return _dynamic_config_manager

async def init_dynamic_config_system(
    remote_server_url: Optional[str] = None,
    instance_id: Optional[str] = None,
    auth_token: Optional[str] = None
) -> DynamicConfigManager:
    """初始化动态配置系统"""
    global _dynamic_config_manager
    
    config_manager = get_config_manager()
    _dynamic_config_manager = DynamicConfigManager(config_manager)
    
    # 设置远程客户端（如果提供了参数）
    if remote_server_url and instance_id and auth_token:
        await _dynamic_config_manager.setup_remote_client(
            remote_server_url, instance_id, auth_token
        )
    
    # 启动动态配置管理器
    await _dynamic_config_manager.start()
    
    logger.info("Dynamic config system initialized successfully")
    return _dynamic_config_manager