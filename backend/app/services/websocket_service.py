"""
WebSocket实时状态推送服务
支持分析进度、结果和系统状态的实时推送
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型"""
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_PROGRESS = "analysis_progress"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    CACHE_UPDATED = "cache_updated"
    SYSTEM_STATUS = "system_status"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class WebSocketEvent:
    """WebSocket事件"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    session_id: Optional[str] = None
    pr_number: Optional[int] = None
    tenant_id: Optional[str] = None
    repo_id: Optional[str] = None
    data: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "pr_number": self.pr_number,
            "tenant_id": self.tenant_id,
            "repo_id": self.repo_id,
            "data": self.data or {}
        }


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 活跃连接 {connection_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}
        
        # 连接元数据 {connection_id: metadata}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # 订阅关系 {subscription_key: Set[connection_id]}
        self.subscriptions: Dict[str, Set[str]] = {}
        
        # 连接到订阅的反向索引 {connection_id: Set[subscription_key]}
        self.connection_subscriptions: Dict[str, Set[str]] = {}
    
    async def connect(
        self, 
        websocket: WebSocket, 
        connection_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """建立WebSocket连接"""
        if connection_id is None:
            connection_id = str(uuid.uuid4())
        
        await websocket.accept()
        
        self.active_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = metadata or {}
        self.connection_subscriptions[connection_id] = set()
        
        logger.info(f"WebSocket connection established: {connection_id}")
        return connection_id
    
    def disconnect(self, connection_id: str):
        """断开WebSocket连接"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
        
        # 清理订阅关系
        if connection_id in self.connection_subscriptions:
            for sub_key in self.connection_subscriptions[connection_id]:
                if sub_key in self.subscriptions:
                    self.subscriptions[sub_key].discard(connection_id)
                    if not self.subscriptions[sub_key]:
                        del self.subscriptions[sub_key]
            
            del self.connection_subscriptions[connection_id]
        
        logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def send_personal_message(
        self, 
        message: Dict[str, Any], 
        connection_id: str
    ):
        """发送个人消息"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to send message to {connection_id}: {e}")
                    self.disconnect(connection_id)
    
    async def broadcast_to_subscription(
        self, 
        message: Dict[str, Any], 
        subscription_key: str
    ):
        """向订阅者广播消息"""
        if subscription_key not in self.subscriptions:
            return
        
        disconnected_connections = []
        
        for connection_id in self.subscriptions[subscription_key]:
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_text(json.dumps(message))
                    except Exception as e:
                        logger.error(f"Failed to broadcast to {connection_id}: {e}")
                        disconnected_connections.append(connection_id)
                else:
                    disconnected_connections.append(connection_id)
        
        # 清理断开的连接
        for connection_id in disconnected_connections:
            self.disconnect(connection_id)
    
    def subscribe(
        self, 
        connection_id: str, 
        subscription_key: str
    ):
        """订阅频道"""
        if connection_id not in self.active_connections:
            return False
        
        if subscription_key not in self.subscriptions:
            self.subscriptions[subscription_key] = set()
        
        self.subscriptions[subscription_key].add(connection_id)
        self.connection_subscriptions[connection_id].add(subscription_key)
        
        logger.info(f"Connection {connection_id} subscribed to {subscription_key}")
        return True
    
    def unsubscribe(
        self, 
        connection_id: str, 
        subscription_key: str
    ):
        """取消订阅"""
        if subscription_key in self.subscriptions:
            self.subscriptions[subscription_key].discard(connection_id)
            if not self.subscriptions[subscription_key]:
                del self.subscriptions[subscription_key]
        
        if connection_id in self.connection_subscriptions:
            self.connection_subscriptions[connection_id].discard(subscription_key)
        
        logger.info(f"Connection {connection_id} unsubscribed from {subscription_key}")
    
    def get_connection_count(self) -> int:
        """获取活跃连接数"""
        return len(self.active_connections)
    
    def get_subscription_count(self, subscription_key: str) -> int:
        """获取订阅者数量"""
        return len(self.subscriptions.get(subscription_key, set()))


class RealtimeEventService:
    """实时事件服务"""
    
    def __init__(self, connection_manager: ConnectionManager, redis_client=None):
        self.connection_manager = connection_manager
        self.redis_client = redis_client
        
        # 事件历史缓存
        self.event_history: List[WebSocketEvent] = []
        self.max_history_size = 1000
        
        # 事件统计
        self.event_stats: Dict[str, int] = {}
    
    async def publish_event(self, event: WebSocketEvent):
        """发布事件"""
        # 记录事件历史
        self._record_event(event)
        
        # 构建订阅键
        subscription_keys = self._build_subscription_keys(event)
        
        # 广播事件
        event_data = event.to_dict()
        
        for sub_key in subscription_keys:
            await self.connection_manager.broadcast_to_subscription(event_data, sub_key)
        
        # 如果有Redis，也发布到Redis
        if self.redis_client:
            try:
                await self.redis_client.publish(
                    "analysis_events",
                    json.dumps(event_data)
                )
            except Exception as e:
                logger.error(f"Failed to publish event to Redis: {e}")
        
        logger.info(f"Event published: {event.event_type.value} for session {event.session_id}")
    
    async def publish_analysis_progress(
        self,
        session_id: str,
        progress: float,
        current_file: str,
        total_files: int,
        processed_files: int,
        pr_number: Optional[int] = None,
        tenant_id: Optional[str] = None,
        repo_id: Optional[str] = None
    ):
        """发布分析进度"""
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ANALYSIS_PROGRESS,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            pr_number=pr_number,
            tenant_id=tenant_id,
            repo_id=repo_id,
            data={
                "progress": progress,
                "current_file": current_file,
                "total_files": total_files,
                "processed_files": processed_files,
                "estimated_remaining": self._calculate_estimated_remaining(
                    progress, processed_files, total_files
                )
            }
        )
        
        await self.publish_event(event)
    
    async def publish_analysis_started(
        self,
        session_id: str,
        pr_number: int,
        files_to_analyze: List[str],
        tenant_id: Optional[str] = None,
        repo_id: Optional[str] = None
    ):
        """发布分析开始事件"""
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ANALYSIS_STARTED,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            pr_number=pr_number,
            tenant_id=tenant_id,
            repo_id=repo_id,
            data={
                "files_to_analyze": files_to_analyze,
                "total_files": len(files_to_analyze),
                "started_at": datetime.utcnow().isoformat()
            }
        )
        
        await self.publish_event(event)
    
    async def publish_analysis_completed(
        self,
        session_id: str,
        pr_number: int,
        analysis_results: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        tenant_id: Optional[str] = None,
        repo_id: Optional[str] = None
    ):
        """发布分析完成事件"""
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ANALYSIS_COMPLETED,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            pr_number=pr_number,
            tenant_id=tenant_id,
            repo_id=repo_id,
            data={
                "analysis_results": analysis_results,
                "performance_metrics": performance_metrics,
                "completed_at": datetime.utcnow().isoformat()
            }
        )
        
        await self.publish_event(event)
    
    async def publish_analysis_failed(
        self,
        session_id: str,
        pr_number: int,
        error_message: str,
        error_details: Dict[str, Any] = None,
        tenant_id: Optional[str] = None,
        repo_id: Optional[str] = None
    ):
        """发布分析失败事件"""
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ANALYSIS_FAILED,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            pr_number=pr_number,
            tenant_id=tenant_id,
            repo_id=repo_id,
            data={
                "error_message": error_message,
                "error_details": error_details or {},
                "failed_at": datetime.utcnow().isoformat()
            }
        )
        
        await self.publish_event(event)
    
    async def publish_cache_updated(
        self,
        tenant_id: str,
        repo_id: str,
        cache_stats: Dict[str, Any]
    ):
        """发布缓存更新事件"""
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.CACHE_UPDATED,
            timestamp=datetime.utcnow(),
            tenant_id=tenant_id,
            repo_id=repo_id,
            data={
                "cache_stats": cache_stats,
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        
        await self.publish_event(event)
    
    async def publish_system_status(
        self,
        system_metrics: Dict[str, Any]
    ):
        """发布系统状态事件"""
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.SYSTEM_STATUS,
            timestamp=datetime.utcnow(),
            data={
                "system_metrics": system_metrics,
                "connection_count": self.connection_manager.get_connection_count(),
                "status_at": datetime.utcnow().isoformat()
            }
        )
        
        await self.publish_event(event)
    
    def _record_event(self, event: WebSocketEvent):
        """记录事件历史"""
        self.event_history.append(event)
        
        # 限制历史记录大小
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]
        
        # 更新统计
        event_type = event.event_type.value
        self.event_stats[event_type] = self.event_stats.get(event_type, 0) + 1
    
    def _build_subscription_keys(self, event: WebSocketEvent) -> List[str]:
        """构建订阅键"""
        keys = []
        
        # 全局事件
        keys.append("global")
        
        # 事件类型
        keys.append(f"event_type:{event.event_type.value}")
        
        # 会话相关
        if event.session_id:
            keys.append(f"session:{event.session_id}")
        
        # PR相关
        if event.pr_number:
            keys.append(f"pr:{event.pr_number}")
        
        # 租户相关
        if event.tenant_id:
            keys.append(f"tenant:{event.tenant_id}")
        
        # 仓库相关
        if event.repo_id:
            keys.append(f"repo:{event.repo_id}")
        
        # 租户+仓库组合
        if event.tenant_id and event.repo_id:
            keys.append(f"tenant_repo:{event.tenant_id}:{event.repo_id}")
        
        return keys
    
    def _calculate_estimated_remaining(
        self,
        progress: float,
        processed_files: int,
        total_files: int
    ) -> Optional[int]:
        """计算预估剩余时间（秒）"""
        if progress <= 0 or processed_files == 0:
            return None
        
        # 简单的线性估算
        avg_time_per_file = 2.0  # 假设平均每个文件2秒
        remaining_files = total_files - processed_files
        
        return int(remaining_files * avg_time_per_file)
    
    def get_event_history(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取事件历史"""
        filtered_events = self.event_history
        
        # 按会话过滤
        if session_id:
            filtered_events = [
                e for e in filtered_events 
                if e.session_id == session_id
            ]
        
        # 按事件类型过滤
        if event_type:
            filtered_events = [
                e for e in filtered_events 
                if e.event_type == event_type
            ]
        
        # 按时间倒序排列并限制数量
        filtered_events = sorted(
            filtered_events,
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        return [event.to_dict() for event in filtered_events[:limit]]
    
    def get_event_stats(self) -> Dict[str, Any]:
        """获取事件统计"""
        return {
            "total_events": len(self.event_history),
            "event_type_stats": self.event_stats.copy(),
            "active_connections": self.connection_manager.get_connection_count(),
            "subscriptions": {
                key: len(subscribers) 
                for key, subscribers in self.connection_manager.subscriptions.items()
            }
        }


# 全局实例
connection_manager = ConnectionManager()
event_service = RealtimeEventService(connection_manager)


async def handle_websocket_connection(
    websocket: WebSocket,
    connection_id: str = None,
    subscriptions: List[str] = None,
    metadata: Dict[str, Any] = None
):
    """处理WebSocket连接"""
    conn_id = await connection_manager.connect(
        websocket, connection_id, metadata
    )
    
    # 订阅指定的频道
    if subscriptions:
        for sub in subscriptions:
            connection_manager.subscribe(conn_id, sub)
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理订阅/取消订阅请求
            if message.get("type") == "subscribe":
                channel = message.get("channel")
                if channel:
                    connection_manager.subscribe(conn_id, channel)
            
            elif message.get("type") == "unsubscribe":
                channel = message.get("channel")
                if channel:
                    connection_manager.unsubscribe(conn_id, channel)
            
            elif message.get("type") == "ping":
                # 响应心跳
                await websocket.send_text(json.dumps({"type": "pong"}))
    
    except WebSocketDisconnect:
        connection_manager.disconnect(conn_id)
    except Exception as e:
        logger.error(f"WebSocket error for {conn_id}: {e}")
        connection_manager.disconnect(conn_id)