"""
无状态服务基类和会话管理 - 支持水平扩展
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib

logger = logging.getLogger(__name__)


class SessionStorage(Enum):
    """会话存储类型"""
    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"
    DISTRIBUTED_CACHE = "distributed_cache"


class ServiceState(Enum):
    """服务状态"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SessionData:
    """会话数据"""
    session_id: str
    user_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查会话是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """更新最后访问时间"""
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'data': self.data,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'expires_at': self.expires_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """从字典创建"""
        return cls(
            session_id=data['session_id'],
            user_id=data.get('user_id'),
            data=data.get('data', {}),
            created_at=data.get('created_at', time.time()),
            last_accessed=data.get('last_accessed', time.time()),
            expires_at=data.get('expires_at'),
            metadata=data.get('metadata', {})
        )


class SessionManager(ABC):
    """会话管理器抽象基类"""
    
    @abstractmethod
    async def create_session(self, user_id: Optional[str] = None, 
                           ttl: Optional[int] = None) -> str:
        """创建会话"""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """获取会话"""
        pass
    
    @abstractmethod
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """更新会话数据"""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        pass
    
    @abstractmethod
    async def cleanup_expired_sessions(self) -> int:
        """清理过期会话"""
        pass


class MemorySessionManager(SessionManager):
    """内存会话管理器"""
    
    def __init__(self, default_ttl: int = 3600):
        self.sessions: Dict[str, SessionData] = {}
        self.default_ttl = default_ttl
        self.lock = asyncio.Lock()
        self.cleanup_task = None
        self.cleanup_interval = 300  # 5分钟清理一次
    
    async def start(self):
        """启动会话管理器"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """停止会话管理器"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            self.cleanup_task = None
    
    async def create_session(self, user_id: Optional[str] = None, 
                           ttl: Optional[int] = None) -> str:
        """创建会话"""
        session_id = str(uuid.uuid4())
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl if ttl > 0 else None
        
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at
        )
        
        async with self.lock:
            self.sessions[session_id] = session
        
        logger.debug(f"Created session: {session_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """获取会话"""
        async with self.lock:
            session = self.sessions.get(session_id)
            
            if session is None:
                return None
            
            if session.is_expired():
                del self.sessions[session_id]
                return None
            
            session.touch()
            return session
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """更新会话数据"""
        async with self.lock:
            session = self.sessions.get(session_id)
            
            if session is None or session.is_expired():
                return False
            
            session.data.update(data)
            session.touch()
            return True
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        async with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.debug(f"Deleted session: {session_id}")
                return True
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """清理过期会话"""
        expired_sessions = []
        
        async with self.lock:
            for session_id, session in self.sessions.items():
                if session.is_expired():
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup loop: {e}")
    
    def get_session_count(self) -> int:
        """获取会话数量"""
        return len(self.sessions)


class DistributedSessionManager(SessionManager):
    """分布式会话管理器"""
    
    def __init__(self, cache_client, default_ttl: int = 3600):
        self.cache_client = cache_client
        self.default_ttl = default_ttl
        self.key_prefix = "session:"
    
    async def create_session(self, user_id: Optional[str] = None, 
                           ttl: Optional[int] = None) -> str:
        """创建会话"""
        session_id = str(uuid.uuid4())
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl if ttl > 0 else None
        
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at
        )
        
        key = f"{self.key_prefix}{session_id}"
        await self.cache_client.set(key, json.dumps(session.to_dict()), ttl)
        
        logger.debug(f"Created distributed session: {session_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """获取会话"""
        key = f"{self.key_prefix}{session_id}"
        data = await self.cache_client.get(key)
        
        if data is None:
            return None
        
        try:
            session_dict = json.loads(data)
            session = SessionData.from_dict(session_dict)
            
            if session.is_expired():
                await self.delete_session(session_id)
                return None
            
            session.touch()
            
            # 更新最后访问时间
            await self.cache_client.set(key, json.dumps(session.to_dict()))
            
            return session
        
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing session data: {e}")
            await self.delete_session(session_id)
            return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """更新会话数据"""
        session = await self.get_session(session_id)
        
        if session is None:
            return False
        
        session.data.update(data)
        session.touch()
        
        key = f"{self.key_prefix}{session_id}"
        await self.cache_client.set(key, json.dumps(session.to_dict()))
        
        return True
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        key = f"{self.key_prefix}{session_id}"
        result = await self.cache_client.delete(key)
        
        if result:
            logger.debug(f"Deleted distributed session: {session_id}")
        
        return result
    
    async def cleanup_expired_sessions(self) -> int:
        """清理过期会话"""
        # 分布式缓存通常有自动过期机制
        # 这里可以实现额外的清理逻辑
        return 0


class StatelessService(ABC):
    """无状态服务基类"""
    
    def __init__(self, service_name: str, session_manager: Optional[SessionManager] = None):
        self.service_name = service_name
        self.service_id = f"{service_name}_{uuid.uuid4().hex[:8]}"
        self.session_manager = session_manager or MemorySessionManager()
        self.state = ServiceState.STOPPED
        self.start_time = None
        self.request_count = 0
        self.error_count = 0
        self.middleware_stack: List[Callable] = []
        self.hooks: Dict[str, List[Callable]] = {
            'before_request': [],
            'after_request': [],
            'on_error': []
        }
    
    async def start(self):
        """启动服务"""
        try:
            self.state = ServiceState.STARTING
            
            # 启动会话管理器
            if hasattr(self.session_manager, 'start'):
                await self.session_manager.start()
            
            # 执行服务特定的启动逻辑
            await self._on_start()
            
            self.state = ServiceState.RUNNING
            self.start_time = time.time()
            
            logger.info(f"Service {self.service_name} started with ID: {self.service_id}")
        
        except Exception as e:
            self.state = ServiceState.ERROR
            logger.error(f"Failed to start service {self.service_name}: {e}")
            raise
    
    async def stop(self):
        """停止服务"""
        try:
            self.state = ServiceState.STOPPING
            
            # 执行服务特定的停止逻辑
            await self._on_stop()
            
            # 停止会话管理器
            if hasattr(self.session_manager, 'stop'):
                await self.session_manager.stop()
            
            self.state = ServiceState.STOPPED
            
            logger.info(f"Service {self.service_name} stopped")
        
        except Exception as e:
            self.state = ServiceState.ERROR
            logger.error(f"Error stopping service {self.service_name}: {e}")
            raise
    
    @abstractmethod
    async def _on_start(self):
        """服务启动时的回调"""
        pass
    
    @abstractmethod
    async def _on_stop(self):
        """服务停止时的回调"""
        pass
    
    async def handle_request(self, request_data: Dict[str, Any], 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理请求"""
        context = context or {}
        request_id = context.get('request_id', str(uuid.uuid4()))
        
        try:
            # 执行前置钩子
            await self._execute_hooks('before_request', request_data, context)
            
            # 应用中间件
            response = await self._apply_middleware(request_data, context)
            
            if response is None:
                # 执行实际的请求处理
                response = await self._handle_request(request_data, context)
            
            # 执行后置钩子
            await self._execute_hooks('after_request', response, context)
            
            self.request_count += 1
            
            return response
        
        except Exception as e:
            self.error_count += 1
            
            # 执行错误钩子
            await self._execute_hooks('on_error', e, context)
            
            logger.error(f"Error handling request {request_id}: {e}")
            
            return {
                'error': str(e),
                'request_id': request_id,
                'service_id': self.service_id
            }
    
    @abstractmethod
    async def _handle_request(self, request_data: Dict[str, Any], 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """处理具体的请求逻辑"""
        pass
    
    async def _apply_middleware(self, request_data: Dict[str, Any], 
                              context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """应用中间件"""
        for middleware in self.middleware_stack:
            try:
                result = await middleware(request_data, context)
                if result is not None:
                    return result
            except Exception as e:
                logger.error(f"Middleware error: {e}")
                raise
        
        return None
    
    async def _execute_hooks(self, hook_name: str, data: Any, context: Dict[str, Any]):
        """执行钩子"""
        hooks = self.hooks.get(hook_name, [])
        
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(data, context)
                else:
                    hook(data, context)
            except Exception as e:
                logger.error(f"Hook {hook_name} error: {e}")
    
    def add_middleware(self, middleware: Callable):
        """添加中间件"""
        self.middleware_stack.append(middleware)
    
    def add_hook(self, hook_name: str, hook: Callable):
        """添加钩子"""
        if hook_name in self.hooks:
            self.hooks[hook_name].append(hook)
    
    async def create_session(self, user_id: Optional[str] = None, 
                           ttl: Optional[int] = None) -> str:
        """创建会话"""
        return await self.session_manager.create_session(user_id, ttl)
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """获取会话"""
        return await self.session_manager.get_session(session_id)
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """更新会话"""
        return await self.session_manager.update_session(session_id, data)
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        return await self.session_manager.delete_session(session_id)
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'service_name': self.service_name,
            'service_id': self.service_id,
            'state': self.state.value,
            'uptime': uptime,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'session_count': getattr(self.session_manager, 'get_session_count', lambda: 0)()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取服务指标"""
        return {
            'service_metrics': self.get_health_status(),
            'session_metrics': {
                'active_sessions': getattr(self.session_manager, 'get_session_count', lambda: 0)()
            }
        }


class TranslationStatelessService(StatelessService):
    """翻译无状态服务示例"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        super().__init__("translation_service", session_manager)
        self.translation_engines = {}
    
    async def _on_start(self):
        """启动时初始化翻译引擎"""
        # 这里可以初始化翻译引擎
        logger.info("Translation service starting...")
    
    async def _on_stop(self):
        """停止时清理资源"""
        # 清理翻译引擎资源
        logger.info("Translation service stopping...")
    
    async def _handle_request(self, request_data: Dict[str, Any], 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """处理翻译请求"""
        action = request_data.get('action')
        
        if action == 'translate':
            return await self._handle_translate(request_data, context)
        elif action == 'detect_language':
            return await self._handle_detect_language(request_data, context)
        elif action == 'get_supported_languages':
            return await self._handle_get_supported_languages(request_data, context)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _handle_translate(self, request_data: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """处理翻译请求"""
        text = request_data.get('text', '')
        source_lang = request_data.get('source_lang', 'auto')
        target_lang = request_data.get('target_lang', 'en')
        
        # 模拟翻译处理
        await asyncio.sleep(0.1)
        
        translated_text = f"[{source_lang}->{target_lang}] {text}"
        
        return {
            'translated_text': translated_text,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'confidence': 0.95
        }
    
    async def _handle_detect_language(self, request_data: Dict[str, Any], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """处理语言检测请求"""
        text = request_data.get('text', '')
        
        # 模拟语言检测
        await asyncio.sleep(0.05)
        
        # 简单的语言检测逻辑
        import re
        if re.search(r'[\u4e00-\u9fff]', text):
            detected_lang = 'zh'
        elif re.search(r'[a-zA-Z]', text):
            detected_lang = 'en'
        else:
            detected_lang = 'auto'
        
        return {
            'detected_language': detected_lang,
            'confidence': 0.9
        }
    
    async def _handle_get_supported_languages(self, request_data: Dict[str, Any], 
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """获取支持的语言列表"""
        return {
            'supported_languages': ['en', 'zh', 'ja', 'ko', 'fr', 'de', 'es', 'ru']
        }


# 中间件示例
async def authentication_middleware(request_data: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """认证中间件"""
    auth_token = request_data.get('auth_token')
    
    if not auth_token:
        return {
            'error': 'Authentication required',
            'code': 401
        }
    
    # 验证token（这里简化处理）
    if auth_token == 'invalid':
        return {
            'error': 'Invalid authentication token',
            'code': 401
        }
    
    # 将用户信息添加到上下文
    context['user_id'] = 'user_123'
    
    return None  # 继续处理


async def rate_limiting_middleware(request_data: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """限流中间件"""
    user_id = context.get('user_id', 'anonymous')
    
    # 这里应该实现实际的限流逻辑
    # 例如使用Redis计数器或令牌桶算法
    
    # 模拟限流检查
    if user_id == 'rate_limited_user':
        return {
            'error': 'Rate limit exceeded',
            'code': 429
        }
    
    return None  # 继续处理


# 工厂函数
def create_session_manager(storage_type: SessionStorage, **kwargs) -> SessionManager:
    """创建会话管理器"""
    if storage_type == SessionStorage.MEMORY:
        return MemorySessionManager(**kwargs)
    elif storage_type == SessionStorage.DISTRIBUTED_CACHE:
        cache_client = kwargs.get('cache_client')
        if not cache_client:
            raise ValueError("cache_client is required for distributed session manager")
        return DistributedSessionManager(cache_client, **kwargs)
    else:
        raise ValueError(f"Unsupported session storage type: {storage_type}")


async def create_translation_service(session_storage: SessionStorage = SessionStorage.MEMORY,
                                   **kwargs) -> TranslationStatelessService:
    """创建翻译服务实例"""
    session_manager = create_session_manager(session_storage, **kwargs)
    service = TranslationStatelessService(session_manager)
    
    # 添加中间件
    service.add_middleware(authentication_middleware)
    service.add_middleware(rate_limiting_middleware)
    
    await service.start()
    return service