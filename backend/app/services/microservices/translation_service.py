"""
翻译微服务 - 核心翻译服务接口定义
支持水平扩展和插件化架构
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """服务状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class TranslationPriority(Enum):
    """翻译优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TranslationRequest:
    """翻译请求数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Union[str, List[str]] = ""
    source_lang: str = "auto"
    target_lang: str = "zh"
    priority: TranslationPriority = TranslationPriority.NORMAL
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout: int = 30  # 超时时间（秒）
    
    def __post_init__(self):
        if isinstance(self.content, str):
            self.content = [self.content]


@dataclass
class TranslationResponse:
    """翻译响应数据结构"""
    request_id: str
    translated_content: List[str]
    source_lang: str
    target_lang: str
    confidence_scores: List[float] = field(default_factory=list)
    processing_time: float = 0.0
    engine_used: str = ""
    cache_hit: bool = False
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class ServiceHealth:
    """服务健康状态"""
    status: ServiceStatus
    response_time: float
    cpu_usage: float
    memory_usage: float
    active_requests: int
    error_rate: float
    last_check: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class TranslationServiceInterface(ABC):
    """翻译服务接口 - 定义所有翻译服务必须实现的方法"""
    
    @abstractmethod
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """执行翻译任务"""
        pass
    
    @abstractmethod
    async def batch_translate(self, requests: List[TranslationRequest]) -> List[TranslationResponse]:
        """批量翻译"""
        pass
    
    @abstractmethod
    async def get_health(self) -> ServiceHealth:
        """获取服务健康状态"""
        pass
    
    @abstractmethod
    async def get_supported_languages(self) -> Dict[str, List[str]]:
        """获取支持的语言对"""
        pass


class BaseTranslationService(TranslationServiceInterface):
    """翻译服务基类 - 提供通用功能实现"""
    
    def __init__(self, service_id: str, config: Dict[str, Any]):
        self.service_id = service_id
        self.config = config
        self.active_requests: Dict[str, TranslationRequest] = {}
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.now()
        
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """翻译实现模板方法"""
        start_time = datetime.now()
        self.active_requests[request.id] = request
        self.request_count += 1
        
        try:
            # 预处理
            processed_request = await self._preprocess_request(request)
            
            # 执行翻译
            response = await self._execute_translation(processed_request)
            
            # 后处理
            final_response = await self._postprocess_response(response)
            
            # 计算处理时间
            final_response.processing_time = (datetime.now() - start_time).total_seconds()
            
            return final_response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Translation failed for request {request.id}: {e}")
            raise
        finally:
            self.active_requests.pop(request.id, None)
    
    async def batch_translate(self, requests: List[TranslationRequest]) -> List[TranslationResponse]:
        """批量翻译实现"""
        # 并发处理，但限制并发数
        semaphore = asyncio.Semaphore(self.config.get('max_concurrent_requests', 10))
        
        async def translate_with_semaphore(req):
            async with semaphore:
                return await self.translate(req)
        
        tasks = [translate_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_health(self) -> ServiceHealth:
        """获取服务健康状态"""
        import psutil
        
        # 计算错误率
        error_rate = self.error_count / max(self.request_count, 1)
        
        # 获取系统资源使用情况
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # 模拟响应时间检查
        start = datetime.now()
        await asyncio.sleep(0.001)  # 模拟健康检查
        response_time = (datetime.now() - start).total_seconds() * 1000
        
        # 确定服务状态
        if error_rate > 0.1 or cpu_usage > 90 or memory_usage > 90:
            status = ServiceStatus.UNHEALTHY
        elif error_rate > 0.05 or cpu_usage > 70 or memory_usage > 70:
            status = ServiceStatus.DEGRADED
        else:
            status = ServiceStatus.HEALTHY
        
        return ServiceHealth(
            status=status,
            response_time=response_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_requests=len(self.active_requests),
            error_rate=error_rate,
            details={
                'service_id': self.service_id,
                'uptime': (datetime.now() - self.start_time).total_seconds(),
                'total_requests': self.request_count,
                'total_errors': self.error_count
            }
        )
    
    @abstractmethod
    async def _preprocess_request(self, request: TranslationRequest) -> TranslationRequest:
        """预处理请求 - 子类实现"""
        pass
    
    @abstractmethod
    async def _execute_translation(self, request: TranslationRequest) -> TranslationResponse:
        """执行翻译 - 子类实现"""
        pass
    
    @abstractmethod
    async def _postprocess_response(self, response: TranslationResponse) -> TranslationResponse:
        """后处理响应 - 子类实现"""
        pass


class TranslationServiceRegistry:
    """翻译服务注册中心"""
    
    def __init__(self):
        self.services: Dict[str, TranslationServiceInterface] = {}
        self.service_weights: Dict[str, float] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
    
    def register_service(self, service_id: str, service: TranslationServiceInterface, weight: float = 1.0):
        """注册翻译服务"""
        self.services[service_id] = service
        self.service_weights[service_id] = weight
        logger.info(f"Registered translation service: {service_id}")
    
    def unregister_service(self, service_id: str):
        """注销翻译服务"""
        self.services.pop(service_id, None)
        self.service_weights.pop(service_id, None)
        self.service_health.pop(service_id, None)
        logger.info(f"Unregistered translation service: {service_id}")
    
    async def get_healthy_services(self) -> List[str]:
        """获取健康的服务列表"""
        healthy_services = []
        
        for service_id, service in self.services.items():
            try:
                health = await service.get_health()
                self.service_health[service_id] = health
                
                if health.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]:
                    healthy_services.append(service_id)
            except Exception as e:
                logger.error(f"Health check failed for service {service_id}: {e}")
        
        return healthy_services
    
    async def select_service(self, request: TranslationRequest) -> Optional[str]:
        """根据负载均衡策略选择服务"""
        healthy_services = await self.get_healthy_services()
        
        if not healthy_services:
            return None
        
        # 基于权重和健康状态的选择策略
        best_service = None
        best_score = float('inf')
        
        for service_id in healthy_services:
            health = self.service_health.get(service_id)
            if not health:
                continue
            
            # 计算服务评分（越低越好）
            weight = self.service_weights.get(service_id, 1.0)
            load_factor = health.active_requests / weight
            error_penalty = health.error_rate * 100
            
            score = load_factor + error_penalty
            
            if score < best_score:
                best_score = score
                best_service = service_id
        
        return best_service


# 全局服务注册中心实例
translation_service_registry = TranslationServiceRegistry()