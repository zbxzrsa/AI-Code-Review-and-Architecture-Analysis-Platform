"""
AI模型推理优化器
支持模型量化、批处理、内存管理和性能监控
"""
import asyncio
import gc
import hashlib
import json
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

logger = logging.getLogger(__name__)

class ModelType(Enum):
    EMBEDDING = "embedding"
    DEFECT_DETECTION = "defect_detection"
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    SIMILARITY = "similarity"

@dataclass
class ModelConfig:
    model_type: ModelType
    model_path: str
    max_batch_size: int = 32
    max_memory_mb: int = 2048
    enable_quantization: bool = True
    enable_caching: bool = True
    timeout_seconds: int = 30

@dataclass
class InferenceMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    cache_hit_rate: float = 0.0
    model_load_time_ms: float = 0.0

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0
        self.memory_history = []
        self.lock = threading.Lock()
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def check_memory_limit(self) -> bool:
        """检查是否超过内存限制"""
        current_memory = self.get_memory_usage()
        with self.lock:
            self.memory_history.append(current_memory)
            if len(self.memory_history) > 100:
                self.memory_history.pop(0)
        
        return current_memory > self.max_memory_mb
    
    def force_garbage_collection(self) -> None:
        """强制垃圾回收"""
        gc.collect()
        logger.info("Forced garbage collection")
    
    def optimize_memory(self) -> None:
        """内存优化"""
        if self.check_memory_limit():
            logger.warning(f"Memory limit exceeded: {self.get_memory_usage():.2f}MB")
            self.force_garbage_collection()

class ModelCache:
    """模型缓存管理"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # 移除最少使用的项
                oldest_key = min(self.access_times.keys(), 
                               key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

class BatchProcessor:
    """批处理器"""
    
    def __init__(self, max_batch_size: int = 32, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = asyncio.Queue()
        self.processing = False
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def add_request(self, request_data: Any) -> Any:
        """添加请求到批处理队列"""
        future = asyncio.Future()
        await self.queue.put((request_data, future))
        
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self) -> None:
        """处理批次"""
        if self.processing:
            return
        
        self.processing = True
        try:
            while not self.queue.empty():
                batch = []
                futures = []
                
                # 收集批次
                start_time = time.time()
                while (len(batch) < self.max_batch_size and 
                       time.time() - start_time < self.max_wait_time):
                    try:
                        request_data, future = await asyncio.wait_for(
                            self.queue.get(), timeout=0.01
                        )
                        batch.append(request_data)
                        futures.append(future)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    # 并行处理批次
                    results = await self._process_batch_parallel(batch)
                    
                    # 设置结果
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)
        
        finally:
            self.processing = False
    
    async def _process_batch_parallel(self, batch: List[Any]) -> List[Any]:
        """并行处理批次"""
        loop = asyncio.get_event_loop()
        tasks = []
        
        for item in batch:
            task = loop.run_in_executor(
                self.executor, self._process_single_item, item
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def _process_single_item(self, item: Any) -> Any:
        """处理单个项目（子类实现）"""
        raise NotImplementedError

class OptimizedAIModel:
    """优化的AI模型基类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.memory_manager = MemoryManager(config.max_memory_mb)
        self.cache = ModelCache() if config.enable_caching else None
        self.batch_processor = BatchProcessor(config.max_batch_size)
        self.metrics = InferenceMetrics()
        self.model_loaded = False
        self.load_lock = threading.Lock()
    
    async def load_model(self) -> None:
        """加载模型"""
        if self.model_loaded:
            return
        
        with self.load_lock:
            if self.model_loaded:
                return
            
            start_time = time.time()
            logger.info(f"Loading model: {self.config.model_path}")
            
            try:
                # 模拟模型加载（实际实现中会加载真实模型）
                await self._load_model_impl()
                
                load_time = (time.time() - start_time) * 1000
                self.metrics.model_load_time_ms = load_time
                self.model_loaded = True
                
                logger.info(f"Model loaded in {load_time:.2f}ms")
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    async def _load_model_impl(self) -> None:
        """模型加载实现（子类实现）"""
        # 模拟加载
        await asyncio.sleep(0.1)
        self.model = "mock_model"
    
    async def predict(self, inputs: Any, use_cache: bool = True) -> Any:
        """预测接口"""
        start_time = time.time()
        self.metrics.total_requests += 1
        
        try:
            # 检查缓存
            if use_cache and self.cache:
                cache_key = self._generate_cache_key(inputs)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self.metrics.cache_hit_rate = (
                        (self.metrics.cache_hit_rate * (self.metrics.total_requests - 1) + 1) /
                        self.metrics.total_requests
                    )
                    return cached_result
            
            # 确保模型已加载
            await self.load_model()
            
            # 内存优化
            self.memory_manager.optimize_memory()
            
            # 执行推理
            result = await self._predict_impl(inputs)
            
            # 缓存结果
            if use_cache and self.cache:
                self.cache.set(cache_key, result)
            
            # 更新指标
            self.metrics.successful_requests += 1
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.avg_latency_ms = (
                (self.metrics.avg_latency_ms * (self.metrics.successful_requests - 1) + latency_ms) /
                self.metrics.successful_requests
            )
            
            # 更新内存峰值
            current_memory = self.memory_manager.get_memory_usage()
            self.metrics.peak_memory_mb = max(
                self.metrics.peak_memory_mb, current_memory
            )
            
            return result
            
        except Exception as e:
            self.metrics.failed_requests += 1
            logger.error(f"Prediction failed: {e}")
            raise
    
    async def _predict_impl(self, inputs: Any) -> Any:
        """预测实现（子类实现）"""
        raise NotImplementedError
    
    def _generate_cache_key(self, inputs: Any) -> str:
        """生成缓存键"""
        if isinstance(inputs, (str, int, float, bool)):
            return f"{self.config.model_type.value}:{inputs}"
        elif isinstance(inputs, (list, dict)):
            content = json.dumps(inputs, sort_keys=True)
            return hashlib.md5(content.encode()).hexdigest()
        else:
            return hashlib.md5(str(inputs).encode()).hexdigest()
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "model_type": self.config.model_type.value,
            "total_requests": self.metrics.total_requests,
            "success_rate": (
                self.metrics.successful_requests / max(self.metrics.total_requests, 1)
            ),
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "peak_memory_mb": self.metrics.peak_memory_mb,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "model_load_time_ms": self.metrics.model_load_time_ms,
            "current_memory_mb": self.memory_manager.get_memory_usage(),
        }
    
    async def unload_model(self) -> None:
        """卸载模型释放内存"""
        if self.model_loaded:
            self.model = None
            self.model_loaded = False
            self.memory_manager.force_garbage_collection()
            logger.info("Model unloaded and memory freed")

class OptimizedEmbeddingModel(OptimizedAIModel):
    """优化的嵌入模型"""
    
    async def _load_model_impl(self) -> None:
        """加载嵌入模型"""
        # 模拟加载嵌入模型
        await asyncio.sleep(0.2)
        self.model = {
            "type": "embedding",
            "dimension": 768,
            "max_length": 512
        }
    
    async def _predict_impl(self, inputs: Any) -> List[float]:
        """嵌入预测"""
        # 模拟嵌入计算
        if isinstance(inputs, dict):
            # 基于代码结构生成嵌入
            content_hash = hashlib.md5(
                json.dumps(inputs, sort_keys=True).encode()
            ).hexdigest()
            np.random.seed(int(content_hash[:8], 16))
        else:
            np.random.seed(hash(str(inputs)) % (2**32))
        
        return list(np.random.rand(768).astype(float))

class OptimizedDefectModel(OptimizedAIModel):
    """优化的缺陷检测模型"""
    
    async def _load_model_impl(self) -> None:
        """加载缺陷检测模型"""
        await asyncio.sleep(0.3)
        self.model = {
            "type": "defect_detection",
            "categories": ["security", "performance", "maintainability"],
            "confidence_threshold": 0.7
        }
    
    async def _predict_impl(self, inputs: Any) -> List[Dict[str, Any]]:
        """缺陷检测预测"""
        # 模拟缺陷检测
        code = inputs.get("code", "") if isinstance(inputs, dict) else str(inputs)
        defects = []
        
        # 简单的规则检测
        if "password" in code.lower():
            defects.append({
                "type": "hardcoded_password",
                "confidence": 0.9,
                "line": code.find("password") // code.count('\n') + 1
            })
        
        if len(code.split('\n')) > 100:
            defects.append({
                "type": "long_function",
                "confidence": 0.7,
                "line": 1
            })
        
        return defects

class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.models: Dict[ModelType, OptimizedAIModel] = {}
        self.model_configs: Dict[ModelType, ModelConfig] = {}
    
    def register_model(self, model_type: ModelType, config: ModelConfig) -> None:
        """注册模型"""
        if model_type == ModelType.EMBEDDING:
            model = OptimizedEmbeddingModel(config)
        elif model_type == ModelType.DEFECT_DETECTION:
            model = OptimizedDefectModel(config)
        else:
            model = OptimizedAIModel(config)
        
        self.models[model_type] = model
        self.model_configs[model_type] = config
        logger.info(f"Registered model: {model_type.value}")
    
    async def get_model(self, model_type: ModelType) -> OptimizedAIModel:
        """获取模型"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type.value} not registered")
        
        model = self.models[model_type]
        await model.load_model()
        return model
    
    async def predict(self, model_type: ModelType, inputs: Any) -> Any:
        """使用指定模型进行预测"""
        model = await self.get_model(model_type)
        return await model.predict(inputs)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有模型的性能指标"""
        metrics = {}
        for model_type, model in self.models.items():
            metrics[model_type.value] = model.get_metrics()
        return metrics
    
    async def unload_all_models(self) -> None:
        """卸载所有模型"""
        for model in self.models.values():
            await model.unload_model()
        logger.info("All models unloaded")

# 全局模型管理器
model_manager = ModelManager()

# 初始化默认模型
async def init_default_models() -> None:
    """初始化默认模型"""
    # 嵌入模型
    model_manager.register_model(
        ModelType.EMBEDDING,
        ModelConfig(
            model_type=ModelType.EMBEDDING,
            model_path="models/embedding",
            max_batch_size=64,
            max_memory_mb=1024
        )
    )
    
    # 缺陷检测模型
    model_manager.register_model(
        ModelType.DEFECT_DETECTION,
        ModelConfig(
            model_type=ModelType.DEFECT_DETECTION,
            model_path="models/defect_detection",
            max_batch_size=32,
            max_memory_mb=1536
        )
    )
    
    logger.info("Default AI models initialized")

# 获取模型管理器
def get_model_manager() -> ModelManager:
    """获取全局模型管理器"""
    return model_manager