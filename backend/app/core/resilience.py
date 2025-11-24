import asyncio
import functools
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast
from app.core.logger import logger

# 定义类型变量
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

class CircuitState(str, Enum):
    """熔断器状态枚举"""
    CLOSED = "CLOSED"  # 正常状态，允许请求通过
    OPEN = "OPEN"      # 熔断状态，拒绝所有请求
    HALF_OPEN = "HALF_OPEN"  # 半开状态，允许部分请求通过以测试服务是否恢复


class CircuitBreaker:
    """
    熔断器实现
    当失败率达到阈值时，熔断器打开，阻止请求通过，防止级联失败
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
    
    def allow_request(self) -> bool:
        """检查是否允许请求通过"""
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            # 检查是否达到恢复超时时间
            if current_time - self.last_failure_time >= self.recovery_timeout:
                logger.info(f"熔断器 {self.name} 进入半开状态")
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        
        elif self.state == CircuitState.HALF_OPEN:
            # 在半开状态下限制请求数量
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        # 闭合状态，允许所有请求
        return True
    
    def record_success(self) -> None:
        """记录成功请求"""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"熔断器 {self.name} 恢复正常，关闭熔断")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
    
    def record_failure(self) -> None:
        """记录失败请求"""
        current_time = time.time()
        self.last_failure_time = current_time
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"熔断器 {self.name} 在半开状态下检测到失败，重新打开熔断")
            self.state = CircuitState.OPEN
            return
        
        self.failure_count += 1
        
        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            logger.warning(f"熔断器 {self.name} 失败次数达到阈值 {self.failure_threshold}，打开熔断")
            self.state = CircuitState.OPEN


class RetryConfig:
    """重试配置"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter


class ResilienceDecorators:
    """弹性装饰器集合，提供重试和熔断功能"""
    
    _circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    @classmethod
    def get_circuit_breaker(cls, name: str) -> CircuitBreaker:
        """获取或创建熔断器实例"""
        if name not in cls._circuit_breakers:
            cls._circuit_breakers[name] = CircuitBreaker(name)
        return cls._circuit_breakers[name]
    
    @classmethod
    def retry(
        cls,
        retryable_exceptions: List[type] = None,
        retry_config: RetryConfig = None,
        circuit_breaker_name: Optional[str] = None
    ):
        """
        重试装饰器
        在遇到指定异常时自动重试函数调用
        """
        retryable_exceptions = retryable_exceptions or [Exception]
        retry_config = retry_config or RetryConfig()
        
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                circuit_breaker = None
                if circuit_breaker_name:
                    circuit_breaker = cls.get_circuit_breaker(circuit_breaker_name)
                
                retry_count = 0
                last_exception = None
                
                while retry_count <= retry_config.max_retries:
                    try:
                        # 检查熔断器状态
                        if circuit_breaker and not circuit_breaker.allow_request():
                            logger.warning(f"熔断器 {circuit_breaker_name} 已打开，拒绝请求")
                            raise Exception(f"服务 {circuit_breaker_name} 暂时不可用，请稍后重试")
                        
                        # 执行函数
                        result = await func(*args, **kwargs)
                        
                        # 记录成功
                        if circuit_breaker:
                            circuit_breaker.record_success()
                        
                        return result
                    
                    except tuple(retryable_exceptions) as e:
                        last_exception = e
                        retry_count += 1
                        
                        # 记录失败
                        if circuit_breaker:
                            circuit_breaker.record_failure()
                        
                        # 达到最大重试次数，抛出异常
                        if retry_count > retry_config.max_retries:
                            logger.error(f"重试次数达到上限 {retry_config.max_retries}，放弃重试", extra={
                                "function": func.__name__,
                                "exception": str(e),
                                "retry_count": retry_count
                            })
                            raise
                        
                        # 计算退避时间
                        backoff = min(
                            retry_config.initial_backoff * (retry_config.backoff_multiplier ** (retry_count - 1)),
                            retry_config.max_backoff
                        )
                        
                        # 添加抖动以避免惊群效应
                        if retry_config.jitter:
                            import random
                            backoff = backoff * (0.5 + random.random())
                        
                        logger.warning(f"操作失败，将在 {backoff:.2f} 秒后重试 ({retry_count}/{retry_config.max_retries})", extra={
                            "function": func.__name__,
                            "exception": str(e),
                            "retry_count": retry_count,
                            "backoff": backoff
                        })
                        
                        # 等待退避时间
                        await asyncio.sleep(backoff)
                
                # 不应该到达这里，但为了类型检查添加
                assert last_exception is not None
                raise last_exception
            
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                circuit_breaker = None
                if circuit_breaker_name:
                    circuit_breaker = cls.get_circuit_breaker(circuit_breaker_name)
                
                retry_count = 0
                last_exception = None
                
                while retry_count <= retry_config.max_retries:
                    try:
                        # 检查熔断器状态
                        if circuit_breaker and not circuit_breaker.allow_request():
                            logger.warning(f"熔断器 {circuit_breaker_name} 已打开，拒绝请求")
                            raise Exception(f"服务 {circuit_breaker_name} 暂时不可用，请稍后重试")
                        
                        # 执行函数
                        result = func(*args, **kwargs)
                        
                        # 记录成功
                        if circuit_breaker:
                            circuit_breaker.record_success()
                        
                        return result
                    
                    except tuple(retryable_exceptions) as e:
                        last_exception = e
                        retry_count += 1
                        
                        # 记录失败
                        if circuit_breaker:
                            circuit_breaker.record_failure()
                        
                        # 达到最大重试次数，抛出异常
                        if retry_count > retry_config.max_retries:
                            logger.error(f"重试次数达到上限 {retry_config.max_retries}，放弃重试", extra={
                                "function": func.__name__,
                                "exception": str(e),
                                "retry_count": retry_count
                            })
                            raise
                        
                        # 计算退避时间
                        backoff = min(
                            retry_config.initial_backoff * (retry_config.backoff_multiplier ** (retry_count - 1)),
                            retry_config.max_backoff
                        )
                        
                        # 添加抖动以避免惊群效应
                        if retry_config.jitter:
                            import random
                            backoff = backoff * (0.5 + random.random())
                        
                        logger.warning(f"操作失败，将在 {backoff:.2f} 秒后重试 ({retry_count}/{retry_config.max_retries})", extra={
                            "function": func.__name__,
                            "exception": str(e),
                            "retry_count": retry_count,
                            "backoff": backoff
                        })
                        
                        # 等待退避时间
                        time.sleep(backoff)
                
                # 不应该到达这里，但为了类型检查添加
                assert last_exception is not None
                raise last_exception
            
            # 根据函数是否为协程函数选择包装器
            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            else:
                return cast(F, sync_wrapper)
        
        return decorator
    
    @classmethod
    def circuit_breaker(cls, name: str, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        """
        熔断器装饰器
        当失败次数达到阈值时，阻止请求通过一段时间
        """
        def decorator(func: F) -> F:
            circuit_breaker = cls.get_circuit_breaker(name)
            circuit_breaker.failure_threshold = failure_threshold
            circuit_breaker.recovery_timeout = recovery_timeout
            
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not circuit_breaker.allow_request():
                    logger.warning(f"熔断器 {name} 已打开，拒绝请求")
                    raise Exception(f"服务 {name} 暂时不可用，请稍后重试")
                
                try:
                    result = await func(*args, **kwargs)
                    circuit_breaker.record_success()
                    return result
                except Exception as e:
                    circuit_breaker.record_failure()
                    raise
            
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not circuit_breaker.allow_request():
                    logger.warning(f"熔断器 {name} 已打开，拒绝请求")
                    raise Exception(f"服务 {name} 暂时不可用，请稍后重试")
                
                try:
                    result = func(*args, **kwargs)
                    circuit_breaker.record_success()
                    return result
                except Exception as e:
                    circuit_breaker.record_failure()
                    raise
            
            # 根据函数是否为协程函数选择包装器
            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            else:
                return cast(F, sync_wrapper)
        
        return decorator
    
    @classmethod
    def timeout(cls, seconds: float):
        """
        超时装饰器
        在指定时间后取消操作
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
                except asyncio.TimeoutError:
                    logger.error(f"操作超时 ({seconds}秒)", extra={
                        "function": func.__name__,
                        "timeout": seconds
                    })
                    raise Exception(f"操作超时，请稍后重试")
            
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                import signal
                
                def timeout_handler(signum, frame):
                    raise Exception(f"操作超时 ({seconds}秒)")
                
                # 设置信号处理器
                original_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(seconds))
                
                try:
                    return func(*args, **kwargs)
                finally:
                    # 恢复原始信号处理器
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, original_handler)
            
            # 根据函数是否为协程函数选择包装器
            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            else:
                return cast(F, sync_wrapper)
        
        return decorator


# 导出便捷函数
retry = ResilienceDecorators.retry
circuit_breaker = ResilienceDecorators.circuit_breaker
timeout = ResilienceDecorators.timeout