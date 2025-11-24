from fastapi import HTTPException, Request
from typing import Optional, Callable, Dict, Any
import time
import asyncio
from datetime import datetime
import json
from functools import wraps

# 简单的内存缓存实现
class Cache:
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """获取缓存值"""
        if key in self.cache:
            item = self.cache[key]
            if item["expire"] > time.time():
                return item["value"]
            else:
                # 过期缓存清理
                del self.cache[key]
        return None
    
    async def set(self, key: str, value: Dict[str, Any], expire: int = 3600) -> None:
        """设置缓存值"""
        self.cache[key] = {
            "value": value,
            "expire": time.time() + expire
        }
    
    async def delete(self, key: str) -> None:
        """删除缓存值"""
        if key in self.cache:
            del self.cache[key]
    
    async def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()

# 全局缓存实例
_cache = Cache()

def get_cache() -> Cache:
    """获取缓存实例"""
    return _cache

# 简单的内存限流器实现
class RateLimiter:
    def __init__(self):
        self.requests: Dict[str, Dict[str, Any]] = {}
    
    async def is_rate_limited(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """检查是否超过限流阈值"""
        now = time.time()
        
        # 初始化或清理过期请求记录
        if key not in self.requests or self.requests[key]["window_end"] < now:
            self.requests[key] = {
                "count": 0,
                "window_end": now + window_seconds
            }
        
        # 检查请求数量
        if self.requests[key]["count"] >= max_requests:
            return True
        
        # 增加请求计数
        self.requests[key]["count"] += 1
        return False

# 全局限流器实例
_rate_limiter = RateLimiter()

def rate_limiter(max_requests: int = 10, window_seconds: int = 60):
    """
    请求限流依赖
    
    Args:
        max_requests: 窗口期内最大请求数
        window_seconds: 窗口期时长(秒)
    """
    async def _rate_limiter(request: Request):
        # 使用客户端IP作为限流键
        client_ip = request.client.host if request.client else "unknown"
        key = f"{client_ip}:{request.url.path}"
        
        # 检查是否超过限流阈值
        is_limited = await _rate_limiter.is_rate_limited(key, max_requests, window_seconds)
        if is_limited:
            raise HTTPException(
                status_code=429,
                detail=f"请求过于频繁，请在{window_seconds}秒后重试"
            )
    
    return _rate_limiter