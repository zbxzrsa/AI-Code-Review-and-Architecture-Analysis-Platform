"""
API Response Caching Middleware
"""

import json
import hashlib
from typing import Any, Dict, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)

class ResponseCache:
    """Intelligent response caching with cache invalidation"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 300  # 5 minutes
        
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key from request"""
        key_data = {
            'method': request.method,
            'url': str(request.url),
            'headers': dict(request.headers),
            'body': await request.body() if request.method in ['POST', 'PUT'] else None
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
        
    async def get_cached_response(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        cache_key = self._generate_cache_key(request)
        try:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None
        
    async def cache_response(self, request: Request, response: Dict[str, Any], ttl: int = None) -> None:
        """Cache response data"""
        cache_key = self._generate_cache_key(request)
        ttl = ttl or self.default_ttl
        
        try:
            await self.redis.setex(
                cache_key, 
                ttl, 
                json.dumps(response, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            
    def should_cache_request(self, request: Request) -> bool:
        """Determine if request should be cached"""
        # Only cache GET requests
        if request.method != 'GET':
            return False
            
        # Don't cache authenticated requests
        if 'authorization' in request.headers:
            return False
            
        # Don't cache API endpoints with side effects
        no_cache_paths = ['/api/v1/auth/', '/api/v1/projects/', '/api/v1/analyze']
        if any(path in str(request.url) for path in no_cache_paths):
            return False
            
        return True

# Cache middleware factory
def create_cache_middleware(cache: ResponseCache):
    async def cache_middleware(request: Request, call_next):
        if not cache.should_cache_request(request):
            return await call_next(request)
            
        # Try to get from cache
        cached_response = await cache.get_cached_response(request)
        if cached_response:
            return JSONResponse(
                content=cached_response['content'],
                status_code=cached_response['status_code'],
                headers=cached_response['headers']
            )
            
        # Get fresh response
        response = await call_next(request)
        
        # Cache successful GET responses
        if response.status_code == 200:
            await cache.cache_response(request, {
                'content': response.body,
                'status_code': response.status_code,
                'headers': dict(response.headers)
            })
            
        return response
        
    return cache_middleware