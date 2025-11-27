import hashlib
import json
import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os

class AICache:
    def __init__(self):
        # Mock Redis for development - replace with actual Redis client
        self.cache_store: Dict[str, Dict[str, Any]] = {}
        self.batch_window = 0.1  # 100ms batching window
        self.pending_requests: Dict[str, List[asyncio.Future]] = {}
        
    def _make_cache_key(self, channel: str, model: str, prompt_version: str, prompt: str) -> str:
        """Create deterministic cache key"""
        # Normalize prompt
        normalized = " ".join(prompt.lower().strip().split())
        
        # Create hash
        key_data = f"{channel}:{model}:v{prompt_version}:{normalized}"
        return f"ai:{hashlib.sha256(key_data.encode()).hexdigest()}"
    
    async def get(self, channel: str, model: str, prompt_version: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        cache_key = self._make_cache_key(channel, model, prompt_version, prompt)
        cached = self.cache_store.get(cache_key)
        
        if cached and cached.get('expires_at', 0) > time.time():
            data = cached["data"]
            return {
                "result": data["result"],
                "meta": {
                    "cache_hit": True,
                    "cached_at": data["cached_at"],
                    "channel": channel,
                    "model": model,
                    "prompt_version": prompt_version
                }
            }
        
        return None
    
    async def set(self, channel: str, model: str, prompt_version: str, prompt: str, result: str, ttl: int = 60):
        """Set cache entry"""
        cache_key = self._make_cache_key(channel, model, prompt_version, prompt)
        data = {
            "result": result,
            "cached_at": datetime.utcnow().isoformat(),
            "channel": channel,
            "model": model,
            "prompt_version": prompt_version
        }
        
        self.cache_store[cache_key] = {
            "data": data,
            "expires_at": time.time() + ttl
        }
    
    async def batch_request(self, cache_key: str, future: asyncio.Future) -> Optional[asyncio.Future]:
        """Batch identical requests within window"""
        if cache_key not in self.pending_requests:
            self.pending_requests[cache_key] = []
            # Schedule batch execution after window
            asyncio.create_task(self._execute_batch(cache_key))
        
        self.pending_requests[cache_key].append(future)
        return future
    
    async def _execute_batch(self, cache_key: str):
        """Execute batched requests"""
        await asyncio.sleep(self.batch_window)
        
        futures = self.pending_requests.pop(cache_key, [])
        if not futures:
            return
        
        # First request triggers generation, others wait
        first_future = futures[0]
        
        try:
            # The first future should be resolved by the actual generation
            result = await first_future
            
            # Resolve all other futures with the same result
            for future in futures[1:]:
                if not future.done():
                    future.set_result(result)
        except Exception as e:
            # Propagate error to all waiting futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)

# Global cache instance
ai_cache = AICache()