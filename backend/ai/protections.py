import time
import hashlib
from typing import Dict, Optional
from fastapi import Request, HTTPException, Header
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os

# Rate limiter using Redis (or in-memory for dev)
limiter = Limiter(key_func=get_remote_address)

class HTTPProtections:
    def __init__(self):
        self.idempotency_store: Dict[str, Dict] = {}
        self.etag_store: Dict[str, Dict] = {}
    
    def generate_etag(self, content: str) -> str:
        """Generate ETag from content"""
        return f'"{hashlib.sha256(content.encode()).hexdigest()}"'
    
    def check_idempotency(self, idempotency_key: str, prompt_hash: str) -> Optional[Dict]:
        """Check if idempotent request exists"""
        key = f"{idempotency_key}:{prompt_hash}"
        stored = self.idempotency_store.get(key)
        
        if stored and stored['expires_at'] > time.time():
            return stored['response']
        
        return None
    
    def store_idempotent_response(self, idempotency_key: str, prompt_hash: str, response: Dict, ttl: int = 300):
        """Store idempotent response"""
        key = f"{idempotency_key}:{prompt_hash}"
        self.idempotency_store[key] = {
            'response': response,
            'expires_at': time.time() + ttl
        }
    
    def check_etag(self, etag: str, content_hash: str) -> bool:
        """Check if ETag matches"""
        stored = self.etag_store.get(content_hash)
        if stored and stored['expires_at'] > time.time():
            return stored['etag'] == etag
        return False
    
    def store_etag(self, content_hash: str, etag: str, ttl: int = 30):
        """Store ETag for content"""
        self.etag_store[content_hash] = {
            'etag': etag,
            'expires_at': time.time() + ttl
        }

http_protections = HTTPProtections()

# Rate limit decorator
@limiter.limit(f"{os.getenv('RATE_LIMIT_REQUESTS_PER_MINUTE', '60')}/minute")
def rate_limit_protected():
    pass

def get_client_ip(request: Request) -> str:
    """Get client IP from request"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def create_rate_limit_response(exc: RateLimitExceeded) -> HTTPException:
    """Create rate limit response"""
    return HTTPException(
        status_code=429,
        detail="Rate limit exceeded",
        headers={"Retry-After": str(exc.detail["retry-after"])}
    )