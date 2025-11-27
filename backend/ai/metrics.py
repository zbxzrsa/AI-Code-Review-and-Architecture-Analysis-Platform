import time
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging

# Prometheus metrics
ai_requests_total = Counter(
    'ai_requests_total',
    'Total AI requests',
    ['channel', 'model', 'cache_hit', 'status']
)

ai_latency_seconds = Histogram(
    'ai_latency_seconds',
    'AI request latency in seconds',
    ['channel', 'model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

ai_tokens_out_total = Counter(
    'ai_tokens_out_total',
    'Total tokens generated',
    ['channel', 'model']
)

ai_cache_size = Gauge(
    'ai_cache_size',
    'AI cache size',
    ['cache_type']
)

ai_eval_score = Gauge(
    'ai_eval_score',
    'AI evaluation score',
    ['channel', 'dataset']
)

class StructuredLogger:
    def __init__(self, service_name: str = "ai-backend"):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        
        # Configure structured logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        channel: str,
        model: str,
        latency_ms: float,
        cache_hit: bool,
        tokens_out: int,
        status: str,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log AI request with structured data"""
        log_entry = {
            "ts": datetime.utcnow().isoformat(),
            "level": "info",
            "request_id": request_id,
            "service": self.service_name,
            "event": "ai_request",
            "method": method,
            "path": path,
            "channel": channel,
            "model": model,
            "latency_ms": round(latency_ms, 2),
            "cache_hit": cache_hit,
            "tokens_out": tokens_out,
            "status": status
        }
        
        if client_ip:
            log_entry["client_ip"] = client_ip
        if user_agent:
            log_entry["user_agent"] = user_agent
        
        self.logger.info(json.dumps(log_entry))
        
        # Update Prometheus metrics
        ai_requests_total.labels(
            channel=channel,
            model=model,
            cache_hit=str(cache_hit),
            status=status
        ).inc()
        
        ai_latency_seconds.labels(
            channel=channel,
            model=model
        ).observe(latency_ms / 1000.0)
        
        if tokens_out > 0:
            ai_tokens_out_total.labels(
                channel=channel,
                model=model
            ).inc(tokens_out)
    
    def log_error(
        self,
        request_id: str,
        error: str,
        channel: Optional[str] = None,
        model: Optional[str] = None,
        stack_trace: Optional[str] = None
    ):
        """Log error with structured data"""
        log_entry = {
            "ts": datetime.utcnow().isoformat(),
            "level": "error",
            "request_id": request_id,
            "service": self.service_name,
            "event": "ai_error",
            "error": error,
            "channel": channel,
            "model": model
        }
        
        if stack_trace:
            log_entry["stack_trace"] = stack_trace
        
        self.logger.error(json.dumps(log_entry))
    
    def log_security_event(
        self,
        request_id: str,
        event_type: str,
        details: Dict[str, Any],
        channel: Optional[str] = None
    ):
        """Log security events"""
        log_entry = {
            "ts": datetime.utcnow().isoformat(),
            "level": "warn",
            "request_id": request_id,
            "service": self.service_name,
            "event": "security_event",
            "event_type": event_type,
            "channel": channel,
            "details": details
        }
        
        self.logger.warning(json.dumps(log_entry))
    
    def log_eval_run(
        self,
        channel: str,
        dataset: str,
        score: float,
        total_items: int,
        successful_items: int,
        avg_latency_ms: float
    ):
        """Log evaluation run"""
        log_entry = {
            "ts": datetime.utcnow().isoformat(),
            "level": "info",
            "service": self.service_name,
            "event": "eval_run",
            "channel": channel,
            "dataset": dataset,
            "score": round(score, 3),
            "total_items": total_items,
            "successful_items": successful_items,
            "avg_latency_ms": round(avg_latency_ms, 2)
        }
        
        self.logger.info(json.dumps(log_entry))
        
        # Update Prometheus gauge
        ai_eval_score.labels(
            channel=channel,
            dataset=dataset
        ).set(score)
    
    def log_cache_metrics(self, cache_type: str, size: int, hits: int, misses: int):
        """Log cache performance metrics"""
        log_entry = {
            "ts": datetime.utcnow().isoformat(),
            "level": "info",
            "service": self.service_name,
            "event": "cache_metrics",
            "cache_type": cache_type,
            "size": size,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / (hits + misses) * 100, 2) if (hits + misses) > 0 else 0
        }
        
        self.logger.info(json.dumps(log_entry))
        
        # Update Prometheus gauge
        ai_cache_size.labels(cache_type=cache_type).set(size)

# Global structured logger
structured_logger = StructuredLogger()

def generate_request_id() -> str:
    """Generate unique request ID"""
    return str(uuid.uuid4())

def get_metrics() -> str:
    """Get Prometheus metrics in text format"""
    return generate_latest()

class MetricsMiddleware:
    """FastAPI middleware for metrics collection"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            request_id = generate_request_id()
            
            # Add request_id to scope for downstream use
            scope["request_id"] = request_id
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Calculate latency
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Log request completion
                    structured_logger.log_request(
                        request_id=request_id,
                        method=scope["method"],
                        path=scope["path"],
                        channel="unknown",  # Will be set by endpoint
                        model="unknown",     # Will be set by endpoint
                        latency_ms=latency_ms,
                        cache_hit=False,
                        tokens_out=0,
                        status=str(message["status"])
                    )
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)