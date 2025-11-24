"""
Structured logging and observability for AI Code Review Platform.
"""

import os
import sys
import time
import json
import logging
import logging.handlers
from typing import Any, Dict, Optional, Union
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import traceback

import structlog
from prometheus_client import Counter, Histogram, Gauge, Info
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor


# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Active database connections'
)

AI_ANALYSIS_COUNT = Counter(
    'ai_analysis_total',
    'Total AI analyses',
    ['model', 'language', 'status']
)

AI_ANALYSIS_DURATION = Histogram(
    'ai_analysis_duration_seconds',
    'AI analysis duration',
    ['model', 'language']
)

CACHE_HIT_RATE = Gauge(
    'cache_hit_rate',
    'Cache hit rate'
)

ERROR_COUNT = Counter(
    'errors_total',
    'Total errors',
    ['error_type', 'component']
)

APP_INFO = Info(
    'app_info',
    'Application information'
)


@dataclass
class LogContext:
    """Structured log context."""
    
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    action: Optional[str] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StructuredLogger:
    """Enhanced structured logger with observability."""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self.component = name
        
    def _log(
        self,
        level: str,
        message: str,
        context: Optional[LogContext] = None,
        **kwargs
    ):
        """Log with structured context."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level.upper(),
            'component': self.component,
            'message': message,
        }
        
        if context:
            context_dict = asdict(context)
            # Remove None values
            context_dict = {k: v for k, v in context_dict.items() if v is not None}
            log_data.update(context_dict)
        
        # Add additional kwargs
        log_data.update(kwargs)
        
        # Log with structlog
        getattr(self.logger, level.lower())(message, **log_data)
        
        # Update Prometheus metrics
        if level.upper() == 'ERROR':
            ERROR_COUNT.labels(
                error_type=context.error if context else 'unknown',
                component=self.component
            ).inc()
    
    def info(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log info message."""
        self._log('info', message, context, **kwargs)
    
    def warning(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log warning message."""
        self._log('warning', message, context, **kwargs)
    
    def error(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log error message."""
        self._log('error', message, context, **kwargs)
    
    def debug(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log debug message."""
        self._log('debug', message, context, **kwargs)
    
    def critical(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log critical message."""
        self._log('critical', message, context, **kwargs)
    
    @contextmanager
    def log_execution(
        self,
        action: str,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **metadata
    ):
        """Context manager for logging execution time."""
        start_time = time.time()
        context = LogContext(
            request_id=request_id,
            user_id=user_id,
            component=self.component,
            action=action,
            metadata=metadata
        )
        
        self.info(f"Starting {action}", context=context)
        
        try:
            yield context
            duration_ms = (time.time() - start_time) * 1000
            context.duration_ms = duration_ms
            self.info(f"Completed {action}", context=context)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            context.duration_ms = duration_ms
            context.error = str(e)
            context.stack_trace = traceback.format_exc()
            self.error(f"Failed {action}", context=context)
            raise


class MetricsCollector:
    """Collect and manage application metrics."""
    
    def __init__(self):
        self.setup_app_info()
    
    def setup_app_info(self):
        """Set application info metrics."""
        from app.core.config import settings
        
        APP_INFO.info({
            'version': getattr(settings, 'VERSION', '1.0.0'),
            'environment': getattr(settings, 'ENVIRONMENT', 'development'),
            'build_date': getattr(settings, 'BUILD_DATE', 'unknown'),
        })
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ):
        """Record HTTP request metrics."""
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_ai_analysis(
        self,
        model: str,
        language: str,
        status: str,
        duration: float
    ):
        """Record AI analysis metrics."""
        AI_ANALYSIS_COUNT.labels(
            model=model,
            language=language,
            status=status
        ).inc()
        
        AI_ANALYSIS_DURATION.labels(
            model=model,
            language=language
        ).observe(duration)
    
    def update_cache_hit_rate(self, hit_rate: float):
        """Update cache hit rate metric."""
        CACHE_HIT_RATE.set(hit_rate)
    
    def update_active_connections(self, count: int):
        """Update active connections metric."""
        ACTIVE_CONNECTIONS.set(count)


class TracingManager:
    """Manage distributed tracing."""
    
    def __init__(self, service_name: str = "ai-code-review-backend"):
        self.service_name = service_name
        self.setup_tracing()
    
    def setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        # Configure OTLP exporter if enabled
        otlp_endpoint = os.getenv('OTLP_ENDPOINT')
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Instrument FastAPI
        FastAPIInstrumentor.instrument()
        
        # Instrument database clients
        SQLAlchemyInstrumentor.instrument()
        RedisInstrumentor.instrument()
        HTTPXClientInstrumentor.instrument()
    
    def get_current_span(self):
        """Get current tracing span."""
        return trace.get_current_span()
    
    def add_span_attribute(self, key: str, value: Union[str, int, float, bool]):
        """Add attribute to current span."""
        span = self.get_current_span()
        if span:
            span.set_attribute(key, value)
    
    def add_span_event(self, name: str, attributes: Dict[str, Any]):
        """Add event to current span."""
        span = self.get_current_span()
        if span:
            span.add_event(name, attributes)


class LogstashHandler(logging.Handler):
    """Custom Logstash handler with structured formatting."""
    
    def __init__(self, host: str, port: int, service_name: str):
        super().__init__()
        self.host = host
        self.port = port
        self.service_name = service_name
        
        # Try to import logstash
        try:
            import logstash
            self.logstash_handler = logstash.TCPLogstashHandler(
                host, port, version=1
            )
        except ImportError:
            self.logstash_handler = None
            print("Warning: logstash not available, using fallback handler")
    
    def emit(self, record):
        """Emit log record to Logstash."""
        if self.logstash_handler:
            # Add service name to record
            record.service = self.service_name
            self.logstash_handler.emit(record)
        else:
            # Fallback to stderr
            print(json.dumps({
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'service': self.service_name,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
            }))


def setup_logging(
    service_name: str = "ai-code-review-backend",
    log_level: str = "INFO",
    enable_logstash: bool = False,
    logstash_host: str = "localhost",
    logstash_port: int = 5000
) -> StructuredLogger:
    """Setup structured logging for the application."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Add Logstash handler if enabled
    if enable_logstash:
        logstash_handler = LogstashHandler(
            logstash_host, logstash_port, service_name
        )
        logging.getLogger().addHandler(logstash_handler)
    
    return StructuredLogger(service_name)


def setup_observability(
    service_name: str = "ai-code-review-backend",
    enable_tracing: bool = False,
    enable_metrics: bool = True
):
    """Setup complete observability stack."""
    
    # Setup logging
    logger = setup_logging(service_name)
    
    # Setup metrics
    metrics = None
    if enable_metrics:
        metrics = MetricsCollector()
    
    # Setup tracing
    tracer = None
    if enable_tracing:
        tracer = TracingManager(service_name)
    
    return logger, metrics, tracer


# Utility functions
def get_correlation_id(request=None) -> Optional[str]:
    """Extract correlation ID from request or generate new one."""
    if request:
        return getattr(request.state, 'correlation_id', None)
    return None


def create_log_context(
    request=None,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    **metadata
) -> LogContext:
    """Create log context from request and additional data."""
    return LogContext(
        request_id=get_correlation_id(request),
        user_id=user_id,
        session_id=getattr(request.state, 'session_id', None) if request else None,
        component='api',
        action=action,
        metadata=metadata
    )


# Performance monitoring decorator
def monitor_performance(logger: StructuredLogger, action: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with logger.log_execution(action=action):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Error tracking
def track_error(
    logger: StructuredLogger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None
):
    """Track and log errors with context."""
    log_context = LogContext(
        error=str(error),
        stack_trace=traceback.format_exc(),
        metadata=context
    )
    logger.error(f"Error occurred: {error}", context=log_context)