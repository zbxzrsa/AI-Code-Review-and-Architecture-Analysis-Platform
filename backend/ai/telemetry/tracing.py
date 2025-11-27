"""
OpenTelemetry distributed tracing for AI code review system.
Provides comprehensive observability across all components.
"""

import os
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry
try:
    from opentelemetry import trace, baggage, context
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.semconv.trace import SpanKind
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not available, using mock tracing")


class TraceStatus(Enum):
    """Status of trace spans."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of components being traced."""
    AI_ENGINE = "ai_engine"
    SAFETY_JUDGE = "safety_judge"
    POLICY_ENGINE = "policy_engine"
    AUTO_FIX = "auto_fix"
    RAG_INDEX = "rag_index"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    API_ENDPOINT = "api_endpoint"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class TraceSpan:
    """Represents a trace span."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    operation_name: str
    component_type: ComponentType
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: TraceStatus = TraceStatus.UNKNOWN
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time is not None:
            return (self.end_time - self.start_time) * 1000
        return None
    
    def finish(self, status: TraceStatus = TraceStatus.SUCCESS, error: Optional[Exception] = None):
        """Finish the span."""
        self.end_time = time.time()
        self.status = status
        
        if error:
            self.attributes["error.message"] = str(error)
            self.attributes["error.type"] = type(error).__name__
            self.events.append({
                "name": "error",
                "timestamp": time.time(),
                "attributes": {"error": str(error)}
            })


class MockTracer:
    """Mock tracer when OpenTelemetry is not available."""
    
    def __init__(self):
        self.spans = []
    
    def start_span(self, name: str, kind: str = "internal") -> 'MockSpan':
        """Start a new span."""
        return MockSpan(name, self)
    
    def get_current_span(self):
        """Get current span."""
        return None


class MockSpan:
    """Mock span when OpenTelemetry is not available."""
    
    def __init__(self, name: str, tracer):
        self.name = name
        self.tracer = tracer
        self.attributes = {}
        self.events = []
    
    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add event to span."""
        event = {"name": name, "timestamp": time.time()}
        if attributes:
            event["attributes"] = attributes
        self.events.append(event)
    
    def set_status(self, status: TraceStatus):
        """Set span status."""
        self.status = status
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.set_status(TraceStatus.ERROR)
            self.add_event("error", {"error": str(exc_val)})
        else:
            self.set_status(TraceStatus.SUCCESS)


class OpenTelemetryTracer:
    """OpenTelemetry tracer implementation."""
    
    def __init__(self, service_name: str = "ai-code-review", config: Optional[Dict[str, Any]] = None):
        self.service_name = service_name
        self.config = config or {}
        self.spans = []
        
        if OPENTELEMETRY_AVAILABLE:
            self._setup_opentelemetry()
        else:
            self.tracer = MockTracer()
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry tracer."""
        try:
            # Configure Jaeger exporter
            jaeger_endpoint = self.config.get("jaeger_endpoint", "http://localhost:14268/api/traces")
            exporter = JaegerExporter(
                endpoint=jaeger_endpoint,
                collector_endpoint=self.config.get("jaeger_collector", "http://localhost:14268")
            )
            
            # Configure span processor
            span_processor = BatchSpanProcessor(exporter)
            
            # Configure resource
            resource = Resource.create(
                attributes={
                    "service.name": self.service_name,
                    "service.version": self.config.get("service_version", "1.0.0"),
                    "deployment.environment": self.config.get("environment", "development")
                }
            )
            
            # Configure tracer provider
            tracer_provider = TracerProvider(
                resource=resource,
                span_processors=[span_processor]
            )
            
            # Set global tracer
            trace.set_tracer_provider(tracer_provider)
            self.tracer = trace.get_tracer(__name__)
            
            # Setup auto-instrumentation
            self._setup_instrumentation()
            
            logger.info(f"OpenTelemetry configured with Jaeger endpoint: {jaeger_endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry: {e}")
            self.tracer = MockTracer()
    
    def _setup_instrumentation(self):
        """Setup automatic instrumentation."""
        try:
            # FastAPI instrumentation
            if self.config.get("instrument_fastapi", True):
                FastAPIInstrumentor().instrument()
            
            # Requests instrumentation
            if self.config.get("instrument_requests", True):
                RequestsInstrumentor().instrument()
            
            # SQLAlchemy instrumentation
            if self.config.get("instrument_sqlalchemy", True):
                SQLAlchemyInstrumentor().instrument()
            
            logger.info("OpenTelemetry auto-instrumentation enabled")
            
        except Exception as e:
            logger.warning(f"Failed to setup auto-instrumentation: {e}")
    
    def start_span(
        self, 
        name: str, 
        component_type: ComponentType = ComponentType.AI_ENGINE,
        parent_span: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Start a new trace span."""
        if OPENTELEMETRY_AVAILABLE:
            span = self.tracer.start_as_current_span(
                name,
                kind=SpanKind.INTERNAL,
                attributes=attributes or {}
            )
            
            # Add component type attribute
            span.set_attribute("component.type", component_type.value)
            
            return span
        else:
            return self.tracer.start_span(name)
    
    def create_trace_context(
        self, 
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Create trace context for propagation."""
        trace_id = trace_id or str(uuid.uuid4())
        span_id = span_id or str(uuid.uuid4())
        
        return {
            "trace_id": trace_id,
            "span_id": span_id
        }
    
    def extract_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Extract trace context from headers."""
        context = {}
        
        if OPENTELEMETRY_AVAILABLE:
            carrier = {}
            for key, value in headers.items():
                carrier[key] = value
            
            extracted = context.extract(carrier)
            context["trace_id"] = extracted.get("traceparent", "").split("-")[0]
            context["span_id"] = extracted.get("traceparent", "").split("-")[1]
        
        return context
    
    def inject_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into headers."""
        if OPENTELEMETRY_AVAILABLE:
            carrier = {}
            context.inject(carrier)
            
            for key, value in carrier.items():
                headers[key] = value
        
        return headers


class DistributedTracer:
    """High-level distributed tracing interface."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tracer = OpenTelemetryTracer("ai-code-review", config)
        self.active_spans = {}
    
    def trace_function(
        self, 
        component_type: ComponentType,
        operation_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Decorator to trace function execution."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                # Start span
                span = self.tracer.start_span(
                    name=func_name,
                    component_type=component_type,
                    attributes=attributes
                )
                
                try:
                    # Execute function
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    
                    # Add function attributes
                    if OPENTELEMETRY_AVAILABLE:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        span.set_attribute("function.duration_ms", (end_time - start_time) * 1000)
                        span.set_attribute("function.args_count", len(args))
                        span.set_attribute("function.kwargs_count", len(kwargs))
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    if OPENTELEMETRY_AVAILABLE:
                        span.set_status(TraceStatus.ERROR)
                        span.add_event("error", {"error": str(e), "type": type(e).__name__})
                    raise
                    
                finally:
                    # End span
                    if OPENTELEMETRY_AVAILABLE:
                        span.__exit__(None, None, None)
            
            return wrapper
        return decorator
    
    def trace_ai_engine(
        self, 
        channel: str, 
        model: str, 
        operation: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Trace AI engine operation."""
        span_attrs = {
            "ai.channel": channel,
            "ai.model": model,
            "ai.operation": operation,
            **(attributes or {})
        }
        
        return self.tracer.start_span(
            name=f"ai.{channel}.{operation}",
            component_type=ComponentType.AI_ENGINE,
            attributes=span_attrs
        )
    
    def trace_safety_evaluation(
        self, 
        prompt: str, 
        response_length: int,
        safety_level: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Trace safety evaluation."""
        span_attrs = {
            "safety.prompt_length": len(prompt),
            "safety.response_length": response_length,
            "safety.level": safety_level,
            **(attributes or {})
        }
        
        return self.tracer.start_span(
            name="safety.evaluation",
            component_type=ComponentType.SAFETY_JUDGE,
            attributes=span_attrs
        )
    
    def trace_policy_execution(
        self, 
        policy_name: str, 
        action: str,
        confidence: float,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Trace policy execution."""
        span_attrs = {
            "policy.name": policy_name,
            "policy.action": action,
            "policy.confidence": confidence,
            **(attributes or {})
        }
        
        return self.tracer.start_span(
            name=f"policy.{policy_name}",
            component_type=ComponentType.POLICY_ENGINE,
            attributes=span_attrs
        )
    
    def trace_code_fix(
        self, 
        issue_type: str, 
        fix_type: str,
        severity: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Trace code fix generation."""
        span_attrs = {
            "fix.issue_type": issue_type,
            "fix.fix_type": fix_type,
            "fix.severity": severity,
            **(attributes or {})
        }
        
        return self.tracer.start_span(
            name="autofix.generation",
            component_type=ComponentType.AUTO_FIX,
            attributes=span_attrs
        )
    
    def trace_rag_operation(
        self, 
        operation: str, 
        query_length: int,
        result_count: int,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Trace RAG operation."""
        span_attrs = {
            "rag.operation": operation,
            "rag.query_length": query_length,
            "rag.result_count": result_count,
            **(attributes or {})
        }
        
        return self.tracer.start_span(
            name=f"rag.{operation}",
            component_type=ComponentType.RAG_INDEX,
            attributes=span_attrs
        )
    
    def trace_api_request(
        self, 
        endpoint: str, 
        method: str,
        user_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Trace API request."""
        span_attrs = {
            "http.endpoint": endpoint,
            "http.method": method,
            "http.user_id": user_id or "anonymous",
            **(attributes or {})
        }
        
        return self.tracer.start_span(
            name=f"api.{method}.{endpoint}",
            component_type=ComponentType.API_ENDPOINT,
            attributes=span_attrs
        )
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        return {
            "opentelemetry_available": OPENTELEMETRY_AVAILABLE,
            "service_name": self.tracer.service_name if OPENTELEMETRY_AVAILABLE else "mock",
            "config": self.config,
            "active_spans": len(self.active_spans)
        }


# Global distributed tracer instance
_distributed_tracer = None


def get_distributed_tracer(config: Optional[Dict[str, Any]] = None) -> DistributedTracer:
    """Get global distributed tracer instance."""
    global _distributed_tracer
    if _distributed_tracer is None:
        # Load config from environment
        env_config = {
            "jaeger_endpoint": os.getenv("JAEGER_ENDPOINT"),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "service_version": os.getenv("SERVICE_VERSION", "1.0.0"),
            "instrument_fastapi": os.getenv("INSTRUMENT_FASTAPI", "true").lower() == "true",
            "instrument_requests": os.getenv("INSTRUMENT_REQUESTS", "true").lower() == "true",
            "instrument_sqlalchemy": os.getenv("INSTRUMENT_SQLALCHEMY", "true").lower() == "true"
        }
        
        # Merge with provided config
        merged_config = {**env_config, **(config or {})}
        _distributed_tracer = DistributedTracer(merged_config)
    
    return _distributed_tracer


# Convenience decorators
def trace_ai_engine(channel: str = "unknown", model: str = "unknown"):
    """Decorator to trace AI engine functions."""
    def decorator(func: Callable) -> Callable:
        tracer = get_distributed_tracer()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace_ai_engine(
                channel=channel,
                model=model,
                operation=func.__name__
            ) as span:
                if OPENTELEMETRY_AVAILABLE:
                    span.set_attribute("function.args", str(len(args)))
                    span.set_attribute("function.kwargs", str(len(kwargs)))
                
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def trace_safety_evaluation():
    """Decorator to trace safety evaluation functions."""
    def decorator(func: Callable) -> Callable:
        tracer = get_distributed_tracer()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace_safety_evaluation(
                prompt=str(args[0]) if args else "",
                response_length=len(str(kwargs.get("response", ""))),
                safety_level=kwargs.get("safety_level", "unknown")
            ) as span:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def trace_policy_execution():
    """Decorator to trace policy execution functions."""
    def decorator(func: Callable) -> Callable:
        tracer = get_distributed_tracer()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace_policy_execution(
                policy_name=kwargs.get("policy_name", "unknown"),
                action=kwargs.get("action", "unknown"),
                confidence=kwargs.get("confidence", 0.0)
            ) as span:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def trace_api_request(endpoint: str = "", method: str = "GET"):
    """Decorator to trace API endpoint functions."""
    def decorator(func: Callable) -> Callable:
        tracer = get_distributed_tracer()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace_api_request(
                endpoint=endpoint,
                method=method,
                user_id=kwargs.get("user_id")
            ) as span:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Middleware for FastAPI
def setup_opentelemetry_middleware(app):
    """Setup OpenTelemetry middleware for FastAPI app."""
    if OPENTELEMETRY_AVAILABLE:
        try:
            FastAPIInstrumentor().instrument_app(app)
            logger.info("OpenTelemetry FastAPI middleware enabled")
        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry middleware: {e}")
    else:
        logger.warning("OpenTelemetry not available, skipping middleware setup")


# Context management
class TraceContext:
    """Context manager for trace operations."""
    
    def __init__(self, name: str, component_type: ComponentType, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.component_type = component_type
        self.attributes = attributes
        self.tracer = get_distributed_tracer()
        self.span = None
    
    def __enter__(self):
        self.span = self.tracer.tracer.start_span(
            name=self.name,
            component_type=self.component_type,
            attributes=self.attributes
        )
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span and OPENTELEMETRY_AVAILABLE:
            if exc_val:
                self.span.set_status(TraceStatus.ERROR)
                self.span.add_event("error", {"error": str(exc_val)})
            else:
                self.span.set_status(TraceStatus.SUCCESS)
            
            self.span.__exit__(exc_type, exc_val, exc_tb)