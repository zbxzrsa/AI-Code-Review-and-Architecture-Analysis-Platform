"""Prometheus metrics integration for FastAPI and Celery."""
from prometheus_client import Counter, Histogram, Gauge, REGISTRY, generate_latest
from fastapi import Response
import os

# Analysis metrics
ANALYSIS_STARTED = Counter(
    "analysis_jobs_started_total",
    "Total analysis jobs started",
    ["tenant"]
)

ANALYSIS_COMPLETED = Counter(
    "analysis_jobs_completed_total",
    "Total analysis jobs completed",
    ["tenant"]
)

ANALYSIS_FAILED = Counter(
    "analysis_jobs_failed_total",
    "Total analysis jobs failed",
    ["tenant"]
)

CACHED_FILES_SKIPPED = Counter(
    "cached_files_skipped_total",
    "Total cached files skipped",
    ["tenant"]
)

RETRY_COUNT = Counter(
    "task_retry_count_total",
    "Total task retries",
    ["tenant", "task_name"]
)

# Queue metrics
QUEUE_BACKLOG = Gauge(
    "queue_backlog_size",
    "Current queue backlog size",
    ["queue_name"]
)

# Performance metrics
ANALYSIS_DURATION = Histogram(
    "analysis_job_duration_seconds",
    "Analysis job duration in seconds",
    ["language", "rulepack"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600)
)

INCREMENTAL_HIT_RATIO = Histogram(
    "incremental_hit_ratio",
    "Cache hit ratio for incremental analysis",
    [],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

S3_UPLOAD_DURATION = Histogram(
    "s3_upload_duration_seconds",
    "S3/MinIO artifact upload duration in seconds",
    ["bucket"],
    buckets=(0.1, 0.5, 1, 5, 10, 30, 60)
)

S3_UPLOAD_SIZE = Histogram(
    "s3_upload_size_bytes",
    "S3/MinIO artifact upload size in bytes",
    ["bucket"],
    buckets=(1000, 10000, 100000, 1000000, 10000000)
)

# API metrics (from existing middleware)
API_REQUESTS = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method"]
)

API_ERRORS = Counter(
    "api_errors_total",
    "Total API errors",
    ["endpoint", "method", "status_code"]
)

API_LATENCY = Histogram(
    "api_latency_seconds",
    "API request latency in seconds",
    ["endpoint", "method"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5)
)


def metrics_handler(request):
    """Handler for /metrics endpoint that returns Prometheus metrics."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain; charset=utf-8"
    )
