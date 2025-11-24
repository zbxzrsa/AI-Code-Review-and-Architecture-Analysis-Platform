"""
性能监控增强 - OpenTelemetry集成
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


class MetricUnit(Enum):
    """指标单位"""
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    BYTES = "bytes"
    COUNT = "count"
    PERCENTAGE = "percentage"


class OpenTelemetryTracer:
    """OpenTelemetry分布式追踪"""
    
    def __init__(self, service_name: str, collector_endpoint: Optional[str] = None):
        self.service_name = service_name
        self.collector_endpoint = collector_endpoint
        self.spans = {}
        
    def start_span(self, operation_name: str, attributes: Dict[str, Any] = None) -> str:
        """开始新的span"""
        span_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        span_data = {
            "span_id": span_id,
            "operation_name": operation_name,
            "service_name": self.service_name,
            "start_time": datetime.utcnow().isoformat(),
            "attributes": attributes or {}
        }
        
        self.spans[span_id] = span_data
        logger.info(f"Started span: {operation_name} ({span_id})")
        
        return span_id
    
    def end_span(self, span_id: str, result: Dict[str, Any] = None, error: Optional[str] = None) -> None:
        """结束span"""
        if span_id not in self.spans:
            logger.warning(f"Span {span_id} not found")
            return
        
        span = self.spans[span_id]
        span["end_time"] = datetime.utcnow().isoformat()
        span["duration_ms"] = (datetime.utcnow() - datetime.fromisoformat(span["start_time"])).total_seconds() * 1000
        span["result"] = result
        span["error"] = error
        
        logger.info(f"Ended span: {span_id} in {span['duration_ms']}ms")
        
        # 发送到收集器（如果配置了）
        if self.collector_endpoint:
            asyncio.create_task(self._send_span_to_collector(span))
        
        del self.spans[span_id]
    
    async def _send_span_to_collector(self, span: Dict[str, Any]) -> None:
        """发送span到收集器"""
        try:
            # 模拟HTTP发送到OpenTelemetry收集器
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.collector_endpoint,
                    json=span,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Span sent to collector: {span['span_id']}")
                    else:
                        logger.warning(f"Failed to send span: {response.status}")
        except Exception as e:
            logger.error(f"Error sending span: {str(e)}")


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, prometheus_gateway: str = "http://prometheus:9090"):
        self.prometheus_gateway = prometheus_gateway
        self.metrics = {}
        
    def record_counter(self, name: str, value: float = 1, labels: Dict[str, str] = None) -> None:
        """记录计数器指标"""
        metric_key = f"counter_{name}"
        if metric_key not in self.metrics:
            self.metrics[metric_key] = {
                "type": MetricType.COUNTER,
                "value": 0,
                "labels": labels or {}
            }
        
        self.metrics[metric_key]["value"] += value
        logger.info(f"Counter {name}: +{value} = {self.metrics[metric_key]['value']}")
    
    def record_histogram(self, name: str, value: float, unit: MetricUnit = MetricUnit.SECONDS, 
                   labels: Dict[str, str] = None) -> None:
        """记录直方图指标"""
        metric_key = f"histogram_{name}"
        if metric_key not in self.metrics:
            self.metrics[metric_key] = {
                "type": MetricType.HISTOGRAM,
                "values": [],
                "unit": unit.value,
                "labels": labels or {}
            }
        
        self.metrics[metric_key]["values"].append(value)
        logger.info(f"Histogram {name}: {value} {unit.value}")
    
    def record_gauge(self, name: str, value: float, unit: MetricUnit = MetricUnit.COUNT, 
                 labels: Dict[str, str] = None) -> None:
        """记录仪表指标"""
        metric_key = f"gauge_{name}"
        self.metrics[metric_key] = {
                "type": MetricType.GAUGE,
                "value": value,
                "unit": unit.value,
                "labels": labels or {}
            }
        logger.info(f"Gauge {name}: {value} {unit.value}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics,
            "total_metrics": len(self.metrics)
        }
        
        # 计算统计信息
        counters = {k: v for k, v in self.metrics.items() if v["type"] == MetricType.COUNTER}
        histograms = {k: v for k, v in self.metrics.items() if v["type"] == MetricType.HISTOGRAM}
        gauges = {k: v for k, v in self.metrics.items() if v["type"] == MetricType.GAUGE}
        
        summary["counters"] = len(counters)
        summary["histograms"] = len(histograms)
        summary["gauges"] = len(gauges)
        
        return summary


class LogAggregator:
    """日志聚合器"""
    
    def __init__(self, elasticsearch_endpoint: str = "http://elasticsearch:9200"):
        self.elasticsearch_endpoint = elasticsearch_endpoint
        self.log_buffer = []
        self.buffer_size = 1000
        
    def log_structured(self, level: str, service: str, message: str, 
                   trace_id: Optional[str] = None, span_id: Optional[str] = None, 
                   fields: Dict[str, Any] = None) -> None:
        """记录结构化日志"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "service": service,
            "message": message,
            "trace_id": trace_id,
            "span_id": span_id,
            "fields": fields or {}
        }
        
        self.log_buffer.append(log_entry)
        
        # 缓冲区满时发送到Elasticsearch
        if len(self.log_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_logs())
    
    async def _flush_logs(self) -> None:
        """刷新日志到Elasticsearch"""
        if not self.log_buffer:
            return
        
        try:
            # 模拟批量发送到Elasticsearch
            bulk_data = "\\n".join([json.dumps(log) for log in self.log_buffer])
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.elasticsearch_endpoint}/_bulk",
                    data=bulk_data,
                    headers={"Content-Type": "application/x-ndjson"}
                ) as response:
                    if response.status == 200:
                        logger.info(f"Flushed {len(self.log_buffer)} logs to Elasticsearch")
                        self.log_buffer.clear()
                    else:
                        logger.warning(f"Failed to flush logs: {response.status}")
        except Exception as e:
            logger.error(f"Error flushing logs: {str(e)}")


class CICDPipeline:
    """CI/CD流水线"""
    
    def __init__(self, github_token: str, registry_url: str = "docker.io/library"):
        self.github_token = github_token
        self.registry_url = registry_url
        self.pipeline_stages = []
        
    def generate_github_workflow(self, app_name: str, dockerfile_path: str) -> str:
        """生成GitHub Actions工作流"""
        workflow = f"""
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ${{{{ secrets.REGISTRY_URL }}}
  DOCKERHUB_USERNAME: ${{{{ secrets.DOCKERHUB_USERNAME }}}
  DOCKERHUB_TOKEN: ${{{{ secrets.DOCKERHUB_TOKEN }}}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run Tests
        run: |
          python -m pytest tests/ -v
          python -m flake8 app/ --max-line-length=100
          python -m mypy app/
      
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - name: Run Security Scan
        run: |
          docker run --rm -v $(pwd):/app:/app \\
            aquasec/trivy:latest \\
            image /app:/. \\
            --format json \\
            --exit-code 0
        continue-on-error: true
      
      - name: Upload Security Results
        uses: actions/upload-artifact@v3
        with:
          name: security-scan-results
          path: trivy-results.json
      
  build-and-deploy:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Build and Push Docker Image
        run: |
          docker build -t {self.registry_url}/{app_name}:${{{{{ github.sha }}}} .
          docker push {self.registry_url}/{app_name}:${{{{{ github.sha }}}}
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/{app_name}-deployment {app_name}={self.registry_url}/{app_name}:${{{{{ github.sha }}}}
          kubectl rollout status deployment/{app_name}-deployment
"""
        
        return workflow
    
    def generate_deploy_script(self, app_name: str, namespace: str = "default") -> str:
        """生成部署脚本"""
        return f"""#!/bin/bash
# 自动部署脚本

set -e

echo "Deploying {app_name} to namespace {namespace}..."

# 应用Kubernetes清单
kubectl apply -f k8s-deployment.yaml -n {namespace}

# 等待部署完成
kubectl rollout status deployment/{app_name}-deployment -n {namespace}
kubectl wait --for condition=available --timeout=300s deployment/{app_name}-deployment -n {namespace}

echo "Deployment completed successfully!"

# 验证部署
kubectl get pods -n {namespace} -l app={app_name}
kubectl get services -n {namespace} -l app={app_name}

echo "All done!"
"""


# 全局实例
telemetry_tracer = OpenTelemetryTracer("code-review-platform")
metrics_collector = MetricsCollector()
log_aggregator = LogAggregator()
cicd_pipeline = CICDPipeline()


# 使用示例
async def example_usage():
    """性能监控和CI/CD使用示例"""
    
    # 1. 分布式追踪示例
    span_id = telemetry_tracer.start_span(
        "code_analysis",
        {"file_path": "src/main.py", "language": "python"}
    )
    
    # 模拟一些工作
    await asyncio.sleep(0.1)
    
    telemetry_tracer.end_span(
        span_id,
        {"issues_found": 5, "analysis_time": 2.5}
    )
    
    # 2. 指标记录示例
    metrics_collector.record_counter("analysis_requests", 1, {"language": "python"})
    metrics_collector.record_histogram("analysis_duration", 2.5, MetricUnit.SECONDS, {"file_type": "python"})
    metrics_collector.record_gauge("active_sessions", 3, MetricUnit.COUNT)
    
    # 3. 结构化日志示例
    log_aggregator.log_structured(
        "INFO",
        "analysis-service",
        "Code analysis completed",
        "trace_123",
        "span_456",
        {"files_processed": 10, "issues_found": 2}
    )
    
    # 4. CI/CD流水线示例
    workflow = cicd_pipeline.generate_github_workflow("code-review-app", "Dockerfile")
    print("Generated GitHub Actions workflow")
    
    # 5. 部署脚本示例
    deploy_script = cicd_pipeline.generate_deploy_script("code-review-app", "production")
    print("Generated deployment script")
    
    # 6. 获取指标摘要
    metrics_summary = metrics_collector.get_metrics_summary()
    print(f"Metrics summary: {metrics_summary}")


if __name__ == "__main__":
    asyncio.run(example_usage())