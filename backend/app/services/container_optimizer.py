"""
容器化改进服务 - 多阶段构建 + 安全扫描
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import os

logger = logging.getLogger(__name__)


class BuildStage(Enum):
    """构建阶段"""
    DEPENDENCIES = "dependencies"
    BASE_IMAGE = "base_image"
    RUNTIME_IMAGE = "runtime_image"
    FINAL_IMAGE = "final_image"


class SecurityScanType(Enum):
    """安全扫描类型"""
    DEPENDENCY_CHECK = "dependency_check"
    VULNERABILITY_SCAN = "vulnerability_scan"
    SAST_SCAN = "sast_scan"
    IMAGE_SCAN = "image_scan"


class ContainerOptimizer:
    """容器优化器"""
    
    def __init__(self, registry_url: str = "docker.io/library"):
        self.registry_url = registry_url
        self.build_cache = {}
        
    def generate_dockerfile(self, app_name: str, base_image: str = "python:3.11-slim", 
                        runtime_image: str = "python:3.11-slim") -> str:
        """
        生成优化的Dockerfile
        
        Args:
            app_name: 应用名称
            base_image: 基础镜像
            runtime_image: 运行时镜像
            
        Returns:
            Dockerfile内容
        """
        return f"""
# 多阶段构建优化
FROM {base_image} AS deps
WORKDIR /app

# 安装依赖并缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 构建应用
COPY . .

# 运行时阶段
FROM {runtime_image} AS runner

# 非root用户
RUN addgroup --system --gid 1001 appgroup && \\
    adduser --system --uid 1001 --gid 1001 --home /app appuser

# 复制构建结果
COPY --from=deps /app /app

# 设置权限
RUN chown -R appuser:appgroup /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    def generate_compose_override(self, service_name: str, replicas: int = 1, 
                             cpu_limit: str = "500m", memory_limit: str = "512Mi") -> str:
        """
        生成Docker Compose覆盖配置
        
        Args:
            service_name: 服务名称
            replicas: 副本数
            cpu_limit: CPU限制
            memory_limit: 内存限制
            
        Returns:
            Docker Compose配置
        """
        return f"""
version: '3.8'

services:
  {service_name}:
    image: {service_name}:latest
    deploy:
      replicas: {replicas}
      resources:
        limits:
          cpus: '{cpu_limit}'
          memory: '{memory_limit}'
        restart_policy: on-failure
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/{service_name}_db
      - REDIS_URL=redis://localhost:6379/0
      - LOG_LEVEL=INFO
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
"""
    
    async def build_image(self, app_name: str, dockerfile_content: str) -> Dict[str, Any]:
        """
        构建Docker镜像
        
        Args:
            app_name: 应用名称
            dockerfile_content: Dockerfile内容
            
        Returns:
            构建结果
        """
        build_id = hashlib.sha256(dockerfile_content.encode()).hexdigest()[:16]
        
        # 模拟Docker构建
        logger.info(f"Building Docker image for {app_name} with build ID: {build_id}")
        
        # 模拟构建时间
        await asyncio.sleep(5)  # 模拟构建过程
        
        return {
            "success": True,
            "build_id": build_id,
            "image_tag": f"{app_name}:latest",
            "build_time_seconds": 5,
            "size_mb": 150,  # 模拟镜像大小
            "layers": 8
        }
    
    async def scan_dependencies(self, app_path: str) -> Dict[str, Any]:
        """
        依赖扫描
        
        Args:
            app_path: 应用路径
            
        Returns:
            扫描结果
        """
        # 模拟依赖扫描
        vulnerabilities = [
            {
                "package": "requests",
                "version": "2.25.1",
                "severity": "high",
                "cve": "CVE-2023-1234",
                "description": "Requests library vulnerable to SSRF"
            },
            {
                "package": "urllib3",
                "version": "1.26.0", 
                "severity": "medium",
                "cve": "CVE-2023-5678",
                "description": "URL parsing vulnerability"
            }
        ]
        
        return {
            "scan_type": SecurityScanType.DEPENDENCY_CHECK,
            "vulnerabilities": vulnerabilities,
            "total_vulnerabilities": len(vulnerabilities),
            "high_severity_count": len([v for v in vulnerabilities if v["severity"] == "high"]),
            "scan_time_seconds": 2,
            "recommendations": [
                "Update requests to version 2.31.0",
                "Use urllib3 with proper URL validation"
            ]
        }
    
    async def scan_image(self, image_tag: str) -> Dict[str, Any]:
        """
        镜像安全扫描
        
        Args:
            image_tag: 镜像标签
            
        Returns:
            扫描结果
        """
        # 模拟镜像扫描
        security_issues = [
            {
                "type": "non_root_user",
                "severity": "high",
                "description": "Container running as root user"
            },
            {
                "type": "excessive_permissions",
                "severity": "medium", 
                "description": "Container has more permissions than necessary"
            },
            {
                "type": "vulnerable_base_image",
                "severity": "high",
                "description": "Base image has known vulnerabilities"
            }
        ]
        
        return {
            "scan_type": SecurityScanType.IMAGE_SCAN,
            "image_tag": image_tag,
            "security_issues": security_issues,
            "total_issues": len(security_issues),
            "scan_time_seconds": 3,
            "recommendations": [
                "Use non-root user",
                "Minimize container permissions",
                "Use minimal base images"
            ]
        }
    
    async def generate_k8s_manifests(self, app_name: str, namespace: str = "default") -> Dict[str, Any]:
        """
        生成Kubernetes部署清单
        
        Args:
            app_name: 应用名称
            namespace: K8s命名空间
            
        Returns:
            K8s清单
        """
        manifests = {}
        
        # Deployment清单
        manifests["deployment"] = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{app_name}-deployment",
                "namespace": namespace,
                "labels": {
                    "app": app_name,
                    "version": "1.0.0"
                }
            },
            "spec": {
                "replicas": 3,
                "selector": {
                    "matchLabels": {
                        "app": app_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": app_name
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": app_name,
                            "image": f"{app_name}:latest",
                            "ports": [{
                                "containerPort": 8000,
                                "protocol": "TCP"
                            }],
                            "resources": {
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "128Mi"
                                },
                                "limits": {
                                    "cpu": "500m",
                                    "memory": "512Mi"
                                }
                            },
                            "securityContext": {
                                "runAsNonRoot": True,
                                "runAsUser": 1001,
                                "runAsGroup": 1001,
                                "fsGroup": 1001
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }]
                }
            }
        }
        
        # Service清单
        manifests["service"] = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{app_name}-service",
                "namespace": namespace,
                "labels": {
                    "app": app_name
                }
            },
            "spec": {
                "selector": {
                    "app": app_name
                },
                "ports": [{
                    "port": 8000,
                    "targetPort": 8000,
                    "protocol": "TCP"
                }],
                "type": "ClusterIP"
            }
        }
        
        # HPA清单
        manifests["hpa"] = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{app_name}-hpa",
                "namespace": namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{app_name}-deployment"
                },
                "minReplicas": 2,
                "maxReplicas": 10,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": 70
                        }
                    }
                }]
            },
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [{
                            "type": "Pods"
                        }]
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300
                    }
                }
            }
        }
        
        return {
            "success": True,
            "namespace": namespace,
            "app_name": app_name,
            "manifests": manifests,
            "generated_at": datetime.utcnow().isoformat()
        }


class SecurityScanner:
    """安全扫描器"""
    
    def __init__(self):
        self.scan_results = {}
    
    async def run_security_scan(self, scan_type: SecurityScanType, target: str) -> Dict[str, Any]:
        """
        运行安全扫描
        
        Args:
            scan_type: 扫描类型
            target: 扫描目标
            
        Returns:
            扫描结果
        """
        scan_id = hashlib.sha256(f"{scan_type.value}:{target}:{datetime.utcnow().isoformat()}").hexdigest()[:16]
        
        try:
            if scan_type == SecurityScanType.DEPENDENCY_CHECK:
                result = await self._dependency_check(target)
            elif scan_type == SecurityScanType.IMAGE_SCAN:
                result = await self._image_scan(target)
            else:
                result = {"success": False, "error": f"Unsupported scan type: {scan_type}"}
            
            self.scan_results[scan_id] = result
            logger.info(f"Security scan completed: {scan_type} for {target}")
            
            return {
                "success": True,
                "scan_id": scan_id,
                "scan_type": scan_type.value,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Security scan failed: {str(e)}")
            return {
                "success": False,
                "scan_id": scan_id,
                "scan_type": scan_type.value,
                "error": str(e)
            }
    
    async def _dependency_check(self, app_path: str) -> Dict[str, Any]:
        """依赖检查实现"""
        # 模拟OWASP Dependency Check
        return {
            "success": True,
            "dependencies_found": 2,
            "high_risk": 1,
            "recommendations": ["Update dependencies", "Use dependency scanning in CI"]
        }
    
    async def _image_scan(self, image_tag: str) -> Dict[str, Any]:
        """镜像扫描实现"""
        # 模拟Trivy扫描
        return {
            "success": True,
            "vulnerabilities": 3,
            "critical_issues": 1,
            "recommendations": ["Update base image", "Use minimal base images"]
        }


# 全局实例
container_optimizer = ContainerOptimizer()
security_scanner = SecurityScanner()


# 使用示例
async def example_usage():
    """容器化和安全扫描使用示例"""
    
    # 1. 生成优化的Dockerfile
    dockerfile = container_optimizer.generate_dockerfile("code-review-app")
    print("Generated optimized Dockerfile")
    
    # 2. 构建镜像
    build_result = await container_optimizer.build_image("code-review-app", dockerfile)
    print(f"Build result: {build_result}")
    
    # 3. 依赖扫描
    dep_scan = await security_scanner.run_security_scan(
        SecurityScanType.DEPENDENCY_CHECK, 
        "/app"
    )
    print(f"Dependency scan: {dep_scan}")
    
    # 4. 镜像扫描
    img_scan = await security_scanner.run_security_scan(
        SecurityScanType.IMAGE_SCAN,
        "code-review-app:latest"
    )
    print(f"Image scan: {img_scan}")
    
    # 5. 生成K8s清单
    k8s_manifests = await container_optimizer.generate_k8s_manifests("code-review-app")
    print(f"Generated K8s manifests for {k8s_manifests['app_name']}")
    
    # 6. 生成Docker Compose覆盖
    compose_override = container_optimizer.generate_compose_override(
        "code-review-app",
        replicas=3,
        cpu_limit="200m",
        memory_limit="256Mi"
    )
    print("Generated Docker Compose override")


if __name__ == "__main__":
    asyncio.run(example_usage())