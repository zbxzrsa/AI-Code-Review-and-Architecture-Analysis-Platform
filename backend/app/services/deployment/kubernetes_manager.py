"""
Kubernetes部署管理器
支持云原生架构和自动化部署
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import yaml
import json
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    """部署策略"""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"

class ServiceType(Enum):
    """服务类型"""
    CLUSTER_IP = "ClusterIP"
    NODE_PORT = "NodePort"
    LOAD_BALANCER = "LoadBalancer"
    EXTERNAL_NAME = "ExternalName"

class ResourceType(Enum):
    """资源类型"""
    DEPLOYMENT = "Deployment"
    SERVICE = "Service"
    CONFIG_MAP = "ConfigMap"
    SECRET = "Secret"
    INGRESS = "Ingress"
    HPA = "HorizontalPodAutoscaler"
    PVC = "PersistentVolumeClaim"

@dataclass
class ResourceRequirements:
    """资源需求配置"""
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    storage_request: str = "1Gi"

@dataclass
class HealthCheck:
    """健康检查配置"""
    path: str = "/health"
    port: int = 8000
    initial_delay: int = 30
    period: int = 10
    timeout: int = 5
    failure_threshold: int = 3
    success_threshold: int = 1

@dataclass
class AutoScaling:
    """自动扩缩容配置"""
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    scale_up_stabilization: int = 60
    scale_down_stabilization: int = 300

@dataclass
class DeploymentConfig:
    """部署配置"""
    name: str
    namespace: str = "default"
    image: str = ""
    replicas: int = 1
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    health_check: HealthCheck = field(default_factory=HealthCheck)
    auto_scaling: Optional[AutoScaling] = None
    environment: Dict[str, str] = field(default_factory=dict)
    config_maps: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

@dataclass
class ServiceConfig:
    """服务配置"""
    name: str
    namespace: str = "default"
    service_type: ServiceType = ServiceType.CLUSTER_IP
    ports: List[Dict[str, Any]] = field(default_factory=list)
    selector: Dict[str, str] = field(default_factory=dict)
    external_ips: List[str] = field(default_factory=list)
    load_balancer_ip: Optional[str] = None

class KubernetesResourceGenerator:
    """Kubernetes资源生成器"""
    
    def __init__(self):
        self.api_version_map = {
            ResourceType.DEPLOYMENT: "apps/v1",
            ResourceType.SERVICE: "v1",
            ResourceType.CONFIG_MAP: "v1",
            ResourceType.SECRET: "v1",
            ResourceType.INGRESS: "networking.k8s.io/v1",
            ResourceType.HPA: "autoscaling/v2",
            ResourceType.PVC: "v1"
        }
    
    def generate_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """生成Deployment资源"""
        deployment = {
            "apiVersion": self.api_version_map[ResourceType.DEPLOYMENT],
            "kind": "Deployment",
            "metadata": {
                "name": config.name,
                "namespace": config.namespace,
                "labels": {
                    "app": config.name,
                    **config.labels
                },
                "annotations": config.annotations
            },
            "spec": {
                "replicas": config.replicas,
                "strategy": self._get_deployment_strategy(config.strategy),
                "selector": {
                    "matchLabels": {
                        "app": config.name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.name,
                            **config.labels
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": config.name,
                            "image": config.image,
                            "ports": [{
                                "containerPort": config.health_check.port,
                                "name": "http"
                            }],
                            "env": [
                                {"name": k, "value": v} 
                                for k, v in config.environment.items()
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": config.resources.cpu_request,
                                    "memory": config.resources.memory_request
                                },
                                "limits": {
                                    "cpu": config.resources.cpu_limit,
                                    "memory": config.resources.memory_limit
                                }
                            },
                            "livenessProbe": self._get_probe_config(config.health_check),
                            "readinessProbe": self._get_probe_config(config.health_check)
                        }]
                    }
                }
            }
        }
        
        # 添加ConfigMap和Secret挂载
        if config.config_maps or config.secrets:
            volumes = []
            volume_mounts = []
            
            for cm in config.config_maps:
                volumes.append({
                    "name": f"{cm}-volume",
                    "configMap": {"name": cm}
                })
                volume_mounts.append({
                    "name": f"{cm}-volume",
                    "mountPath": f"/etc/config/{cm}"
                })
            
            for secret in config.secrets:
                volumes.append({
                    "name": f"{secret}-volume",
                    "secret": {"secretName": secret}
                })
                volume_mounts.append({
                    "name": f"{secret}-volume",
                    "mountPath": f"/etc/secrets/{secret}"
                })
            
            deployment["spec"]["template"]["spec"]["volumes"] = volumes
            deployment["spec"]["template"]["spec"]["containers"][0]["volumeMounts"] = volume_mounts
        
        return deployment
    
    def generate_service(self, config: ServiceConfig) -> Dict[str, Any]:
        """生成Service资源"""
        service = {
            "apiVersion": self.api_version_map[ResourceType.SERVICE],
            "kind": "Service",
            "metadata": {
                "name": config.name,
                "namespace": config.namespace
            },
            "spec": {
                "type": config.service_type.value,
                "ports": config.ports,
                "selector": config.selector or {"app": config.name}
            }
        }
        
        if config.external_ips:
            service["spec"]["externalIPs"] = config.external_ips
        
        if config.load_balancer_ip:
            service["spec"]["loadBalancerIP"] = config.load_balancer_ip
        
        return service
    
    def generate_hpa(self, config: DeploymentConfig) -> Optional[Dict[str, Any]]:
        """生成HorizontalPodAutoscaler资源"""
        if not config.auto_scaling:
            return None
        
        hpa = {
            "apiVersion": self.api_version_map[ResourceType.HPA],
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{config.name}-hpa",
                "namespace": config.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": config.name
                },
                "minReplicas": config.auto_scaling.min_replicas,
                "maxReplicas": config.auto_scaling.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": config.auto_scaling.target_cpu_utilization
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": config.auto_scaling.target_memory_utilization
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": config.auto_scaling.scale_up_stabilization
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": config.auto_scaling.scale_down_stabilization
                    }
                }
            }
        }
        
        return hpa
    
    def generate_config_map(self, name: str, namespace: str, data: Dict[str, str]) -> Dict[str, Any]:
        """生成ConfigMap资源"""
        return {
            "apiVersion": self.api_version_map[ResourceType.CONFIG_MAP],
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "data": data
        }
    
    def generate_secret(self, name: str, namespace: str, data: Dict[str, str], secret_type: str = "Opaque") -> Dict[str, Any]:
        """生成Secret资源"""
        return {
            "apiVersion": self.api_version_map[ResourceType.SECRET],
            "kind": "Secret",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "type": secret_type,
            "data": data  # 注意：实际使用时需要base64编码
        }
    
    def _get_deployment_strategy(self, strategy: DeploymentStrategy) -> Dict[str, Any]:
        """获取部署策略配置"""
        if strategy == DeploymentStrategy.ROLLING_UPDATE:
            return {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxUnavailable": "25%",
                    "maxSurge": "25%"
                }
            }
        elif strategy == DeploymentStrategy.RECREATE:
            return {"type": "Recreate"}
        else:
            return {"type": "RollingUpdate"}
    
    def _get_probe_config(self, health_check: HealthCheck) -> Dict[str, Any]:
        """获取探针配置"""
        return {
            "httpGet": {
                "path": health_check.path,
                "port": health_check.port
            },
            "initialDelaySeconds": health_check.initial_delay,
            "periodSeconds": health_check.period,
            "timeoutSeconds": health_check.timeout,
            "failureThreshold": health_check.failure_threshold,
            "successThreshold": health_check.success_threshold
        }

class KubernetesManager:
    """Kubernetes部署管理器"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path
        self.resource_generator = KubernetesResourceGenerator()
        self.deployed_resources: Dict[str, List[Dict[str, Any]]] = {}
    
    async def deploy_application(self, app_name: str, configs: List[DeploymentConfig]) -> bool:
        """部署应用"""
        try:
            resources = []
            
            for config in configs:
                # 生成Deployment
                deployment = self.resource_generator.generate_deployment(config)
                resources.append(deployment)
                
                # 生成Service
                service_config = ServiceConfig(
                    name=f"{config.name}-service",
                    namespace=config.namespace,
                    ports=[{
                        "name": "http",
                        "port": 80,
                        "targetPort": config.health_check.port,
                        "protocol": "TCP"
                    }]
                )
                service = self.resource_generator.generate_service(service_config)
                resources.append(service)
                
                # 生成HPA（如果配置了自动扩缩容）
                hpa = self.resource_generator.generate_hpa(config)
                if hpa:
                    resources.append(hpa)
            
            # 应用资源
            success = await self._apply_resources(resources)
            if success:
                self.deployed_resources[app_name] = resources
                logger.info(f"Successfully deployed application: {app_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to deploy application {app_name}: {e}")
            return False
    
    async def update_application(self, app_name: str, configs: List[DeploymentConfig]) -> bool:
        """更新应用"""
        try:
            # 执行滚动更新
            for config in configs:
                deployment = self.resource_generator.generate_deployment(config)
                success = await self._apply_resource(deployment)
                if not success:
                    return False
            
            logger.info(f"Successfully updated application: {app_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update application {app_name}: {e}")
            return False
    
    async def delete_application(self, app_name: str) -> bool:
        """删除应用"""
        try:
            if app_name not in self.deployed_resources:
                logger.warning(f"Application {app_name} not found in deployed resources")
                return False
            
            resources = self.deployed_resources[app_name]
            success = await self._delete_resources(resources)
            
            if success:
                del self.deployed_resources[app_name]
                logger.info(f"Successfully deleted application: {app_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete application {app_name}: {e}")
            return False
    
    async def scale_application(self, app_name: str, service_name: str, replicas: int) -> bool:
        """扩缩容应用"""
        try:
            # 这里应该调用Kubernetes API来扩缩容
            # 模拟实现
            logger.info(f"Scaling {service_name} in {app_name} to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale application {app_name}: {e}")
            return False
    
    async def get_application_status(self, app_name: str) -> Dict[str, Any]:
        """获取应用状态"""
        try:
            # 这里应该调用Kubernetes API获取实际状态
            # 模拟返回状态
            return {
                "name": app_name,
                "status": "Running",
                "replicas": {
                    "desired": 3,
                    "current": 3,
                    "ready": 3
                },
                "services": [
                    {
                        "name": f"{app_name}-service",
                        "type": "ClusterIP",
                        "cluster_ip": "10.96.0.1",
                        "ports": [{"port": 80, "target_port": 8000}]
                    }
                ],
                "pods": [
                    {
                        "name": f"{app_name}-pod-1",
                        "status": "Running",
                        "ready": True,
                        "restarts": 0
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get application status {app_name}: {e}")
            return {}
    
    async def export_manifests(self, app_name: str, output_dir: str) -> bool:
        """导出Kubernetes清单文件"""
        try:
            if app_name not in self.deployed_resources:
                logger.warning(f"Application {app_name} not found")
                return False
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            resources = self.deployed_resources[app_name]
            for i, resource in enumerate(resources):
                filename = f"{resource['kind'].lower()}-{resource['metadata']['name']}.yaml"
                filepath = output_path / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(resource, f, default_flow_style=False)
            
            logger.info(f"Exported manifests for {app_name} to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export manifests for {app_name}: {e}")
            return False
    
    async def _apply_resources(self, resources: List[Dict[str, Any]]) -> bool:
        """应用资源列表"""
        try:
            for resource in resources:
                success = await self._apply_resource(resource)
                if not success:
                    return False
            return True
        except Exception as e:
            logger.error(f"Failed to apply resources: {e}")
            return False
    
    async def _apply_resource(self, resource: Dict[str, Any]) -> bool:
        """应用单个资源"""
        try:
            # 这里应该调用Kubernetes API
            # 模拟实现
            logger.info(f"Applying {resource['kind']}: {resource['metadata']['name']}")
            await asyncio.sleep(0.1)  # 模拟API调用延迟
            return True
        except Exception as e:
            logger.error(f"Failed to apply resource: {e}")
            return False
    
    async def _delete_resources(self, resources: List[Dict[str, Any]]) -> bool:
        """删除资源列表"""
        try:
            for resource in reversed(resources):  # 反向删除
                success = await self._delete_resource(resource)
                if not success:
                    return False
            return True
        except Exception as e:
            logger.error(f"Failed to delete resources: {e}")
            return False
    
    async def _delete_resource(self, resource: Dict[str, Any]) -> bool:
        """删除单个资源"""
        try:
            # 这里应该调用Kubernetes API
            # 模拟实现
            logger.info(f"Deleting {resource['kind']}: {resource['metadata']['name']}")
            await asyncio.sleep(0.1)  # 模拟API调用延迟
            return True
        except Exception as e:
            logger.error(f"Failed to delete resource: {e}")
            return False

# 全局Kubernetes管理器实例
_kubernetes_manager: Optional[KubernetesManager] = None

def get_kubernetes_manager() -> KubernetesManager:
    """获取Kubernetes管理器实例"""
    global _kubernetes_manager
    if _kubernetes_manager is None:
        _kubernetes_manager = KubernetesManager()
    return _kubernetes_manager

def init_kubernetes_manager(kubeconfig_path: Optional[str] = None) -> KubernetesManager:
    """初始化Kubernetes管理器"""
    global _kubernetes_manager
    _kubernetes_manager = KubernetesManager(kubeconfig_path)
    return _kubernetes_manager

# 预定义的部署配置
def create_translation_service_config() -> DeploymentConfig:
    """创建翻译服务部署配置"""
    return DeploymentConfig(
        name="translation-service",
        namespace="translation",
        image="translation-service:latest",
        replicas=3,
        strategy=DeploymentStrategy.ROLLING_UPDATE,
        resources=ResourceRequirements(
            cpu_request="200m",
            cpu_limit="1000m",
            memory_request="256Mi",
            memory_limit="1Gi"
        ),
        auto_scaling=AutoScaling(
            min_replicas=2,
            max_replicas=10,
            target_cpu_utilization=70
        ),
        environment={
            "ENV": "production",
            "LOG_LEVEL": "INFO",
            "DATABASE_URL": "postgresql://postgres:password@postgres:5432/translation_db"
        },
        labels={
            "component": "translation",
            "tier": "backend"
        }
    )

def create_monitoring_service_config() -> DeploymentConfig:
    """创建监控服务部署配置"""
    return DeploymentConfig(
        name="monitoring-service",
        namespace="monitoring",
        image="monitoring-service:latest",
        replicas=2,
        resources=ResourceRequirements(
            cpu_request="100m",
            cpu_limit="500m",
            memory_request="128Mi",
            memory_limit="512Mi"
        ),
        environment={
            "ENV": "production",
            "METRICS_RETENTION": "7d"
        },
        labels={
            "component": "monitoring",
            "tier": "infrastructure"
        }
    )