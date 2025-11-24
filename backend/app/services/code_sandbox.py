"""
容器化代码执行沙箱环境
提供安全的代码执行环境，隔离潜在危险的代码执行
"""

import asyncio
import json
import logging
import tempfile
import shutil
import subprocess
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import docker
from docker.errors import DockerException
import docker.models.containers
from docker.models.networks
import docker.types
import aiodocker

logger = logging.getLogger(__name__)


class SandboxStatus(Enum):
    """沙箱状态"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ExecutionResult:
    """执行结果"""
    execution_id: str
    status: SandboxStatus
    exit_code: Optional[int] = None
    stdout: str
    stderr: str
    error_message: Optional[str] = None
    execution_time: float
    resource_usage: Dict[str, Any]
    created_files: List[str]
    deleted_files: List[str]


@dataclass
class SandboxConfig:
    """沙箱配置"""
    image_name: str = "python:3.11-slim"
    memory_limit: str = "512m"
    cpu_limit: str = "1.0"
    timeout_seconds: int = 30
    network_enabled: bool = False
    read_only_filesystem: bool = True
    temp_dir: Optional[str] = None
    allowed_commands: List[str] = None
    environment_variables: Dict[str, str] = None
    resource_limits: Dict[str, Any] = None


class CodeSandbox:
    """代码执行沙箱"""
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.client = None
        self.active_containers: Dict[str, aiodocker.containers.Container] = {}
        self.execution_history: List[ExecutionResult] = []
        
        # 默认允许的命令
        if self.config.allowed_commands is None:
            self.config.allowed_commands = [
                "python", "python3", "pip", "pip3",
                "node", "npm", "yarn",
                "java", "javac",
                "go", "go run",
                "rustc", "cargo",
                "ruby", "gem",
                "php", "php -r",
                "bash", "sh",
                "cat", "ls", "echo", "date"
            ]
        
        # 默认环境变量
        if self.config.environment_variables is None:
            self.config.environment_variables = {
                "PYTHONPATH": "/usr/local/bin:/usr/bin:/bin",
                "NODE_PATH": "/usr/local/bin:/usr/bin",
                "JAVA_HOME": "/usr/lib/jvm/default-java-11-openjdk-amd64",
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
            }
        
        # 默认资源限制
        if self.config.resource_limits is None:
            self.config.resource_limits = {
                "memory": self.config.memory_limit,
                "cpu": self.config.cpu_limit,
                "pids": 10,
                "disk_size": "1g"
            }
    
    async def initialize(self):
        """初始化沙箱环境"""
        try:
            self.client = aiodocker.from_env()
            await self.client.ping()
            logger.info("Docker client initialized successfully")
            
            # 拉取基础镜像
            await self._pull_base_image()
            
            logger.info("Code sandbox initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize sandbox: {e}")
            raise
    
    async def _pull_base_image(self):
        """拉取基础镜像"""
        try:
            logger.info(f"Pulling base image: {self.config.image_name}")
            
            await self.client.images.pull(self.config.image_name)
            logger.info("Base image pulled successfully")
            
        except Exception as e:
            logger.error(f"Failed to pull base image: {e}")
            raise
    
    async def execute_code(
        self,
        code: str,
        language: str = "python",
        files: Optional[Dict[str, str]] = None,
        environment: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
        execution_id: Optional[str] = None
    ) -> ExecutionResult:
        """执行代码"""
        execution_id = execution_id or str(uuid.uuid4())
        
        if timeout is None:
            timeout = self.config.timeout_seconds
        
        start_time = asyncio.get_event_loop_time()
        
        try:
            # 创建临时目录
            temp_dir = tempfile.mkdtempdir(prefix="sandbox_")
            
            # 准备代码文件
            code_file = Path(temp_dir) / f"code.{self._get_file_extension(language)}"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # 准备额外文件
            file_paths = []
            if files:
                for file_path, content in files.items():
                    file_full_path = Path(temp_dir) / file_path
                    file_paths.append(file_full_path)
                    
                    with open(file_full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
            
            # 构建Docker命令
            docker_command = self._build_docker_command(
                language, code_file.name, file_paths, environment
            )
            
            # 创建容器
            container_config = self._create_container_config(temp_dir)
            
            logger.info(f"Starting container for execution {execution_id}")
            
            container = await self.client.containers.run(
                image=self.config.image_name,
                command=docker_command,
                volumes=[f"{temp_dir}:/workspace"],
                environment=self._merge_environment(environment),
                **container_config,
                detach=True,
                remove=True
            )
            
            self.active_containers[execution_id] = container
            
            # 等待容器完成
            result = await self._wait_for_completion(
                container, execution_id, start_time, timeout
            )
            
            # 清理临时目录
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Execution {execution_id} timed out after {timeout} seconds")
            return ExecutionResult(
                execution_id=execution_id,
                status=SandboxStatus.TIMEOUT,
                exit_code=None,
                stdout="",
                stderr="Execution timed out",
                error_message=f"Execution timed out after {timeout} seconds",
                execution_time=timeout,
                resource_usage={},
                created_files=[],
                deleted_files=[]
            )
        
        except Exception as e:
            logger.error(f"Error in execution {execution_id}: {e}")
            return ExecutionResult(
                execution_id=execution_id,
                status=SandboxStatus.FAILED,
                exit_code=None,
                stdout="",
                stderr=str(e),
                error_message=str(e),
                execution_time=asyncio.get_event_loop_time() - start_time,
                resource_usage={},
                created_files=[],
                deleted_files=[]
            )
    
    async def _wait_for_completion(
        self,
        container: aiodocker.containers.Container,
        execution_id: str,
        start_time: float,
        timeout: int
    ) -> ExecutionResult:
        """等待容器完成"""
        try:
            # 等待容器退出
            exit_code = 0
            stdout = ""
            stderr = ""
            
            async for log in container.logs(stream=True):
                if log:
                    if log.get("stream") == "stdout":
                        stdout += log.get("data", "")
                    elif log.get("stream") == "stderr":
                        stderr += log.get("data", "")
            
            # 等待容器退出
            await container.wait()
            
            # 获取退出码
            inspect = await container.inspect()
            exit_code = inspect["State"]["ExitCode"]
            
            execution_time = asyncio.get_event_loop_time() - start_time
            
            # 获取资源使用情况
            stats = await container.stats()
            resource_usage = {
                "cpu_usage": stats.get("CPUStats", {}),
                "memory_usage": stats.get("MemoryStats", {}),
                "network_io": stats.get("NetworkIO", {}),
                "block_io": stats.get("BlockIO", {})
            }
            
            return ExecutionResult(
                execution_id=execution_id,
                status=SandboxStatus.COMPLETED,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                error_message=None,
                execution_time=execution_time,
                resource_usage=resource_usage,
                created_files=[],
                deleted_files=[]
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Container execution {execution_id} timed out")
            return ExecutionResult(
                execution_id=execution_id,
                status=SandboxStatus.TIMEOUT,
                exit_code=None,
                stdout="",
                stderr="",
                error_message=f"Execution timed out after {timeout} seconds",
                execution_time=timeout,
                resource_usage={},
                created_files=[],
                deleted_files=[]
            )
        
        except Exception as e:
            logger.error(f"Error waiting for container completion {execution_id}: {e}")
            return ExecutionResult(
                execution_id=execution_id,
                status=SandboxStatus.FAILED,
                exit_code=None,
                stdout="",
                stderr=str(e),
                error_message=str(e),
                execution_time=asyncio.get_event_loop_time() - start_time,
                resource_usage={},
                created_files=[],
                deleted_files=[]
            )
    
    def _build_docker_command(
        self,
        language: str,
        code_file: str,
        additional_files: List[Path],
        environment: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """构建Docker命令"""
        command = []
        
        # 基础命令
        if language == "python":
            command = ["python", code_file.name]
        elif language == "python3":
            command = ["python3", code_file.name]
        elif language == "node":
            command = ["node", code_file.name]
        elif language == "npm":
            command = ["npm", "install"]
        elif language == "yarn":
            command = ["yarn", "install"]
        elif language == "java":
            command = ["java", code_file.name]
        elif language == "javac":
            command = ["javac", code_file.name]
        elif language == "go":
            command = ["go", "run", code_file.name]
        elif language == "rustc":
            command = ["rustc", code_file.name]
        elif language == "cargo":
            command = ["cargo", "run"]
        elif language == "ruby":
            command = ["ruby", code_file.name]
        elif language == "php":
            command = ["php", "-r", code_file.name]
        elif language == "bash":
            command = ["bash", code_file.name]
        elif language == "sh":
            command = ["sh", code_file.name]
        else:
            command = ["python", code_file.name]
        
        # 添加额外文件参数
        for file_path in additional_files:
            command.extend([str(file_path)])
        
        return command
    
    def _create_container_config(self, temp_dir: str) -> Dict[str, Any]:
        """创建容器配置"""
        config = {
            "working_dir": "/workspace",
            "mem_limit": self.config.memory_limit,
            "cpu_quota": self.config.cpu_limit,
            "network_disabled": not self.config.network_enabled,
            "read_only": self.config.read_only_filesystem,
            "tmpfs_size": "100m",
            "security_opt": [
                "no-new-privileges:true",
                "no-suid:true",
                "no-root:true",
                "seccomp:default",
                "apparmor:docker-default"
            ]
        }
        
        return config
    
    def _merge_environment(
        self,
        additional_env: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """合并环境变量"""
        env = self.config.environment_variables.copy()
        
        if additional_env:
            env.update(additional_env)
        
        return env
    
    def _get_file_extension(self, language: str) -> str:
        """获取文件扩展名"""
        extensions = {
            "python": ".py",
            "python3": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "node": ".js",
            "npm": ".json",
            "yarn": ".json",
            "java": ".java",
            "javac": ".java",
            "go": ".go",
            "rustc": ".rs",
            "cargo": ".rs",
            "ruby": ".rb",
            "php": ".php",
            "bash": ".sh",
            "shell": ".sh"
        }
        
        return extensions.get(language, ".py")
    
    async def get_container_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """获取容器状态"""
        if execution_id not in self.active_containers:
            return None
        
        container = self.active_containers[execution_id]
        
        try:
            # 获取容器状态
            status = container.status
            inspect = await container.inspect()
            stats = await container.stats()
            
            return {
                "execution_id": execution_id,
                "status": status,
                "exit_code": inspect["State"]["ExitCode"],
                "created": inspect["Created"],
                "started": inspect["State"]["StartedAt"],
                "finished": inspect["State"]["FinishedAt"],
                "stats": stats,
                "resource_usage": {
                    "cpu_usage": stats.get("CPUStats", {}),
                    "memory_usage": stats.get("MemoryStats", {}),
                    "network_io": stats.get("NetworkIO", {}),
                    "block_io": stats.get("BlockIO", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting container status for {execution_id}: {e}")
            return None
    
    async def stop_execution(self, execution_id: str) -> bool:
        """停止执行"""
        if execution_id not in self.active_containers:
            return False
        
        try:
            container = self.active_containers[execution_id]
            await container.stop()
            del self.active_containers[execution_id]
            
            logger.info(f"Stopped execution {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping execution {execution_id}: {e}")
            return False
    
    async def cleanup(self):
        """清理所有活跃容器"""
        for execution_id, container in list(self.active_containers.items()):
            try:
                await container.stop()
                logger.info(f"Cleaned up execution {execution_id}")
            except Exception as e:
                logger.error(f"Error cleaning up execution {execution_id}: {e}")
        
        self.active_containers.clear()
        
        # 清理Docker客户端
        if self.client:
            await self.client.close()
            self.client = None
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """获取执行历史"""
        return self.execution_history.copy()
    
    def get_active_executions(self) -> List[str]:
        """获取活跃的执行ID列表"""
        return list(self.active_containers.keys())


# 全局实例
code_sandbox = CodeSandbox()