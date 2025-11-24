#!/usr/bin/env python3
"""
项目初始化脚本 - 快速设置开发环境

用法:
    python scripts/init_project.py --env development
    python scripts/init_project.py --env test
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, Any

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"


class ProjectInitializer:
    """项目初始化器"""

    def __init__(self, env: str = "development"):
        self.env = env
        self.project_root = PROJECT_ROOT
        self.backend_root = BACKEND_ROOT

    def check_requirements(self) -> bool:
        """检查系统依赖"""
        print("🔍 检查系统依赖...")

        requirements = {
            "python": "3.11+",
            "docker": "latest",
            "docker compose": "latest",
        }

        for tool, version in requirements.items():
            try:
                if tool == "python":
                    result = subprocess.run(["python3", "--version"],
                                          capture_output=True, text=True)
                elif tool == "docker compose":
                    result = subprocess.run(["docker", "compose", "--version"],
                                          capture_output=True, text=True)
                else:
                    result = subprocess.run([tool, "--version"],
                                          capture_output=True, text=True)

                if result.returncode == 0:
                    print(f"  ✓ {tool}: {result.stdout.strip()}")
                else:
                    print(f"  ✗ {tool}: 未找到")
                    return False
            except Exception as e:
                print(f"  ✗ {tool}: {e}")
                return False

        return True

    def setup_docker_services(self) -> bool:
        """启动 Docker 服务"""
        print("\n🐳 启动 Docker 服务...")

        compose_file = self.backend_root / "docker-compose.yml"
        if self.env == "test":
            compose_file = self.backend_root / "docker-compose.test.yml"

        if not compose_file.exists():
            print(f"  ✗ Docker Compose 文件不存在: {compose_file}")
            return False

        try:
            # 启动服务
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "up", "-d"],
                cwd=str(self.backend_root),
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                print("  ✓ Docker 服务启动成功")

                # 等待服务就绪
                self._wait_for_services()
                return True
            else:
                print(f"  ✗ Docker 服务启动失败:")
                print(f"    {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("  ✗ Docker 服务启动超时")
            return False
        except Exception as e:
            print(f"  ✗ Docker 服务启动异常: {e}")
            return False

    def setup_python_environment(self) -> bool:
        """设置 Python 虚拟环境"""
        print("\n🐍 设置 Python 虚拟环境...")

        venv_path = self.backend_root / "venv"

        try:
            # 创建虚拟环境
            if not venv_path.exists():
                subprocess.run(
                    ["python3", "-m", "venv", str(venv_path)],
                    check=True,
                    capture_output=True
                )
                print(f"  ✓ 虚拟环境创建: {venv_path}")

            # 激活并安装依赖
            pip_path = venv_path / "bin" / "pip"
            requirements_file = self.backend_root / "requirements.txt"

            if requirements_file.exists():
                subprocess.run(
                    [str(pip_path), "install", "-r", str(requirements_file)],
                    check=True,
                    capture_output=True,
                    timeout=300
                )
                print(f"  ✓ 依赖包安装完成")

            return True

        except subprocess.CalledProcessError as e:
            print(f"  ✗ 虚拟环境设置失败: {e}")
            return False
        except Exception as e:
            print(f"  ✗ 虚拟环境设置异常: {e}")
            return False

    def initialize_database(self) -> bool:
        """初始化数据库"""
        print("\n🗄️  初始化数据库...")

        try:
            # 等待 PostgreSQL 启动
            self._wait_for_postgres()

            # 运行迁移
            alembic_ini = self.backend_root / "alembic.ini"
            if alembic_ini.exists():
                subprocess.run(
                    ["alembic", "upgrade", "head"],
                    cwd=str(self.backend_root),
                    capture_output=True,
                    timeout=60
                )
                print("  ✓ 数据库迁移完成")

            return True

        except subprocess.TimeoutExpired:
            print("  ✗ 数据库初始化超时")
            return False
        except Exception as e:
            print(f"  ✗ 数据库初始化异常: {e}")
            return False

    def create_env_file(self) -> bool:
        """创建环境变量文件"""
        print("\n📝 创建环境变量文件...")

        env_file = self.backend_root / ".env"

        env_config = {
            "DEVELOPMENT": {
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG",
                "DATABASE_URL": "postgresql://postgres:password@localhost:5432/code_review",
                "REDIS_URL": "redis://localhost:6379/0",
                "API_HOST": "127.0.0.1",
                "API_PORT": "8000",
            },
            "TEST": {
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG",
                "DATABASE_URL": "postgresql://postgres:password@localhost:5433/code_review_test",
                "REDIS_URL": "redis://localhost:6380/0",
                "API_HOST": "127.0.0.1",
                "API_PORT": "8001",
            },
            "PRODUCTION": {
                "DEBUG": "false",
                "LOG_LEVEL": "INFO",
                "DATABASE_URL": "${DATABASE_URL}",
                "REDIS_URL": "${REDIS_URL}",
                "API_HOST": "0.0.0.0",
                "API_PORT": "8000",
            }
        }

        config = env_config.get(self.env.upper(), env_config["DEVELOPMENT"])

        try:
            with open(env_file, "w") as f:
                for key, value in config.items():
                    f.write(f"{key}={value}\n")

            print(f"  ✓ 环境变量文件创建: {env_file}")
            return True

        except Exception as e:
            print(f"  ✗ 环境变量文件创建失败: {e}")
            return False

    def run_tests(self) -> bool:
        """运行测试"""
        print("\n🧪 运行单元测试...")

        try:
            pytest_args = [
                "pytest",
                "tests/",
                "-v",
                "--tb=short",
                "-x",  # 第一个失败后停止
            ]

            if self.env == "test":
                pytest_args.append("--cov=app")  # 覆盖率检查

            result = subprocess.run(
                pytest_args,
                cwd=str(self.backend_root),
                timeout=300
            )

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print("  ✗ 测试超时")
            return False
        except Exception as e:
            print(f"  ✗ 测试异常: {e}")
            return False

    def print_summary(self, success: bool) -> None:
        """打印初始化摘要"""
        print("\n" + "=" * 60)

        if success:
            print("✅ 项目初始化成功！")
            print("\n📌 后续步骤:")
            print(f"  1. 进入项目: cd {self.backend_root}")
            print(f"  2. 激活虚拟环境: source venv/bin/activate")
            print(f"  3. 运行应用: python -m app.main")
            print(f"  4. 访问 API: http://127.0.0.1:8000/docs")
        else:
            print("❌ 项目初始化失败！")
            print("\n⚠️  请检查上述错误信息并重试。")

        print("=" * 60)

    def _wait_for_services(self, max_retries: int = 30) -> None:
        """等待 Docker 服务就绪"""
        import time

        print("  ⏳ 等待 Docker 服务启动...", end="", flush=True)

        for i in range(max_retries):
            try:
                # 检查 PostgreSQL
                subprocess.run(
                    ["pg_isready", "-h", "localhost", "-p", "5432"],
                    capture_output=True,
                    check=True,
                    timeout=5
                )

                # 检查 Redis
                subprocess.run(
                    ["redis-cli", "-h", "localhost", "-p", "6379", "ping"],
                    capture_output=True,
                    check=True,
                    timeout=5
                )

                print(" ✓")
                return

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print(".", end="", flush=True)
                time.sleep(1)

        print(" ⏱️ (继续执行，可能需要再等待...)")

    def _wait_for_postgres(self, max_retries: int = 30) -> None:
        """等待 PostgreSQL 启动"""
        import time

        print("  ⏳ 等待 PostgreSQL 启动...", end="", flush=True)

        for i in range(max_retries):
            try:
                subprocess.run(
                    ["pg_isready", "-h", "localhost", "-p", "5432"],
                    capture_output=True,
                    check=True,
                    timeout=5
                )
                print(" ✓")
                return

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print(".", end="", flush=True)
                time.sleep(1)

        print(" ⏱️")

    def run(self) -> int:
        """执行初始化"""
        print(f"🚀 开始项目初始化 (环境: {self.env})")
        print("=" * 60)

        steps = [
            ("系统依赖检查", self.check_requirements),
            ("创建环境变量", self.create_env_file),
            ("启动 Docker 服务", self.setup_docker_services),
            ("Python 环境设置", self.setup_python_environment),
            ("数据库初始化", self.initialize_database),
        ]

        if self.env == "test":
            steps.append(("运行单元测试", self.run_tests))

        for step_name, step_func in steps:
            if not step_func():
                self.print_summary(False)
                return 1

        self.print_summary(True)
        return 0


def main():
    parser = argparse.ArgumentParser(description="智能代码审查平台 - 项目初始化")
    parser.add_argument(
        "--env",
        choices=["development", "test", "production"],
        default="development",
        help="环境"
    )

    args = parser.parse_args()

    initializer = ProjectInitializer(env=args.env)
    return initializer.run()


if __name__ == "__main__":
    sys.exit(main())
