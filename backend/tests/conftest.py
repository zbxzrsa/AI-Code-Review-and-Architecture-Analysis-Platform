"""
Pytest configuration and fixtures for the AI Code Review Platform.
"""

import asyncio
import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock
from httpx import AsyncClient
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from neo4j import GraphDatabase
import redis.asyncio as redis
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.db.session import get_db
from app.models.base import Base
from app.models.project import Project
from app.models.analysis_session import AnalysisSession
from app.models.baseline import Baseline


# Test database configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"
TEST_REDIS_URL = "redis://localhost:6379/15"
TEST_NEO4J_URI = "bolt://localhost:7687"
TEST_NEO4J_USER = "neo4j"
TEST_NEO4J_PASSWORD = "password"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def async_engine() -> AsyncGenerator[AsyncSession, None]:
    """Create async database engine for testing."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
        pool_pre_ping=True,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Drop all tables after tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing."""
    async_session = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with database dependency override."""
    
    def override_get_db():
        return db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest.fixture
def sync_client() -> Generator[TestClient, None, None]:
    """Create a synchronous test client."""
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Create a Redis client for testing."""
    client = redis.from_url(TEST_REDIS_URL, decode_responses=True)
    
    # Clear test database
    await client.flushdb()
    
    yield client
    
    await client.flushdb()
    await client.close()


@pytest.fixture
def neo4j_driver() -> Generator[Mock, None, None]:
    """Create a mock Neo4j driver for testing."""
    mock_driver = Mock()
    mock_session = Mock()
    mock_result = Mock()
    
    mock_result.data.return_value = []
    mock_session.run.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    mock_driver.session.return_value = mock_session
    mock_driver.verify_connectivity = AsyncMock()
    
    yield mock_driver


# Factory fixtures
@pytest.fixture
def project_factory(db_session: AsyncSession):
    """Factory for creating test projects."""
    created_projects = []
    
    async def create_project(**kwargs):
        defaults = {
            "name": "Test Project",
            "description": "A test project",
            "repository_url": "https://github.com/test/repo.git",
            "language": "python",
            "is_active": True,
        }
        defaults.update(kwargs)
        
        project = Project(**defaults)
        db_session.add(project)
        await db_session.commit()
        await db_session.refresh(project)
        
        created_projects.append(project)
        return project
    
    yield create_project
    
    # Cleanup
    for project in created_projects:
        await db_session.delete(project)
    await db_session.commit()


@pytest.fixture
def analysis_session_factory(db_session: AsyncSession, project_factory):
    """Factory for creating test analysis sessions."""
    created_sessions = []
    
    async def create_session(**kwargs):
        # Create a project if not provided
        if "project_id" not in kwargs:
            project = await project_factory()
            kwargs["project_id"] = project.id
        
        defaults = {
            "status": "pending",
            "analysis_type": "code_review",
            "config": {},
        }
        defaults.update(kwargs)
        
        session = AnalysisSession(**defaults)
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        created_sessions.append(session)
        return session
    
    yield create_session
    
    # Cleanup
    for session in created_sessions:
        await db_session.delete(session)
    await db_session.commit()


@pytest.fixture
def baseline_factory(db_session: AsyncSession, project_factory):
    """Factory for creating test baselines."""
    created_baselines = []
    
    async def create_baseline(**kwargs):
        # Create a project if not provided
        if "project_id" not in kwargs:
            project = await project_factory()
            kwargs["project_id"] = project.id
        
        defaults = {
            "name": "Test Baseline",
            "description": "A test baseline",
            "metrics": {"complexity": 10, "coverage": 80},
            "thresholds": {"complexity_max": 15, "coverage_min": 70},
            "is_active": True,
        }
        defaults.update(kwargs)
        
        baseline = Baseline(**defaults)
        db_session.add(baseline)
        await db_session.commit()
        await db_session.refresh(baseline)
        
        created_baselines.append(baseline)
        return baseline
    
    yield create_baseline
    
    # Cleanup
    for baseline in created_baselines:
        await db_session.delete(baseline)
    await db_session.commit()


# Mock fixtures
@pytest.fixture
def mock_ai_service():
    """Mock AI analysis service."""
    service = Mock()
    service.analyze_code = AsyncMock(return_value={
        "complexity": 10,
        "quality_score": 85,
        "suggestions": ["Add docstring", "Extract method"],
        "security_issues": [],
        "performance_issues": [],
    })
    service.get_code_metrics = AsyncMock(return_value={
        "lines_of_code": 100,
        "cyclomatic_complexity": 5,
        "maintainability_index": 75,
    })
    return service


@pytest.fixture
def mock_github_service():
    """Mock GitHub integration service."""
    service = Mock()
    service.get_repository_info = AsyncMock(return_value={
        "name": "test-repo",
        "full_name": "user/test-repo",
        "language": "Python",
        "stars": 42,
        "forks": 10,
    })
    service.get_pull_requests = AsyncMock(return_value=[])
    service.create_webhook = AsyncMock(return_value={"id": "webhook-123"})
    return service


@pytest.fixture
def mock_celery_worker():
    """Mock Celery worker for testing."""
    worker = Mock()
    worker.apply_async = Mock(return_value=Mock(id="task-123"))
    worker.state = "SUCCESS"
    worker.result = {"status": "completed", "data": {}}
    return worker


# Data fixtures
@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
'''


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for testing."""
    return '''
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
}
'''


@pytest.fixture
def sample_analysis_result():
    """Sample analysis result for testing."""
    return {
        "file_path": "src/calculator.py",
        "language": "python",
        "metrics": {
            "lines_of_code": 15,
            "cyclomatic_complexity": 3,
            "maintainability_index": 85,
            "technical_debt_ratio": 0.1,
        },
        "issues": [
            {
                "type": "style",
                "severity": "info",
                "message": "Missing docstring for class",
                "line": 8,
                "column": 1,
            },
            {
                "type": "complexity",
                "severity": "warning",
                "message": "Function complexity is high",
                "line": 2,
                "column": 1,
            },
        ],
        "suggestions": [
            "Add type hints for better code documentation",
            "Consider using memoization for fibonacci function",
        ],
        "security_issues": [],
        "performance_issues": [],
        "quality_score": 78,
    }


# Environment fixtures
@pytest.fixture
def test_env_vars(monkeypatch):
    """Set test environment variables."""
    test_vars = {
        "DATABASE_URL": TEST_DATABASE_URL,
        "REDIS_URL": TEST_REDIS_URL,
        "SECRET_KEY": "test-secret-key",
        "JWT_SECRET_KEY": "test-jwt-secret",
        "DEBUG": "true",
        "TESTING": "true",
        "LOG_LEVEL": "DEBUG",
    }
    
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)
    
    yield test_vars


# Performance testing fixtures
@pytest.fixture
def performance_monitor():
    """Monitor performance during tests."""
    import time
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class Metric:
        name: str
        duration: float
        memory_usage: int = 0
    
    metrics: List[Metric] = []
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            
        def start(self):
            self.start_time = time.time()
            
        def stop(self, name: str):
            if self.start_time:
                duration = time.time() - self.start_time
                metrics.append(Metric(name=name, duration=duration))
                self.start_time = None
                
        def get_metrics(self) -> List[Metric]:
            return metrics.copy()
    
    return PerformanceMonitor()


# Integration testing fixtures
@pytest.fixture
async def docker_services():
    """Ensure Docker services are running for integration tests."""
    import docker
    from docker.errors import NotFound
    
    client = docker.from_env()
    
    # Check if required services are running
    required_services = ["postgres", "redis", "neo4j"]
    running_services = []
    
    for service in required_services:
        try:
            container = client.containers.get(f"ai-code-review-{service}")
            if container.status == "running":
                running_services.append(service)
        except NotFound:
            pass
    
    if len(running_services) < len(required_services):
        pytest.skip(f"Required services not running: {set(required_services) - set(running_services)}")
    
    yield running_services
    
    # Cleanup if needed
    client.close()


# Security testing fixtures
@pytest.fixture
def security_test_data():
    """Data for security testing."""
    return {
        "malicious_code": '''
import os
import subprocess

def malicious_function():
    os.system("rm -rf /")
    subprocess.Popen(["malicious_command"])
    
# SQL injection attempt
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute_query(query)
''',
        "xss_payload": "<script>alert('xss')</script>",
        "sql_injection": "'; DROP TABLE users; --",
        "path_traversal": "../../../etc/passwd",
        "command_injection": "; cat /etc/passwd",
    }


# Error handling fixtures
@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing."""
    return {
        "database_error": Exception("Database connection failed"),
        "network_timeout": TimeoutError("Request timed out"),
        "authentication_error": PermissionError("Invalid credentials"),
        "rate_limit_error": Exception("Rate limit exceeded"),
        "validation_error": ValueError("Invalid input data"),
        "service_unavailable": ConnectionError("Service unavailable"),
    }