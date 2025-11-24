"""
第三方服务集成测试
测试Redis、Neo4j、外部API等第三方服务集成
"""
import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock
import redis.asyncio as redis
from neo4j import AsyncGraphDatabase

from app.core.config import settings
from app.core.dependencies import Cache, get_cache


class TestRedisIntegration:
    """Redis集成测试"""
    
    @pytest.mark.asyncio
    async def test_redis_connection(self):
        """测试Redis连接"""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True
            
            # 测试连接
            redis_client = redis.from_url(settings.REDIS_URL)
            result = await redis_client.ping()
            
            assert result is True
            mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_cache_operations(self):
        """测试Redis缓存操作"""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            # 模拟缓存操作
            mock_client.set.return_value = True
            mock_client.get.return_value = json.dumps({"result": "cached_data"})
            mock_client.delete.return_value = 1
            mock_client.exists.return_value = 1
            mock_client.expire.return_value = True
            
            cache = Cache()
            
            # 测试设置缓存
            await cache.set("test_key", {"result": "cached_data"}, expire=3600)
            mock_client.set.assert_called_once()
            
            # 测试获取缓存
            result = await cache.get("test_key")
            assert result == {"result": "cached_data"}
            mock_client.get.assert_called_once()
            
            # 测试删除缓存
            await cache.delete("test_key")
            mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_pub_sub(self):
        """测试Redis发布订阅"""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_pubsub = AsyncMock()
            mock_client.pubsub.return_value = mock_pubsub
            
            # 模拟订阅
            mock_pubsub.subscribe.return_value = None
            mock_pubsub.listen.return_value = [
                {"type": "message", "data": b'{"event": "analysis_complete", "project_id": 1}'}
            ]
            
            redis_client = redis.from_url(settings.REDIS_URL)
            pubsub = redis_client.pubsub()
            
            # 订阅频道
            await pubsub.subscribe("analysis_events")
            
            # 监听消息
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    assert data["event"] == "analysis_complete"
                    assert data["project_id"] == 1
                    break
            
            mock_pubsub.subscribe.assert_called_once_with("analysis_events")

    @pytest.mark.asyncio
    async def test_redis_rate_limiting(self):
        """测试Redis限流"""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            # 模拟限流逻辑
            mock_client.incr.return_value = 1
            mock_client.expire.return_value = True
            mock_client.get.return_value = b'5'  # 当前请求数
            
            redis_client = redis.from_url(settings.REDIS_URL)
            
            # 测试限流检查
            key = "rate_limit:user:123"
            current_requests = await redis_client.incr(key)
            
            if current_requests == 1:
                await redis_client.expire(key, 60)  # 设置过期时间
            
            # 检查是否超过限制
            total_requests = int(await redis_client.get(key))
            is_rate_limited = total_requests > 100  # 假设限制为100次/分钟
            
            assert not is_rate_limited
            mock_client.incr.assert_called_once_with(key)

    @pytest.mark.asyncio
    async def test_redis_session_storage(self):
        """测试Redis会话存储"""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            session_data = {
                "user_id": 123,
                "username": "testuser",
                "permissions": ["read", "write"],
                "last_activity": "2024-01-01T12:00:00Z"
            }
            
            # 模拟会话操作
            mock_client.hset.return_value = 1
            mock_client.hgetall.return_value = {
                k.encode(): json.dumps(v).encode() if not isinstance(v, str) else v.encode()
                for k, v in session_data.items()
            }
            mock_client.expire.return_value = True
            
            redis_client = redis.from_url(settings.REDIS_URL)
            session_key = "session:abc123"
            
            # 存储会话
            for key, value in session_data.items():
                await redis_client.hset(session_key, key, json.dumps(value))
            await redis_client.expire(session_key, 3600)
            
            # 获取会话
            stored_session = await redis_client.hgetall(session_key)
            
            assert len(stored_session) == len(session_data)
            mock_client.hset.assert_called()
            mock_client.expire.assert_called_once_with(session_key, 3600)


class TestNeo4jIntegration:
    """Neo4j集成测试"""
    
    @pytest.mark.asyncio
    async def test_neo4j_connection(self):
        """测试Neo4j连接"""
        with patch('neo4j.AsyncGraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            mock_session.run.return_value.single.return_value = {"result": 1}
            
            # 测试连接
            driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            
            async with driver.session() as session:
                result = await session.run("RETURN 1 as result")
                record = await result.single()
                assert record["result"] == 1

    @pytest.mark.asyncio
    async def test_neo4j_code_graph_operations(self):
        """测试Neo4j代码图操作"""
        with patch('neo4j.AsyncGraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            # 模拟创建节点
            mock_session.run.return_value.single.return_value = {
                "id": 1,
                "name": "UserService",
                "type": "class"
            }
            
            driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            
            async with driver.session() as session:
                # 创建代码节点
                result = await session.run("""
                    CREATE (n:CodeNode {
                        name: $name,
                        type: $type,
                        file_path: $file_path,
                        line_number: $line_number
                    })
                    RETURN id(n) as id, n.name as name, n.type as type
                """, {
                    "name": "UserService",
                    "type": "class",
                    "file_path": "/src/services/user_service.py",
                    "line_number": 10
                })
                
                record = await result.single()
                assert record["name"] == "UserService"
                assert record["type"] == "class"

    @pytest.mark.asyncio
    async def test_neo4j_dependency_analysis(self):
        """测试Neo4j依赖分析"""
        with patch('neo4j.AsyncGraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            # 模拟依赖查询结果
            mock_records = [
                {"source": "UserService", "target": "DatabaseService", "type": "DEPENDS_ON"},
                {"source": "UserService", "target": "EmailService", "type": "USES"},
                {"source": "DatabaseService", "target": "ConfigService", "type": "DEPENDS_ON"}
            ]
            mock_session.run.return_value.data.return_value = mock_records
            
            driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            
            async with driver.session() as session:
                # 查询依赖关系
                result = await session.run("""
                    MATCH (source:CodeNode)-[r:DEPENDS_ON|USES]->(target:CodeNode)
                    WHERE source.name = $service_name
                    RETURN source.name as source, target.name as target, type(r) as type
                """, {"service_name": "UserService"})
                
                dependencies = await result.data()
                assert len(dependencies) >= 2
                
                # 验证依赖关系
                dep_targets = [dep["target"] for dep in dependencies]
                assert "DatabaseService" in dep_targets
                assert "EmailService" in dep_targets

    @pytest.mark.asyncio
    async def test_neo4j_circular_dependency_detection(self):
        """测试Neo4j循环依赖检测"""
        with patch('neo4j.AsyncGraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            # 模拟循环依赖查询结果
            mock_records = [
                {"cycle": ["ServiceA", "ServiceB", "ServiceC", "ServiceA"]}
            ]
            mock_session.run.return_value.data.return_value = mock_records
            
            driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            
            async with driver.session() as session:
                # 检测循环依赖
                result = await session.run("""
                    MATCH path = (start:CodeNode)-[:DEPENDS_ON*]->(start)
                    WHERE length(path) > 2
                    RETURN [node in nodes(path) | node.name] as cycle
                    LIMIT 10
                """)
                
                cycles = await result.data()
                
                if cycles:
                    cycle = cycles[0]["cycle"]
                    assert cycle[0] == cycle[-1]  # 循环的起点和终点相同
                    assert len(cycle) > 3  # 至少包含3个不同的节点

    @pytest.mark.asyncio
    async def test_neo4j_code_metrics_calculation(self):
        """测试Neo4j代码指标计算"""
        with patch('neo4j.AsyncGraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            # 模拟指标计算结果
            mock_session.run.return_value.single.return_value = {
                "total_nodes": 150,
                "total_relationships": 300,
                "avg_dependencies": 2.5,
                "max_depth": 5
            }
            
            driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            
            async with driver.session() as session:
                # 计算代码指标
                result = await session.run("""
                    MATCH (n:CodeNode)
                    OPTIONAL MATCH (n)-[r:DEPENDS_ON]->()
                    WITH count(DISTINCT n) as total_nodes, 
                         count(r) as total_relationships,
                         avg(size((n)-[:DEPENDS_ON]->())) as avg_dependencies
                    MATCH path = (start:CodeNode)-[:DEPENDS_ON*]->(end:CodeNode)
                    WHERE NOT (end)-[:DEPENDS_ON]->()
                    RETURN total_nodes, total_relationships, avg_dependencies,
                           max(length(path)) as max_depth
                """)
                
                metrics = await result.single()
                assert metrics["total_nodes"] > 0
                assert metrics["total_relationships"] >= 0
                assert metrics["avg_dependencies"] >= 0


class TestExternalAPIIntegration:
    """外部API集成测试"""
    
    @pytest.mark.asyncio
    async def test_github_api_integration(self):
        """测试GitHub API集成"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "name": "test-repo",
                "full_name": "user/test-repo",
                "description": "Test repository",
                "language": "Python",
                "stargazers_count": 100,
                "forks_count": 20
            }
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            # 测试GitHub API调用
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.github.com/repos/user/test-repo",
                    headers={"Authorization": f"token {settings.GITHUB_TOKEN}"}
                )
                
                assert response.status_code == 200
                repo_data = response.json()
                assert repo_data["name"] == "test-repo"
                assert repo_data["language"] == "Python"

    @pytest.mark.asyncio
    async def test_openai_api_integration(self):
        """测试OpenAI API集成"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
                ],
                "model": "text-embedding-ada-002",
                "usage": {"total_tokens": 10}
            }
            
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            # 测试OpenAI API调用
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "input": "def hello_world(): return 'Hello, World!'",
                        "model": "text-embedding-ada-002"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "data" in data
                assert len(data["data"][0]["embedding"]) == 5

    @pytest.mark.asyncio
    async def test_webhook_integration(self):
        """测试Webhook集成"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = "OK"
            
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            # 测试发送Webhook
            webhook_payload = {
                "event": "analysis_completed",
                "project_id": 123,
                "results": {
                    "defects_found": 5,
                    "security_issues": 2,
                    "code_quality_score": 85
                },
                "timestamp": "2024-01-01T12:00:00Z"
            }
            
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://webhook.example.com/analysis-complete",
                    json=webhook_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                assert response.status_code == 200
                assert response.text == "OK"

    @pytest.mark.asyncio
    async def test_email_service_integration(self):
        """测试邮件服务集成"""
        with patch('smtplib.SMTP_SSL') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            # 测试发送邮件
            mock_server.login.return_value = None
            mock_server.send_message.return_value = {}
            
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = settings.SMTP_USER
            msg['To'] = "user@example.com"
            msg['Subject'] = "Analysis Complete"
            
            body = "Your code analysis has been completed."
            msg.attach(MIMEText(body, 'plain'))
            
            # 发送邮件
            with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(msg)
            
            mock_server.login.assert_called_once()
            mock_server.send_message.assert_called_once()


class TestServiceHealthChecks:
    """服务健康检查测试"""
    
    @pytest.mark.asyncio
    async def test_redis_health_check(self):
        """测试Redis健康检查"""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            # 测试健康的Redis
            mock_client.ping.return_value = True
            redis_client = redis.from_url(settings.REDIS_URL)
            is_healthy = await redis_client.ping()
            assert is_healthy is True
            
            # 测试不健康的Redis
            mock_client.ping.side_effect = Exception("Connection failed")
            try:
                await redis_client.ping()
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Connection failed" in str(e)

    @pytest.mark.asyncio
    async def test_neo4j_health_check(self):
        """测试Neo4j健康检查"""
        with patch('neo4j.AsyncGraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            # 测试健康的Neo4j
            mock_session.run.return_value.single.return_value = {"result": 1}
            
            driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            
            async with driver.session() as session:
                result = await session.run("RETURN 1 as result")
                record = await result.single()
                assert record["result"] == 1
            
            # 测试不健康的Neo4j
            mock_session.run.side_effect = Exception("Database unavailable")
            
            try:
                async with driver.session() as session:
                    await session.run("RETURN 1 as result")
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Database unavailable" in str(e)

    @pytest.mark.asyncio
    async def test_external_api_health_check(self):
        """测试外部API健康检查"""
        with patch('httpx.AsyncClient') as mock_client:
            # 测试健康的API
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok"}
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("https://api.example.com/health")
                assert response.status_code == 200
                assert response.json()["status"] == "ok"
            
            # 测试不健康的API
            mock_response.status_code = 503
            mock_response.json.return_value = {"status": "error", "message": "Service unavailable"}
            
            async with httpx.AsyncClient() as client:
                response = await client.get("https://api.example.com/health")
                assert response.status_code == 503
                assert response.json()["status"] == "error"


class TestServiceFailover:
    """服务故障转移测试"""
    
    @pytest.mark.asyncio
    async def test_redis_failover(self):
        """测试Redis故障转移"""
        with patch('redis.asyncio.from_url') as mock_redis:
            # 模拟主Redis失败，备用Redis正常
            primary_client = AsyncMock()
            backup_client = AsyncMock()
            
            primary_client.ping.side_effect = Exception("Primary Redis down")
            backup_client.ping.return_value = True
            backup_client.get.return_value = json.dumps({"data": "from_backup"})
            
            # 故障转移逻辑
            try:
                primary_redis = redis.from_url(settings.REDIS_URL)
                await primary_redis.ping()
                redis_client = primary_redis
            except Exception:
                # 切换到备用Redis
                backup_redis = redis.from_url(settings.REDIS_BACKUP_URL)
                await backup_redis.ping()
                redis_client = backup_redis
            
            # 验证使用备用Redis
            mock_redis.return_value = backup_client
            result = await redis_client.get("test_key")
            data = json.loads(result)
            assert data["data"] == "from_backup"

    @pytest.mark.asyncio
    async def test_database_connection_retry(self):
        """测试数据库连接重试"""
        with patch('sqlalchemy.ext.asyncio.create_async_engine') as mock_engine:
            mock_conn = AsyncMock()
            
            # 模拟前两次连接失败，第三次成功
            mock_engine.return_value.begin.side_effect = [
                Exception("Connection failed"),
                Exception("Connection failed"),
                mock_conn
            ]
            
            # 重试逻辑
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    engine = mock_engine.return_value
                    async with engine.begin() as conn:
                        # 连接成功
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(1)  # 等待重试
            
            # 验证最终连接成功
            assert mock_engine.return_value.begin.call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self):
        """测试断路器模式"""
        class CircuitBreaker:
            def __init__(self, failure_threshold=3, timeout=60):
                self.failure_threshold = failure_threshold
                self.timeout = timeout
                self.failure_count = 0
                self.last_failure_time = None
                self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
            
            async def call(self, func, *args, **kwargs):
                if self.state == "OPEN":
                    if self.last_failure_time and \
                       (asyncio.get_event_loop().time() - self.last_failure_time) > self.timeout:
                        self.state = "HALF_OPEN"
                    else:
                        raise Exception("Circuit breaker is OPEN")
                
                try:
                    result = await func(*args, **kwargs)
                    if self.state == "HALF_OPEN":
                        self.state = "CLOSED"
                        self.failure_count = 0
                    return result
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = asyncio.get_event_loop().time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"
                    
                    raise e
        
        # 测试断路器
        circuit_breaker = CircuitBreaker(failure_threshold=2)
        
        async def failing_service():
            raise Exception("Service error")
        
        # 第一次失败
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_service)
        assert circuit_breaker.state == "CLOSED"
        
        # 第二次失败，断路器打开
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_service)
        assert circuit_breaker.state == "OPEN"
        
        # 第三次调用，断路器阻止调用
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await circuit_breaker.call(failing_service)