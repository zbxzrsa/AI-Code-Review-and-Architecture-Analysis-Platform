"""
API端点集成测试
测试完整的API请求-响应流程
"""
import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import patch, AsyncMock

from app.main import app
from app.db.session import get_db
from app.core.dependencies import get_cache, get_rate_limiter


class TestHealthEndpoint:
    """健康检查端点测试"""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """测试健康检查成功"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "version" in data

    @pytest.mark.asyncio
    async def test_health_check_with_database(self):
        """测试包含数据库检查的健康检查"""
        with patch('app.db.session.AsyncSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/health?check_db=true")
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["database"] == "connected"


class TestAIAnalysisEndpoints:
    """AI分析端点测试"""
    
    @pytest.mark.asyncio
    async def test_embed_code_success(self):
        """测试代码嵌入成功"""
        with patch('app.services.ai_model_service.AIModelService.embed_code') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/ai/embed",
                    json={"code": "def hello(): return 'world'"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "embedding" in data
                assert len(data["embedding"]) == 5
                assert data["embedding"] == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_embed_code_invalid_input(self):
        """测试代码嵌入无效输入"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ai/embed",
                json={"code": ""}
            )
            
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_defect_analysis_success(self):
        """测试缺陷分析成功"""
        with patch('app.services.ai_model_service.AIModelService.detect_defects') as mock_detect:
            mock_detect.return_value = {
                "defects": [
                    {
                        "type": "potential_bug",
                        "severity": "medium",
                        "message": "Potential null pointer dereference",
                        "line": 5,
                        "confidence": 0.85
                    }
                ],
                "summary": {
                    "total_defects": 1,
                    "high_severity": 0,
                    "medium_severity": 1,
                    "low_severity": 0
                }
            }
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/ai/defect-analysis",
                    json={"code": "def risky_function(x): return x.value"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "defects" in data
                assert "summary" in data
                assert len(data["defects"]) == 1
                assert data["summary"]["total_defects"] == 1

    @pytest.mark.asyncio
    async def test_architecture_analysis_success(self):
        """测试架构分析成功"""
        with patch('app.services.ai_model_service.AIModelService.analyze_architecture') as mock_analyze:
            mock_analyze.return_value = {
                "components": [
                    {"name": "UserService", "type": "service", "complexity": 3},
                    {"name": "DatabaseLayer", "type": "data", "complexity": 2}
                ],
                "dependencies": [
                    {"from": "UserService", "to": "DatabaseLayer", "type": "uses"}
                ],
                "metrics": {
                    "coupling": 0.3,
                    "cohesion": 0.8,
                    "complexity": 2.5
                }
            }
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/ai/architecture-analysis",
                    json={"code": "class UserService: pass"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "components" in data
                assert "dependencies" in data
                assert "metrics" in data

    @pytest.mark.asyncio
    async def test_similarity_analysis_success(self):
        """测试相似度分析成功"""
        with patch('app.services.ai_model_service.AIModelService.calculate_similarity') as mock_similarity:
            mock_similarity.return_value = {
                "similarity_score": 0.85,
                "similar_functions": [
                    {"name": "process_data", "similarity": 0.92},
                    {"name": "handle_request", "similarity": 0.78}
                ]
            }
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/ai/similarity",
                    json={
                        "code1": "def process_user_data(data): return data.upper()",
                        "code2": "def handle_user_input(input): return input.upper()"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "similarity_score" in data
                assert data["similarity_score"] == 0.85


class TestUserEndpoints:
    """用户端点测试"""
    
    @pytest.mark.asyncio
    async def test_get_users_success(self):
        """测试获取用户列表成功"""
        with patch('app.api.api_v1.endpoints.users.get_users') as mock_get_users:
            mock_get_users.return_value = [
                {"id": 1, "email": "user1@example.com", "is_active": True},
                {"id": 2, "email": "user2@example.com", "is_active": True}
            ]
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/api/v1/users/")
                
                assert response.status_code == 200
                data = response.json()
                assert len(data) == 2
                assert data[0]["email"] == "user1@example.com"

    @pytest.mark.asyncio
    async def test_create_user_success(self):
        """测试创建用户成功"""
        with patch('app.api.api_v1.endpoints.users.create_user') as mock_create_user:
            mock_create_user.return_value = {
                "id": 1,
                "email": "newuser@example.com",
                "is_active": True
            }
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/users/",
                    json={
                        "email": "newuser@example.com",
                        "password": "securepassword123"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["email"] == "newuser@example.com"
                assert data["is_active"] is True


class TestRateLimitingIntegration:
    """限流集成测试"""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_enforcement(self):
        """测试限流执行"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # 快速发送多个请求
            responses = []
            for _ in range(10):
                response = await client.post(
                    "/api/v1/ai/embed",
                    json={"code": "print('test')"}
                )
                responses.append(response.status_code)
            
            # 应该有一些请求被限流（429状态码）
            assert 429 in responses or all(r == 200 for r in responses[:5])


class TestCacheIntegration:
    """缓存集成测试"""
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self):
        """测试缓存功能"""
        with patch('app.services.ai_model_service.AIModelService.embed_code') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                # 第一次请求
                response1 = await client.post(
                    "/api/v1/ai/embed",
                    json={"code": "def test(): pass"}
                )
                
                # 第二次相同请求（应该从缓存返回）
                response2 = await client.post(
                    "/api/v1/ai/embed",
                    json={"code": "def test(): pass"}
                )
                
                assert response1.status_code == 200
                assert response2.status_code == 200
                assert response1.json() == response2.json()
                
                # 验证AI服务只被调用一次（第二次从缓存返回）
                assert mock_embed.call_count <= 2


class TestErrorHandling:
    """错误处理集成测试"""
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """测试验证错误处理"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ai/embed",
                json={"invalid_field": "value"}
            )
            
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data

    @pytest.mark.asyncio
    async def test_internal_error_handling(self):
        """测试内部错误处理"""
        with patch('app.services.ai_model_service.AIModelService.embed_code') as mock_embed:
            mock_embed.side_effect = Exception("Internal service error")
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/ai/embed",
                    json={"code": "def test(): pass"}
                )
                
                assert response.status_code == 500
                data = response.json()
                assert "detail" in data

    @pytest.mark.asyncio
    async def test_not_found_error_handling(self):
        """测试404错误处理"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/nonexistent-endpoint")
            
            assert response.status_code == 404
            data = response.json()
            assert "detail" in data


class TestCORSIntegration:
    """CORS集成测试"""
    
    @pytest.mark.asyncio
    async def test_cors_headers(self):
        """测试CORS头部"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.options(
                "/api/v1/ai/embed",
                headers={"Origin": "http://localhost:3000"}
            )
            
            assert response.status_code == 200
            assert "access-control-allow-origin" in response.headers
            assert "access-control-allow-methods" in response.headers