"""
工具函数单元测试
测试缓存、限流、配置管理等工具函数
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi import Request

from app.core.dependencies import Cache, RateLimiter, rate_limiter, get_cache
from app.core.config import Settings


class TestCache:
    """缓存功能测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.cache = Cache()

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """测试设置和获取缓存"""
        key = "test_key"
        value = {"data": "test_value"}
        
        await self.cache.set(key, value, expire=3600)
        result = await self.cache.get(key)
        
        assert result == value

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        result = await self.cache.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """测试缓存过期"""
        key = "expire_test"
        value = {"data": "expire_value"}
        
        # 设置1秒过期
        await self.cache.set(key, value, expire=1)
        
        # 立即获取应该成功
        result = await self.cache.get(key)
        assert result == value
        
        # 等待过期
        await asyncio.sleep(1.1)
        
        # 过期后应该返回None
        result = await self.cache.get(key)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_cache(self):
        """测试删除缓存"""
        key = "delete_test"
        value = {"data": "delete_value"}
        
        await self.cache.set(key, value)
        await self.cache.delete(key)
        
        result = await self.cache.get(key)
        assert result is None

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """测试清空缓存"""
        await self.cache.set("key1", {"data": "value1"})
        await self.cache.set("key2", {"data": "value2"})
        
        await self.cache.clear()
        
        assert await self.cache.get("key1") is None
        assert await self.cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_cache_overwrite(self):
        """测试缓存覆写"""
        key = "overwrite_test"
        value1 = {"data": "value1"}
        value2 = {"data": "value2"}
        
        await self.cache.set(key, value1)
        await self.cache.set(key, value2)
        
        result = await self.cache.get(key)
        assert result == value2


class TestRateLimiter:
    """限流器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.rate_limiter = RateLimiter()

    @pytest.mark.asyncio
    async def test_rate_limit_within_threshold(self):
        """测试在阈值内的请求"""
        key = "test_user"
        max_requests = 5
        window_seconds = 60
        
        # 前5个请求应该都通过
        for i in range(max_requests):
            is_limited = await self.rate_limiter.is_rate_limited(key, max_requests, window_seconds)
            assert not is_limited

    @pytest.mark.asyncio
    async def test_rate_limit_exceed_threshold(self):
        """测试超过阈值的请求"""
        key = "test_user_exceed"
        max_requests = 3
        window_seconds = 60
        
        # 前3个请求通过
        for i in range(max_requests):
            is_limited = await self.rate_limiter.is_rate_limited(key, max_requests, window_seconds)
            assert not is_limited
        
        # 第4个请求应该被限制
        is_limited = await self.rate_limiter.is_rate_limited(key, max_requests, window_seconds)
        assert is_limited

    @pytest.mark.asyncio
    async def test_rate_limit_window_reset(self):
        """测试时间窗口重置"""
        key = "test_user_reset"
        max_requests = 2
        window_seconds = 1
        
        # 用完配额
        for i in range(max_requests):
            is_limited = await self.rate_limiter.is_rate_limited(key, max_requests, window_seconds)
            assert not is_limited
        
        # 应该被限制
        is_limited = await self.rate_limiter.is_rate_limited(key, max_requests, window_seconds)
        assert is_limited
        
        # 等待窗口重置
        await asyncio.sleep(1.1)
        
        # 新窗口应该可以再次请求
        is_limited = await self.rate_limiter.is_rate_limited(key, max_requests, window_seconds)
        assert not is_limited

    @pytest.mark.asyncio
    async def test_different_keys_independent(self):
        """测试不同键的独立性"""
        key1 = "user1"
        key2 = "user2"
        max_requests = 2
        window_seconds = 60
        
        # 用户1用完配额
        for i in range(max_requests):
            is_limited = await self.rate_limiter.is_rate_limited(key1, max_requests, window_seconds)
            assert not is_limited
        
        # 用户1被限制
        is_limited = await self.rate_limiter.is_rate_limited(key1, max_requests, window_seconds)
        assert is_limited
        
        # 用户2不受影响
        is_limited = await self.rate_limiter.is_rate_limited(key2, max_requests, window_seconds)
        assert not is_limited


class TestRateLimiterDependency:
    """限流依赖测试"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_dependency_success(self):
        """测试限流依赖成功情况"""
        # 创建模拟请求
        mock_request = Mock(spec=Request)
        mock_request.client.host = "127.0.0.1"
        mock_request.url.path = "/test"
        
        # 创建限流依赖
        limiter_func = rate_limiter(max_requests=5, window_seconds=60)
        
        with patch('app.core.dependencies._rate_limiter') as mock_limiter:
            mock_limiter.is_rate_limited.return_value = False
            
            # 应该正常通过
            result = await limiter_func(mock_request)
            assert result is None

    @pytest.mark.asyncio
    async def test_rate_limiter_dependency_blocked(self):
        """测试限流依赖阻止情况"""
        from fastapi import HTTPException
        
        # 创建模拟请求
        mock_request = Mock(spec=Request)
        mock_request.client.host = "127.0.0.1"
        mock_request.url.path = "/test"
        
        # 创建限流依赖
        limiter_func = rate_limiter(max_requests=5, window_seconds=60)
        
        with patch('app.core.dependencies._rate_limiter') as mock_limiter:
            mock_limiter.is_rate_limited.return_value = True
            
            # 应该抛出HTTP异常
            with pytest.raises(HTTPException) as exc_info:
                await limiter_func(mock_request)
            
            assert exc_info.value.status_code == 429


class TestSettings:
    """配置测试"""
    
    def test_default_settings(self):
        """测试默认配置"""
        settings = Settings()
        
        assert settings.API_V1_STR == "/api/v1"
        assert settings.PROJECT_NAME == "CodeInsight"
        assert settings.POSTGRES_SERVER == "postgres"
        assert settings.POSTGRES_USER == "postgres"
        assert settings.POSTGRES_DB == "codeinsight"

    def test_cors_origins_validation(self):
        """测试CORS源验证"""
        # 测试字符串格式
        settings = Settings(BACKEND_CORS_ORIGINS="http://localhost:3000,http://localhost:8000")
        assert len(settings.BACKEND_CORS_ORIGINS) == 2
        
        # 测试列表格式
        settings = Settings(BACKEND_CORS_ORIGINS=["http://localhost:3000"])
        assert len(settings.BACKEND_CORS_ORIGINS) == 1

    def test_database_uri_assembly(self):
        """测试数据库URI组装"""
        settings = Settings(
            POSTGRES_SERVER="localhost",
            POSTGRES_USER="testuser",
            POSTGRES_PASSWORD="testpass",
            POSTGRES_DB="testdb"
        )
        
        assert "postgresql://" in str(settings.SQLALCHEMY_DATABASE_URI)
        assert "testuser" in str(settings.SQLALCHEMY_DATABASE_URI)
        assert "testdb" in str(settings.SQLALCHEMY_DATABASE_URI)

    @patch.dict('os.environ', {
        'POSTGRES_SERVER': 'custom_host',
        'POSTGRES_USER': 'custom_user',
        'POSTGRES_PASSWORD': 'custom_pass',
        'POSTGRES_DB': 'custom_db'
    })
    def test_environment_variable_override(self):
        """测试环境变量覆盖"""
        settings = Settings()
        
        assert settings.POSTGRES_SERVER == "custom_host"
        assert settings.POSTGRES_USER == "custom_user"
        assert settings.POSTGRES_PASSWORD == "custom_pass"
        assert settings.POSTGRES_DB == "custom_db"

    def test_secret_key_generation(self):
        """测试密钥生成"""
        settings1 = Settings()
        settings2 = Settings()
        
        # 每次创建应该生成不同的密钥
        assert settings1.SECRET_KEY != settings2.SECRET_KEY
        assert len(settings1.SECRET_KEY) > 0

    def test_redis_url_default(self):
        """测试Redis URL默认值"""
        settings = Settings()
        assert settings.REDIS_URL == "redis://redis:6379/0"

    def test_neo4j_config_default(self):
        """测试Neo4j配置默认值"""
        settings = Settings()
        assert settings.NEO4J_URI == "bolt://neo4j:7687"
        assert settings.NEO4J_USER == "neo4j"
        assert settings.NEO4J_PASSWORD == "password"


class TestUtilityFunctions:
    """工具函数测试"""
    
    def test_get_cache_singleton(self):
        """测试缓存单例"""
        cache1 = get_cache()
        cache2 = get_cache()
        
        # 应该返回同一个实例
        assert cache1 is cache2

    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """测试缓存集成"""
        cache = get_cache()
        
        # 清空缓存确保测试环境干净
        await cache.clear()
        
        # 测试基本功能
        await cache.set("integration_test", {"value": "test"})
        result = await cache.get("integration_test")
        
        assert result == {"value": "test"}


if __name__ == "__main__":
    pytest.main([__file__])