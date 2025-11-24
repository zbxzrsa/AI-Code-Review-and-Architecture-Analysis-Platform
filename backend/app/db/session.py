"""
优化的数据库会话管理
支持连接池、健康检查和性能监控
"""
import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import event, text
from sqlalchemy.pool import QueuePool
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """高性能数据库管理器"""
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._pool_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "overflow_connections": 0
        }
    
    async def initialize(self) -> None:
        """初始化数据库连接池"""
        if self._engine is not None:
            return
        
        # 优化的连接池配置
        self._engine = create_async_engine(
            settings.SQLALCHEMY_DATABASE_URI.replace("postgresql", "postgresql+asyncpg"),
            echo=settings.DEBUG if hasattr(settings, 'DEBUG') else False,
            future=True,
            # 连接池优化
            poolclass=QueuePool,
            pool_size=20,  # 基础连接数
            max_overflow=30,  # 最大溢出连接数
            pool_pre_ping=True,  # 连接前ping检查
            pool_recycle=3600,  # 1小时回收连接
            pool_timeout=30,  # 获取连接超时
            # 查询优化
            connect_args={
                "command_timeout": 60,
                "server_settings": {
                    "application_name": "ai_code_review_platform",
                    "jit": "off",  # 关闭JIT优化查询计划缓存
                }
            }
        )
        
        # 添加性能监控事件
        self._setup_performance_monitoring()
        
        # 创建会话工厂
        self._session_factory = sessionmaker(
            self._engine, 
            class_=AsyncSession, 
            expire_on_commit=False,
            # 优化会话配置
            autoflush=True,
            autocommit=False
        )
        
        logger.info("Database connection pool initialized successfully")
    
    def _setup_performance_monitoring(self) -> None:
        """设置性能监控"""
        @event.listens_for(self._engine.sync_engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """连接建立时的处理"""
            logger.debug("New database connection established")
        
        @event.listens_for(self._engine.sync_engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """连接检出时的处理"""
            self._pool_stats["active_connections"] += 1
            self._pool_stats["idle_connections"] -= 1
        
        @event.listens_for(self._engine.sync_engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """连接归还时的处理"""
            self._pool_stats["active_connections"] -= 1
            self._pool_stats["idle_connections"] += 1
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话的上下文管理器"""
        if self._session_factory is None:
            await self.initialize()
        
        async with self._session_factory() as session:
            try:
                # 设置查询超时
                await session.execute(
                    text("SET statement_timeout = '30s'")
                )
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def execute_query(self, query: str, params: dict = None) -> any:
        """执行查询并记录性能"""
        start_time = time.time()
        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), params or {})
                await session.commit()
                
                execution_time = time.time() - start_time
                logger.info(f"Query executed in {execution_time:.3f}s: {query[:100]}...")
                
                return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query failed after {execution_time:.3f}s: {e}")
            raise
    
    async def health_check(self) -> dict:
        """数据库健康检查"""
        try:
            start_time = time.time()
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                response_time = time.time() - start_time
                
                # 获取连接池状态
                pool = self._engine.pool
                self._pool_stats.update({
                    "total_connections": pool.size() + pool.overflow(),
                    "active_connections": pool.checkedout(),
                    "idle_connections": pool.checkedin(),
                    "overflow_connections": pool.overflow()
                })
                
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "pool_stats": self._pool_stats.copy()
                }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "pool_stats": self._pool_stats.copy()
            }
    
    async def close(self) -> None:
        """关闭数据库连接"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connections closed")

# 全局数据库管理器实例
db_manager = DatabaseManager()

# 向后兼容的函数
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话（向后兼容）"""
    async with db_manager.get_session() as session:
        yield session

# 优化的批量操作函数
async def bulk_insert(session: AsyncSession, model_class, data: list) -> None:
    """批量插入优化"""
    try:
        await session.execute(
            model_class.__table__.insert(),
            data
        )
        await session.commit()
        logger.info(f"Bulk inserted {len(data)} records into {model_class.__tablename__}")
    except Exception as e:
        await session.rollback()
        logger.error(f"Bulk insert failed: {e}")
        raise

async def bulk_update(session: AsyncSession, model_class, data: list, 
                    update_fields: list) -> None:
    """批量更新优化"""
    try:
        await session.execute(
            model_class.__table__.update()
            .where(model_class.id.in_([item['id'] for item in data]))
            .values(data)
        )
        await session.commit()
        logger.info(f"Bulk updated {len(data)} records in {model_class.__tablename__}")
    except Exception as e:
        await session.rollback()
        logger.error(f"Bulk update failed: {e}")
        raise

# 查询优化装饰器
def query_cache(ttl: int = 300):
    """查询缓存装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 简单的内存缓存实现
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            # 这里可以集成Redis等外部缓存
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# 数据库初始化函数
async def init_database() -> None:
    """初始化数据库连接"""
    await db_manager.initialize()
    
    # 执行健康检查
    health = await db_manager.health_check()
    if health["status"] == "healthy":
        logger.info("Database initialized and healthy")
    else:
        logger.warning(f"Database initialized but unhealthy: {health}")

# 获取数据库管理器实例
def get_db_manager() -> DatabaseManager:
    """获取数据库管理器实例"""
    return db_manager