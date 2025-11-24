"""
Database connection pooling optimization
"""

import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)

class DatabasePool:
    """Optimized database connection pool"""
    
    def __init__(self, engine, pool_size=20, max_overflow=30):
        self.engine = engine
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        
        # Configure connection pool
        self.engine.pool_size = pool_size
        self.engine.max_overflow = max_overflow
        self.engine.pool_timeout = 30
        self.engine.pool_recycle = 3600
        
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with connection pooling"""
        async_session = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        try:
            async with async_session() as session:
                yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await self.engine.dispose()

# Usage example
db_pool = DatabasePool(engine)