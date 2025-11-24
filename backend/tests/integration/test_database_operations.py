"""
数据库操作集成测试
测试数据库连接、事务、CRUD操作等
"""
import pytest
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from unittest.mock import patch, AsyncMock

from app.db.session import get_db, engine
from app.db.base import Base
from app.core.config import settings


class TestDatabaseConnection:
    """数据库连接测试"""
    
    @pytest.mark.asyncio
    async def test_database_connection_success(self):
        """测试数据库连接成功"""
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_database_connection_pool(self):
        """测试数据库连接池"""
        # 创建多个并发连接
        async def test_connection():
            async with engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                return result.scalar()
        
        tasks = [test_connection() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(result == 1 for result in results)

    @pytest.mark.asyncio
    async def test_database_session_dependency(self):
        """测试数据库会话依赖"""
        async for session in get_db():
            assert isinstance(session, AsyncSession)
            # 测试会话可以执行查询
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
            break


class TestDatabaseTransactions:
    """数据库事务测试"""
    
    @pytest.mark.asyncio
    async def test_transaction_commit(self):
        """测试事务提交"""
        async with engine.begin() as conn:
            # 创建临时表
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_table (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(50)
                )
            """))
            
            # 插入数据
            await conn.execute(text("""
                INSERT INTO test_table (name) VALUES ('test_name')
            """))
            
            # 验证数据存在
            result = await conn.execute(text("""
                SELECT name FROM test_table WHERE name = 'test_name'
            """))
            assert result.scalar() == 'test_name'

    @pytest.mark.asyncio
    async def test_transaction_rollback(self):
        """测试事务回滚"""
        try:
            async with engine.begin() as conn:
                # 创建临时表
                await conn.execute(text("""
                    CREATE TEMPORARY TABLE test_rollback (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(50)
                    )
                """))
                
                # 插入数据
                await conn.execute(text("""
                    INSERT INTO test_rollback (name) VALUES ('test_rollback')
                """))
                
                # 故意引发异常
                raise Exception("Test rollback")
                
        except Exception:
            pass
        
        # 验证事务已回滚，表不存在
        async with engine.begin() as conn:
            try:
                await conn.execute(text("SELECT * FROM test_rollback"))
                assert False, "Table should not exist after rollback"
            except Exception:
                # 预期的异常，表示回滚成功
                pass

    @pytest.mark.asyncio
    async def test_nested_transactions(self):
        """测试嵌套事务"""
        async with engine.begin() as conn:
            # 创建临时表
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_nested (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(50)
                )
            """))
            
            # 外层事务
            await conn.execute(text("""
                INSERT INTO test_nested (name) VALUES ('outer')
            """))
            
            # 内层事务（保存点）
            savepoint = await conn.begin_nested()
            try:
                await conn.execute(text("""
                    INSERT INTO test_nested (name) VALUES ('inner')
                """))
                await savepoint.commit()
            except Exception:
                await savepoint.rollback()
            
            # 验证数据
            result = await conn.execute(text("""
                SELECT COUNT(*) FROM test_nested
            """))
            assert result.scalar() == 2


class TestDatabaseCRUDOperations:
    """数据库CRUD操作测试"""
    
    @pytest.mark.asyncio
    async def test_create_operation(self):
        """测试创建操作"""
        async with engine.begin() as conn:
            # 创建临时表
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(100) UNIQUE,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # 插入用户
            result = await conn.execute(text("""
                INSERT INTO test_users (email) 
                VALUES ('test@example.com') 
                RETURNING id, email, is_active
            """))
            
            user = result.fetchone()
            assert user.email == 'test@example.com'
            assert user.is_active is True
            assert user.id is not None

    @pytest.mark.asyncio
    async def test_read_operation(self):
        """测试读取操作"""
        async with engine.begin() as conn:
            # 创建临时表并插入数据
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_projects (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    description TEXT,
                    status VARCHAR(20) DEFAULT 'active'
                )
            """))
            
            await conn.execute(text("""
                INSERT INTO test_projects (name, description) VALUES 
                ('Project 1', 'Description 1'),
                ('Project 2', 'Description 2'),
                ('Project 3', 'Description 3')
            """))
            
            # 测试单条查询
            result = await conn.execute(text("""
                SELECT * FROM test_projects WHERE name = 'Project 1'
            """))
            project = result.fetchone()
            assert project.name == 'Project 1'
            assert project.description == 'Description 1'
            
            # 测试多条查询
            result = await conn.execute(text("""
                SELECT * FROM test_projects ORDER BY id
            """))
            projects = result.fetchall()
            assert len(projects) == 3
            assert projects[0].name == 'Project 1'

    @pytest.mark.asyncio
    async def test_update_operation(self):
        """测试更新操作"""
        async with engine.begin() as conn:
            # 创建临时表并插入数据
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_analysis (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER,
                    status VARCHAR(20) DEFAULT 'pending',
                    result JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # 插入分析记录
            result = await conn.execute(text("""
                INSERT INTO test_analysis (project_id, status) 
                VALUES (1, 'pending') 
                RETURNING id
            """))
            analysis_id = result.scalar()
            
            # 更新分析结果
            await conn.execute(text("""
                UPDATE test_analysis 
                SET status = 'completed', 
                    result = '{"defects": 5, "score": 85}',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :id
            """), {"id": analysis_id})
            
            # 验证更新
            result = await conn.execute(text("""
                SELECT status, result FROM test_analysis WHERE id = :id
            """), {"id": analysis_id})
            
            analysis = result.fetchone()
            assert analysis.status == 'completed'
            assert analysis.result['defects'] == 5

    @pytest.mark.asyncio
    async def test_delete_operation(self):
        """测试删除操作"""
        async with engine.begin() as conn:
            # 创建临时表并插入数据
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_logs (
                    id SERIAL PRIMARY KEY,
                    message TEXT,
                    level VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # 插入日志记录
            await conn.execute(text("""
                INSERT INTO test_logs (message, level) VALUES 
                ('Info message', 'INFO'),
                ('Warning message', 'WARN'),
                ('Error message', 'ERROR')
            """))
            
            # 删除特定级别的日志
            result = await conn.execute(text("""
                DELETE FROM test_logs WHERE level = 'ERROR'
                RETURNING id
            """))
            deleted_ids = result.fetchall()
            assert len(deleted_ids) == 1
            
            # 验证删除
            result = await conn.execute(text("""
                SELECT COUNT(*) FROM test_logs
            """))
            assert result.scalar() == 2
            
            # 验证特定记录已删除
            result = await conn.execute(text("""
                SELECT COUNT(*) FROM test_logs WHERE level = 'ERROR'
            """))
            assert result.scalar() == 0


class TestDatabaseIndexes:
    """数据库索引测试"""
    
    @pytest.mark.asyncio
    async def test_index_performance(self):
        """测试索引性能"""
        async with engine.begin() as conn:
            # 创建带索引的临时表
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_performance (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(100),
                    project_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # 创建索引
            await conn.execute(text("""
                CREATE INDEX idx_test_email ON test_performance(email)
            """))
            
            await conn.execute(text("""
                CREATE INDEX idx_test_project_created ON test_performance(project_id, created_at)
            """))
            
            # 插入测试数据
            for i in range(100):
                await conn.execute(text("""
                    INSERT INTO test_performance (email, project_id) 
                    VALUES (:email, :project_id)
                """), {
                    "email": f"user{i}@example.com",
                    "project_id": i % 10
                })
            
            # 测试索引查询
            result = await conn.execute(text("""
                SELECT * FROM test_performance 
                WHERE email = 'user50@example.com'
            """))
            user = result.fetchone()
            assert user.email == 'user50@example.com'
            
            # 测试复合索引查询
            result = await conn.execute(text("""
                SELECT COUNT(*) FROM test_performance 
                WHERE project_id = 5 AND created_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
            """))
            count = result.scalar()
            assert count >= 0


class TestDatabaseConstraints:
    """数据库约束测试"""
    
    @pytest.mark.asyncio
    async def test_unique_constraint(self):
        """测试唯一约束"""
        async with engine.begin() as conn:
            # 创建带唯一约束的临时表
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_unique (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(100) UNIQUE,
                    username VARCHAR(50) UNIQUE
                )
            """))
            
            # 插入第一条记录
            await conn.execute(text("""
                INSERT INTO test_unique (email, username) 
                VALUES ('test@example.com', 'testuser')
            """))
            
            # 尝试插入重复邮箱（应该失败）
            with pytest.raises(Exception):
                await conn.execute(text("""
                    INSERT INTO test_unique (email, username) 
                    VALUES ('test@example.com', 'anotheruser')
                """))

    @pytest.mark.asyncio
    async def test_foreign_key_constraint(self):
        """测试外键约束"""
        async with engine.begin() as conn:
            # 创建主表
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_projects_fk (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100)
                )
            """))
            
            # 创建从表
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_analyses_fk (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER REFERENCES test_projects_fk(id),
                    status VARCHAR(20)
                )
            """))
            
            # 插入主表记录
            result = await conn.execute(text("""
                INSERT INTO test_projects_fk (name) 
                VALUES ('Test Project') 
                RETURNING id
            """))
            project_id = result.scalar()
            
            # 插入从表记录（应该成功）
            await conn.execute(text("""
                INSERT INTO test_analyses_fk (project_id, status) 
                VALUES (:project_id, 'pending')
            """), {"project_id": project_id})
            
            # 尝试插入不存在的外键（应该失败）
            with pytest.raises(Exception):
                await conn.execute(text("""
                    INSERT INTO test_analyses_fk (project_id, status) 
                    VALUES (99999, 'pending')
                """))

    @pytest.mark.asyncio
    async def test_check_constraint(self):
        """测试检查约束"""
        async with engine.begin() as conn:
            # 创建带检查约束的临时表
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_scores (
                    id SERIAL PRIMARY KEY,
                    score INTEGER CHECK (score >= 0 AND score <= 100),
                    grade CHAR(1) CHECK (grade IN ('A', 'B', 'C', 'D', 'F'))
                )
            """))
            
            # 插入有效数据
            await conn.execute(text("""
                INSERT INTO test_scores (score, grade) 
                VALUES (85, 'A')
            """))
            
            # 尝试插入无效分数（应该失败）
            with pytest.raises(Exception):
                await conn.execute(text("""
                    INSERT INTO test_scores (score, grade) 
                    VALUES (150, 'A')
                """))
            
            # 尝试插入无效等级（应该失败）
            with pytest.raises(Exception):
                await conn.execute(text("""
                    INSERT INTO test_scores (score, grade) 
                    VALUES (85, 'X')
                """))


class TestDatabaseMigrations:
    """数据库迁移测试"""
    
    @pytest.mark.asyncio
    async def test_schema_migration(self):
        """测试模式迁移"""
        async with engine.begin() as conn:
            # 创建初始表结构
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_migration (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(50)
                )
            """))
            
            # 插入初始数据
            await conn.execute(text("""
                INSERT INTO test_migration (name) VALUES ('initial_data')
            """))
            
            # 模拟迁移：添加新列
            await conn.execute(text("""
                ALTER TABLE test_migration 
                ADD COLUMN email VARCHAR(100)
            """))
            
            # 验证迁移后的结构
            result = await conn.execute(text("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'test_migration'
            """))
            columns = [row[0] for row in result.fetchall()]
            assert 'email' in columns
            
            # 验证原有数据仍然存在
            result = await conn.execute(text("""
                SELECT name FROM test_migration WHERE name = 'initial_data'
            """))
            assert result.scalar() == 'initial_data'


class TestDatabasePerformance:
    """数据库性能测试"""
    
    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self):
        """测试批量插入性能"""
        async with engine.begin() as conn:
            # 创建临时表
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_bulk (
                    id SERIAL PRIMARY KEY,
                    data VARCHAR(100),
                    value INTEGER
                )
            """))
            
            # 批量插入数据
            import time
            start_time = time.time()
            
            # 使用批量插入
            values = [(f"data_{i}", i) for i in range(1000)]
            await conn.execute(text("""
                INSERT INTO test_bulk (data, value) 
                VALUES (unnest(ARRAY[:data_array]), unnest(ARRAY[:value_array]))
            """), {
                "data_array": [v[0] for v in values],
                "value_array": [v[1] for v in values]
            })
            
            end_time = time.time()
            insert_time = end_time - start_time
            
            # 验证插入结果
            result = await conn.execute(text("""
                SELECT COUNT(*) FROM test_bulk
            """))
            assert result.scalar() == 1000
            
            # 性能应该在合理范围内（小于1秒）
            assert insert_time < 1.0

    @pytest.mark.asyncio
    async def test_query_performance(self):
        """测试查询性能"""
        async with engine.begin() as conn:
            # 创建临时表并插入数据
            await conn.execute(text("""
                CREATE TEMPORARY TABLE test_query_perf (
                    id SERIAL PRIMARY KEY,
                    category VARCHAR(20),
                    value INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # 插入测试数据
            for i in range(10000):
                await conn.execute(text("""
                    INSERT INTO test_query_perf (category, value) 
                    VALUES (:category, :value)
                """), {
                    "category": f"cat_{i % 10}",
                    "value": i
                })
            
            # 测试查询性能
            import time
            start_time = time.time()
            
            result = await conn.execute(text("""
                SELECT category, COUNT(*), AVG(value) 
                FROM test_query_perf 
                GROUP BY category 
                ORDER BY category
            """))
            
            end_time = time.time()
            query_time = end_time - start_time
            
            # 验证查询结果
            results = result.fetchall()
            assert len(results) == 10
            
            # 查询时间应该在合理范围内
            assert query_time < 1.0