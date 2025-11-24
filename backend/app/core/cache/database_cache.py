"""
数据库缓存实现 (L3)

特点:
- 持久化存储
- 支持复杂查询
- 生命周期管理
- 审计追踪
"""

import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from sqlalchemy import Column, String, Integer, DateTime, JSON, Index, func, and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import JSONB

logger = logging.getLogger(__name__)


# SQLAlchemy 模型

class CacheRecord:
    """缓存记录模型 - 使用现有的 analysis_cache 表"""

    __tablename__ = 'analysis_cache'

    tenant_id = Column(String(36), primary_key=True)
    repo_id = Column(String(36), primary_key=True)
    file_path = Column(String(500), primary_key=True)
    rulepack_version = Column(String(32), primary_key=True)
    file_hash = Column(String(64))
    ast_fingerprint = Column(String(64))
    result_hash = Column(String(64))
    payload_url = Column(String(1000))
    created_at = Column(DateTime, server_default=func.now())
    last_access_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime)

    # For compatibility, add computed cache_key
    @property
    def cache_key(self):
        return f"RULE_RESULT:{self.repo_id}:{self.tenant_id}:{self.rulepack_version}:{self.file_path}"

    @property
    def value(self):
        return {
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "ast_fingerprint": self.ast_fingerprint,
            "payload_url": self.payload_url,
            "etag": self.result_hash,
            "findings": []  # Placeholder
        }


class DatabaseCache:
    """数据库缓存实现"""

    def __init__(self, session: Session):
        """
        初始化数据库缓存

        Args:
            session: SQLAlchemy Session
        """
        self.session = session

    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键 (格式: RULE_RESULT:repo_id:commit_sha:rulepack_version:file_path)

        Returns:
            缓存值，或 None 如果不存在或已过期
        """
        try:
            # Parse cache key
            parts = key.split(":")
            if len(parts) < 5 or parts[0] != "RULE_RESULT":
                logger.error(f"Invalid cache key format: {key}")
                return None

            repo_id = parts[1]
            commit_sha = parts[2]
            rulepack_version = parts[3]
            file_path = ":".join(parts[4:])  # Handle file paths with colons

            record = self.session.query(CacheRecord).filter(
                CacheRecord.repo_id == repo_id,
                CacheRecord.tenant_id == repo_id,  # Using repo_id as tenant_id for now
                CacheRecord.rulepack_version == rulepack_version,
                CacheRecord.file_path == file_path,
                or_(
                    CacheRecord.expires_at.is_(None),
                    CacheRecord.expires_at > datetime.utcnow()
                )
            ).first()

            if record:
                # 更新访问时间
                record.last_access_at = datetime.utcnow()
                self.session.commit()
                logger.debug(f"Database cache hit: {key}")
                return record.value

            logger.debug(f"Database cache miss: {key}")
            return None

        except Exception as e:
            logger.error(f"Database get error: {key}: {e}")
            return None

    async def put(self,
                   key: str,
                   value: Any,
                   cache_type: str,
                   repo_id: str,
                   commit_sha: str,
                   expires_at: Optional[datetime] = None) -> bool:
        """
        设置缓存值

        Args:
            key: 缓存键 (格式: RULE_RESULT:repo_id:commit_sha:rulepack_version:file_path)
            value: 缓存值 (dict with file_path, file_hash, etc.)
            cache_type: 缓存类型
            repo_id: 仓库 ID
            commit_sha: 提交 SHA
            expires_at: 过期时间

        Returns:
            是否成功设置
        """
        try:
            # Parse cache key
            parts = key.split(":")
            if len(parts) < 5 or parts[0] != "RULE_RESULT":
                logger.error(f"Invalid cache key format: {key}")
                return False

            parsed_repo_id = parts[1]
            parsed_commit_sha = parts[2]
            rulepack_version = parts[3]
            file_path = ":".join(parts[4:])

            # Use parsed values if repo_id/commit_sha not provided
            actual_repo_id = repo_id or parsed_repo_id
            actual_commit_sha = commit_sha or parsed_commit_sha

            # Upsert using ON CONFLICT
            from sqlalchemy.dialects.postgresql import insert

            insert_stmt = insert(CacheRecord).values(
                tenant_id=actual_repo_id,
                repo_id=actual_repo_id,
                file_path=file_path,
                rulepack_version=rulepack_version,
                file_hash=value.get('file_hash', ''),
                ast_fingerprint=value.get('ast_fingerprint', ''),
                result_hash=value.get('etag', ''),
                payload_url=value.get('payload_url', ''),
                expires_at=expires_at
            )

            update_stmt = insert_stmt.on_conflict_do_update(
                index_elements=['tenant_id', 'repo_id', 'rulepack_version', 'file_path'],
                set_={
                    'file_hash': insert_stmt.excluded.file_hash,
                    'ast_fingerprint': insert_stmt.excluded.ast_fingerprint,
                    'result_hash': insert_stmt.excluded.result_hash,
                    'payload_url': insert_stmt.excluded.payload_url,
                    'last_access_at': datetime.utcnow(),
                    'expires_at': insert_stmt.excluded.expires_at
                }
            )

            self.session.execute(update_stmt)
            self.session.commit()
            logger.debug(f"Database cache put: {key}")
            return True

        except Exception as e:
            logger.error(f"Database put error: {key}: {e}")
            self.session.rollback()
            return False

    async def delete(self, key: str) -> bool:
        """
        删除缓存值

        Args:
            key: 缓存键

        Returns:
            是否成功删除
        """
        try:
            count = self.session.query(CacheRecord).filter(
                CacheRecord.cache_key == key
            ).delete()

            self.session.commit()
            if count > 0:
                logger.debug(f"Database cache delete: {key}")
            return count > 0

        except Exception as e:
            logger.error(f"Database delete error: {key}: {e}")
            self.session.rollback()
            return False

    async def delete_by_pattern(self, pattern: str) -> int:
        """
        删除匹配的所有键

        Args:
            pattern: 键的模式（% 为通配符）

        Returns:
            删除的记录数
        """
        try:
            count = self.session.query(CacheRecord).filter(
                CacheRecord.cache_key.like(pattern)
            ).delete()

            self.session.commit()
            logger.info(f"Database delete by pattern: {pattern} ({count} records)")
            return count

        except Exception as e:
            logger.error(f"Database delete by pattern error: {pattern}: {e}")
            self.session.rollback()
            return 0

    async def delete_expired(self) -> int:
        """
        删除已过期的缓存记录

        Returns:
            删除的记录数
        """
        try:
            count = self.session.query(CacheRecord).filter(
                and_(
                    CacheRecord.expires_at.isnot(None),
                    CacheRecord.expires_at < datetime.utcnow()
                )
            ).delete()

            self.session.commit()
            logger.info(f"Deleted {count} expired cache records")
            return count

        except Exception as e:
            logger.error(f"Database cleanup error: {e}")
            self.session.rollback()
            return 0

    async def delete_old_records(self, days: int = 7) -> int:
        """
        删除超过 N 天的缓存记录

        Args:
            days: 天数

        Returns:
            删除的记录数
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            count = self.session.query(CacheRecord).filter(
                CacheRecord.created_at < cutoff_time
            ).delete()

            self.session.commit()
            logger.info(f"Deleted {count} old cache records (>{days} days)")
            return count

        except Exception as e:
            logger.error(f"Database delete old records error: {e}")
            self.session.rollback()
            return 0

    async def get_by_repo_and_commit(self, repo_id: str, commit_sha: str) -> List[Dict]:
        """
        获取指定仓库和提交的所有缓存记录

        Args:
            repo_id: 仓库 ID
            commit_sha: 提交 SHA

        Returns:
            缓存记录列表
        """
        try:
            records = self.session.query(CacheRecord).filter(
                and_(
                    CacheRecord.repo_id == repo_id,
                    CacheRecord.commit_sha == commit_sha,
                    or_(
                        CacheRecord.expires_at.is_(None),
                        CacheRecord.expires_at > datetime.utcnow()
                    )
                )
            ).all()

            return [
                {
                    'cache_key': r.cache_key,
                    'cache_type': r.cache_type,
                    'created_at': r.created_at.isoformat(),
                    'access_count': r.access_count,
                    'size_bytes': r.size_bytes,
                }
                for r in records
            ]

        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []

    async def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            total_count = self.session.query(func.count(CacheRecord.id)).scalar()
            total_size = self.session.query(func.sum(CacheRecord.size_bytes)).scalar() or 0

            by_type = self.session.query(
                CacheRecord.cache_type,
                func.count(CacheRecord.id).label('count'),
                func.sum(CacheRecord.size_bytes).label('total_size')
            ).group_by(CacheRecord.cache_type).all()

            return {
                'total_records': total_count,
                'total_size_bytes': total_size,
                'by_type': [
                    {
                        'cache_type': t[0],
                        'count': t[1],
                        'total_size_bytes': t[2] or 0,
                    }
                    for t in by_type
                ]
            }

        except Exception as e:
            logger.error(f"Database stats error: {e}")
            return {}

    async def compact(self) -> int:
        """
        清理和优化数据库

        Returns:
            清理的记录数
        """
        try:
            # 删除过期记录
            expired_count = await self.delete_expired()

            # 删除超过 30 天的记录
            old_count = await self.delete_old_records(days=30)

            # 数据库 VACUUM（PostgreSQL 特定）
            self.session.execute("VACUUM ANALYZE cache_records")
            self.session.commit()

            total = expired_count + old_count
            logger.info(f"Database compact completed: {total} records removed")
            return total

        except Exception as e:
            logger.error(f"Database compact error: {e}")
            return 0
