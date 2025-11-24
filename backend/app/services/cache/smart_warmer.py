"""
智能缓存预热服务 (Smart Cache Warmer)

实现基于历史访问模式的智能预热策略，提升缓存命中率
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import hashlib

from app.core.cache.cache_manager import CacheManager
from app.core.metrics import CACHE_HITS_TOTAL, CACHE_MISS_TOTAL, WARMUP_JOBS_TOTAL, WARMUP_LATENCY_SECONDS
from app.db.session import AsyncSessionLocal
from sqlalchemy import text, select

logger = logging.getLogger(__name__)


class SmartCacheWarmer:
    """智能缓存预热器"""
    
    def __init__(self, cache_manager: CacheManager, repo_store: Any, metrics: Any):
        self.cache = cache_manager
        self.repo_store = repo_store
        self.metrics = metrics
        
    async def preload(self, tenant_id: str, repo_id: str, file_path: str) -> bool:
        """
        预加载单个文件到缓存
        
        Args:
            tenant_id: 租户ID
            repo_id: 仓库ID
            file_path: 文件路径
            
        Returns:
            是否成功预加载
        """
        try:
            # 生成缓存键
            cache_key = f"an:cache:{tenant_id}:{repo_id}:default:{file_path}:default:default"
            
            # 检查是否已存在
            existing = await self.cache.get(cache_key)
            if existing:
                logger.info(f"Cache entry already exists for {file_path}")
                return True
            
            # 从仓库获取文件内容（这里需要实际的文件读取逻辑）
            # 在真实实现中，这里从 repo_store 获取文件内容
            file_content = await self._get_file_content(repo_id, file_path)
            if not file_content:
                logger.warning(f"Could not get content for {file_path}")
                return False
            
            # 计算文件哈希和AST指纹
            file_hash = hashlib.sha256(file_content.encode()).hexdigest()[:64]
            ast_fp = hashlib.sha1(file_content.encode()).hexdigest()[:64]
            
            # 构建缓存条目
            cache_entry = {
                "file_hash": file_hash,
                "ast_fingerprint": ast_fp,
                "result_hash": file_hash,  # 简化：使用文件哈希作为结果哈希
                "payload_url": f"s3://artifacts/preloaded/{tenant_id}/{repo_id}/{file_path.replace('/', '_')}.json",
                "created_at": datetime.utcnow().isoformat(),
                "last_access_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat()
            }
            
            # 存储到缓存
            success = await self.cache.set(cache_key, cache_entry)
            if success:
                self.metrics.CACHE_HITS_TOTAL.labels(operation="preload").inc()
                logger.info(f"Preloaded cache entry for {file_path}")
                return True
            else:
                self.metrics.CACHE_MISS_TOTAL.labels(operation="preload").inc()
                logger.error(f"Failed to preload cache for {file_path}")
                return False
                
        except Exception as e:
            logger.exception(f"Error preloading {file_path}: {str(e)}")
            return False
    
    async def warmup_by_repo_pattern(self, tenant_id: str, repo_id: str, pattern: str = "src/**") -> Dict[str, Any]:
        """
        按路径模式预热文件
        
        Args:
            tenant_id: 租户ID
            repo_id: 仓库ID
            pattern: 文件路径模式（如 src/**, core/**）
            
        Returns:
            预热结果统计
        """
        start_time = datetime.utcnow()
        
        try:
            # 获取匹配模式的文件列表
            files = await self._get_files_by_pattern(repo_id, pattern, limit=200)
            logger.info(f"Found {len(files)} files for pattern '{pattern}' in repo {repo_id}")
            
            if not files:
                return {
                    "status": "no_files",
                    "files_count": 0,
                    "preloaded_count": 0,
                    "duration_seconds": 0
                }
            
            # 并发预热文件
            preload_tasks = []
            for file_path in files:
                task = self.preload(tenant_id, repo_id, file_path)
                preload_tasks.append(task)
            
            # 等待所有预热任务完成
            results = await asyncio.gather(*preload_tasks, return_exceptions=True)
            
            # 统计结果
            preloaded_count = sum(1 for r in results if r)
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # 记录指标
            self.metrics.WARMUP_JOBS_TOTAL.labels(tenant=tenant_id, repo=repo_id).inc(len(files))
            self.metrics.WARMUP_LATENCY_SECONDS.labels(tenant=tenant_id, repo=repo_id).observe(duration)
            
            logger.info(f"Warmed up {preloaded_count}/{len(files)} files for repo {repo_id} in {duration:.2f}s")
            
            return {
                "status": "completed",
                "files_count": len(files),
                "preloaded_count": preloaded_count,
                "duration_seconds": duration,
                "success_rate": preloaded_count / len(files) if files else 0
            }
            
        except Exception as e:
            logger.exception(f"Error in warmup_by_repo_pattern for repo {repo_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def warmup_by_recent_changes(self, tenant_id: str, repo_id: str, days: int = 7) -> Dict[str, Any]:
        """
        基于最近变更预热文件
        
        Args:
            tenant_id: 租户ID
            repo_id: 仓库ID
            days: 最近天数
            
        Returns:
            预热结果统计
        """
        start_time = datetime.utcnow()
        
        try:
            # 获取最近变更的文件
            changed_files = await self._get_recently_changed_files(repo_id, days)
            logger.info(f"Found {len(changed_files)} recently changed files in repo {repo_id}")
            
            if not changed_files:
                return {
                    "status": "no_changes",
                    "files_count": 0,
                    "preloaded_count": 0,
                    "duration_seconds": 0
                }
            
            # 预热变更文件
            preload_tasks = []
            for file_path in changed_files:
                task = self.preload(tenant_id, repo_id, file_path)
                preload_tasks.append(task)
            
            results = await asyncio.gather(*preload_tasks, return_exceptions=True)
            preloaded_count = sum(1 for r in results if r)
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # 记录指标
            self.metrics.WARMUP_JOBS_TOTAL.labels(tenant=tenant_id, repo=repo_id).inc(len(changed_files))
            self.metrics.WARMUP_LATENCY_SECONDS.labels(tenant=tenant_id, repo=repo_id).observe(duration)
            
            return {
                "status": "completed",
                "files_count": len(changed_files),
                "preloaded_count": preloaded_count,
                "duration_seconds": duration,
                "success_rate": preloaded_count / len(changed_files) if changed_files else 0
            }
            
        except Exception as e:
            logger.exception(f"Error in warmup_by_recent_changes for repo {repo_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def _get_file_content(self, repo_id: str, file_path: str) -> Optional[str]:
        """
        获取文件内容（占位符实现）
        
        在真实实现中，这里应该：
        1. 从Git仓库读取文件
        2. 从文件系统缓存读取
        3. 从对象存储获取
        """
        # 占位符实现：返回示例内容
        return f"// Content of {file_path}\\n// This would be actual file content from repository"
    
    async def _get_files_by_pattern(self, repo_id: str, pattern: str, limit: int = 200) -> List[str]:
        """
        按模式获取文件列表（占位符实现）
        
        在真实实现中，这里应该：
        1. 查询数据库获取文件列表
        2. 扫描Git仓库
        3. 从文件系统获取
        """
        # 占位符实现：返回示例文件列表
        common_patterns = {
            "src/**": [
                "src/main.py", "src/app.py", "src/utils.py", "src/models.py",
                "src/components/auth.py", "src/services/user.py"
            ],
            "core/**": [
                "core/config.py", "core/database.py", "core/cache.py",
                "core/middleware.py", "core/auth.py"
            ],
            "tests/**": [
                "tests/test_main.py", "tests/test_auth.py", "tests/test_api.py"
            ]
        }
        
        return common_patterns.get(pattern, [])
    
    async def _get_recently_changed_files(self, repo_id: str, days: int) -> List[str]:
        """
        获取最近变更的文件（占位符实现）
        
        在真实实现中，这里应该：
        1. 查询Git历史获取最近变更
        2. 从分析结果表获取最近分析的文件
        3. 从CI/CD系统获取最近构建的文件
        """
        # 占位符实现：返回示例变更文件
        return [
            "src/main.py", "src/app.py", "src/utils.py",
            "src/components/auth.py", "src/services/user.py",
            "README.md", "package.json"
        ]


# 注册为Celery任务
from app.worker import celery_app

@celery_app.task(bind=True, max_retries=3, autoretry_for=(Exception,), retry_backoff=2)
def smart_warmup_task(tenant_id: str, repo_id: str, strategy: str = "pattern", 
                     pattern: Optional[str] = None, days: Optional[int] = None) -> Dict[str, Any]:
    """
    智能预热任务
    
    Args:
        tenant_id: 租户ID
        repo_id: 仓库ID
        strategy: 预热策略 (pattern, recent_changes)
        pattern: 路径模式
        days: 天数（用于recent_changes策略）
        
    Returns:
        预热结果
    """
    from app.services.cache.smart_warmer import SmartCacheWarmer
    from app.core.cache.cache_manager import get_cache_manager
    from app.core.metrics import get_metrics
    
    cache_manager = get_cache_manager()
    metrics = get_metrics()
    warmer = SmartCacheWarmer(cache_manager, None, metrics)
    
    if strategy == "pattern" and pattern:
        result = await warmer.warmup_by_repo_pattern(tenant_id, repo_id, pattern)
    elif strategy == "recent_changes" and days:
        result = await warmer.warmup_by_recent_changes(tenant_id, repo_id, days)
    else:
        # 默认使用模式策略
        result = await warmer.warmup_by_repo_pattern(tenant_id, repo_id, "src/**")
    
    logger.info(f"Smart warmup completed for {tenant_id}/{repo_id}: {result}")
    return result


@celery_app.task(bind=True)
def schedule_periodic_warmup():
    """
    定期预热任务（每天执行一次）
    """
    # 这里可以添加定期预热逻辑
    # 例如：预热所有活跃仓库的热门文件
    pass