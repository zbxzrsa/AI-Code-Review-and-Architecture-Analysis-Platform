"""
智能预热服务 - 简化版本
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import hashlib

logger = logging.getLogger(__name__)


class SmartCacheService:
    """智能缓存预热服务"""
    
    def __init__(self):
        self.cache_stats = {
            "total_preloads": 0,
            "successful_preloads": 0,
            "failed_preloads": 0,
            "total_files_preloaded": 0
        }
    
    def _generate_cache_key(self, tenant_id: str, repo_id: str, file_path: str) -> str:
        """生成缓存键"""
        return f"an:cache:{tenant_id}:{repo_id}:default:{file_path}:default:default"
    
    def _simulate_file_content(self, file_path: str) -> str:
        """模拟文件内容（实际实现中从仓库读取）"""
        return f"// Content of {file_path}\\n// Generated at {datetime.utcnow().isoformat()}\\n"
    
    async def preload_file(self, tenant_id: str, repo_id: str, file_path: str) -> Dict[str, Any]:
        """预加载单个文件到缓存"""
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(tenant_id, repo_id, file_path)
            
            # 模拟文件内容
            file_content = self._simulate_file_content(file_path)
            file_hash = hashlib.sha256(file_content.encode()).hexdigest()[:64]
            ast_fp = hashlib.sha1(file_content.encode()).hexdigest()[:64]
            
            # 构建缓存条目
            cache_entry = {
                "file_hash": file_hash,
                "ast_fingerprint": ast_fp,
                "result_hash": file_hash,  # 简化：使用文件哈希
                "payload_url": f"s3://artifacts/preloaded/{tenant_id}/{repo_id}/{file_path.replace('/', '_')}.json",
                "created_at": datetime.utcnow().isoformat(),
                "last_access_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat()
            }
            
            # 模拟存储到缓存（实际实现中会调用Redis）
            logger.info(f"Preloaded cache entry for {file_path}")
            
            self.cache_stats["total_preloads"] += 1
            self.cache_stats["successful_preloads"] += 1
            self.cache_stats["total_files_preloaded"] += 1
            
            return {
                "success": True,
                "cache_key": cache_key,
                "file_path": file_path,
                "action": "preloaded"
            }
            
        except Exception as e:
            logger.error(f"Error preloading {file_path}: {str(e)}")
            self.cache_stats["total_preloads"] += 1
            self.cache_stats["failed_preloads"] += 1
            
            return {
                "success": False,
                "cache_key": cache_key,
                "file_path": file_path,
                "action": "preload_failed",
                "error": str(e)
            }
    
    async def warmup_by_pattern(self, tenant_id: str, repo_id: str, pattern: str = "src/**", limit: int = 50) -> Dict[str, Any]:
        """按路径模式预热文件"""
        try:
            # 模拟获取文件列表
            files = self._get_files_by_pattern(pattern, limit)
            logger.info(f"Found {len(files)} files for pattern '{pattern}'")
            
            results = []
            for file_path in files:
                result = await self.preload_file(tenant_id, repo_id, file_path)
                results.append(result)
            
            successful_count = sum(1 for r in results if r.get("success", False))
            
            return {
                "success": True,
                "tenant_id": tenant_id,
                "repo_id": repo_id,
                "pattern": pattern,
                "files_count": len(files),
                "successful_count": successful_count,
                "results": results,
                "message": f"Warmed up {successful_count}/{len(files)} files for pattern '{pattern}'"
            }
            
        except Exception as e:
            logger.error(f"Error in warmup_by_pattern: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "pattern": pattern
            }
    
    async def warmup_recent_changes(self, tenant_id: str, repo_id: str, days: int = 7) -> Dict[str, Any]:
        """基于最近变更预热文件"""
        try:
            # 模拟获取最近变更的文件
            changed_files = self._get_recently_changed_files(days)
            logger.info(f"Found {len(changed_files)} recently changed files")
            
            results = []
            for file_path in changed_files:
                result = await self.preload_file(tenant_id, repo_id, file_path)
                results.append(result)
            
            successful_count = sum(1 for r in results if r.get("success", False))
            
            return {
                "success": True,
                "tenant_id": tenant_id,
                "repo_id": repo_id,
                "days": days,
                "files_count": len(changed_files),
                "successful_count": successful_count,
                "results": results,
                "message": f"Warmed up {successful_count}/{len(changed_files)} recently changed files"
            }
            
        except Exception as e:
            logger.error(f"Error in warmup_recent_changes: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "days": days
            }
    
    def _get_files_by_pattern(self, pattern: str, limit: int = 50) -> List[str]:
        """模拟按模式获取文件列表"""
        # 简化实现：返回预定义的文件列表
        if pattern == "src/**":
            return [
                "src/main.py", "src/app.py", "src/utils.py", "src/models.py",
                "src/components/auth.py", "src/services/user.py", "src/config.py"
            ]
        elif pattern == "core/**":
            return [
                "core/config.py", "core/database.py", "core/cache.py",
                "core/middleware.py", "core/auth.py"
            ]
        elif pattern == "tests/**":
            return [
                "tests/test_main.py", "tests/test_auth.py", "tests/test_api.py"
            ]
        else:
            return []
    
    def _get_recently_changed_files(self, days: int) -> List[str]:
        """模拟获取最近变更的文件"""
        # 简化实现：返回一些常见的变更文件
        return [
            "src/main.py", "src/app.py", "src/utils.py",
            "README.md", "package.json"
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取预热统计信息"""
        return {
            "stats": self.cache_stats,
            "timestamp": datetime.utcnow().isoformat()
        }


# 全局服务实例
smart_cache_service = SmartCacheService()


def get_smart_cache_service() -> SmartCacheService:
    """获取智能缓存预热服务实例"""
    return smart_cache_service