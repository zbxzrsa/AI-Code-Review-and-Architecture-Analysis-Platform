import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
import json

try:
    from app.worker import celery_app
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    celery_app = None

# Simple imports with fallbacks
from app.db.session import get_db
from sqlalchemy import text, select
from sqlalchemy.exc import SQLAlchemyError
from app.core.logger import logger

# Simple metrics with fallbacks
try:
    from app.core.metrics import ANALYSIS_STARTED, ANALYSIS_COMPLETED, CACHED_FILES_SKIPPED, ANALYSIS_DURATION, INCREMENTAL_HIT
except ImportError:
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    CACHED_FILES_SKIPPED = "cached_files_skipped"
    ANALYSIS_DURATION = "analysis_duration"
    INCREMENTAL_HIT = "incremental_hit"
from app.db.session import get_db
from app.core.logger import logger
# Import metrics with fallback
try:
    from app.core.metrics import ANALYSIS_STARTED, ANALYSIS_COMPLETED, CACHED_FILES_SKIPPED, ANALYSIS_DURATION, INCREMENTAL_HIT
except ImportError:
    # Fallback metrics if metrics module has issues
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    CACHED_FILES_SKIPPED = "cached_files_skipped"
    ANALYSIS_DURATION = "analysis_duration"
    INCREMENTAL_HIT = "incremental_hit"
from app.services.s3_storage import get_s3_storage
from app.services.glm_service import glm_service
from app.core.cache import CacheManager, LocalLRUCache, RedisCache, DatabaseCache

TTL_DEFAULT_DAYS = 7


def compute_file_meta(file_path: str) -> dict:
    """Placeholder: compute file_hash and ast_fingerprint for file.
    In real implementation this reads file contents and computes AST fingerprint.
    """
    # For now use hashed file_path as stub
    import hashlib
    h = hashlib.sha256(file_path.encode('utf-8')).hexdigest()
    return {
        'file_hash': h[:64],
        'ast_fingerprint': hashlib.sha1(file_path.encode('utf-8')).hexdigest()
    }


async def emit_cached_result(db, result_data, session_id: int, file_path: str):
    """Emit analysis result by upserting into session artifacts.
    This is a simplified placeholder: store a session artifact pointing to payload_url.
    """
    try:
        # insert session_artifact
        insert_sql = text(
            "INSERT INTO session_artifact (session_id, type, path, size, created_at) VALUES (:sid, :type, :path, :size, now())"
        )
        await db.execute(insert_sql.bindparams(sid=session_id, type='analysis_result', path=result_data.get('payload_url', ''), size=0))
        await db.commit()
    except Exception:
        logger.error("Failed to emit result for %s", file_path)


def persist_artifacts(session_id: int, file_path: str, result: dict) -> Tuple[str, str]:
    """Persist analysis result to S3/MinIO and return (object_url, etag).

    Args:
        session_id: Analysis session ID
        file_path: File path being analyzed
        result: Analysis result dict (will be serialized to JSON)

    Returns:
        Tuple of (object_url, etag)
    """
    try:
        storage = get_s3_storage()
        content = json.dumps(result, default=str).encode('utf-8')
        metadata = {
            "session_id": str(session_id),
            "file_path": file_path,
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }
        object_url, etag = storage.upload_artifact(session_id, file_path, content, metadata)
        logger.info("Artifacts persisted: file=%s, url=%s, etag=%s", file_path, object_url, etag)
        return object_url, etag
    except Exception:
        logger.exception("Failed to persist artifacts for %s (session %s)", file_path, session_id)
        # Fallback: return stub URL for graceful degradation
        return f"s3://artifacts/sessions/{session_id}/{file_path.replace('/', '_')}.json", "unknown"


# Use dummy_task decorator if celery is not available
task_decorator = celery_app.task if celery_app else dummy_task

@task_decorator(bind=True, max_retries=5, autoretry_for=(Exception,), retry_backoff=2)
def run_analysis(self, session_id: int):
    """Celery task to run incremental analysis for a session.

    This task demonstrates:
    - loading session context from DB
    - computing changed files (stub)
    - expanding reverse dependencies (stub)
    - checking analysis_cache for hits and emitting results
    - analyzing changed files and writing cache entries
    """
    # Use synchronous DB operations via get_db() in an asyncio loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        async for db in get_db():
            try:
                # load session context (stub: reading project_id and commit if present)
                q = text("SELECT id, project_id, label FROM analysis_session WHERE id = :sid")
                res = await db.execute(q.bindparams(sid=session_id))
                row = res.first()
                if not row:
                    logger.error("Session %s not found", session_id)
                    return
                project_id = row[1]

                tenant = str(project_id)
                rulepack = 'default'

                # Initialize cache components
                import redis
                redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                l1_cache = LocalLRUCache(max_size=10000, ttl_seconds=300)
                l2_cache = RedisCache(redis_client, prefix="analysis:")
                l3_cache = DatabaseCache(db)
                cache_manager = CacheManager(l1_cache, l2_cache, l3_cache)

                # start metrics
                ANALYSIS_STARTED.labels(tenant=tenant).inc()

                # compute changed files (placeholder - in prod compute git diff)
                changed = ["src/main.py", "src/lib/util.py"]

                # expand reverse deps (placeholder - in prod use dependency graph)
                affected = changed  # simple passthrough

                total = len(affected)
                skipped = 0

                # Assume a commit_sha for caching (in real implementation, get from session)
                commit_sha = "abc123def456"

                for file_path in affected:
                    meta = compute_file_meta(file_path)

                    # Define cache key
                    cache_key = f"RULE_RESULT:{project_id}:{commit_sha}:{rulepack}:{file_path}"

                    # Define analysis fallback function
                    async def analyze_file():
                        # analyze file (placeholder - in real implementation, call GLM service)
                        result = {"file": file_path, "findings": []}

                        # persist artifacts with real S3 upload
                        object_url, etag = persist_artifacts(session_id, file_path, result)

                        # Return cached result structure
                        return {
                            "file_path": file_path,
                            "file_hash": meta['file_hash'],
                            "ast_fingerprint": meta['ast_fingerprint'],
                            "payload_url": object_url,
                            "etag": etag,
                            "findings": result["findings"]
                        }

                    # Get from cache or compute
                    try:
                        cached_result = await cache_manager.get(
                            key=cache_key,
                            fallback=analyze_file
                        )

                        if cached_result:
                            # Emit result (whether from cache or computed)
                            await emit_cached_result(db, cached_result, session_id, file_path)
                            skipped += 1 if 'cached' in str(cached_result).lower() else 0  # Simple heuristic
                            CACHED_FILES_SKIPPED.labels(tenant=tenant).inc()
                        else:
                            logger.warning("Cache manager returned None for %s", file_path)

                    except Exception:
                        logger.exception("Cache operation failed for %s", file_path)
                        # Fallback to direct analysis
                        result = await analyze_file()
                        await emit_cached_result(db, result, session_id, file_path)

                # update session status to completed
                try:
                    await db.execute(text("UPDATE analysis_session SET status='completed', completed_at=now() WHERE id = :sid").bindparams(sid=session_id))
                    await db.commit()
                except Exception:
                    logger.exception("Failed to mark session completed %s", session_id)

                # metrics
                ANALYSIS_COMPLETED.labels(tenant=tenant).inc()
                if total > 0:
                    INCREMENTAL_HIT.observe(skipped / total)

            except Exception:
                logger.exception("Unexpected error in run_analysis %s", session_id)

    loop.run_until_complete(_run())


@task_decorator(bind=True, max_retries=3, autoretry_for=(Exception,), retry_backoff=2)
def analyze_file_with_glm(self, file_path: str, content: str, language: str = "python", 
                        focus_areas: list = None) -> dict:
    """
    Analyze a single file using GLM-4.6:cloud model
    
    Args:
        file_path: Path to the file being analyzed
        content: File content to analyze
        language: Programming language
        focus_areas: Specific areas to focus on (security, performance, etc.)
        
    Returns:
        Analysis result dictionary
    """
    try:
        # Get AI review from GLM service
        result = glm_service.review_code(content, language, focus_areas)
        
        if result["success"]:
            review_data = result["review"]
            
            # Extract structured data if available
            if isinstance(review_data, dict):
                issues = review_data.get("issues", [])
                improvements = review_data.get("improvements", [])
                overall_score = review_data.get("overall_score", 75)
            else:
                # Raw text response
                issues = []
                improvements = []
                overall_score = 75
                review_data = {
                    "summary": review_data,
                    "raw_text": result.get("raw_response", "")
                }
            
            return {
                "file_path": file_path,
                "analysis_type": "glm_ai_review",
                "status": "completed",
                "issues_found": len(issues),
                "overall_score": overall_score,
                "review_data": {
                    "model": result["model"],
                    "tokens_used": result["tokens_used"],
                    "duration": result.get("duration", 0),
                    "review": review_data,
                    "issues": issues,
                    "improvements": improvements,
                    "reviewed_at": result["timestamp"]
                }
            }
        else:
            logger.error(f"GLM analysis failed for {file_path}: {result.get('error')}")
            return {
                "file_path": file_path,
                "analysis_type": "glm_ai_review",
                "status": "failed",
                "error": result.get("error"),
                "issues_found": 0
            }
            
    except Exception as e:
        logger.exception(f"GLM analysis error for {file_path}: {str(e)}")
        return {
            "file_path": file_path,
            "analysis_type": "glm_ai_review",
            "status": "error",
            "error": str(e),
            "issues_found": 0
        }


@task_decorator(bind=True, max_retries=3, autoretry_for=(Exception,), retry_backoff=2)
def batch_glm_analysis(self, session_id: int, file_paths: list, focus_areas: list = None) -> dict:
    """
    Run batch GLM analysis for multiple files
    
    Args:
        session_id: Analysis session ID
        file_paths: List of file paths to analyze
        focus_areas: Common focus areas for all files
        
    Returns:
        Batch analysis summary
    """
    results = []
    total_issues = 0
    total_files = len(file_paths)
    
    for file_path in file_paths:
        try:
            # Read file content (placeholder - in real implementation read from git/workspace)
            content = f"# Content of {file_path}\\n# This would be actual file content"
            language = detect_language_from_path(file_path)
            
            # Analyze with GLM
            result = analyze_file_with_glm(self, file_path, content, language, focus_areas)
            results.append(result)
            
            if result.get("status") == "completed":
                total_issues += result.get("issues_found", 0)
                
        except Exception as e:
            logger.error(f"Failed to analyze {file_path} with GLM: {str(e)}")
            results.append({
                "file_path": file_path,
                "status": "error",
                "error": str(e),
                "issues_found": 0
            })
    
    return {
        "session_id": session_id,
        "analysis_type": "glm_batch_review",
        "total_files": total_files,
        "completed_files": len([r for r in results if r.get("status") == "completed"]),
        "total_issues_found": total_issues,
        "results": results,
        "summary": f"GLM analysis completed: {total_files} files, {total_issues} issues found"
    }


def detect_language_from_path(file_path: str) -> str:
    """Detect programming language from file extension"""
    extension = file_path.split('.')[-1].lower() if '.' in file_path else ''
    
    language_map = {
        'py': 'python',
        'js': 'javascript',
        'ts': 'typescript',
        'java': 'java',
        'cpp': 'cpp',
        'c': 'c',
        'go': 'go',
        'rs': 'rust',
        'php': 'php',
        'rb': 'ruby',
        'swift': 'swift',
        'kt': 'kotlin',
        'scala': 'scala',
        'sh': 'bash',
        'sql': 'sql',
        'html': 'html',
        'css': 'css',
        'json': 'json',
        'yaml': 'yaml',
        'yml': 'yaml',
        'xml': 'xml',
        'md': 'markdown'
    }
    
    return language_map.get(extension, 'text')
