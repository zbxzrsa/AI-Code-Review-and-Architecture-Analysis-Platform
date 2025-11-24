import logging
import hashlib

logger = logging.getLogger(__name__)


def enqueue_analysis_task(session_id: int, payload: dict):
    """Try to enqueue a Celery task if celery app is available.
    If not, log and return False.
    """
    try:
        # Lazy import to avoid hard dependency if Celery not installed/configured
        from app.worker import celery_app  # type: ignore
        celery_app.send_task("analysis.run", args=[session_id], kwargs=payload)
        return True
    except Exception as e:
        logger.debug("Celery not available or enqueue failed: %s", e)
        # Fallback: write to Redis, or leave as TODO. For now we log.
        logger.info("Enqueue fallback: session=%s payload=%s", session_id, payload)
        return False


def compute_idempotency_key(tenant_id: str, repo_id: str, commit_sha: str, rulepack_version: str, config_hash: str) -> str:
    s = f"{tenant_id}:{repo_id}:{commit_sha}:{rulepack_version}:{config_hash}"
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:64]
