from fastapi import APIRouter, Depends, HTTPException, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import hashlib
import json
import logging
from typing import Optional

from app.db.session import get_db
from app.tasks.queue import enqueue_analysis_task, compute_idempotency_key

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/projects/{project_id}/versions/{commit_sha}/analyze")
async def analyze_version(
    project_id: int,
    commit_sha: str,
    request: Request,
    rulepack_version: str,
    analysis_config_hash: Optional[str] = None,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    db: AsyncSession = Depends(get_db),
):
    """Trigger analysis for a project version with idempotency handling.

    Behavior:
    - Idempotency-Key header is required; if missing, compute one from project_id,commit_sha,rulepack_version,analysis_config_hash
    - If a session exists with the same idempotency_key (or same tuple), return it and mark idem=True
    - Otherwise create a new session, write an audit record, enqueue the task and return idem=False
    """

    payload = await request.json() if request.method in ("POST",) else {}

    # compute fallback idempotency key if not provided
    if not idempotency_key:
        idempotency_key = compute_idempotency_key(str(project_id), str(project_id), commit_sha, rulepack_version, analysis_config_hash or "")

    # Try to find existing session by idempotency_key
    try:
        q = text("SELECT id, status FROM analysis_session WHERE idempotency_key = :ik LIMIT 1")
        res = await db.execute(q.bindparams(ik=idempotency_key))
        row = res.first()
        if row:
            session_id = row[0]
            return {"session_id": session_id, "idem": True}
    except Exception:
        logger.exception("Failed to query analysis_session by idempotency_key")

    # Fallback: try to find by tuple (project_id, commit_sha, rulepack_version) if columns exist
    try:
        q2 = text(
            "SELECT id FROM information_schema.columns WHERE table_name='analysis_session' AND column_name='commit_sha'"
        )
        colres = await db.execute(q2)
        if colres.first():
            q3 = text(
                "SELECT id FROM analysis_session WHERE project_id = :pid AND commit_sha = :sha AND rulepack_version = :rp LIMIT 1"
            )
            res3 = await db.execute(q3.bindparams(pid=project_id, sha=commit_sha, rp=rulepack_version))
            r = res3.first()
            if r:
                return {"session_id": r[0], "idem": True}
    except Exception:
        # ignore; not all deployments have these columns
        logger.debug("Tuple lookup unavailable or failed; continuing to create session")

    # Create new analysis_session
    try:
        insert_sql = text(
            "INSERT INTO analysis_session (project_id, label, status, started_at, idempotency_key) VALUES (:pid, :label, :status, now(), :ik) RETURNING id"
        )
        label = f"analysis:{commit_sha}:{rulepack_version}"
        r = await db.execute(insert_sql.bindparams(pid=project_id, label=label, status='pending', ik=idempotency_key))
        await db.commit()
        new_id = r.scalar()

        # write audit record
        try:
            audit_sql = text(
                "INSERT INTO analysis_audit (session_id, event_type, actor, project_id, commit_sha, trace_id, payload) VALUES (:sid, :evt, :actor, :pid, :sha, :trace, :payload)"
            )
            trace = request.headers.get("X-Trace-Id") or request.headers.get("trace-id") or None
            actor = request.client.host if request.client else None
            await db.execute(audit_sql.bindparams(sid=new_id, evt='ANALYZE_REQUESTED', actor=actor, pid=project_id, sha=commit_sha, trace=trace, payload=json.dumps(payload)))
            await db.commit()
        except Exception:
            logger.exception("Failed to write audit record")

        # enqueue task (best-effort)
        try:
            enqueued = enqueue_analysis_task(new_id, {"project_id": project_id, "commit_sha": commit_sha, "rulepack_version": rulepack_version})
            logger.info("Enqueue result for session %s: %s", new_id, enqueued)
        except Exception:
            logger.exception("Enqueue failed")

        return {"session_id": new_id, "idem": False}
    except Exception:
        logger.exception("Failed to create analysis_session")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create analysis session")
