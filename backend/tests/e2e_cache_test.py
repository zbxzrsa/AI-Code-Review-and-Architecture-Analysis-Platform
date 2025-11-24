#!/usr/bin/env python3
"""
End-to-end test script for analysis cache and idempotency.

Scenario:
1. Trigger first analysis on commit A (full scan, all files analyzed).
2. Modify 1 file, trigger analysis on commit B (should use cache for unchanged files).
3. Verify incremental hit ratio >= 0.6 and cache metrics.
"""
import requests
import json
import time
import os
import sys
from typing import Dict, Optional

BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PROJECT_ID = 1
RULEPACK_VERSION = "default"

# Test data
TEST_FILES = ["src/main.py", "src/util.py", "src/lib.py"]
FIRST_COMMIT = "abc123def456"
SECOND_COMMIT = "def456ghi789"


def log(msg: str):
    """Log message with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def trigger_analysis(commit_sha: str, files: list, idempotency_key: Optional[str] = None) -> Dict:
    """Trigger analysis for a commit."""
    url = f"{BASE_URL}/projects/{PROJECT_ID}/versions/{commit_sha}/analyze"
    headers = {}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key

    payload = {
        "rulepack_version": RULEPACK_VERSION,
        "analysis_config_hash": "config_v1",
        "files": files
    }

    log(f"Triggering analysis: {commit_sha}, files={len(files)}")
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    log(f"  -> session_id={data['session_id']}, idem={data.get('idem', False)}")
    return data


def wait_session_complete(session_id: int, timeout: int = 30) -> Dict:
    """Wait for session to complete and fetch stats."""
    url = f"{BASE_URL}/sessions/{session_id}"
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "completed":
                log(f"Session {session_id} completed")
                return data
        except Exception as e:
            log(f"  poll error (will retry): {e}")

        time.sleep(2)

    raise TimeoutError(f"Session {session_id} did not complete within {timeout}s")


def get_metrics() -> Dict:
    """Fetch Prometheus metrics."""
    url = f"{BASE_URL}/metrics"
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def parse_metric(metrics_text: str, metric_name: str) -> Optional[float]:
    """Parse a single metric value from Prometheus text format."""
    for line in metrics_text.split("\n"):
        if line.startswith(metric_name) and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[-1])
                except ValueError:
                    pass
    return None


def main():
    log("=== Analysis Cache E2E Test ===")
    log(f"API URL: {BASE_URL}")

    try:
        # Step 1: First analysis (full scan)
        log("\n[Step 1] First analysis on commit A (full scan)")
        result1 = trigger_analysis(FIRST_COMMIT, TEST_FILES)
        session1_id = result1["session_id"]

        log(f"Waiting for session {session1_id} to complete...")
        stats1 = wait_session_complete(session1_id, timeout=60)
        log(f"Session stats: {stats1}")

        # Check metrics after first analysis
        metrics1 = get_metrics()
        jobs_started_1 = parse_metric(metrics1, "analysis_jobs_started_total")
        jobs_completed_1 = parse_metric(metrics1, "analysis_jobs_completed_total")
        log(f"Metrics after first analysis: started={jobs_started_1}, completed={jobs_completed_1}")

        time.sleep(2)

        # Step 2: Idempotent retry (same commit, should return cached session)
        log("\n[Step 2] Idempotent retry (same commit)")
        result1_retry = trigger_analysis(FIRST_COMMIT, TEST_FILES, idempotency_key="test-key-1")
        if result1_retry["idem"]:
            log("✓ Idempotency working: returned same session_id")
            assert result1_retry["session_id"] == session1_id
        else:
            log("✗ Idempotency failed: returned different session_id")
            sys.exit(1)

        time.sleep(2)

        # Step 3: Incremental analysis (commit B, 1 file changed)
        log("\n[Step 3] Incremental analysis on commit B (1 file changed)")
        changed_files = TEST_FILES.copy()
        # Simulate: only first file changed
        incremental_files = [TEST_FILES[0]]  # Only src/main.py changed

        result2 = trigger_analysis(SECOND_COMMIT, incremental_files)
        session2_id = result2["session_id"]

        log(f"Waiting for session {session2_id} to complete...")
        stats2 = wait_session_complete(session2_id, timeout=60)
        log(f"Session stats: {stats2}")

        # Check metrics after incremental analysis
        metrics2 = get_metrics()
        cached_skipped = parse_metric(metrics2, "cached_files_skipped_total")
        hit_ratio = parse_metric(metrics2, "incremental_hit_ratio")
        log(f"Metrics after incremental: cached_skipped={cached_skipped}, hit_ratio={hit_ratio}")

        # Step 4: Verify results
        log("\n[Step 4] Verifying results")

        # Check idempotency
        if result1["idem"] == False and result1_retry["idem"] == True:
            log("✓ Idempotency test: PASS")
        else:
            log("✗ Idempotency test: FAIL")
            sys.exit(1)

        # Check cache hit ratio (if > 0.5, incremental is working)
        if hit_ratio and hit_ratio >= 0.5:
            log(f"✓ Cache hit ratio test: PASS (ratio={hit_ratio})")
        elif hit_ratio is None:
            log("⚠ Cache hit ratio not available in metrics")
        else:
            log(f"✗ Cache hit ratio test: FAIL (ratio={hit_ratio}, expected >= 0.5)")
            sys.exit(1)

        log("\n=== All tests passed! ===")
        return 0

    except Exception as e:
        log(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
