import os
import time
import json
import math
import pytest
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return d0 + d1


def _do_request(method, url, payload=None, timeout=10, headers=None):
    start = time.perf_counter()
    try:
        if method.upper() == "GET":
            resp = requests.get(url, timeout=timeout, headers=headers)
        else:
            resp = requests.post(url, json=payload, timeout=timeout, headers=headers)
        dur = time.perf_counter() - start
        proc_header = resp.headers.get("X-Process-Time")
        proc_time = None
        if proc_header:
            try:
                proc_time = float(proc_header)
            except Exception:
                proc_time = None
        return {
            "status": resp.status_code,
            "duration": dur,
            "process_time": proc_time,
        }
    except requests.RequestException:
        dur = time.perf_counter() - start
        return {
            "status": None,
            "duration": dur,
            "process_time": None,
        }


def run_load(path, method="POST", payload=None, total_requests=10, concurrency=2, timeout=10):
    url = BASE_URL.rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as exe:
        futures = [
            exe.submit(_do_request, method, url, payload, timeout, headers)
            for _ in range(total_requests)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    durations = [r["duration"] for r in results if r["duration"] is not None]
    proc_times = [r["process_time"] for r in results if r["process_time"] is not None]

    success_count = sum(1 for r in results if r["status"] == 200)
    rate_limit_count = sum(1 for r in results if r["status"] == 429)
    other_error_count = sum(1 for r in results if (r["status"] is None) or (r["status"] not in (200, 429)))

    metrics = {
        "total": total_requests,
        "success": success_count,
        "rate_limited": rate_limit_count,
        "errors": other_error_count,
        "avg_latency": (sum(durations) / len(durations)) if durations else float("nan"),
        "p50_latency": percentile(durations, 50) if durations else float("nan"),
        "p95_latency": percentile(durations, 95) if durations else float("nan"),
        "avg_process_time_header": (sum(proc_times) / len(proc_times)) if proc_times else None,
    }
    return metrics


@pytest.fixture(scope="session")
def ensure_server():
    url = BASE_URL.rstrip("/") + "/health"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            pytest.skip(f"健康检查未通过：{resp.status_code}")
    except requests.RequestException as e:
        pytest.skip(f"无法连接到后端服务：{e}")


@pytest.mark.performance
def test_embed_under_limit(ensure_server):
    # /api/v1/embed 速率限制：10/60s
    payload = {
        "code_structure": {
            "files": [
                {
                    "path": "main.py",
                    "functions": [
                        {"name": "foo", "loc": 10},
                        {"name": "bar", "loc": 20}
                    ]
                }
            ]
        },
        "language": "python"
    }
    metrics = run_load(path="/api/v1/embed", method="POST", payload=payload, total_requests=10, concurrency=2)
    assert metrics["errors"] == 0, f"出现网络或服务器错误：{metrics}"
    assert metrics["rate_limited"] == 0, f"不应触发速率限制：{metrics}"
    assert metrics["success"] == 10, f"成功请求数不符合预期：{metrics}"
    assert metrics["p95_latency"] < 2.0, f"P95 延迟过高：{metrics}"


@pytest.mark.performance
def test_defects_under_limit(ensure_server):
    # /api/v1/analyze/defects 速率限制：5/60s
    payload = {
        "code": "def risky(x):\n  return x.foo()\n",
        "vector": None,
        "language": "python"
    }
    metrics = run_load(path="/api/v1/analyze/defects", method="POST", payload=payload, total_requests=5, concurrency=1)
    assert metrics["errors"] == 0, f"出现网络或服务器错误：{metrics}"
    assert metrics["rate_limited"] == 0, f"不应触发速率限制：{metrics}"
    assert metrics["success"] == 5, f"成功请求数不符合预期：{metrics}"
    assert metrics["p95_latency"] < 2.0, f"P95 延迟过高：{metrics}"


@pytest.mark.performance
def test_architecture_under_limit(ensure_server):
    # /api/v1/analyze/architecture 速率限制：2/300s
    payload = {
        "project_structure": {
            "root": "repo",
            "components": ["controllers", "models", "views"],
            "edges": [["controllers", "models"], ["views", "controllers"]]
        },
        "vectors": None
    }
    metrics = run_load(path="/api/v1/analyze/architecture", method="POST", payload=payload, total_requests=2, concurrency=1)
    assert metrics["errors"] == 0, f"出现网络或服务器错误：{metrics}"
    assert metrics["rate_limited"] == 0, f"不应触发速率限制：{metrics}"
    assert metrics["success"] == 2, f"成功请求数不符合预期：{metrics}"
    assert metrics["p95_latency"] < 2.5, f"P95 延迟过高：{metrics}"


@pytest.mark.performance
def test_similarity_under_limit(ensure_server):
    # /api/v1/similarity 速率限制：10/60s
    payload = {
        "code1": "def add(a,b):\n  return a+b\n",
        "code2": "def sum(x,y):\n  return x+y\n",
        "language": "python",
        "detailed_analysis": True
    }
    metrics = run_load(path="/api/v1/similarity", method="POST", payload=payload, total_requests=10, concurrency=2)
    assert metrics["errors"] == 0, f"出现网络或服务器错误：{metrics}"
    assert metrics["rate_limited"] == 0, f"不应触发速率限制：{metrics}"
    assert metrics["success"] == 10, f"成功请求数不符合预期：{metrics}"
    assert metrics["p95_latency"] < 2.0, f"P95 延迟过高：{metrics}"


@pytest.mark.performance
def test_similarity_burst_rate_limit(ensure_server):
    # 有意超出速率限制，验证限流行为（429）
    payload = {
        "code1": "def add(a,b): return a+b",
        "code2": "def sum(x,y): return x+y",
        "language": "python",
        "detailed_analysis": False
    }
    metrics = run_load(path="/api/v1/similarity", method="POST", payload=payload, total_requests=20, concurrency=10)
    assert metrics["success"] > 0, f"应有部分请求成功：{metrics}"
    assert metrics["rate_limited"] > 0, f"应触发速率限制：{metrics}"
    # 其他错误不应过多
    assert metrics["errors"] == 0, f"除限流外不应有其他错误：{metrics}"