#!/usr/bin/env python3
"""
P2 性能测试脚本

目标：
1. P95 < 60s（完整 PR 分析）
2. 缓存命中率 >= 60%
3. 依赖图计算 < 2s

测试场景：
- 100 个文件的 PR
- 20 个文件变更
- 50 个文件受影响（通过依赖图）
"""

import time
import json
import subprocess
import statistics
from typing import List, Dict
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8000"
PERF_RESULTS = []


def log(msg: str):
    """记录日志"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def trigger_analysis(pr_number: int, head_sha: str) -> Dict:
    """触发分析"""
    payload = {
        "rulepack_version": "default",
        "include_categories": ["security", "performance"],
    }

    response = requests.post(
        f"{API_URL}/api/v1/pr/{pr_number}/analyze",
        json=payload,
        headers={"X-GitHub-Token": "test-token"}
    )

    if response.status_code != 200:
        raise Exception(f"Failed to trigger analysis: {response.text}")

    return response.json()


def wait_analysis_complete(pr_number: int, head_sha: str, timeout: int = 120) -> Dict:
    """等待分析完成"""
    start = time.time()

    while time.time() - start < timeout:
        response = requests.get(
            f"{API_URL}/api/v1/pr/{pr_number}/analysis",
            params={"sha": head_sha}
        )

        if response.status_code != 200:
            time.sleep(1)
            continue

        result = response.json()

        if result['status'] in ['completed', 'failed']:
            return result

        elapsed = time.time() - start
        progress = result.get('progress', 0)
        log(f"Progress: {progress}% ({elapsed:.1f}s)")

        time.sleep(2)

    raise TimeoutError(f"Analysis timeout after {timeout}s")


def benchmark_pr_analysis():
    """基准测试：PR 分析时间"""
    log("=== Benchmark 1: PR Analysis Time ===")

    times = []

    for i in range(3):  # 运行 3 次
        log(f"\nRound {i+1}/3")

        pr_number = 1000 + i
        head_sha = f"abc{i}" * 10  # 模拟 SHA

        # 触发分析
        log("Triggering analysis...")
        analyze_response = trigger_analysis(pr_number, head_sha)

        # 等待完成
        start = time.time()
        log("Waiting for completion...")
        result = wait_analysis_complete(pr_number, head_sha)
        elapsed = time.time() - start

        times.append(elapsed)

        log(f"✓ Completed in {elapsed:.2f}s")
        log(f"  - Total issues: {result.get('summary', {}).get('total_issues', 0)}")
        log(f"  - Cache hit ratio: {result.get('performance', {}).get('cache_hit_ratio', 0):.1%}")

    # 统计
    avg_time = statistics.mean(times)
    p95_time = sorted(times)[-1]  # 3 次取最大值作为 P95 估计

    log(f"\n📊 Results:")
    log(f"  Average: {avg_time:.2f}s")
    log(f"  P95 (est): {p95_time:.2f}s")
    log(f"  Target: P95 < 60s ✓ PASS" if p95_time < 60 else f"  Target: P95 < 60s ✗ FAIL")

    PERF_RESULTS.append({
        "test": "pr_analysis_time",
        "p95": p95_time,
        "average": avg_time,
        "passed": p95_time < 60
    })


def benchmark_cache_hit_ratio():
    """基准测试：缓存命中率"""
    log("\n=== Benchmark 2: Cache Hit Ratio ===")

    hit_ratios = []

    for i in range(3):
        log(f"\nRound {i+1}/3")

        pr_number = 2000 + i
        head_sha = f"def{i}" * 10

        # 第一次分析（冷启动）
        log("Cold start analysis...")
        trigger_analysis(pr_number, head_sha)
        result1 = wait_analysis_complete(pr_number, head_sha)
        cache_ratio_1 = result1.get('performance', {}).get('cache_hit_ratio', 0)
        log(f"  Cache hit ratio: {cache_ratio_1:.1%}")

        # 第二次分析（相同提交）- 应该有缓存
        log("Warm cache analysis...")
        pr_number += 100
        trigger_analysis(pr_number, head_sha)
        result2 = wait_analysis_complete(pr_number, head_sha)
        cache_ratio_2 = result2.get('performance', {}).get('cache_hit_ratio', 0)
        log(f"  Cache hit ratio: {cache_ratio_2:.1%}")

        hit_ratios.append(cache_ratio_2)

    avg_ratio = statistics.mean(hit_ratios)

    log(f"\n📊 Results:")
    log(f"  Average cache hit ratio: {avg_ratio:.1%}")
    log(f"  Target: >= 60% ✓ PASS" if avg_ratio >= 0.6 else f"  Target: >= 60% ✗ FAIL")

    PERF_RESULTS.append({
        "test": "cache_hit_ratio",
        "average": avg_ratio,
        "passed": avg_ratio >= 0.6
    })


def benchmark_dependency_graph():
    """基准测试：依赖图计算"""
    log("\n=== Benchmark 3: Dependency Graph Analysis ===")

    times = []

    for i in range(3):
        log(f"\nRound {i+1}/3")

        changed_files = [f"src/module{j}.py" for j in range(i * 5 + 5, i * 5 + 25)]

        start = time.time()
        response = requests.post(
            f"{API_URL}/api/v1/dependency-graph/analyze-change",
            json={"changed_files": changed_files, "max_depth": 10}
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            log(f"✓ Analyzed {len(changed_files)} files in {elapsed:.3f}s")
            log(f"  Impact: {result.get('affected_count', 0)} affected files")
            times.append(elapsed)
        else:
            log(f"✗ Failed: {response.text}")

    avg_time = statistics.mean(times) if times else 0

    log(f"\n📊 Results:")
    log(f"  Average: {avg_time:.3f}s")
    log(f"  Target: < 2s ✓ PASS" if avg_time < 2 else f"  Target: < 2s ✗ FAIL")

    PERF_RESULTS.append({
        "test": "dependency_graph",
        "average": avg_time,
        "passed": avg_time < 2
    })


def benchmark_rule_filtering():
    """基准测试：规则过滤"""
    log("\n=== Benchmark 4: Rule Filtering ===")

    times = []

    # 生成模拟 issue 集合
    test_cases = [
        {"name": "100 issues", "count": 100},
        {"name": "1000 issues", "count": 1000},
        {"name": "5000 issues", "count": 5000},
    ]

    for test_case in test_cases:
        issues = [
            {
                "rule_id": f"rule_{i % 10}",
                "file_path": f"src/module_{i % 20}.py",
                "line": (i % 100) + 1,
                "message": f"Test issue {i}",
                "severity": ["error", "warning", "info"][i % 3],
                "category": ["security", "performance", "style"][i % 3]
            }
            for i in range(test_case["count"])
        ]

        payload = {
            "issues": issues,
            "severity_threshold": "warning"
        }

        start = time.time()
        response = requests.post(
            f"{API_URL}/api/v1/rules/filter",
            json=payload
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            log(f"{test_case['name']}: {elapsed:.3f}s ({result.get('total_output')} output)")
            times.append(elapsed)
        else:
            log(f"{test_case['name']}: FAILED")

    PERF_RESULTS.append({
        "test": "rule_filtering",
        "times": times,
        "passed": all(t < 1.0 for t in times)  # 每个应该 < 1s
    })


def benchmark_concurrent_pr_analysis():
    """基准测试：并发 PR 分析"""
    log("\n=== Benchmark 5: Concurrent PR Analysis ===")

    num_concurrent = 5

    def analyze_pr(pr_id):
        try:
            pr_number = 3000 + pr_id
            head_sha = f"ghi{pr_id}" * 10

            trigger_analysis(pr_number, head_sha)
            start = time.time()
            result = wait_analysis_complete(pr_number, head_sha, timeout=120)
            elapsed = time.time() - start

            return {
                "pr_id": pr_id,
                "time": elapsed,
                "success": result.get('status') == 'completed'
            }
        except Exception as e:
            return {
                "pr_id": pr_id,
                "time": None,
                "success": False,
                "error": str(e)
            }

    log(f"Starting {num_concurrent} concurrent analyses...")
    results = []

    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(analyze_pr, i) for i in range(num_concurrent)]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result['success']:
                log(f"  PR {result['pr_id']}: {result['time']:.2f}s ✓")
            else:
                log(f"  PR {result['pr_id']}: FAILED ✗")

    successful = [r for r in results if r['success']]
    if successful:
        avg_time = statistics.mean(r['time'] for r in successful)
        p95_time = sorted([r['time'] for r in successful])[-1]

        log(f"\n📊 Results:")
        log(f"  Successful: {len(successful)}/{num_concurrent}")
        log(f"  Average: {avg_time:.2f}s")
        log(f"  P95: {p95_time:.2f}s")

        PERF_RESULTS.append({
            "test": "concurrent_analysis",
            "p95": p95_time,
            "passed": p95_time < 60
        })


def print_summary():
    """打印总结"""
    log("\n" + "=" * 60)
    log("PERFORMANCE TEST SUMMARY")
    log("=" * 60)

    passed = sum(1 for r in PERF_RESULTS if r.get('passed', False))
    total = len(PERF_RESULTS)

    for result in PERF_RESULTS:
        test_name = result['test']
        passed_str = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
        log(f"{test_name}: {passed_str}")

    log(f"\nTotal: {passed}/{total} tests passed")

    # 保存结果
    with open("p2_performance_results.json", "w") as f:
        json.dump(PERF_RESULTS, f, indent=2, default=str)

    log(f"Results saved to p2_performance_results.json")


if __name__ == "__main__":
    log("Starting P2 Performance Benchmarks")
    log(f"Target API: {API_URL}")

    try:
        benchmark_pr_analysis()
        benchmark_cache_hit_ratio()
        benchmark_dependency_graph()
        benchmark_rule_filtering()
        benchmark_concurrent_pr_analysis()
    except Exception as e:
        log(f"✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print_summary()
