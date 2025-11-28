#!/usr/bin/env python3
"""
AI Version Benchmark Script
Compares v1 vs v2 performance on code review tasks
"""

import json
import time
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIBenchmark:
    def __init__(self, v1_path: str, v2_path: str):
        self.v1_path = Path(v1_path)
        self.v2_path = Path(v2_path)
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test code snippets for benchmarking"""
        return [
            {
                "id": "python_security",
                "language": "python",
                "code": """
import os
import subprocess

def execute_command(user_input):
    # Potential security issue - command injection
    return subprocess.run(user_input, shell=True, capture_output=True)
                """,
                "expected_issues": ["security", "command_injection"],
                "description": "Python code with command injection vulnerability"
            },
            {
                "id": "javascript_complexity",
                "language": "javascript",
                "code": """
function processComplexData(data) {
    var result = [];
    for (var i = 0; i < data.length; i++) {
        for (var j = 0; j < data[i].items.length; j++) {
            for (var k = 0; k < data[i].items[j].features.length; k++) {
                result.push({
                    id: data[i].items[j].features[k].id,
                    value: data[i].items[j].features[k].value * 2
                });
            }
        }
    }
    return result;
}
                """,
                "expected_issues": ["complexity", "performance"],
                "description": "High complexity JavaScript with nested loops"
            },
            {
                "id": "python_style",
                "language": "python",
                "code": """
def Calculate(a,b):
    x=a+b
    return x
                """,
                "expected_issues": ["style", "naming"],
                "description": "Python code with style issues"
            }
        ]
    
    def _run_benchmark(self, version_path: Path, version_name: str) -> Dict[str, Any]:
        """Run benchmark against a specific version"""
        logger.info(f"Running benchmark for {version_name}")
        
        config_file = version_path / "config.yaml"
        if not config_file.exists():
            return {"error": f"Config not found: {config_file}"}
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            return {"error": f"Failed to load config: {e}"}
        
        results = {
            "version": version_name,
            "model": config.get("model", {}),
            "test_results": [],
            "total_time": 0,
            "avg_latency": 0,
            "accuracy": 0,
            "false_positives": 0,
            "cost": 0
        }
        
        for i, test_case in enumerate(self.test_cases):
            logger.info(f"Running test case {i+1}/{len(self.test_cases)}: {test_case['id']}")
            
            start_time = time.time()
            
            # Simulate running the AI version
            try:
                test_result = self._simulate_ai_call(config, test_case)
                end_time = time.time()
                latency = end_time - start_time
                
                test_result["latency"] = latency
                test_result["test_case"] = test_case["id"]
                
                results["test_results"].append(test_result)
                results["total_time"] += latency
                
            except Exception as e:
                logger.error(f"Test case {test_case['id']} failed: {e}")
                results["test_results"].append({
                    "test_case": test_case["id"],
                    "error": str(e),
                    "latency": float('inf')
                })
        
        # Calculate metrics
        if results["test_results"]:
            valid_results = [r for r in results["test_results"] if "error" not in r]
            if valid_results:
                results["avg_latency"] = sum(r["latency"] for r in valid_results) / len(valid_results)
                
                # Calculate accuracy (simplified)
                correct = 0
                for result in valid_results:
                    expected = next(tc for tc in self.test_cases if tc["id"] == result["test_case"])
                    if self._evaluate_accuracy(result, expected):
                        correct += 1
                
                results["accuracy"] = correct / len(valid_results)
                results["false_positives"] = self._count_false_positives(valid_results, self.test_cases)
                
                # Calculate cost (simulated)
                results["cost"] = self._calculate_cost(config, results["total_time"])
        
        return results
    
    def _simulate_ai_call(self, config: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate an AI call based on version configuration"""
        model_type = config.get("model", {}).get("type", "transformers")
        
        if model_type == "transformers":
            # Simulate CodeBERT (v1) - faster but less accurate
            base_latency = 0.8
            accuracy = 0.85
            return self._generate_review_result(test_case, base_latency, accuracy)
            
        elif model_type == "ollama":
            # Simulate Llama2 (v2) - slower but more accurate
            base_latency = 1.2
            accuracy = 0.88
            return self._generate_review_result(test_case, base_latency, accuracy)
            
        else:
            # Fallback
            return self._generate_review_result(test_case, 2.0, 0.75)
    
    def _generate_review_result(self, test_case: Dict[str, Any], latency: float, accuracy: float) -> Dict[str, Any]:
        """Generate a simulated review result"""
        import random
        
        # Simulate finding issues based on test case
        found_issues = []
        if test_case["id"] == "python_security":
            if random.random() < accuracy:
                found_issues = ["security", "command_injection"]
        elif test_case["id"] == "javascript_complexity":
            if random.random() < accuracy:
                found_issues = ["complexity", "performance"]
        elif test_case["id"] == "python_style":
            if random.random() < accuracy:
                found_issues = ["style", "naming"]
        
        return {
            "review": f"Code review completed for {test_case['description']}",
            "issues": found_issues,
            "confidence": accuracy,
            "tokens_used": int(latency * 100)  # Simulated
        }
    
    def _evaluate_accuracy(self, result: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Evaluate if AI result matches expected issues"""
        if "error" in result:
            return False
        
        found_issues = set(result.get("issues", []))
        expected_issues = set(expected.get("expected_issues", []))
        
        # At least 50% match expected issues
        overlap = len(found_issues.intersection(expected_issues))
        return overlap >= len(expected_issues) * 0.5
    
    def _count_false_positives(self, results: List[Dict[str, Any]], test_cases: List[Dict[str, Any]]) -> int:
        """Count false positives in results"""
        false_positives = 0
        
        for result in results:
            test_case = next(tc for tc in test_cases if tc["id"] == result["test_case"])
            expected_issues = set(test_case.get("expected_issues", []))
            found_issues = set(result.get("issues", []))
            
            # Issues found that weren't expected
            false_positive_issues = found_issues - expected_issues
            false_positives += len(false_positive_issues)
        
        return false_positives
    
    def _calculate_cost(self, config: Dict[str, Any], total_time: float) -> float:
        """Calculate cost based on model type and usage"""
        model_type = config.get("model", {}).get("type", "transformers")
        
        if model_type == "transformers":
            # Free open source model
            return 0.001 * total_time  # Minimal cost
        elif model_type == "ollama":
            # Free local model
            return 0.0  # No cost
        else:
            # Paid API
            return 0.05 * total_time  # Higher cost
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run full benchmark comparison"""
        logger.info("Starting AI version benchmark comparison")
        
        v1_results = self._run_benchmark(self.v1_path, "v1_stable")
        v2_results = self._run_benchmark(self.v2_path, "v2_experimental")
        
        comparison = {
            "v1": v1_results,
            "v2": v2_results,
            "recommendation": self._make_recommendation(v1_results, v2_results),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        logger.info(f"Benchmark complete: {comparison['recommendation']}")
        return comparison
    
    def _make_recommendation(self, v1: Dict[str, Any], v2: Dict[str, Any]) -> str:
        """Make promotion/demotion recommendation"""
        if "error" in v1 or "error" in v2:
            return "error - both versions failed"
        
        v1_latency = v1.get("avg_latency", float('inf'))
        v2_latency = v2.get("avg_latency", float('inf'))
        v1_accuracy = v1.get("accuracy", 0)
        v2_accuracy = v2.get("accuracy", 0)
        
        # Promote v2 if it meets criteria
        if (v2_latency <= v1_latency * 1.1 and v2_accuracy >= v1_accuracy):
            return "promote_v2_to_v1"
        
        # Demote v1 if v2 is significantly better
        if (v2_latency < v1_latency * 0.8 and v2_accuracy > v1_accuracy * 1.1):
            return "demote_v1_to_v3"
        
        return "keep_current"

def main():
    parser = argparse.ArgumentParser(description="AI Version Benchmark")
    parser.add_argument("--v1_path", required=True, help="Path to v1 stable")
    parser.add_argument("--v2_path", required=True, help="Path to v2 experimental")
    parser.add_argument("--output", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    benchmark = AIBenchmark(args.v1_path, args.v2_path)
    results = benchmark.run_comparison()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmark complete: {results['recommendation']}")
    return results['recommendation']

if __name__ == "__main__":
    main()