#!/usr/bin/env python3
"""
AI Evaluation Harness - Offline evaluation for three-channel AI system
Usage: python -m backend.ai.eval --channel next --dataset datasets/ai_eval/base.jsonl --offline
"""
import argparse
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from backend.ai.router import pick


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return " ".join(text.lower().strip().split())


def keyword_score(output: str, expected: List[str], negative: List[str], weights: Dict[str, float]) -> float:
    """Calculate keyword-based score with weights"""
    output_norm = normalize_text(output)
    score = 0.0
    total_weight = 0.0
    
    # Positive keywords
    for keyword in expected:
        keyword_norm = normalize_text(keyword)
        weight = weights.get(keyword, 1.0)
        total_weight += weight
        if keyword_norm in output_norm:
            score += weight
    
    # Negative keywords (penalty)
    for neg_keyword in negative:
        neg_norm = normalize_text(neg_keyword)
        if neg_norm in output_norm:
            score -= 1.0  # Fixed penalty for negative matches
    
    return max(0.0, score / total_weight) if total_weight > 0 else 0.0


def semantic_score(output: str, expected: List[str], model: SentenceTransformer) -> float:
    """Calculate semantic similarity using sentence transformers"""
    if not TRANSFORMERS_AVAILABLE:
        return 0.0
    
    # Encode output and expected keywords
    output_emb = model.encode([output])
    expected_emb = model.encode(expected)
    
    # Calculate max similarity
    similarities = cosine_similarity(output_emb, expected_emb)[0]
    return float(max(similarities))


def evaluate_item(item: Dict[str, Any], engine, model: Optional[SentenceTransformer] = None, offline: bool = True) -> Dict[str, Any]:
    """Evaluate a single evaluation item"""
    input_text = item["input"]
    expected = item["expected"]
    negative = item.get("negative", [])
    weights = item.get("weights", {})
    
    start_time = time.time()
    
    try:
        response = engine.review(f"Review this code change and suggest improvements:\n{input_text}")
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate scores
        if offline or not TRANSFORMERS_AVAILABLE:
            score = keyword_score(response, expected, negative, weights)
            semantic_sim = 0.0
        else:
            score = keyword_score(response, expected, negative, weights)
            semantic_sim = semantic_score(response, expected, model)
            # Combine scores (70% keyword, 30% semantic)
            score = 0.7 * score + 0.3 * semantic_sim
        
        return {
            "id": item["id"],
            "input": input_text,
            "output": response,
            "expected": expected,
            "negative": negative,
            "score": score,
            "semantic_similarity": semantic_sim,
            "latency_ms": latency_ms,
            "tokens_out": len(response.split()),
            "error": None
        }
    except Exception as e:
        return {
            "id": item["id"],
            "input": input_text,
            "output": None,
            "expected": expected,
            "negative": negative,
            "score": 0.0,
            "semantic_similarity": 0.0,
            "latency_ms": (time.time() - start_time) * 1000,
            "tokens_out": 0,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="AI Evaluation Harness")
    parser.add_argument("--channel", choices=["stable", "next", "legacy"], default="stable", help="AI channel to evaluate")
    parser.add_argument("--dataset", required=True, help="Path to evaluation dataset (.jsonl)")
    parser.add_argument("--offline", action="store_true", help="Force offline keyword-only evaluation")
    parser.add_argument("--output", help="Output directory for reports (default: ./ai_models/eval_reports)")
    args = parser.parse_args()
    
    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset {dataset_path} not found")
        return 1
    
    with open(dataset_path, 'r') as f:
        items = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(items)} evaluation items from {dataset_path}")
    
    # Initialize model if available and not offline
    model = None
    if not offline and TRANSFORMERS_AVAILABLE:
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get engine for channel
    engine = pick(args.channel)
    print(f"Evaluating channel: {args.channel}")
    
    # Run evaluation
    results = []
    for i, item in enumerate(items, 1):
        print(f"Evaluating item {i}/{len(items)}: {item['id']}")
        result = evaluate_item(item, engine, model, args.offline)
        results.append(result)
    
    # Calculate summary statistics
    scores = [r["score"] for r in results if r["error"] is None]
    latencies = [r["latency_ms"] for r in results if r["error"] is None]
    tokens = [r["tokens_out"] for r in results if r["error"] is None]
    
    summary = {
        "channel": args.channel,
        "dataset": str(dataset_path),
        "timestamp": datetime.utcnow().isoformat(),
        "total_items": len(items),
        "successful_items": len(scores),
        "failed_items": len(items) - len(scores),
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "avg_tokens_out": sum(tokens) / len(tokens) if tokens else 0.0,
        "offline_mode": args.offline or not TRANSFORMERS_AVAILABLE
    }
    
    # Prepare report
    report = {
        "summary": summary,
        "results": results
    }
    
    # Save report
    output_dir = Path(args.output) if args.output else Path("./ai_models/eval_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{args.channel}_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Channel: {summary['channel']}")
    print(f"Dataset: {summary['dataset']}")
    print(f"Items: {summary['successful_items']}/{summary['total_items']} successful")
    print(f"Average Score: {summary['avg_score']:.3f}")
    print(f"Score Range: {summary['min_score']:.3f} - {summary['max_score']:.3f}")
    print(f"Average Latency: {summary['avg_latency_ms']:.1f}ms")
    print(f"Average Tokens: {summary['avg_tokens_out']:.1f}")
    print(f"Report saved to: {report_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())