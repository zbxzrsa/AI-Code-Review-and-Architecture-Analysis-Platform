#!/usr/bin/env python3
"""
SLO gate script for next→stable promotions.
Evaluates performance, safety, and evaluation SLOs.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy.promotion_gate import get_promotion_gate


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SLO gate for next→stable promotions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="%(prog)s: %(message)s"
    )
    
    parser.add_argument(
        "--channel",
        choices=["next", "stable"],
        default="next",
        help="Channel to evaluate (default: next)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry run - don't save results"
    )
    
    parser.add_argument(
        "--output",
        default="promotion_evaluation.json",
        help="Output file for evaluation results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main SLO gate execution."""
    args = parse_args()
    
    print("🚀 SLO Gate Evaluation")
    print(f"Channel: {args.channel}")
    print(f"Dry run: {args.dry_run}")
    print(f"Output: {args.output}")
    
    if args.verbose:
        print("📊 Loading promotion policy...")
    
    gate = get_promotion_gate()
    
    if args.verbose:
        print(f"✅ Policy loaded: {gate.policy.name}")
        print(f"📊 Observation window: {gate.policy.observation_window_hours}h")
        print(f"📊 Auto-promote: {gate.policy.auto_promote}")
    
    # Evaluate promotion
    evaluation = gate.evaluate_promotion(args.channel)
    
    if args.verbose:
        print(f"\n📊 Evaluating {len(evaluation['gate_results'])} gates...")
        for result in evaluation['gate_results']:
            status_emoji = {
                "PASS": "✅",
                "FAIL": "❌", 
                "WARNING": "⚠️",
                "SKIP": "⏭️"
            }.get(result.status.value, "❓")
            
            print(f"  {status_emoji} {result.gate_type.value}: {result.message}")
            if result.details.get("failed_gates"):
                for failed_gate in result.details["failed_gates"]:
                    print(f"    - {failed_gate['metric']}: {failed_gate['actual']:.3f} (threshold: {failed_gate['threshold']})")
    
    print(f"  📊 Overall decision: {evaluation['decision']}")
    
    if args.verbose:
        print(f"\n📊 Gate results summary:")
        for result in evaluation['gate_results']:
            print(f"  - {result.gate_type.value}: {result.status.value}")
    
    # Check if auto-promotion should be triggered
    if evaluation['decision'] == "approve" and gate.should_auto_promote(args.channel):
        print(f"\n🚀 Auto-promotion conditions met for {args.channel}")
        print("📊 Would auto-promote next→stable")
    else:
        print(f"⏸️ Auto-promotion not triggered")
    
    # Save evaluation results
    if not args.dry_run:
        output_data = {
            "timestamp": time.time(),
            "channel": args.channel,
            "policy": evaluation["policy"],
            "decision": evaluation["decision"],
            "gate_results": [gate._gate_result_to_dict(r) for r in evaluation['gate_results']],
            "evaluation": evaluation
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n📊 Evaluation saved to: {args.output}")
        
        # Exit with appropriate code
        if evaluation['decision'] == "reject":
            sys.exit(1)
        elif evaluation['decision'] == "warning":
            sys.exit(2)
        else:
            sys.exit(0)


def _gate_result_to_dict(result: GateResult) -> Dict[str, Any]:
    """Convert gate result to dictionary."""
    return {
        "gate_type": result.gate_type.value,
        "status": result.status.value,
        "threshold": result.threshold,
        "actual_value": result.actual_value,
        "message": result.message,
        "details": result.details
    }


if __name__ == "__main__":
    main()