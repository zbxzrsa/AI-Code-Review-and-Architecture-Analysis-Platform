#!/usr/bin/env python3
"""
Self-Healing AI System
Generates patches for v1 when v2 fails
"""

import json
import yaml
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfHealingSystem:
    def __init__(self, v1_path: str, v2_metrics_file: str):
        self.v1_path = Path(v1_path)
        self.v2_metrics_file = Path(v2_metrics_file)
        self.v1_config_file = self.v1_path / "config.yaml"
        
    def detect_failure(self) -> bool:
        """Detect if v2 has failed based on metrics"""
        try:
            with open(self.v2_metrics_file, 'r') as f:
                v2_metrics = json.load(f)
            
            # Check for failure conditions
            v2_false_positives = v2_metrics.get("v2", {}).get("false_positives", 0)
            v2_latency = v2_metrics.get("v2", {}).get("avg_latency", float('inf'))
            
            # Load v1 metrics for comparison
            v1_metrics = self._load_v1_metrics()
            v1_false_positives = v1_metrics.get("false_positives", 0)
            
            # Failure detection: v2 has 50% more false positives than v1
            if v2_false_positives > v1_false_positives * 1.5:
                logger.warning(f"v2 failure detected: {v2_false_positives} vs {v1_false_positives} false positives")
                return True
            
            # Failure detection: v2 latency is too high
            if v2_latency > 5.0:  # 5 seconds threshold
                logger.warning(f"v2 failure detected: latency {v2_latency}s exceeds threshold")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error detecting failure: {e}")
            return False
    
    def _load_v1_metrics(self) -> Dict[str, Any]:
        """Load v1 metrics from config"""
        try:
            with open(self.v1_config_file, 'r') as f:
                config = yaml.safe_load(f)
                return config.get("metrics", {})
        except Exception as e:
            logger.warning(f"Error loading v1 metrics: {e}")
            return {}
    
    def generate_patch(self) -> Dict[str, Any]:
        """Generate a patch for v1 to handle v2's failure"""
        logger.info("🔧 Generating self-healing patch for v1")
        
        try:
            with open(self.v2_metrics_file, 'r') as f:
                v2_metrics = json.load(f)
            
            failure_reason = self._analyze_failure_reason(v2_metrics)
            patch = self._create_patch_for_failure(failure_reason)
            
            # Apply patch to v1 config
            self._apply_patch_to_v1(patch)
            
            logger.info(f"✅ Patch generated: {patch['description']}")
            return patch
            
        except Exception as e:
            logger.error(f"Error generating patch: {e}")
            return {"error": str(e)}
    
    def _analyze_failure_reason(self, v2_metrics: Dict[str, Any]) -> str:
        """Analyze why v2 failed"""
        v2_data = v2_metrics.get("v2", {})
        
        false_positives = v2_data.get("false_positives", 0)
        latency = v2_data.get("avg_latency", 0)
        
        if false_positives > 10:
            return "high_false_positives"
        elif latency > 5.0:
            return "high_latency"
        elif v2_data.get("accuracy", 0) < 0.7:
            return "low_accuracy"
        else:
            return "unknown"
    
    def _create_patch_for_failure(self, failure_reason: str) -> Dict[str, Any]:
        """Create a patch based on failure reason"""
        patches = {
            "high_false_positives": {
                "description": "Add post-processing filter for false positives",
                "rule": "post_process:filter_false_positives",
                "threshold": 0.8,
                "action": "Apply confidence threshold to reduce false positives"
            },
            "high_latency": {
                "description": "Optimize model inference",
                "rule": "optimize:inference_speed",
                "action": "Reduce model complexity and add caching"
            },
            "low_accuracy": {
                "description": "Improve prompt engineering",
                "rule": "prompt:enhance_clarity",
                "action": "Add context and examples to prompts"
            }
        }
        
        return patches.get(failure_reason, {
            "description": "Generic patch",
            "rule": "generic:improvement",
            "action": "Apply general improvements"
        })
    
    def _apply_patch_to_v1(self, patch: Dict[str, Any]):
        """Apply patch to v1 configuration"""
        try:
            with open(self.v1_config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Add patch to rules
            if "rules" not in config:
                config["rules"] = []
            
            config["rules"].append(patch["rule"])
            
            # Add patch metadata
            config["self_healing"] = {
                "patch_applied": True,
                "patch_description": patch["description"],
                "patch_date": datetime.datetime.now().isoformat(),
                "patch_reason": "v2_failure_recovery"
            }
            
            with open(self.v1_config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Patch applied to v1: {patch['rule']}")
            
        except Exception as e:
            logger.error(f"Error applying patch: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Self-Healing AI System")
    parser.add_argument("--v1_path", required=True, help="Path to v1 stable")
    parser.add_argument("--v2_metrics", required=True, help="V2 metrics JSON file")
    parser.add_argument("--apply", action="store_true", help="Apply patch if failure detected")
    
    args = parser.parse_args()
    
    healer = SelfHealingSystem(args.v1_path, args.v2_metrics)
    
    if healer.detect_failure():
        if args.apply:
            patch = healer.generate_patch()
            print(json.dumps(patch, indent=2))
        else:
            print("Failure detected but --apply not specified")
    else:
        print("No failure detected")

if __name__ == "__main__":
    main()