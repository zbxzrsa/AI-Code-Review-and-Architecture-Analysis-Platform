#!/usr/bin/env python3
"""
V2 Configuration Validation Script
Validates v2 experimental config against blocklist and constraints
"""

import yaml
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

class V2Validator:
    def __init__(self, v2_config_path: str, blocklist_path: str):
        self.v2_config_path = Path(v2_config_path)
        self.blocklist_path = Path(blocklist_path)
        
    def validate(self) -> bool:
        """Validate v2 configuration"""
        try:
            # Load blocklist
            with open(self.blocklist_path, 'r') as f:
                blocklist = yaml.safe_load(f)
            
            # Load v2 config
            with open(self.v2_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check model blocklist
            model_name = config.get("model", {}).get("name", "")
            blocked_models = blocklist.get("blocked_models", [])
            
            if model_name in blocked_models:
                print(f"❌ Validation failed: Model '{model_name}' is blocked")
                print(f"Reason: {self._get_block_reason(model_name, blocked_models)}")
                return False
            
            # Check constraints
            constraints = config.get("constraints", {})
            issues = []
            
            # Check GPU requirement
            if constraints.get("gpu_required", False) and not self._has_gpu_support():
                issues.append("GPU required but not available")
            
            # Check memory requirement
            max_memory = constraints.get("max_memory", "0GB")
            if not self._validate_memory(max_memory):
                issues.append(f"Insufficient memory: requires {max_memory}")
            
            # Check timeout
            timeout = constraints.get("timeout", 30)
            if timeout > 300:  # 5 minutes max
                issues.append(f"Timeout too high: {timeout}s (max: 300s)")
            
            if issues:
                print(f"❌ Validation failed:")
                for issue in issues:
                    print(f"  - {issue}")
                return False
            
            print(f"✅ V2 configuration is valid")
            return True
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False
    
    def _get_block_reason(self, model_name: str, blocked_models: List[str]) -> str:
        """Get reason why model is blocked"""
        reasons = {
            "gpt-3.5-turbo": "high_latency",
            "mistral-7b": "cost"
        }
        return reasons.get(model_name, "unknown")
    
    def _has_gpu_support(self) -> bool:
        """Check if GPU support is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            # Fallback check
            import subprocess
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                return result.returncode == 0
            except FileNotFoundError:
                return False
    
    def _validate_memory(self, max_memory: str) -> bool:
        """Validate memory requirement"""
        try:
            import psutil
            available_gb = psutil.virtual_memory().available // (1024**3)
            required_gb = int(max_memory.replace("GB", ""))
            return available_gb >= required_gb
        except ImportError:
            # If psutil not available, assume sufficient
            return True

def main():
    parser = argparse.ArgumentParser(description="V2 Configuration Validation")
    parser.add_argument("--v2_config", required=True, help="Path to v2 config.yaml")
    parser.add_argument("--blocklist", required=True, help="Path to blocklist.yaml")
    
    args = parser.parse_args()
    
    validator = V2Validator(args.v2_config, args.blocklist)
    success = validator.validate()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()