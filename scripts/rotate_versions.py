#!/usr/bin/env python3
"""
AI Version Rotation Script
Handles promotion and demotion of AI versions based on benchmark results
"""

import json
import yaml
import shutil
import logging
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VersionRotator:
    def __init__(self, v1_path: str, v2_path: str, v3_path: str):
        self.v1_path = Path(v1_path)
        self.v2_path = Path(v2_path)
        self.v3_path = Path(v3_path)
        self.base_path = self.v1_path.parent
        
    def rotate_versions(self, metrics_file: str, force_promote: bool = False):
        """Rotate versions based on metrics or force flag"""
        
        # Load benchmark results
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        recommendation = metrics.get("recommendation", "keep_current")
        
        if force_promote:
            recommendation = "promote_v2_to_v1"
            logger.info("Force promoting v2 to v1")
        
        if recommendation == "promote_v2_to_v1":
            self._promote_v2_to_v1(metrics)
        elif recommendation == "demote_v1_to_v3":
            self._demote_v1_to_v3(metrics)
        else:
            logger.info("Keeping current version configuration")
        
        self._commit_changes(recommendation, metrics)
    
    def _promote_v2_to_v1(self, metrics: Dict[str, Any]):
        """Promote v2 to v1 stable position"""
        logger.info("🚀 Promoting v2 to v1 stable")
        
        # Backup current v1
        v1_backup = self.base_path / "v1_backup"
        if v1_backup.exists():
            shutil.rmtree(v1_backup)
        shutil.copytree(self.v1_path, v1_backup)
        
        # Move v2 to v1
        if self.v1_path.exists():
            shutil.rmtree(self.v1_path)
        shutil.copytree(self.v2_path, self.v1_path)
        
        # Update v1 config to mark as promoted
        self._update_v1_config(metrics.get("v2", {}))
        
        logger.info("✅ v2 promoted to v1 successfully")
    
    def _demote_v1_to_v3(self, metrics: Dict[str, Any]):
        """Demote v1 to v3 deprecated position"""
        logger.info("📚 Demoting v1 to v3 deprecated")
        
        # Move current v1 to v3
        if self.v3_path.exists():
            shutil.rmtree(self.v3_path)
        shutil.copytree(self.v1_path, self.v3_path)
        
        # Restore v1 from backup if available
        v1_backup = self.base_path / "v1_backup"
        if v1_backup.exists():
            if self.v1_path.exists():
                shutil.rmtree(self.v1_path)
            shutil.copytree(v1_backup, self.v1_path)
            shutil.rmtree(v1_backup)
        
        # Update v3 metrics
        self._update_v3_metrics(metrics.get("v1", {}))
        
        logger.info("✅ v1 demoted to v3 successfully")
    
    def _update_v1_config(self, v2_metrics: Dict[str, Any]):
        """Update v1 config with v2 metrics"""
        config_file = self.v1_path / "config.yaml"
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error reading v1 config: {e}")
            return
        
        # Update with v2 metrics (now v1 metrics)
        config["metrics"] = v2_metrics.get("metrics", {})
        config["promoted_from"] = "v2_experimental"
        config["promotion_date"] = metrics.get("timestamp")
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _update_v3_metrics(self, v1_metrics: Dict[str, Any]):
        """Update v3 with failure metrics"""
        config_file = self.v3_path / "config.yaml"
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error reading v3 config: {e}")
            return
        
        # Add failure reason
        config["failure_reason"] = "outperformed_by_v2"
        config["demotion_date"] = v1_metrics.get("timestamp", "")
        config["previous_version"] = "v1_stable"
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _commit_changes(self, action: str, metrics: Dict[str, Any]):
        """Commit version changes to git"""
        try:
            import subprocess
            
            # Add all changes
            subprocess.run(["git", "add", "ai_versions/"], check=True)
            
            # Create commit message
            commit_msg = f"chore: {action}"
            if action == "promote_v2_to_v1":
                v2_metrics = metrics.get("v2", {})
                commit_msg += f" - v2 latency: {v2_metrics.get('avg_latency', 'N/A')}s"
            elif action == "demote_v1_to_v3":
                v1_metrics = metrics.get("v1", {})
                commit_msg += f" - v1 latency: {v1_metrics.get('avg_latency', 'N/A')}s"
            
            # Commit changes
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            
            logger.info(f"Changes committed: {commit_msg}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit changes: {e}")

def main():
    parser = argparse.ArgumentParser(description="AI Version Rotation")
    parser.add_argument("--metrics", required=True, help="Benchmark metrics JSON file")
    parser.add_argument("--v1_path", required=True, help="Path to v1 stable")
    parser.add_argument("--v2_path", required=True, help="Path to v2 experimental")
    parser.add_argument("--v3_path", required=True, help="Path to v3 deprecated")
    parser.add_argument("--force_promote", action="store_true", help="Force promote v2 to v1")
    
    args = parser.parse_args()
    
    rotator = VersionRotator(args.v1_path, args.v2_path, args.v3_path)
    rotator.rotate_versions(args.metrics, args.force_promote)

if __name__ == "__main__":
    main()