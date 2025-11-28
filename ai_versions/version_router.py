#!/usr/bin/env python3
"""
Dynamic AI Version Router
Routes requests to appropriate AI version based on flags and performance metrics.
"""

import os
import sys
import yaml
import json
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VersionRouter:
    def __init__(self, base_path: str = "ai_versions"):
        self.base_path = Path(base_path)
        self.v1_path = self.base_path / "v1_stable"
        self.v2_path = self.base_path / "v2_experimental"
        self.v3_path = self.base_path / "v3_deprecated"
        self.current_version = self._detect_current_version()
        
    def _detect_current_version(self) -> str:
        """Detect which version is currently active based on environment or symlink"""
        # Check for version flag
        version_flag = os.getenv("AI_VERSION", "auto")
        if version_flag in ["v1", "v2", "v3"]:
            return version_flag
            
        # Auto-detect based on performance metrics
        return self._select_best_version()
    
    def _select_best_version(self) -> str:
        """Select best version based on benchmark metrics"""
        try:
            v1_metrics = self._load_metrics(self.v1_path)
            v2_metrics = self._load_metrics(self.v2_path)
            
            # Prefer v2 if it meets criteria
            if (v2_metrics.get("avg_latency", float('inf')) <= v1_metrics.get("avg_latency", float('inf')) * 1.1 and
                v2_metrics.get("accuracy", 0) >= v1_metrics.get("accuracy", 0)):
                logger.info("Selecting v2 (better performance)")
                return "v2"
            
            logger.info("Defaulting to v1 (stable)")
            return "v1"
            
        except Exception as e:
            logger.warning(f"Error selecting version: {e}, defaulting to v1")
            return "v1"
    
    def _load_metrics(self, version_path: Path) -> Dict[str, Any]:
        """Load metrics from version's config.yaml"""
        config_file = version_path / "config.yaml"
        if not config_file.exists():
            return {}
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                return config.get("metrics", {})
        except Exception as e:
            logger.warning(f"Error loading metrics from {config_file}: {e}")
            return {}
    
    def get_active_config(self) -> Dict[str, Any]:
        """Get configuration for the currently active version"""
        version_paths = {
            "v1": self.v1_path,
            "v2": self.v2_path,
            "v3": self.v3_path
        }
        
        active_path = version_paths.get(self.current_version, self.v1_path)
        config_file = active_path / "config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add version info
        config["version"] = self.current_version
        config["version_path"] = str(active_path)
        
        return config
    
    def route_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Route a request to the appropriate AI version"""
        # Add routing metadata
        request_data["routed_to"] = self.current_version
        request_data["router_version"] = "1.0.0"
        
        logger.info(f"Routing request to {self.current_version}")
        return request_data
    
    def _parse_metric(self, metric_value: Any) -> float:
        """Parse metric value to float, handling string formats like '0.8s'"""
        if isinstance(metric_value, (int, float)):
            return float(metric_value)
        elif isinstance(metric_value, str):
            # Remove common suffixes and convert
            cleaned = metric_value.strip()
            if cleaned.endswith('s'):
                cleaned = cleaned[:-1]  # Remove 's' from seconds
            if cleaned.endswith('ms'):
                cleaned = cleaned[:-2]  # Remove 'ms'
                return float(cleaned) / 1000  # Convert to seconds
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        return 0.0
    
    def compare_versions(self, v1: str, v2: str) -> Dict[str, Any]:
        """Compare two versions and return performance differences"""
        v1_config = self.get_version_config(v1)
        v2_config = self.get_version_config(v2)
        
        v1_metrics = v1_config.get("metrics", {})
        v2_metrics = v2_config.get("metrics", {})
        
        # Convert string metrics to numbers for comparison
        v1_latency = self._parse_metric(v1_metrics.get("avg_latency", "0"))
        v2_latency = self._parse_metric(v2_metrics.get("avg_latency", "0"))
        v1_accuracy = self._parse_metric(v1_metrics.get("accuracy", "0"))
        v2_accuracy = self._parse_metric(v2_metrics.get("accuracy", "0"))
        v1_throughput = self._parse_metric(v1_metrics.get("throughput", "0"))
        v2_throughput = self._parse_metric(v2_metrics.get("throughput", "0"))
        v1_error_rate = self._parse_metric(v1_metrics.get("error_rate", "0"))
        v2_error_rate = self._parse_metric(v2_metrics.get("error_rate", "0"))
        
        comparison = {
            "v1": v1,
            "v2": v2,
            "v1_metrics": v1_metrics,
            "v2_metrics": v2_metrics,
            "differences": {
                "latency": v2_latency - v1_latency,
                "accuracy": v2_accuracy - v1_accuracy,
                "throughput": v2_throughput - v1_throughput,
                "error_rate": v2_error_rate - v1_error_rate
            },
            "recommendation": self._get_promotion_recommendation(v1, v2, v1_metrics, v2_metrics)
        }
        
        return comparison
    
    def get_version_config(self, version: str) -> Dict[str, Any]:
        """Get configuration for a specific version"""
        version_paths = {
            "v1": self.v1_path,
            "v2": self.v2_path,
            "v3": self.v3_path
        }
        
        version_path = version_paths.get(version, self.v1_path)
        config_file = version_path / "config.yaml"
        
        if not config_file.exists():
            return {"version": version, "status": "config_not_found"}
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                config["version"] = version
                config["version_path"] = str(version_path)
                return config
        except Exception as e:
            logger.error(f"Error loading config for {version}: {e}")
            return {"version": version, "status": "config_error", "error": str(e)}
    
    def _get_promotion_recommendation(self, v1: str, v2: str, v1_metrics: Dict, v2_metrics: Dict) -> str:
        """Get recommendation for version promotion"""
        v1_latency = self._parse_metric(v1_metrics.get("avg_latency", "0"))
        v2_latency = self._parse_metric(v2_metrics.get("avg_latency", "0"))
        v1_accuracy = self._parse_metric(v1_metrics.get("accuracy", "0"))
        v2_accuracy = self._parse_metric(v2_metrics.get("accuracy", "0"))
        
        latency_improvement = v1_latency - v2_latency
        accuracy_improvement = v2_accuracy - v1_accuracy
        
        if latency_improvement > 0.1 and accuracy_improvement > 0.05:
            return f"Promote {v2} to stable - significant improvements in latency and accuracy"
        elif latency_improvement > 0.2:
            return f"Consider promoting {v2} - major latency improvement"
        elif accuracy_improvement > 0.1:
            return f"Consider promoting {v2} - major accuracy improvement"
        else:
            return f"Keep {v1} as stable - {v2} doesn't show significant improvements"
    
    def promote_version(self, from_version: str, to_version: str) -> Dict[str, Any]:
        """Promote a version (e.g., v2 to v1)"""
        if from_version == to_version:
            return {"success": False, "error": "Cannot promote to same version"}
        
        version_paths = {
            "v1": self.v1_path,
            "v2": self.v2_path,
            "v3": self.v3_path
        }
        
        from_path = version_paths.get(from_version)
        to_path = version_paths.get(to_version)
        
        if not from_path or not to_path:
            return {"success": False, "error": "Invalid version specified"}
        
        try:
            # Backup current target version
            backup_path = to_path.parent / f"{to_version}_backup_{int(time.time())}"
            if to_path.exists():
                import shutil
                shutil.move(str(to_path), str(backup_path))
            
            # Move from_version to to_version
            import shutil
            shutil.move(str(from_path), str(to_path))
            
            # Create new from_version (empty)
            from_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Promoted {from_version} to {to_version}")
            return {
                "success": True,
                "from_version": from_version,
                "to_version": to_version,
                "backup_path": str(backup_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to promote {from_version} to {to_version}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_all_versions_status(self) -> Dict[str, Any]:
        """Get status of all versions"""
        versions = {}
        
        for version_name, version_path in [
            ("v1", self.v1_path),
            ("v2", self.v2_path),
            ("v3", self.v3_path)
        ]:
            config = self.get_version_config(version_name)
            versions[version_name] = {
                "path": str(version_path),
                "exists": version_path.exists(),
                "config": config,
                "status": "active" if version_name == self.current_version else "inactive"
            }
        
        return {
            "current_version": self.current_version,
            "versions": versions,
            "router_status": "healthy"
        }

# CLI interface
def main():
    if len(sys.argv) < 2:
        print("Usage: python version_router.py <command>")
        print("Commands:")
        print("  get-config       Get active version config")
        print("  route            Route JSON request from stdin")
        print("  compare <v1> <v2> Compare two versions")
        print("  promote <from> <to> Promote version (e.g., v2 to v1)")
        print("  status           Get all versions status")
        print("  version-config <v> Get config for specific version")
        sys.exit(1)
    
    command = sys.argv[1]
    router = VersionRouter()
    
    if command == "get-config":
        config = router.get_active_config()
        print(json.dumps(config, indent=2))
    elif command == "route":
        # Read JSON from stdin
        try:
            request_data = json.loads(sys.stdin.read())
            routed = router.route_request(request_data)
            print(json.dumps(routed, indent=2))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            sys.exit(1)
    elif command == "compare" and len(sys.argv) >= 4:
        v1, v2 = sys.argv[2], sys.argv[3]
        comparison = router.compare_versions(v1, v2)
        print(json.dumps(comparison, indent=2))
    elif command == "promote" and len(sys.argv) >= 4:
        from_version, to_version = sys.argv[2], sys.argv[3]
        result = router.promote_version(from_version, to_version)
        print(json.dumps(result, indent=2))
    elif command == "status":
        status = router.get_all_versions_status()
        print(json.dumps(status, indent=2))
    elif command == "version-config" and len(sys.argv) >= 3:
        version = sys.argv[2]
        config = router.get_version_config(version)
        print(json.dumps(config, indent=2))
    else:
        logger.error(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()