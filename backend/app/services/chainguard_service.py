# Chainguard Container Security Integration
# Feature flag: USE_CHAINGUARD=true

import os
import subprocess
import json
from typing import Dict, Any, List
from pathlib import Path

from app.core.feature_flags import FeatureFlags

class ChainguardService:
    """Container security service with Chainguard integration."""
    
    def __init__(self):
        self.use_chainguard = FeatureFlags.USE_CHAINGUARD
        self.chainguard_binary = "chainguard"
    
    def scan_image(self, image_name: str) -> Dict[str, Any]:
        """Scan Docker image for vulnerabilities."""
        if not self.use_chainguard:
            return self._mock_scan(image_name)
        
        try:
            # Run Chainguard image scan
            cmd = [
                self.chainguard_binary,
                "image", "scan",
                "--format", "json",
                "--output", "scan-results.json",
                image_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse results
            if Path("scan-results.json").exists():
                with open("scan-results.json", 'r') as f:
                    scan_data = json.load(f)
                return scan_data
            
            return {"status": "completed", "output": result.stdout}
            
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "error": str(e),
                "output": e.stdout
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_sbom(self, image_name: str) -> Dict[str, Any]:
        """Generate Software Bill of Materials."""
        if not self.use_chainguard:
            return self._mock_sbom(image_name)
        
        try:
            # Generate SBOM
            cmd = [
                self.chainguard_binary,
                "image", "sbom",
                "--format", "json",
                "--output", "sbom.json",
                image_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse SBOM
            if Path("sbom.json").exists():
                with open("sbom.json", 'r') as f:
                    sbom_data = json.load(f)
                return sbom_data
            
            return {"status": "completed", "output": result.stdout}
            
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "error": str(e),
                "output": e.stdout
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def sign_image(self, image_name: str, key_name: str = "default") -> Dict[str, Any]:
        """Sign Docker image with Chainguard."""
        if not self.use_chainguard:
            return self._mock_sign(image_name)
        
        try:
            # Sign image
            cmd = [
                self.chainguard_binary,
                "image", "sign",
                "--key", key_name,
                image_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return {
                "status": "completed",
                "output": result.stdout,
                "signature": f"signed-with-{key_name}"
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "error": str(e),
                "output": e.stdout
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def verify_image(self, image_name: str) -> Dict[str, Any]:
        """Verify Docker image signature."""
        if not self.use_chainguard:
            return self._mock_verify(image_name)
        
        try:
            # Verify signature
            cmd = [
                self.chainguard_binary,
                "image", "verify",
                image_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return {
                "status": "verified" if result.returncode == 0 else "failed",
                "output": result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "error": str(e),
                "output": e.stdout
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _mock_scan(self, image_name: str) -> Dict[str, Any]:
        """Mock scan when Chainguard is disabled."""
        return {
            "status": "mock",
            "image": image_name,
            "vulnerabilities": [],
            "message": "Chainguard scanning is disabled"
        }
    
    def _mock_sbom(self, image_name: str) -> Dict[str, Any]:
        """Mock SBOM generation when Chainguard is disabled."""
        return {
            "status": "mock",
            "image": image_name,
            "components": [],
            "message": "Chainguard SBOM generation is disabled"
        }
    
    def _mock_sign(self, image_name: str) -> Dict[str, Any]:
        """Mock image signing when Chainguard is disabled."""
        return {
            "status": "mock",
            "image": image_name,
            "message": "Chainguard signing is disabled"
        }
    
    def _mock_verify(self, image_name: str) -> Dict[str, Any]:
        """Mock signature verification when Chainguard is disabled."""
        return {
            "status": "mock",
            "image": image_name,
            "message": "Chainguard verification is disabled"
        }