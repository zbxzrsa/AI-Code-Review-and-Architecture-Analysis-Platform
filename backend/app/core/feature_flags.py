# Feature Flags Configuration
import os
from typing import Dict, Any

class FeatureFlags:
    """Centralized feature flag management."""
    
    # Batch 1 replacements
    USE_UV = os.getenv("USE_UV", "false").lower() == "true"
    USE_SEMGREP = os.getenv("USE_SEMGREP", "false").lower() == "true"
    USE_PYG = os.getenv("USE_PYG", "false").lower() == "true"
    USE_PGVECTOR = os.getenv("USE_PGVECTOR", "false").lower() == "true"
    USE_CHAINGUARD = os.getenv("USE_CHAINGUARD", "false").lower() == "true"
    
    # Batch 2 replacements
    USE_DRAGONFLY = os.getenv("USE_DRAGONFLY", "false").lower() == "true"
    USE_OPENTELEMETRY = os.getenv("USE_OPENTELEMETRY", "false").lower() == "true"
    USE_REDPANDA = os.getenv("USE_REDPANDA", "false").lower() == "true"
    USE_NEXTJS = os.getenv("USE_NEXTJS", "false").lower() == "true"
    
    # Batch 3 replacements
    USE_TEMPORAL = os.getenv("USE_TEMPORAL", "false").lower() == "true"
    USE_VLLM = os.getenv("USE_VLLM", "false").lower() == "true"
    USE_GRPC = os.getenv("USE_GRPC", "false").lower() == "true"
    USE_KONG = os.getenv("USE_KONG", "false").lower() == "true"
    USE_SOPS = os.getenv("USE_SOPS", "false").lower() == "true"
    
    @classmethod
    def all_flags(cls) -> Dict[str, bool]:
        """Get all feature flags as dictionary."""
        return {
            name: getattr(cls, name) for name in dir(cls) 
            if not name.startswith('_') and isinstance(getattr(cls, name), bool)
        }
    
    @classmethod
    def is_enabled(cls, flag_name: str) -> bool:
        """Check if a feature flag is enabled."""
        return getattr(cls, flag_name.upper(), False)