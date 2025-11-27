"""
Pydantic schemas for structured AI review outputs.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator


class KeyFinding(BaseModel):
    area: Literal["security", "performance", "reliability", "architecture", "correctness", "perf"]
    title: str
    severity: Literal["info", "minor", "major", "critical"]
    evidence: str
    source: Optional[str] = None


class Recommendation(BaseModel):
    title: str
    rationale: str
    est_impact: Optional[Literal["latency", "reliability", "safety", "maintainability"]] = None
    effort: Literal["S", "M", "L"]
    expected_gain: Optional[str] = None
    validate: Optional[str] = None
    hint: Optional[str] = None


class SafePatchSketch(BaseModel):
    file: str
    change_type: Literal["add", "edit", "delete"]
    diff_hint: str


class Test(BaseModel):
    name: str
    type: Literal["unit", "integration", "e2e", "bench"]
    focus: Optional[str] = None
    sketch: str
    success_criteria: Optional[str] = None


class Observability(BaseModel):
    metric_or_log: str
    why: str
    placement: str


class Citation(BaseModel):
    source: str
    reason: str


class Experiment(BaseModel):
    name: str
    toggle: Literal["on", "off"]
    reason: str


class Guardrail(BaseModel):
    risk: str
    mitigation: str


class Meta(BaseModel):
    channel: Literal["stable", "next", "legacy"]
    style: Literal["conservative", "experimental", "baseline-fast"]
    constraints: List[str]
    params_hint: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = None


class StableOutput(BaseModel):
    summary: str
    risk_level: Literal["low", "medium", "high"]
    key_findings: List[KeyFinding]
    recommendations: List[Recommendation]
    safe_patch_sketch: List[SafePatchSketch] = []
    tests: List[Test] = []
    observability: List[Observability] = []
    citations: List[Citation] = []
    meta: Meta


class NextOutput(BaseModel):
    summary: str
    risk_level: Literal["low", "medium", "high"]
    experiments_considered: List[Experiment] = []
    key_findings: List[KeyFinding]
    recommendations: List[Recommendation]
    safe_patch_sketch: List[SafePatchSketch] = []
    tests: List[Test] = []
    guardrails: List[Guardrail] = []
    citations: List[Citation] = []
    meta: Meta


class LegacyOutput(BaseModel):
    summary: str
    risk_level: Literal["low", "medium", "high"]
    key_findings: List[KeyFinding]
    recommendations: List[Recommendation]
    meta: Meta


# Channel-specific rule mappings for legacy
class LegacyKeyFinding(BaseModel):
    rule: str
    evidence: str
    severity: Literal["minor", "major", "critical"]
    source: Optional[str] = None


class LegacyRecommendation(BaseModel):
    title: str
    effort: Literal["S"]
    hint: str


class LegacyOutputStrict(BaseModel):
    summary: str
    risk_level: Literal["low", "medium", "high"]
    key_findings: List[LegacyKeyFinding]
    recommendations: List[LegacyRecommendation]
    meta: Meta


def parse_and_validate(channel: str, raw_text: str) -> Dict[str, Any]:
    """
    Parse and validate JSON output against channel-specific schema.
    
    Args:
        channel: The AI channel (stable, next, legacy)
        raw_text: Raw JSON text from the model
        
    Returns:
        Validated dictionary
        
    Raises:
        ValueError: If JSON is invalid or doesn't match schema
    """
    from .json_utils import safe_json_loads, ensure_json_only
    
    # Clean and extract JSON
    clean_json = ensure_json_only(raw_text)
    
    try:
        data = safe_json_loads(clean_json)
    except Exception as e:
        raise ValueError(f"Invalid JSON: {e}")
    
    # Validate against channel-specific schema
    try:
        if channel == "stable":
            output = StableOutput(**data)
        elif channel == "next":
            output = NextOutput(**data)
        elif channel == "legacy":
            # Try strict legacy schema first, fall back to flexible
            try:
                output = LegacyOutputStrict(**data)
            except Exception:
                output = LegacyOutput(**data)
        else:
            raise ValueError(f"Unknown channel: {channel}")
            
        return output.dict()
    except Exception as e:
        raise ValueError(f"Schema validation failed for {channel}: {e}")