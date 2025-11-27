"""
Policy engine for safety thresholds and automated decision making.
Configurable rules for content moderation and safety enforcement.
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

from .judge import SafetyJudgment, SafetyLevel, RiskCategory

logger = logging.getLogger(__name__)


class PolicyAction(Enum):
    """Actions that can be taken based on policy evaluation."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"
    LOG_ONLY = "log_only"


class PolicyScope(Enum):
    """Scope of policy application."""
    GLOBAL = "global"
    CHANNEL = "channel"
    USER = "user"
    ORGANIZATION = "organization"
    SESSION = "session"


@dataclass
class PolicyRule:
    """Individual policy rule."""
    name: str
    description: str
    scope: PolicyScope
    conditions: Dict[str, Any]
    action: PolicyAction
    threshold: float
    enabled: bool = True
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if rule matches given context."""
        for key, expected in self.conditions.items():
            if key not in context:
                return False
            
            actual = context[key]
            
            # Handle different comparison types
            if isinstance(expected, dict):
                # Range or complex condition
                if "min" in expected and actual < expected["min"]:
                    return False
                if "max" in expected and actual > expected["max"]:
                    return False
                if "in" in expected and actual not in expected["in"]:
                    return False
                if "not_in" in expected and actual in expected["not_in"]:
                    return False
            elif isinstance(expected, list):
                # Must be one of the values
                if actual not in expected:
                    return False
            else:
                # Direct equality
                if actual != expected:
                    return False
        
        return True


@dataclass
class PolicyDecision:
    """Result of policy evaluation."""
    action: PolicyAction
    confidence: float
    matched_rules: List[PolicyRule]
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PolicyEngine:
    """Main policy engine for safety enforcement."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.rules: List[PolicyRule] = []
        self.action_handlers: Dict[PolicyAction, Callable] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        else:
            self._load_default_policies()
        
        # Initialize default action handlers
        self._init_default_handlers()
    
    def _load_default_policies(self):
        """Load default safety policies."""
        
        # Critical safety policies
        self.rules.extend([
            PolicyRule(
                name="critical_content_block",
                description="Block content with critical safety level",
                scope=PolicyScope.GLOBAL,
                conditions={"safety_level": SafetyLevel.CRITICAL.value},
                action=PolicyAction.BLOCK,
                threshold=0.8,
                priority=100
            ),
            PolicyRule(
                name="unsafe_content_block",
                description="Block content with unsafe safety level and high confidence",
                scope=PolicyScope.GLOBAL,
                conditions={
                    "safety_level": SafetyLevel.UNSAFE.value,
                    "confidence": {"min": 0.7}
                },
                action=PolicyAction.BLOCK,
                threshold=0.7,
                priority=90
            ),
            PolicyRule(
                name="malicious_intent_block",
                description="Block content with malicious intent",
                scope=PolicyScope.GLOBAL,
                conditions={"risk_categories": ["malicious_intent"]},
                action=PolicyAction.BLOCK,
                threshold=0.6,
                priority=95
            ),
            PolicyRule(
                name="system_compromise_block",
                description="Block system compromise attempts",
                scope=PolicyScope.GLOBAL,
                conditions={"risk_categories": ["system_compromise"]},
                action=PolicyAction.BLOCK,
                threshold=0.7,
                priority=95
            ),
            PolicyRule(
                name="data_exfiltration_block",
                description="Block data exfiltration attempts",
                scope=PolicyScope.GLOBAL,
                conditions={"risk_categories": ["data_exfiltration"]},
                action=PolicyAction.BLOCK,
                threshold=0.8,
                priority=90
            )
        ])
        
        # Warning policies
        self.rules.extend([
            PolicyRule(
                name="unsafe_content_warn",
                description="Warn for unsafe content with low confidence",
                scope=PolicyScope.GLOBAL,
                conditions={
                    "safety_level": SafetyLevel.UNSAFE.value,
                    "confidence": {"max": 0.7}
                },
                action=PolicyAction.WARN,
                threshold=0.5,
                priority=70
            ),
            PolicyRule(
                name="warning_level_warn",
                description="Warn for warning level content",
                scope=PolicyScope.GLOBAL,
                conditions={"safety_level": SafetyLevel.WARNING.value},
                action=PolicyAction.WARN,
                threshold=0.4,
                priority=60
            ),
            PolicyRule(
                name="security_risk_warn",
                description="Warn for security risks",
                scope=PolicyScope.GLOBAL,
                conditions={"risk_categories": ["security_risk"]},
                action=PolicyAction.WARN,
                threshold=0.5,
                priority=65
            )
        ])
        
        # Channel-specific policies
        self.rules.extend([
            PolicyRule(
                name="stable_channel_strict",
                description="Stricter policies for stable channel",
                scope=PolicyScope.CHANNEL,
                conditions={
                    "channel": "stable",
                    "safety_level": {"in": [SafetyLevel.WARNING.value, SafetyLevel.UNSAFE.value]}
                },
                action=PolicyAction.QUARANTINE,
                threshold=0.6,
                priority=80
            ),
            PolicyRule(
                name="legacy_channel_permissive",
                description="More permissive for legacy channel",
                scope=PolicyScope.CHANNEL,
                conditions={
                    "channel": "legacy",
                    "safety_level": SafetyLevel.WARNING.value,
                    "confidence": {"min": 0.8}
                },
                action=PolicyAction.ALLOW,
                threshold=0.3,
                priority=40
            )
        ])
        
        # Rate limiting policies
        self.rules.extend([
            PolicyRule(
                name="high_frequency_warnings",
                description="Escalate if too many warnings in short time",
                scope=PolicyScope.USER,
                conditions={
                    "recent_warnings": {"min": 5},
                    "time_window_minutes": {"max": 10}
                },
                action=PolicyAction.ESCALATE,
                threshold=0.8,
                priority=85
            ),
            PolicyRule(
                name="repeated_blocks",
                description="Escalate repeated blocks",
                scope=PolicyScope.USER,
                conditions={
                    "recent_blocks": {"min": 3},
                    "time_window_minutes": {"max": 30}
                },
                action=PolicyAction.ESCALATE,
                threshold=0.9,
                priority=88
            )
        ])
    
    def _init_default_handlers(self):
        """Initialize default action handlers."""
        
        def handle_allow(decision: PolicyDecision, context: Dict[str, Any]):
            logger.info(f"Policy ALLOW: {decision.reasoning}")
            return {"allowed": True, "action": "allow"}
        
        def handle_warn(decision: PolicyDecision, context: Dict[str, Any]):
            logger.warning(f"Policy WARN: {decision.reasoning}")
            return {
                "allowed": True, 
                "action": "warn",
                "warning": decision.reasoning,
                "matched_rules": [rule.name for rule in decision.matched_rules]
            }
        
        def handle_block(decision: PolicyDecision, context: Dict[str, Any]):
            logger.error(f"Policy BLOCK: {decision.reasoning}")
            return {
                "allowed": False,
                "action": "block",
                "reason": decision.reasoning,
                "matched_rules": [rule.name for rule in decision.matched_rules]
            }
        
        def handle_quarantine(decision: PolicyDecision, context: Dict[str, Any]):
            logger.warning(f"Policy QUARANTINE: {decision.reasoning}")
            return {
                "allowed": False,
                "action": "quarantine",
                "reason": decision.reasoning,
                "requires_review": True
            }
        
        def handle_escalate(decision: PolicyDecision, context: Dict[str, Any]):
            logger.critical(f"Policy ESCALATE: {decision.reasoning}")
            return {
                "allowed": False,
                "action": "escalate",
                "reason": decision.reasoning,
                "requires_human_review": True,
                "urgency": "high"
            }
        
        def handle_log_only(decision: PolicyDecision, context: Dict[str, Any]):
            logger.info(f"Policy LOG_ONLY: {decision.reasoning}")
            return {"allowed": True, "action": "logged"}
        
        # Register handlers
        self.action_handlers = {
            PolicyAction.ALLOW: handle_allow,
            PolicyAction.WARN: handle_warn,
            PolicyAction.BLOCK: handle_block,
            PolicyAction.QUARANTINE: handle_quarantine,
            PolicyAction.ESCALATE: handle_escalate,
            PolicyAction.LOG_ONLY: handle_log_only
        }
    
    def evaluate(
        self, 
        judgment: SafetyJudgment, 
        context: Dict[str, Any]
    ) -> PolicyDecision:
        """
        Evaluate safety judgment against policies.
        
        Args:
            judgment: Safety judgment from judge model
            context: Additional context (user, channel, etc.)
            
        Returns:
            PolicyDecision with action and reasoning
        """
        # Build evaluation context
        eval_context = {
            "safety_level": judgment.safety_level.value,
            "confidence": judgment.confidence,
            "risk_categories": [cat.value for cat in judgment.risk_categories],
            "flagged_content_count": len(judgment.flagged_content),
            **context
        }
        
        # Add user statistics if available
        if "user_id" in context:
            user_stats = self._get_user_statistics(context["user_id"])
            eval_context.update(user_stats)
        
        # Find matching rules
        matching_rules = []
        for rule in self.rules:
            if rule.enabled and rule.matches(eval_context):
                matching_rules.append(rule)
        
        # Sort by priority (higher first)
        matching_rules.sort(key=lambda r: r.priority, reverse=True)
        
        if not matching_rules:
            # No rules match, allow by default
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                confidence=1.0,
                matched_rules=[],
                reasoning="No policies matched, allowing by default"
            )
        
        # Select highest priority rule
        selected_rule = matching_rules[0]
        
        # Calculate confidence based on rule threshold and judgment confidence
        rule_confidence = min(judgment.confidence / selected_rule.threshold, 1.0)
        
        # Build reasoning
        reasoning_parts = [
            f"Rule '{selected_rule.name}' matched",
            f"Safety level: {judgment.safety_level.value}",
            f"Confidence: {judgment.confidence:.2f}",
            f"Rule threshold: {selected_rule.threshold}"
        ]
        
        if judgment.risk_categories:
            reasoning_parts.append(f"Risk categories: {[cat.value for cat in judgment.risk_categories]}")
        
        reasoning = "; ".join(reasoning_parts)
        
        decision = PolicyDecision(
            action=selected_rule.action,
            confidence=rule_confidence,
            matched_rules=[selected_rule],
            reasoning=reasoning,
            metadata={
                "eval_context": eval_context,
                "all_matching_rules": [rule.name for rule in matching_rules]
            }
        )
        
        # Store in history
        self._store_evaluation(judgment, context, decision)
        
        return decision
    
    def execute_action(
        self, 
        decision: PolicyDecision, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the action determined by policy evaluation."""
        handler = self.action_handlers.get(decision.action)
        
        if handler:
            return handler(decision, context)
        else:
            logger.error(f"No handler for action: {decision.action}")
            return {"allowed": False, "error": f"No handler for {decision.action}"}
    
    def _get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics for policy evaluation."""
        # Filter evaluations for this user
        user_evals = [
            eval_ for eval_ in self.evaluation_history
            if eval_.get("context", {}).get("user_id") == user_id
        ]
        
        # Calculate recent statistics
        now = datetime.utcnow()
        recent_warnings = 0
        recent_blocks = 0
        
        for eval_ in user_evals:
            eval_time = eval_["timestamp"]
            time_diff = (now - eval_time).total_seconds() / 60  # minutes
            
            if time_diff <= 10:  # Last 10 minutes
                if eval_["decision"].action in [PolicyAction.WARN, PolicyAction.BLOCK]:
                    recent_warnings += 1
                if eval_["decision"].action == PolicyAction.BLOCK:
                    recent_blocks += 1
        
        return {
            "recent_warnings": recent_warnings,
            "recent_blocks": recent_blocks,
            "time_window_minutes": 10,
            "total_evaluations": len(user_evals)
        }
    
    def _store_evaluation(
        self, 
        judgment: SafetyJudgment, 
        context: Dict[str, Any], 
        decision: PolicyDecision
    ):
        """Store evaluation in history."""
        self.evaluation_history.append({
            "timestamp": datetime.utcnow(),
            "judgment": judgment,
            "context": context,
            "decision": decision
        })
        
        # Limit history size
        if len(self.evaluation_history) > 10000:
            self.evaluation_history = self.evaluation_history[-5000:]
    
    def add_rule(self, rule: PolicyRule):
        """Add a new policy rule."""
        self.rules.append(rule)
        logger.info(f"Added policy rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a policy rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                logger.info(f"Removed policy rule: {rule_name}")
                return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a policy rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"Enabled policy rule: {rule_name}")
                return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a policy rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"Disabled policy rule: {rule_name}")
                return True
        return False
    
    def load_config(self, config_path: str):
        """Load policies from configuration file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            # Clear existing rules
            self.rules.clear()
            
            # Load rules from config
            for rule_data in config.get("rules", []):
                rule = PolicyRule(
                    name=rule_data["name"],
                    description=rule_data["description"],
                    scope=PolicyScope(rule_data["scope"]),
                    conditions=rule_data["conditions"],
                    action=PolicyAction(rule_data["action"]),
                    threshold=rule_data["threshold"],
                    enabled=rule_data.get("enabled", True),
                    priority=rule_data.get("priority", 0),
                    metadata=rule_data.get("metadata", {})
                )
                self.rules.append(rule)
            
            logger.info(f"Loaded {len(self.rules)} policy rules from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load policy config: {e}")
            self._load_default_policies()
    
    def save_config(self, config_path: str):
        """Save current policies to configuration file."""
        config = {
            "version": "1.0",
            "description": "AI Safety Policy Configuration",
            "rules": []
        }
        
        for rule in self.rules:
            rule_data = {
                "name": rule.name,
                "description": rule.description,
                "scope": rule.scope.value,
                "conditions": rule.conditions,
                "action": rule.action.value,
                "threshold": rule.threshold,
                "enabled": rule.enabled,
                "priority": rule.priority,
                "metadata": rule.metadata
            }
            config["rules"].append(rule_data)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(config, f, default_flow_style=False)
            else:
                json.dump(config, f, indent=2)
        
        logger.info(f"Saved {len(self.rules)} policy rules to {config_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get policy engine statistics."""
        if not self.evaluation_history:
            return {"total_evaluations": 0}
        
        # Count actions
        action_counts = {}
        for eval_ in self.evaluation_history:
            action = eval_["decision"].action.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Count rule matches
        rule_matches = {}
        for eval_ in self.evaluation_history:
            for rule in eval_["decision"].matched_rules:
                rule_matches[rule.name] = rule_matches.get(rule.name, 0) + 1
        
        total = len(self.evaluation_history)
        
        return {
            "total_evaluations": total,
            "action_distribution": action_counts,
            "most_triggered_rules": sorted(rule_matches.items(), key=lambda x: x[1], reverse=True)[:10],
            "enabled_rules": len([r for r in self.rules if r.enabled]),
            "total_rules": len(self.rules),
            "average_confidence": sum(eval_["decision"].confidence for eval_ in self.evaluation_history) / total
        }


# Global policy engine instance
_policy_engine = None


def get_policy_engine() -> PolicyEngine:
    """Get global policy engine instance."""
    global _policy_engine
    if _policy_engine is None:
        config_path = os.getenv("SAFETY_POLICY_CONFIG")
        _policy_engine = PolicyEngine(config_path)
    return _policy_engine


def evaluate_safety_policy(
    judgment: SafetyJudgment, 
    context: Dict[str, Any]
) -> PolicyDecision:
    """Evaluate safety policy for a judgment."""
    engine = get_policy_engine()
    return engine.evaluate(judgment, context)


def execute_safety_action(
    decision: PolicyDecision, 
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute safety action based on policy decision."""
    engine = get_policy_engine()
    return engine.execute_action(decision, context)