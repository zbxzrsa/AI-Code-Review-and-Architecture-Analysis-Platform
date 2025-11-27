"""
Auto-fix generator for code remediation.
Automatically generates secure fixes for identified vulnerabilities and issues.
"""

import ast
import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FixType(Enum):
    """Types of code fixes."""
    SECURITY_PATCH = "security_patch"
    INPUT_VALIDATION = "input_validation"
    ERROR_HANDLING = "error_handling"
    SANITIZATION = "sanitization"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENCRYPTION = "encryption"
    PARAMETERIZATION = "parameterization"
    ESCAPING = "escaping"
    RATE_LIMITING = "rate_limiting"
    LOGGING = "logging"
    CONFIGURATION = "configuration"


class FixSeverity(Enum):
    """Severity levels for fixes."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CodeIssue:
    """Identified code issue."""
    issue_type: str
    severity: FixSeverity
    description: str
    line_number: int
    code_snippet: str
    cwe_id: Optional[str] = None
    confidence: float = 0.8


@dataclass
class CodeFix:
    """Generated code fix."""
    fix_type: FixType
    severity: FixSeverity
    original_code: str
    fixed_code: str
    description: str
    explanation: str
    line_number: int
    confidence: float
    security_impact: str
    testing_notes: List[str]


class SecurityPatternDetector:
    """Detects security vulnerabilities in code."""
    
    def __init__(self):
        self.patterns = {
            "sql_injection": [
                {
                    "pattern": r'(execute|executemany|query)\s*\(\s*["\'].*?\+.*?["\']',
                    "description": "String concatenation in SQL query",
                    "cwe": "CWE-89",
                    "severity": FixSeverity.CRITICAL
                }
            ],
            "command_injection": [
                {
                    "pattern": r'os\.system\s*\(\s*["\'].*?\+',
                    "description": "Command injection via string concatenation",
                    "cwe": "CWE-78",
                    "severity": FixSeverity.CRITICAL
                }
            ],
            "path_traversal": [
                {
                    "pattern": r'open\s*\(\s*["\'].*?\.\./',
                    "description": "Path traversal via ../",
                    "cwe": "CWE-22",
                    "severity": FixSeverity.HIGH
                }
            ],
            "hardcoded_secrets": [
                {
                    "pattern": r'(password|passwd|pwd|secret|key|token)\s*=\s*["\'][^"\']+["\']',
                    "description": "Hardcoded password or secret",
                    "cwe": "CWE-798",
                    "severity": FixSeverity.HIGH
                }
            ],
            "weak_crypto": [
                {
                    "pattern": r'md5\s*\(',
                    "description": "Weak MD5 hash algorithm",
                    "cwe": "CWE-327",
                    "severity": FixSeverity.MEDIUM
                }
            ],
            "missing_error_handling": [
                {
                    "pattern": r'(open|read|write|connect)\s*\([^)]*\)\s*$',
                    "description": "File operation without error handling",
                    "cwe": "CWE-703",
                    "severity": FixSeverity.MEDIUM
                }
            ],
            "eval_usage": [
                {
                    "pattern": r'eval\s*\(',
                    "description": "Use of eval() function",
                    "cwe": "CWE-94",
                    "severity": FixSeverity.HIGH
                }
            ]
        }
    
    def detect_issues(self, code: str) -> List[CodeIssue]:
        """Detect security issues in code."""
        issues = []
        lines = code.split('\n')
        
        for category, patterns in self.patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issue = CodeIssue(
                            issue_type=category,
                            severity=pattern_info["severity"],
                            description=pattern_info["description"],
                            line_number=line_num,
                            code_snippet=line.strip(),
                            cwe_id=pattern_info["cwe"]
                        )
                        issues.append(issue)
        
        return issues


class CodeFixGenerator:
    """Generates fixes for identified code issues."""
    
    def __init__(self):
        self.detector = SecurityPatternDetector()
    
    def generate_fix(self, issue: CodeIssue, full_code: str) -> Optional[CodeFix]:
        """Generate a fix for a specific issue."""
        
        if issue.issue_type == "sql_injection":
            return self._fix_sql_injection(issue, full_code)
        elif issue.issue_type == "command_injection":
            return self._fix_command_injection(issue, full_code)
        elif issue.issue_type == "path_traversal":
            return self._fix_path_traversal(issue, full_code)
        elif issue.issue_type == "hardcoded_secrets":
            return self._fix_hardcoded_secrets(issue, full_code)
        elif issue.issue_type == "weak_crypto":
            return self._fix_weak_crypto(issue, full_code)
        elif issue.issue_type == "missing_error_handling":
            return self._fix_missing_error_handling(issue, full_code)
        elif issue.issue_type == "eval_usage":
            return self._fix_eval_usage(issue, full_code)
        else:
            return None
    
    def _fix_sql_injection(self, issue: CodeIssue, full_code: str) -> CodeFix:
        """Generate fix for SQL injection."""
        original_line = issue.code_snippet
        
        # Generic parameterization
        fixed_line = re.sub(
            r'(["\'][^"\']*?\+[^"\']*?["\'])',
            r'%s',
            original_line
        )
        fixed_line = re.sub(
            r'(execute|executemany|query)\s*\(\s*["\'].*?["\']',
            r'\1("""REPLACE_WITH_PARAMETERIZED_QUERY""", (user_input,))',
            fixed_line
        )
        
        return CodeFix(
            fix_type=FixType.PARAMETERIZATION,
            severity=issue.severity,
            original_code=original_line,
            fixed_code=fixed_line,
            description="Parameterized SQL query to prevent injection",
            explanation="Using parameterized queries prevents SQL injection by separating SQL logic from user input",
            line_number=issue.line_number,
            confidence=0.9,
            security_impact="Prevents SQL injection attacks",
            testing_notes=[
                "Test with various SQL injection payloads",
                "Verify query still returns expected results",
                "Check error handling for invalid input"
            ]
        )
    
    def _fix_command_injection(self, issue: CodeIssue, full_code: str) -> CodeFix:
        """Generate fix for command injection."""
        original_line = issue.code_snippet
        
        if "os.system" in original_line:
            fixed_line = re.sub(
                r'os\.system\s*\(\s*["\']([^"\']*)["\']',
                r'subprocess.run([\1], shell=False, check=True)',
                original_line
            )
        else:
            fixed_line = f"# FIXED: Use subprocess.run with argument list instead of: {original_line}"
        
        return CodeFix(
            fix_type=FixType.SANITIZATION,
            severity=issue.severity,
            original_code=original_line,
            fixed_code=fixed_line,
            description="Replaced dangerous system calls with safe subprocess usage",
            explanation="Using subprocess.run with shell=False and argument lists prevents command injection",
            line_number=issue.line_number,
            confidence=0.85,
            security_impact="Prevents command injection attacks",
            testing_notes=[
                "Test with various command injection payloads",
                "Verify command executes correctly",
                "Check error handling for invalid commands"
            ]
        )
    
    def _fix_path_traversal(self, issue: CodeIssue, full_code: str) -> CodeFix:
        """Generate fix for path traversal."""
        original_line = issue.code_snippet
        
        fixed_code = f'''import os

# Original (vulnerable):
# {original_line}

# Fixed version - with path validation:
def safe_open_path(filename, base_dir="/allowed/path"):
    """Safely open file within base directory."""
    full_path = os.path.normpath(os.path.join(base_dir, filename))
    
    if not full_path.startswith(os.path.normpath(base_dir)):
        raise ValueError("Path traversal attempt detected")
    
    return open(full_path, 'r')

# Usage:
safe_open_path("user_input.txt")'''
        
        return CodeFix(
            fix_type=FixType.INPUT_VALIDATION,
            severity=issue.severity,
            original_code=original_line,
            fixed_code=fixed_code,
            description="Added path validation to prevent traversal attacks",
            explanation="Path normalization and validation prevents directory traversal attacks",
            line_number=issue.line_number,
            confidence=0.9,
            security_impact="Prevents path traversal attacks",
            testing_notes=[
                "Test with ../ sequences",
                "Test with absolute paths",
                "Verify legitimate paths still work"
            ]
        )
    
    def _fix_hardcoded_secrets(self, issue: CodeIssue, full_code: str) -> CodeFix:
        """Generate fix for hardcoded secrets."""
        original_line = issue.code_snippet
        
        var_match = re.search(r'(\w+)\s*=\s*["\'][^"\']+["\']', original_line)
        var_name = var_match.group(1) if var_match else "secret"
        
        fixed_code = f'''import os

# Original (vulnerable):
# {original_line}

# Fixed version - use environment variables:
def get_secret(secret_name):
    """Retrieve secret from environment."""
    secret_value = os.getenv(secret_name)
    if not secret_value:
        raise ValueError(f"Secret {{secret_name}} not found in environment")
    return secret_value

# Usage:
{var_name} = get_secret("{var_name.upper()}_SECRET")'''
        
        return CodeFix(
            fix_type=FixType.ENCRYPTION,
            severity=issue.severity,
            original_code=original_line,
            fixed_code=fixed_code,
            description="Replaced hardcoded secret with environment variable retrieval",
            explanation="Using environment variables prevents secret exposure in source code",
            line_number=issue.line_number,
            confidence=0.95,
            security_impact="Prevents secret exposure in version control",
            testing_notes=[
                "Set environment variables before testing",
                "Test with missing environment variable",
                "Verify secret retrieval works"
            ]
        )
    
    def _fix_weak_crypto(self, issue: CodeIssue, full_code: str) -> CodeFix:
        """Generate fix for weak cryptography."""
        original_line = issue.code_snippet
        
        fixed_code = f'''import hashlib

# Original (weak):
# {original_line}

# Fixed version - use SHA-256:
def secure_hash(data):
    """Generate secure hash using SHA-256."""
    return hashlib.sha256(data.encode()).hexdigest()

# Usage:
hash_result = secure_hash("input_data")'''
        
        return CodeFix(
            fix_type=FixType.ENCRYPTION,
            severity=issue.severity,
            original_code=original_line,
            fixed_code=fixed_code,
            description="Replaced weak cryptographic algorithm with strong alternative",
            explanation="Using strong cryptographic algorithms prevents collision and brute force attacks",
            line_number=issue.line_number,
            confidence=0.9,
            security_impact="Prevents cryptographic attacks",
            testing_notes=[
                "Verify hash output is different but consistent",
                "Test with various input sizes",
                "Performance test the new algorithm"
            ]
        )
    
    def _fix_missing_error_handling(self, issue: CodeIssue, full_code: str) -> CodeFix:
        """Generate fix for missing error handling."""
        original_line = issue.code_snippet
        
        fixed_code = f'''# Original (no error handling):
# {original_line}

# Fixed version - with proper error handling:
try:
    {original_line}
except FileNotFoundError as e:
    print(f"File not found: {{e}}")
except PermissionError as e:
    print(f"Permission denied: {{e}}")
except Exception as e:
    print(f"Unexpected error: {{e}}")
    import logging
    logging.error(f"Operation failed: {{e}}", exc_info=True)'''
        
        return CodeFix(
            fix_type=FixType.ERROR_HANDLING,
            severity=issue.severity,
            original_code=original_line,
            fixed_code=fixed_code,
            description="Added comprehensive error handling",
            explanation="Proper error handling prevents crashes and improves debugging",
            line_number=issue.line_number,
            confidence=0.8,
            security_impact="Prevents information disclosure and improves reliability",
            testing_notes=[
                "Test with missing files",
                "Test with permission errors",
                "Test with various error conditions"
            ]
        )
    
    def _fix_eval_usage(self, issue: CodeIssue, full_code: str) -> CodeFix:
        """Generate fix for eval/exec usage."""
        original_line = issue.code_snippet
        
        fixed_code = f'''# Original (dangerous):
# {original_line}

# Fixed version - avoid eval, use safer alternatives:
# NEVER use eval() with user input
# Use specific, validated operations instead

# Example: For mathematical evaluation
import ast
import operator

def safe_eval(expr):
    """Safely evaluate mathematical expressions."""
    allowed_operators = {{
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }}
    
    try:
        tree = ast.parse(expr, mode='eval')
        return safe_eval_node(tree.body, allowed_operators)
    except (SyntaxError, TypeError, ValueError):
        raise ValueError("Invalid expression")

def safe_eval_node(node, allowed_ops):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        left = safe_eval_node(node.left, allowed_ops)
        right = safe_eval_node(node.right, allowed_ops)
        op_type = type(node.op)
        if op_type in allowed_ops:
            return allowed_ops[op_type](left, right)
        else:
            raise ValueError("Operation not allowed")
    else:
        raise ValueError("Expression type not allowed")

# Usage:
result = safe_eval("2 + 3")  # Safe evaluation'''
        
        return CodeFix(
            fix_type=FixType.SANITIZATION,
            severity=issue.severity,
            original_code=original_line,
            fixed_code=fixed_code,
            description="Replaced dangerous eval() with safe alternative",
            explanation="Avoiding eval() prevents code injection attacks",
            line_number=issue.line_number,
            confidence=0.95,
            security_impact="Prevents code injection attacks",
            testing_notes=[
                "Test with various mathematical expressions",
                "Test with malicious input attempts",
                "Verify only allowed operations work"
            ]
        )


class AutoFixEngine:
    """Main engine for automatic code fix generation."""
    
    def __init__(self):
        self.detector = SecurityPatternDetector()
        self.fix_generator = CodeFixGenerator()
        self.fix_history = []
    
    def analyze_and_fix(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze code and generate fixes for identified issues.
        
        Args:
            code: Source code to analyze
            context: Additional context (language, framework, etc.)
            
        Returns:
            Dictionary with issues and fixes
        """
        # Detect issues
        issues = self.detector.detect_issues(code)
        
        # Generate fixes
        fixes = []
        for issue in issues:
            fix = self.fix_generator.generate_fix(issue, code)
            if fix:
                fixes.append(fix)
        
        # Store in history
        analysis_result = {
            "timestamp": self._get_timestamp(),
            "code": code,
            "context": context or {},
            "issues": [self._issue_to_dict(issue) for issue in issues],
            "fixes": [self._fix_to_dict(fix) for fix in fixes],
            "summary": {
                "total_issues": len(issues),
                "critical_issues": len([i for i in issues if i.severity == FixSeverity.CRITICAL]),
                "high_issues": len([i for i in issues if i.severity == FixSeverity.HIGH]),
                "medium_issues": len([i for i in issues if i.severity == FixSeverity.MEDIUM]),
                "low_issues": len([i for i in issues if i.severity == FixSeverity.LOW]),
                "fixes_generated": len(fixes),
                "fix_success_rate": len(fixes) / len(issues) if issues else 1.0
            }
        }
        
        self.fix_history.append(analysis_result)
        
        return analysis_result
    
    def _issue_to_dict(self, issue: CodeIssue) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return {
            "issue_type": issue.issue_type,
            "severity": issue.severity.value,
            "description": issue.description,
            "line_number": issue.line_number,
            "code_snippet": issue.code_snippet,
            "cwe_id": issue.cwe_id,
            "confidence": issue.confidence
        }
    
    def _fix_to_dict(self, fix: CodeFix) -> Dict[str, Any]:
        """Convert fix to dictionary."""
        return {
            "fix_type": fix.fix_type.value,
            "severity": fix.severity.value,
            "original_code": fix.original_code,
            "fixed_code": fix.fixed_code,
            "description": fix.description,
            "explanation": fix.explanation,
            "line_number": fix.line_number,
            "confidence": fix.confidence,
            "security_impact": fix.security_impact,
            "testing_notes": fix.testing_notes
        }
    
    def apply_fixes(self, code: str, fixes: List[CodeFix]) -> str:
        """Apply fixes to code (simple line replacement)."""
        lines = code.split('\n')
        
        # Sort fixes by line number in reverse order to maintain line numbers
        sorted_fixes = sorted(fixes, key=lambda f: f.line_number, reverse=True)
        
        for fix in sorted_fixes:
            if 1 <= fix.line_number <= len(lines):
                # Replace the problematic line with fixed code
                lines[fix.line_number - 1] = f"# AUTO-FIXED: {fix.description}\n{fix.fixed_code}"
        
        return '\n'.join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis and fix statistics."""
        if not self.fix_history:
            return {"total_analyses": 0}
        
        total = len(self.fix_history)
        total_issues = sum(len(analysis["issues"]) for analysis in self.fix_history)
        total_fixes = sum(len(analysis["fixes"]) for analysis in self.fix_history)
        
        # Issue type distribution
        issue_types = {}
        for analysis in self.fix_history:
            for issue in analysis["issues"]:
                issue_type = issue["issue_type"]
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        # Severity distribution
        severity_dist = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for analysis in self.fix_history:
            for issue in analysis["issues"]:
                severity = issue["severity"]
                severity_dist[severity] = severity_dist.get(severity, 0) + 1
        
        return {
            "total_analyses": total,
            "total_issues_detected": total_issues,
            "total_fixes_generated": total_fixes,
            "average_issues_per_analysis": total_issues / total if total > 0 else 0,
            "average_fixes_per_analysis": total_fixes / total if total > 0 else 0,
            "fix_success_rate": total_fixes / total_issues if total_issues > 0 else 1.0,
            "issue_type_distribution": issue_types,
            "severity_distribution": severity_dist
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()


# Global auto-fix engine instance
_auto_fix_engine = None


def get_auto_fix_engine() -> AutoFixEngine:
    """Get global auto-fix engine instance."""
    global _auto_fix_engine
    if _auto_fix_engine is None:
        _auto_fix_engine = AutoFixEngine()
    return _auto_fix_engine


def analyze_and_fix_code(
    code: str, 
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function to analyze code and generate fixes."""
    engine = get_auto_fix_engine()
    return engine.analyze_and_fix(code, context)