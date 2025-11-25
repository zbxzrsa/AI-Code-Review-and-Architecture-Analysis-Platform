"""
Static Analysis Engine - Simplified version
"""
import os
import re
import json
from typing import List, Dict, Any
from pathlib import Path


class StaticAnalyzer:
    """Simplified static code analysis engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules_enabled = config.get('rules', [])
        self.severity_thresholds = config.get('severity_thresholds', {
            'critical': 0,
            'high': 5,
            'medium': 20,
            'low': 50
        })
    
    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze a single file for static issues"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic security patterns
                security_patterns = [
                    (r'password\s*=\s*["\'][^\'"]', 'hardcoded_secret'),
                    (r'api[_-]?key\s*=\s*["\'][^\'"]', 'api_key'),
                    (r'token\s*=\s*["\'][^\'"]', 'auth_token'),
                    (r'secret\s*=\s*["\'][^\'"]', 'auth_secret'),
                    (r'private[_-]?key\s*=\s*["\'][^\'"]', 'private_key'),
                    (r'aws[_-]?access[_-]?key[_-]?id\s*=\s*["\'][^\'"]', 'aws_access_key'),
                    (r'github[_-]?token\s*=\s*["\'][^\'"]', 'github_token'),
                    (r'database[_-]?url\s*=\s*["\'][^\'"]', 'database_url'),
                ]
            
            line_number = 0
            for line in content.split('\n'):
                line_number = line_number + 1
                
                # Security checks
                for pattern_name, pattern in security_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        findings.append({
                            'type': 'security',
                            'rule_id': 'hardcoded_secret',
                            'rule_name': pattern_name,
                            'severity': 'critical',
                            'confidence': 'high',
                            'title': f'Hardcoded {pattern_name} detected',
                            'description': f'Potential hardcoded {pattern_name} found',
                            'file_path': file_path,
                            'line_number': line_number,
                            'code_snippet': line.strip(),
                            'recommendation': f'Move {pattern_name} to environment variables'
                        })
                        break
                
                # Code quality checks
                if len(line.strip()) > 200:
                    findings.append({
                        'type': 'quality',
                        'rule_id': 'long_line',
                        'rule_name': 'long_line',
                        'severity': 'low',
                        'confidence': 'medium',
                        'title': 'Very long line detected',
                        'description': f'Line {line_number} is very long ({len(line.strip())} chars)',
                        'file_path': file_path,
                        'line_number': line_number,
                        'code_snippet': line.strip()[:100] + '...' if len(line.strip()) > 100 else line.strip(),
                        'recommendation': 'Consider breaking up long lines'
                    })
                
                # Complexity indicators
                if 'def ' in line and 'return ' in line:
                    findings.append({
                        'type': 'quality',
                        'rule_id': 'complex_function',
                        'rule_name': 'complex_function',
                        'severity': 'medium',
                        'confidence': 'medium',
                        'title': 'Complex function detected',
                        'description': f'Complex function found at line {line_number}',
                        'file_path': file_path,
                        'line_number': line_number,
                        'code_snippet': line.strip(),
                        'recommendation': 'Consider breaking down complex functions'
                    })
                
                # Pattern-based checks
                if 'eval(' in line.lower() or 'exec(' in line.lower():
                    findings.append({
                        'type': 'security',
                        'rule_id': 'insecure_eval',
                        'rule_name': 'insecure_eval',
                        'severity': 'critical',
                        'confidence': 'high',
                        'title': 'Insecure eval usage detected',
                        'description': f'Potentially insecure eval usage at line {line_number}',
                        'file_path': file_path,
                        'line_number': line_number,
                        'code_snippet': line.strip(),
                        'recommendation': 'Avoid using eval() with user input'
                    })
                
                line_number += 1
            
            return findings
            
        except Exception as e:
            return [{
                'type': 'error',
                'severity': 'high',
                'message': f'Analysis failed: {str(e)}',
                'file_path': file_path,
            }]
    
    def _get_code_snippet(self, file_path: str, line_number: int) -> str:
        """Extract code snippet from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if line_number <= len(lines) and line_number > 0:
                return lines[line_number - 1]
            return ''
            
            return ''
            
        except Exception:
            return ''
    
    def _map_severity(self, severity_or_code: str, default_severity: str = 'medium') -> str:
        """Map severity or code to standardized severity levels"""
        severity_mapping = {
            'critical': 'critical',
            'high': 'high',
            'medium': 'medium',
            'low': 'low',
            'info': 'info',
        }
        
        if isinstance(severity_or_code, (int, float)):
            if severity_or_code >= 20:
                return 'critical'
            elif severity_or_code >= 10:
                return 'high'
            elif severity_or_code >= 5:
                return 'medium'
            else:
                return 'low'
        
        return severity_mapping.get(severity_or_code, default_severity)
    
    def _get_security_recommendation(self, pattern_name: str) -> str:
        """Get security recommendation based on issue type"""
        recommendations = {
            'hardcoded_secret': 'Move hardcoded secrets to environment variables or secure storage',
            'insecure_eval': 'Avoid using eval() with user input',
            'insecure_exec': 'Avoid executing shell commands with user input',
            'insecure_subprocess': 'Avoid subprocess calls with user input',
            'insecure_http': 'Validate all URLs and user input',
            'private_key': 'Use proper key management',
            'aws_access_key': 'Use IAM roles instead of access keys',
            'github_token': 'Use GitHub OAuth tokens instead',
            'database_url': 'Use connection pooling and parameterized queries',
        }
        
        return recommendations.get(pattern_name, 'Follow security best practices')
    
    def _get_quality_recommendation(self, msg_type: str) -> str:
        """Get quality recommendation based on message type"""
        recommendations = {
            'long_line': 'Consider breaking up long lines',
            'duplicate_code': 'Extract common code into reusable functions',
            'missing_docstring': 'Add docstrings to document functions',
            'too_many_args': 'Reduce function arguments',
            'complex_function': 'Refactor complex functions',
        }
        
        return recommendations.get(msg_type, 'Improve code structure and readability')