"""
Static Analysis Engine
Handles various static analysis tools and rule engines
"""
import ast
import os
import re
import subprocess
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

try:
    from radon.complexity import cc_visit
    from bandit.core import manager as bandit_manager
    from pylint.lint import Run as PylintRun
except ImportError:
    # Fallback for environments without these tools
    cc_visit = None
    bandit_manager = None
    PylintRun = None


class StaticAnalyzer:
    """Static code analysis engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules_enabled = config.get('rules', ['security', 'quality', 'complexity'])
        self.severity_thresholds = config.get('severity_thresholds', {
            'critical': 0,
            'high': 5,
            'medium': 20,
            'low': 50
        })
        self.language = config.get('language', 'python')
        self.tools = self._get_available_tools()
    
    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze a single file for static issues"""
        findings = []
        
        # Detect language if not set
        if not self.language or self.language == 'auto':
            self.language = self._detect_language_from_path(file_path)
            self.tools = self._get_available_tools()
        
        try:
            # Language-specific analysis
            if self.language == 'python':
                # Security analysis with Bandit
                if 'security' in self.rules_enabled and bandit_manager:
                    security_findings = self._run_bandit_analysis(file_path)
                    findings.extend(security_findings)
                
                # Code quality analysis with Pylint
                if 'quality' in self.rules_enabled and PylintRun:
                    quality_findings = self._run_pylint_analysis(file_path)
                    findings.extend(quality_findings)
                
                # Complexity analysis with Radon
                if 'complexity' in self.rules_enabled and cc_visit:
                    complexity_findings = self._run_complexity_analysis(file_path)
                    findings.extend(complexity_findings)
            
            elif self.language in ['javascript', 'typescript']:
                # ESLint analysis for JS/TS
                if 'quality' in self.rules_enabled and 'eslint' in self.tools:
                    eslint_findings = self._run_eslint_analysis(file_path)
                    findings.extend(eslint_findings)
            
            # Pattern-based analysis (always available)
            pattern_findings = self._run_pattern_analysis(file_path)
            findings.extend(pattern_findings)
            
        except Exception as e:
            findings.append({
                'type': 'error',
                'severity': 'high',
                'message': f'Analysis failed: {str(e)}',
                'file_path': file_path,
                'line_number': 1
            })
        
        return findings
    
    def _run_bandit_analysis(self, file_path: str) -> List[Dict[str, Any]]:
        """Run Bandit security analysis"""
        if not bandit_manager:
            return []
        
        try:
            # Use bandit manager for more control
            bandit_mgr = bandit_manager.BanditManager()
            
            # Configure bandit
            bandit_config = {
                'exclude_dirs': ['tests', 'test', '__pycache__', '.git', '.venv'],
                'severity_levels': ['medium', 'high', 'critical'],
                'output_format': 'json',
                'output_file': None,  # We'll capture output
            }
            
            # Run bandit
            result = bandit_mgr.run([file_path], bandit_config)
            
            findings = []
            if result and result.get('results'):
                for issue in result['results'].get(file_path, []):
                    findings.append({
                        'type': 'security',
                        'rule_id': issue.get('test_id', ''),
                        'rule_name': issue.get('test_name', ''),
                        'severity': self._map_severity(issue.get('issue_severity', 'medium')),
                        'confidence': issue.get('issue_cwe', {}).get('confidence', 'medium'),
                        'title': issue.get('issue_text', ''),
                        'description': issue.get('issue_text', ''),
                        'cwe_id': issue.get('issue_cwe', {}).get('id', ''),
                        'file_path': file_path,
                        'line_number': issue.get('line_number', 0),
                        'end_line_number': issue.get('end_line_number', 0),
                        'code_snippet': self._get_code_snippet(file_path, issue.get('line_number', 0), issue.get('end_line_number', 0)),
                        'recommendation': self._get_security_recommendation(issue)
                    })
            
            return findings
            
        except Exception as e:
            return [{
                'type': 'error',
                'severity': 'high',
                'message': f'Bandit analysis failed: {str(e)}',
                'file_path': file_path
            }]
    
    def _run_pylint_analysis(self, file_path: str) -> List[Dict[str, Any]]:
        """Run Pylint code quality analysis"""
        if not PylintRun:
            return []
        
        try:
            # Configure Pylint
            pylint_opts = [
                '--disable=C0111',  # Missing docstring
                '--disable=R0901',  # Too many return statements
                '--disable=R0913',  # Too many arguments
                '--disable=R0914',  # Too many arguments
                '--disable=R0915',  # Too many arguments
                '--output-format=json',
                '--reports=json',
                '--score=no',
            ]
            
            # Run Pylint
            run = PylintRun([file_path] + pylint_opts)
            run.prepare()
            
            findings = []
            for msg in run.linter.reporter.messages:
                if msg.msg_id == 'R0801':  # Similar lines in file
                    findings.append({
                        'type': 'quality',
                        'rule_id': 'duplicate_code',
                        'rule_name': 'duplicate-code',
                        'severity': 'low',
                        'confidence': 'high',
                        'title': 'Duplicate code detected',
                        'description': f'Duplicate code found: {msg.msg}',
                        'file_path': file_path,
                        'line_number': msg.line,
                        'end_line_number': msg.line,
                        'code_snippet': self._get_code_snippet(file_path, msg.line, msg.line),
                        'recommendation': 'Refactor to eliminate code duplication'
                    })
                elif msg.msg_id == 'C0111':  # Missing docstring
                    findings.append({
                        'type': 'quality',
                        'rule_id': 'missing_docstring',
                        'rule_name': 'missing-docstring',
                        'severity': 'low',
                        'confidence': 'medium',
                        'title': 'Missing docstring',
                        'description': f'Missing docstring for {msg.symbol}',
                        'file_path': file_path,
                        'line_number': msg.line,
                        'end_line_number': msg.line,
                        'code_snippet': self._get_code_snippet(file_path, msg.line, msg.line),
                        'recommendation': 'Add docstring to document the function/class'
                    })
                elif msg.msg_id.startswith('R'):  # Other Pylint issues
                    findings.append({
                        'type': 'quality',
                        'rule_id': msg.msg_id,
                        'rule_name': msg.msg_id,
                        'severity': self._map_severity(msg.msg_id, 'medium'),
                        'confidence': 'medium',
                        'title': f'Code quality issue: {msg.msg}',
                        'description': msg.msg,
                        'file_path': file_path,
                        'line_number': msg.line,
                        'end_line_number': msg.line,
                        'code_snippet': self._get_code_snippet(file_path, msg.line, msg.line),
                        'recommendation': self._get_quality_recommendation(msg.msg_id)
                    })
            
            return findings
            
        except Exception as e:
            return [{
                'type': 'error',
                'severity': 'high',
                'message': f'Pylint analysis failed: {str(e)}',
                'file_path': file_path,
            }]
    
    def _run_complexity_analysis(self, file_path: str) -> List[Dict[str, Any]]:
        """Run Radon complexity analysis"""
        if not cc_visit:
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Analyze complexity
            complexity = cc_visit(code, show=False)
            
            findings = []
            for item in complexity:
                if item.complexity > self.severity_thresholds.get('complexity', 10):
                    findings.append({
                        'type': 'quality',
                        'rule_id': 'high_complexity',
                        'rule_name': 'high-complexity',
                        'severity': self._map_severity('high_complexity', item.complexity),
                        'confidence': 'high',
                        'title': f'High complexity function: {item.name}',
                        'description': f'Function {item.name} has cyclomatic complexity of {item.complexity}',
                        'file_path': file_path,
                        'line_number': item.lineno,
                        'end_line_number': item.end_lineno,
                        'code_snippet': self._get_code_snippet(file_path, item.lineno, item.end_lineno),
                        'recommendation': 'Refactor function to reduce complexity'
                    })
            
            return findings
            
        except Exception as e:
            return [{
                'type': 'error',
                'severity': 'high',
                'message': f'Complexity analysis failed: {str(e)}',
                'file_path': file_path,
            }]
    
    def _run_pattern_analysis(self, file_path: str) -> List[Dict[str, Any]]:
        """Run pattern-based analysis"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Hardcoded secrets detection
            secret_patterns = [
                (r'password\s*=\s*["\'][^\'"]', 'hardcoded_secret'),
                (r'api[_-]?key\s*=\s*["\'][^\'"]', 'api_key'),
                (r'token\s*=\s*["\'][^\'"]', 'auth_token'),
                (r'secret\s*=\s*["\'][^\'"]', 'auth_secret'),
                (r'private[_-]?key\s*=\s*["\'][^\'"]', 'private_key'),
                (r'aws[_-]?access[_-]?key[_-]?id\s*=\s*["\'][^\'"]', 'aws_access_key'),
                (r'github[_-]?token\s*=\s*["\'][^\'"]', 'github_token'),
                (r'database[_-]?url\s*=\s*["\'][^\'"]', 'database_url'),
                (r'connection[_-]?string\s*=\s*["\'][^\'"]', 'connection_string'),
            ]
            
            # Insecure patterns
            insecure_patterns = [
                (r'eval\s*\(', 'insecure_eval'),
                (r'exec\s*\(', 'insecure_exec'),
                (r'shell=True\s*\(', 'insecure_shell'),
                (r'subprocess\.call\s*\(', 'insecure_subprocess'),
                (r'pickle\.loads\s*\(', 'insecure_pickle'),
                (r'os\.system\s*\(', 'insecure_system'),
                (r'input\s*\(', 'insecure_input'),
                (r'urllib\.open\s*\(', 'insecure_http'),
                (r'httplib\.', 'insecure_http'),
            ]
            
            for line_num, line in enumerate(code.split('\n'), 1):
                for pattern, pattern_name in secret_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        findings.append({
                            'type': 'security',
                            'rule_id': 'hardcoded_secret',
                            'rule_name': pattern_name,
                            'severity': 'critical',
                            'confidence': 'high',
                            'title': f'Hardcoded {pattern_name} detected',
                            'description': f'Potential hardcoded {pattern_name} found in code',
                            'file_path': file_path,
                            'line_number': line_num,
                            'end_line_number': line_num,
                            'code_snippet': line.strip(),
                            'recommendation': f'Move {pattern_name} to environment variables or secure storage'
                        })
                        break
                
                # Insecure patterns
                for pattern, pattern_name in insecure_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        findings.append({
                            'type': 'security',
                            'rule_id': 'insecure_pattern',
                            'rule_name': pattern_name,
                            'severity': 'critical',
                            'confidence': 'high',
                            'title': f'Insecure {pattern_name} usage detected',
                            'description': f'Potentially insecure {pattern_name} usage found',
                            'file_path': file_path,
                            'line_number': line_num,
                            'end_line_number': line_num,
                            'code_snippet': line.strip(),
                            'recommendation': f'Avoid using {pattern_name} with user input'
                        })
                        break
            
            return findings
            
        except Exception as e:
            return [{
                'type': 'error',
                'severity': 'high',
                'message': f'Pattern analysis failed: {str(e)}',
                'file_path': file_path,
            }]
    
    def _get_available_tools(self) -> List[str]:
        """Get available tools for detected language"""
        # Detect language from file extension if not specified
        if not self.language or self.language == 'auto':
            self.language = self._detect_language_from_config()
        
        # Language-specific tools
        language_tools = {
            'python': ['pylint', 'bandit', 'radon'],
            'javascript': ['eslint', 'jshint'],
            'typescript': ['eslint', 'tsc'],
            'java': ['checkstyle', 'spotbugs', 'pmd'],
            'cpp': ['cppcheck', 'clang-tidy'],
            'c': ['cppcheck', 'clang-tidy'],
            'go': ['golint', 'go vet'],
            'rust': ['clippy', 'rustfmt']
        }
        
        available_tools = language_tools.get(self.language, [])
        
        # Filter by what's actually available
        actual_tools = []
        if self.language == 'python':
            if PylintRun:
                actual_tools.append('pylint')
            if bandit_manager:
                actual_tools.append('bandit')
            if cc_visit:
                actual_tools.append('radon')
        elif self.language in ['javascript', 'typescript']:
            # Check for ESLint
            try:
                subprocess.run(['eslint', '--version'], capture_output=True, check=True)
                actual_tools.append('eslint')
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        return actual_tools or ['pattern']  # Always have pattern analysis
    
    def _detect_language_from_config(self) -> str:
        """Detect language from config or default to python"""
        return self.config.get('language', 'python')
    
    def _detect_language_from_path(self, file_path: str) -> str:
        """Detect programming language from file path"""
        file_ext = Path(file_path).suffix.lower()
        
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
        }
        
        return ext_map.get(file_ext, 'unknown')
    
    def _run_eslint_analysis(self, file_path: str) -> List[Dict[str, Any]]:
        """Run ESLint analysis for JavaScript/TypeScript"""
        try:
            result = subprocess.run(
                ['eslint', file_path, '--format', 'json'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0 and result.stdout:
                eslint_output = json.loads(result.stdout)
                findings = []
                
                for issue in eslint_output:
                    findings.append({
                        'type': 'quality',
                        'rule_id': issue.get('ruleId', 'unknown'),
                        'rule_name': issue.get('ruleId', 'unknown'),
                        'severity': self._map_eslint_severity(issue.get('severity', 1)),
                        'confidence': 'medium',
                        'title': issue.get('message', 'ESLint issue'),
                        'description': issue.get('message', 'Code quality issue detected'),
                        'file_path': file_path,
                        'line_number': issue.get('line', 0),
                        'end_line_number': issue.get('endLine', issue.get('line', 0)),
                        'column': issue.get('column', 0),
                        'end_column': issue.get('endColumn', 0),
                        'code_snippet': self._get_code_snippet(file_path, issue.get('line', 0), issue.get('endLine', issue.get('line', 0))),
                        'recommendation': self._get_eslint_recommendation(issue.get('ruleId', ''))
                    })
                
                return findings
            
            return []
            
        except Exception as e:
            return [{
                'type': 'error',
                'severity': 'medium',
                'message': f'ESLint analysis failed: {str(e)}',
                'file_path': file_path
            }]
    
    def _map_eslint_severity(self, eslint_severity: int) -> str:
        """Map ESLint severity to standard severity levels"""
        severity_map = {
            1: 'low',      # Warning
            2: 'medium',   # Error
        }
        return severity_map.get(eslint_severity, 'medium')
    
    def _get_eslint_recommendation(self, rule_id: str) -> str:
        """Get ESLint recommendation based on rule"""
        recommendations = {
            'no-unused-vars': 'Remove unused variables or use underscore prefix',
            'no-console': 'Remove console.log statements in production code',
            'no-undef': 'Define variables before using them',
            'semi': 'Add or remove semicolons according to style guide',
            'quotes': 'Use consistent quote style (single or double)',
            'indent': 'Fix indentation to be consistent',
            'no-trailing-spaces': 'Remove trailing whitespace',
            'eol-last': 'Add newline at end of file'
        }
        
        return recommendations.get(rule_id, 'Fix the ESLint violation according to the rule documentation')
    
    def _get_code_snippet(self, file_path: str, start_line: int, end_line: int) -> str:
        """Extract code snippet from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if start_line <= len(lines) and end_line <= len(lines):
                snippet_lines = lines[start_line - 1:end_line]
                return '\n'.join(snippet_lines)
            return ''
            
        except Exception:
            return ''
    
    def _map_severity(self, severity_or_code: Union[str, int, float], default_severity: str = 'medium') -> str:
        """Map severity or code to standardized severity levels"""
        severity_mapping = {
            # Pylint severity levels
            'fatal': 'critical',
            'error': 'high',
            'warning': 'medium',
            'refactor': 'low',
            'convention': 'info',
            'info': 'info',
            
            # Bandit severity levels
            'low': 'low',
            'medium': 'medium',
            'high': 'high',
            'critical': 'critical',
            
            # Radon complexity levels
            'A': 'low',
            'B': 'medium',
            'C': 'high',
            'D': 'critical',
            'E': 'critical',
            'F': 'critical',
        }
        
        # Handle numeric complexity
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
    
    def _get_security_recommendation(self, issue: Dict[str, Any]) -> str:
        """Get security recommendation based on issue type"""
        cwe_id = issue.get('cwe_id', '')
        
        recommendations = {
            'CWE-79': 'Use parameterized queries or prepared statements',
            'CWE-89': 'Validate and sanitize all user input',
            'CWE-200': 'Avoid exposing sensitive data in error messages',
            'CWE-327': 'Use proper input validation and encoding',
            'CWE-22': 'Always validate file paths and user input',
            'CWE-78': 'Use os.path.join() for path construction',
            'CWE-20': 'Always validate file paths and user input',
        }
        
        return recommendations.get(cwe_id, 'Follow security best practices')
    
    def _get_quality_recommendation(self, msg_id: str) -> str:
        """Get quality recommendation based on Pylint message"""
        recommendations = {
            'C0111': 'Add comprehensive docstrings to document functions and classes',
            'R0801': 'Refactor duplicate code into reusable functions or classes',
            'R0901': 'Consider breaking down large functions into smaller ones',
            'R0913': 'Reduce the number of function arguments',
            'R0914': 'Consider using a dictionary or dataclass for related values',
            'R0915': 'Simplify complex conditional expressions',
            'duplicate-code': 'Extract common code into reusable functions',
        }
        
        return recommendations.get(msg_id, 'Improve code structure and readability')