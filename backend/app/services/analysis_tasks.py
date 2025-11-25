"""
Analysis Tasks for Celery
"""
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from app.models import AnalysisSession, Finding
from app.db.session import get_db
from app.services.static_analyzer import StaticAnalyzer


class AnalysisTask:
    """Celery task for code analysis"""
    
    def __init__(self):
        self.static_analyzer = StaticAnalyzer({
            'rules_enabled': ['security', 'quality', 'complexity', 'patterns'],
            'severity_thresholds': {
                'critical': 0,
                'high': 5,
                'medium': 20,
                'low': 50
            }
        })
    
    async def run_full_analysis(self, session_id: str, repository_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run full repository analysis"""
        try:
            # Get repository details
            from app.models import Repository
            db = next(get_db())
            repo = db.query(Repository).filter(Repository.id == repository_id).first()
            
            if not repo:
                return {
                    'status': 'error',
                    'message': f'Repository {repository_id} not found'
                }
            
            # Create analysis session
            session = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
            if not session:
                session = AnalysisSession(
                    repository_id=repository_id,
                    analysis_type='full',
                    config=config,
                    status='running'
                )
                db.add(session)
                db.commit()
                db.refresh(session)
            
            # Get repository files
            repo_path = repo.clone_url.replace('https://github.com/', '')
            try:
                import subprocess
                subprocess.run([
                    'git', 'clone', repo_path, repo_path
                ], 
                    capture_output=True, text=True, check=True
                ], timeout=300)
            except subprocess.TimeoutExpired:
                return {
                    'status': 'error',
                    'message': 'Failed to clone repository'
                }
            
            # Start analysis
            findings = await self._analyze_repository(repo_path, session)
            
            # Update session with results
            session.total_findings = len(findings)
            session.critical_findings = len([f for f in findings if f['severity'] == 'critical'])
            session.high_findings = len([f for f in findings if f['severity'] == 'high'])
            session.medium_findings = len([f for f in findings if f['severity'] == 'medium'])
            session.low_findings = len([f for f in findings if f['severity'] == 'low'])
            
            # Update session status
            session.status = 'completed'
            session.completed_at = datetime.utcnow()
            session.duration_seconds = int((datetime.utcnow() - session.started_at).total_seconds())
            
            db.commit()
            db.refresh(session)
            
            return {
                'status': 'completed',
                'session_id': str(session.id),
                'findings': findings,
                'total_findings': session.total_findings,
                'critical_findings': session.critical_findings,
                'high_findings': session.high_findings,
                'medium_findings': session.medium_findings,
                'low_findings': session.low_findings,
                'duration_seconds': session.duration_seconds
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Analysis failed: {str(e)}',
                'session_id': session_id,
            }
    
    async def _analyze_repository(self, repo_path: str, session: AnalysisSession) -> List[Dict[str, Any]]:
        """Analyze repository and return findings"""
        findings = []
        
        try:
            # Get all source files
            source_files = []
            for root, dirs, files in os.walk(repo_path):
                if root.startswith('.git'):
                    continue
                
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.tsx', '.jsx'):
                        source_files.append(os.path.join(root, file))
            
            # Analyze each file
            for file_path in source_files:
                file_findings = self.static_analyzer.analyze_file(file_path)
                findings.extend(file_findings)
            
            return findings
            
        except Exception as e:
            return [{
                'type': 'error',
                'severity': 'high',
                'message': f'Repository analysis failed: {str(e)}',
                'repo_path': repo_path,
            }]
    
    async def run_pr_analysis(self, session_id: str, repository_id: str, pr_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pull request"""
        try:
            # Get PR details
            from app.models import PullRequest, Repository
            db = next(get_db())
            
            pr = db.query(PullRequest).filter(PullRequest.id == pr_id).first()
            
            if not pr:
                return {
                    'status': 'error',
                    'message': f'Pull request {pr_id} not found'
                }
            
            # Get PR diff
            repo_path = pr.repository.clone_url.replace('https://github.com/', '')
            pr_diff_url = pr.diff_url
            
            # Create analysis session
            session = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
            if not session:
                session = AnalysisSession(
                    repository_id=repository_id,
                    pull_request_id=pr_id,
                    analysis_type='pr',
                    config=config,
                    status='running'
                )
                db.add(session)
                db.commit()
                db.refresh(session)
            
            # Get PR diff
            try:
                import subprocess
                result = subprocess.run([
                    'git', 'diff', pr_diff_url, '--name-only', '--no-color', '--exit-code'],
                    ], 
                    capture_output=True, text=True, check=True
                    ], timeout=300)
                
                diff_content = result.stdout
            except subprocess.TimeoutExpired:
                diff_content = None
            
            # Analyze PR diff
            pr_findings = []
            if diff_content:
                pr_findings = self._analyze_diff_content(diff_content, session)
            
            # Update session with results
            session.total_findings = len(pr_findings)
            session.critical_findings = len([f for f in pr_findings if f['severity'] == 'critical'])
            session.high_findings = len([f for f in pr_findings if f['severity'] == 'high'])
            session.medium_findings = len([f for f in pr_findings if f['severity'] == 'medium'])
            session.low_findings = len([f for f in pr_findings if f['severity'] == 'low'])
            
            # Update session status
            session.status = 'completed'
            session.completed_at = datetime.utcnow()
            session.duration_seconds = int((datetime.utcnow() - session.started_at).total_seconds())
            
            db.commit()
            db.refresh(session)
            
            return {
                'status': 'completed',
                'session_id': str(session.id),
                'findings': pr_findings,
                'total_findings': session.total_findings,
                'critical_findings': session.critical_findings,
                'high_findings': session.high_findings,
                'medium_findings': session.medium_findings,
                'low_findings: session.low_findings,
                'duration_seconds': session.duration_seconds,
                'diff_content': diff_content
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'PR analysis failed: {str(e)}',
                'session_id': session_id,
            }
    
    def _analyze_diff_content(self, diff_content: str, session: AnalysisSession) -> List[Dict[str, Any]]:
        """Analyze PR diff content"""
        findings = []
        
        if not diff_content:
            return findings
        
        try:
            lines = diff_content.split('\n')
            file_number = 0
            
            for line in lines:
                line_number = line_number + 1
                
                # Security checks in diff
                if 'password' in line.lower() or 'secret' in line.lower():
                    findings.append({
                        'type': 'security',
                        'rule_id': 'pr_secret',
                        'rule_name': 'pr_secret',
                        'severity': 'critical',
                        'confidence': 'high',
                        'title': 'Potential secret in PR diff',
                        'description': f'Potential secret found in PR diff at line {line_number}',
                        'file_path': 'pr_diff_url',
                        'line_number': line_number,
                        'code_snippet': line.strip(),
                        'recommendation': 'Remove secrets from code before merging'
                    })
                
                # Code quality checks
                if len(line.strip()) > 150:
                    findings.append({
                        'type': 'quality',
                        'rule_id': 'pr_long_line',
                        'rule_name': 'pr_long_line',
                        'severity': 'medium',
                        'confidence': 'medium',
                        'title': 'Long line in PR diff',
                        'description': f'Long line ({len(line.strip())} chars) at line {line_number}',
                        'file_path': 'pr_diff_url',
                        'line_number': line_number,
                        'code_snippet': line.strip()[:100] + '...' if len(line.strip()) > 100 else line.strip(),
                        'recommendation': 'Consider breaking up long lines'
                    })
                
                # Architecture checks
                if 'import' in line.lower() and ('from' in line.lower() or 'import *' in line.lower()):
                    findings.append({
                        'type': 'architecture',
                        'rule_id': 'pr_import',
                        'rule_name': 'pr_import',
                        'severity': 'medium',
                        'confidence': 'high',
                        'title': 'Import statement in PR diff',
                        'description': f'Import statement found in PR diff at line {line_number}',
                        'file_path': 'pr_diff_url',
                        'line_number': line_number,
                        'code_snippet': line.strip(),
                        'recommendation': 'Review imports for security issues'
                    })
                
                # Performance checks
                if 'performance' in line.lower() and ('alloc' in line.lower() or 'malloc' in line.lower()):
                    findings.append({
                        'type': 'performance',
                        'rule_id': 'pr_performance',
                        'rule_name': 'pr_performance',
                        'severity': 'medium',
                        'confidence': 'medium',
                        'title': 'Performance concern in PR diff',
                        'description': f'Performance issue found at line {line_number}',
                        'file_path': 'pr_diff_url',
                        'line_number': line_number,
                        'code_snippet': line.strip(),
                        'recommendation': 'Review performance implications'
                    })
                
                line_number += 1
            
            return findings
            
        except Exception as e:
            return [{
                'type': 'error',
                'severity': 'high',
                'message': f'Diff analysis failed: {str(e)}',
                'file_path': 'pr_diff_url',
            }]
    
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
    
    def _get_recommendation(self, issue: Dict[str, Any]) -> str:
        """Get recommendation based on issue type"""
        recommendations = {
            'pr_secret': 'Remove secrets from code before merging',
            'pr_import': 'Review imports for security issues',
            'pr_performance': 'Review performance implications',
            'pr_long_line': 'Consider breaking up long lines',
            'duplicate-code': 'Extract common code into reusable functions',
        }
        
        return recommendations.get(issue.get('rule_id', 'Follow security best practices')
    
    def _get_quality_recommendation(self, msg_type: str) -> str:
        """Get quality recommendation based on message type"""
        recommendations = {
            'pr_long_line': 'Consider breaking up long lines',
            'duplicate-code': 'Extract common code into reusable functions',
            'missing_docstring': 'Add docstrings to document functions',
            'too_many_args': 'Reduce function arguments',
            'complex_function': 'Refactor complex functions',
        }
        
        return recommendations.get(msg_type, 'Improve code structure and readability')