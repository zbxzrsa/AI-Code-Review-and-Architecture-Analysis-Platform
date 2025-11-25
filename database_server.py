#!/usr/bin/env python3
import http.server
import socketserver
import json
import sqlite3
import os
import re
import ast
import tempfile
from urllib.parse import urlparse, parse_qs, unquote
import subprocess
import threading
import time
from datetime import datetime

class DatabaseManager:
    """Simple SQLite database manager for the platform"""
    
    def __init__(self, db_path='code_review_platform.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_analysis TIMESTAMP,
                issues_count INTEGER DEFAULT 0
            )
        ''')
        
        # Analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                file_path TEXT,
                language TEXT,
                metrics TEXT,  -- JSON string
                issues TEXT,   -- JSON string
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
        
        # Issues table (for detailed tracking)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                line_number INTEGER,
                issue_type TEXT,
                message TEXT,
                severity TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_project(self, name, description=''):
        """Create a new project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO projects (name, description, created_at)
            VALUES (?, ?, ?)
        ''', (name, description, datetime.now().isoformat()))
        
        project_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return project_id
    
    def get_projects(self):
        """Get all projects"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, description, status, created_at, last_analysis, issues_count
            FROM projects
            ORDER BY created_at DESC
        ''')
        
        projects = []
        for row in cursor.fetchall():
            projects.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'status': row[3],
                'created_at': row[4],
                'last_analysis': row[5],
                'issues_count': row[6]
            })
        
        conn.close()
        return projects
    
    def save_analysis(self, project_id, file_path, language, metrics, issues, status):
        """Save analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert to JSON strings
        metrics_json = json.dumps(metrics)
        issues_json = json.dumps(issues)
        
        # Insert analysis
        cursor.execute('''
            INSERT INTO analyses (project_id, file_path, language, metrics, issues, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (project_id, file_path, language, metrics_json, issues_json, status, datetime.now().isoformat()))
        
        analysis_id = cursor.lastrowid
        
        # Insert individual issues
        for issue in issues:
            cursor.execute('''
                INSERT INTO issues (analysis_id, line_number, issue_type, message, severity, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (analysis_id, issue.get('line'), issue.get('type'), issue.get('message'), issue.get('severity'), datetime.now().isoformat()))
        
        # Update project issues count
        cursor.execute('''
            UPDATE projects 
            SET issues_count = issues_count + ?, last_analysis = ?
            WHERE id = ?
        ''', (len(issues), datetime.now().isoformat(), project_id))
        
        conn.commit()
        conn.close()
        
        return analysis_id
    
    def get_project_analyses(self, project_id):
        """Get all analyses for a project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, file_path, language, metrics, issues, status, created_at
            FROM analyses
            WHERE project_id = ?
            ORDER BY created_at DESC
        ''', (project_id,))
        
        analyses = []
        for row in cursor.fetchall():
            analyses.append({
                'id': row[0],
                'file_path': row[1],
                'language': row[2],
                'metrics': json.loads(row[3]) if row[3] else {},
                'issues': json.loads(row[4]) if row[4] else [],
                'status': row[5],
                'created_at': row[6]
            })
        
        conn.close()
        return analyses
    
    def get_stats(self):
        """Get platform statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Project count
        cursor.execute('SELECT COUNT(*) FROM projects')
        project_count = cursor.fetchone()[0]
        
        # Analysis count
        cursor.execute('SELECT COUNT(*) FROM analyses')
        analysis_count = cursor.fetchone()[0]
        
        # Total issues
        cursor.execute('SELECT COUNT(*) FROM issues')
        issues_count = cursor.fetchone()[0]
        
        # Recent analyses
        cursor.execute('''
            SELECT COUNT(*) FROM analyses 
            WHERE created_at >= date('now', '-1 day')
        ''')
        recent_analyses = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'projects': project_count,
            'total_analyses': analysis_count,
            'total_issues': issues_count,
            'recent_analyses': recent_analyses
        }

class CodeAnalyzer:
    """Simple code analysis utilities"""
    
    @staticmethod
    def analyze_python(code):
        """Analyze Python code for basic metrics and issues"""
        try:
            tree = ast.parse(code)
            issues = []
            metrics = {
                'lines': len(code.splitlines()),
                'functions': 0,
                'classes': 0,
                'imports': 0,
                'complexity': 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics['functions'] += 1
                    # Simple complexity calculation
                    metrics['complexity'] += len([n for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler))])
                elif isinstance(node, ast.ClassDef):
                    metrics['classes'] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    metrics['imports'] += 1
            
            # Basic issue detection
            lines = code.splitlines()
            for i, line in enumerate(lines, 1):
                if 'print(' in line and not line.strip().startswith('#'):
                    issues.append({
                        'line': i,
                        'type': 'warning',
                        'message': 'Debug print statement found',
                        'severity': 'low'
                    })
                if len(line) > 120:
                    issues.append({
                        'line': i,
                        'type': 'style',
                        'message': 'Line too long (>120 characters)',
                        'severity': 'medium'
                    })
            
            return {
                'language': 'python',
                'metrics': metrics,
                'issues': issues,
                'status': 'success'
            }
        except SyntaxError as e:
            return {
                'language': 'python',
                'error': f'Syntax error: {str(e)}',
                'status': 'error'
            }
        except Exception as e:
            return {
                'language': 'python',
                'error': f'Analysis error: {str(e)}',
                'status': 'error'
            }
    
    @staticmethod
    def analyze_javascript(code):
        """Analyze JavaScript code for basic metrics"""
        issues = []
        metrics = {
            'lines': len(code.splitlines()),
            'functions': 0,
            'classes': 0,
            'complexity': 0
        }
        
        # Count functions
        metrics['functions'] = len(re.findall(r'\bfunction\s+\w+|=>\s*{|\w+\s*:\s*function', code))
        
        # Count classes
        metrics['classes'] = len(re.findall(r'\bclass\s+\w+', code))
        
        # Basic issue detection
        lines = code.splitlines()
        for i, line in enumerate(lines, 1):
            if 'console.log(' in line and not line.strip().startswith('//'):
                issues.append({
                    'line': i,
                    'type': 'warning',
                    'message': 'Console.log statement found',
                    'severity': 'low'
                })
            if 'var ' in line:
                issues.append({
                    'line': i,
                    'type': 'style',
                    'message': 'Consider using let or const instead of var',
                    'severity': 'medium'
                })
        
        return {
            'language': 'javascript',
            'metrics': metrics,
            'issues': issues,
            'status': 'success'
        }
    
    @staticmethod
    def detect_language(code):
        """Simple language detection"""
        if re.search(r'^(import|from|def|class)\s', code, re.MULTILINE):
            return 'python'
        elif re.search(r'function\s+\w+|var\s+\w+|const\s+\w+|let\s+\w+', code):
            return 'javascript'
        elif re.search(r'public\s+class|private\s+\w+|protected\s+\w+', code):
            return 'java'
        elif re.search(r'#include|int\s+main|printf\s*\(', code):
            return 'c'
        else:
            return 'unknown'

class DatabaseAPIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.analyzer = CodeAnalyzer()
        self.db = DatabaseManager()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/health':
            self.send_json_response({"status": "ok", "service": "database-enabled-api"})
        elif self.path == '/api/status':
            stats = self.db.get_stats()
            self.send_json_response({
                "status": "running",
                "services": {
                    "api": "healthy",
                    "database": "connected",
                    "analyzer": "active"
                },
                "stats": stats
            })
        elif self.path == '/api/v1/projects':
            projects = self.db.get_projects()
            self.send_json_response(projects)
        elif self.path.startswith('/api/v1/projects/') and self.path.endswith('/analyses'):
            # Get analyses for a specific project
            try:
                project_id = int(self.path.split('/')[3])
                analyses = self.db.get_project_analyses(project_id)
                self.send_json_response(analyses)
            except (ValueError, IndexError):
                self.send_json_response({"error": "Invalid project ID"}, 400)
        elif self.path == '/api/v1/stats':
            stats = self.db.get_stats()
            self.send_json_response(stats)
        elif self.path == '/api/v1/analysis/languages':
            self.send_json_response({
                "supported_languages": [
                    {"name": "Python", "extension": "py", "status": "enabled"},
                    {"name": "JavaScript", "extension": "js", "status": "enabled"},
                    {"name": "Java", "extension": "java", "status": "planned"},
                    {"name": "C/C++", "extension": "c", "status": "planned"}
                ]
            })
        else:
            super().do_GET()
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, 400)
            return
        
        if self.path == '/api/v1/analyze/code':
            self.handle_code_analysis(data)
        elif self.path == '/api/v1/analyze/project':
            self.handle_project_analysis(data)
        elif self.path == '/api/v1/projects':
            self.handle_create_project(data)
        else:
            self.send_json_response({"error": "Endpoint not found"}, 404)
    
    def handle_code_analysis(self, data):
        """Handle code analysis request"""
        if 'code' not in data:
            self.send_json_response({"error": "Code is required"}, 400)
            return
        
        code = data['code']
        language = data.get('language', 'auto')
        project_id = data.get('project_id')
        
        if language == 'auto':
            language = self.analyzer.detect_language(code)
        
        # Analyze based on language
        if language == 'python':
            result = self.analyzer.analyze_python(code)
        elif language == 'javascript':
            result = self.analyzer.analyze_javascript(code)
        else:
            result = {
                'language': language,
                'error': f'Analysis not yet supported for {language}',
                'status': 'unsupported'
            }
        
        # Save to database if project_id is provided and analysis was successful
        if project_id and result.get('status') == 'success':
            self.db.save_analysis(
                project_id=project_id,
                file_path=data.get('file_path', 'inline_code'),
                language=language,
                metrics=result.get('metrics', {}),
                issues=result.get('issues', []),
                status=result['status']
            )
        
        self.send_json_response(result)
    
    def handle_project_analysis(self, data):
        """Handle project-level analysis"""
        if 'project_id' not in data or 'code' not in data:
            self.send_json_response({"error": "Project ID and code are required"}, 400)
            return
        
        project_id = data['project_id']
        code = data['code']
        file_path = data.get('file_path', 'unknown_file')
        
        # Verify project exists
        projects = self.db.get_projects()
        if not any(p['id'] == project_id for p in projects):
            self.send_json_response({"error": "Project not found"}, 404)
            return
        
        # Analyze code
        language = data.get('language', 'auto')
        if language == 'auto':
            language = self.analyzer.detect_language(code)
        
        if language == 'python':
            result = self.analyzer.analyze_python(code)
        elif language == 'javascript':
            result = self.analyzer.analyze_javascript(code)
        else:
            result = {
                'language': language,
                'error': f'Analysis not yet supported for {language}',
                'status': 'unsupported'
            }
        
        # Save analysis to database
        if result.get('status') == 'success':
            analysis_id = self.db.save_analysis(
                project_id=project_id,
                file_path=file_path,
                language=language,
                metrics=result.get('metrics', {}),
                issues=result.get('issues', []),
                status=result['status']
            )
            result['analysis_id'] = str(analysis_id)
        
        self.send_json_response(result)
    
    def handle_create_project(self, data):
        """Handle project creation"""
        if 'name' not in data:
            self.send_json_response({"error": "Project name is required"}, 400)
            return
        
        name = data['name']
        description = data.get('description', '')
        
        try:
            project_id = self.db.create_project(name, description)
            
            project = {
                "id": project_id,
                "name": name,
                "description": description,
                "status": "created",
                "created_at": datetime.now().isoformat(),
                "last_analysis": None,
                "issues_count": 0
            }
            
            self.send_json_response(project, 201)
        except Exception as e:
            self.send_json_response({"error": f"Failed to create project: {str(e)}"}, 500)
    
    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

PORT = 8001

def print_banner():
    """Print startup banner"""
    print("""
    🚀 AI Code Review Platform - Database Enabled
    ==========================================
    
    Features:
    ✅ Real-time code analysis
    ✅ SQLite database storage
    ✅ Project management
    ✅ Analysis history
    ✅ Issue tracking
    ✅ Platform statistics
    
    Database: SQLite (code_review_platform.db)
    
    Available endpoints:
    📡 GET  /health - Health check
    📡 GET  /api/status - Service status with stats
    📡 GET  /api/v1/projects - List projects
    📡 GET  /api/v1/projects/{id}/analyses - Project analyses
    📡 GET  /api/v1/stats - Platform statistics
    📡 POST /api/v1/analyze/code - Analyze code
    📡 POST /api/v1/analyze/project - Analyze for project
    📡 POST /api/v1/projects - Create project
    
    Server running on http://localhost:{}".format(PORT)
    """)

if __name__ == "__main__":
    print_banner()
    
    with socketserver.TCPServer(("", PORT), DatabaseAPIHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")