#!/usr/bin/env python3
import http.server
import socketserver
import json
import urllib.request
import urllib.error
import re
import ast
import os
import tempfile
from urllib.parse import urlparse, parse_qs, unquote
import subprocess
import threading
import time

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

class EnhancedAPIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.analyzer = CodeAnalyzer()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/health':
            self.send_json_response({"status": "ok", "service": "enhanced-ai-analysis-api"})
        elif self.path == '/api/status':
            self.send_json_response({
                "status": "running",
                "services": {
                    "api": "healthy",
                    "analyzer": "active",
                    "supported_languages": ["python", "javascript", "java", "c"]
                }
            })
        elif self.path == '/api/v1/projects':
            self.send_json_response([
                {
                    "id": 1,
                    "name": "Sample Project",
                    "description": "A sample project for testing",
                    "status": "active",
                    "last_analysis": "2025-11-25T15:00:00Z",
                    "issues_count": 3
                }
            ])
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
        elif self.path == '/api/v1/analyze/file':
            self.handle_file_analysis(data)
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
        
        self.send_json_response(result)
    
    def handle_file_analysis(self, data):
        """Handle file analysis request"""
        if 'file_path' not in data:
            self.send_json_response({"error": "File path is required"}, 400)
            return
        
        file_path = data['file_path']
        
        try:
            # For security, only analyze files in current directory
            if not os.path.abspath(file_path).startswith(os.getcwd()):
                self.send_json_response({"error": "Access denied"}, 403)
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
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
            
            result['file_path'] = file_path
            self.send_json_response(result)
            
        except FileNotFoundError:
            self.send_json_response({"error": "File not found"}, 404)
        except Exception as e:
            self.send_json_response({"error": f"Analysis failed: {str(e)}"}, 500)
    
    def handle_create_project(self, data):
        """Handle project creation"""
        if 'name' not in data:
            self.send_json_response({"error": "Project name is required"}, 400)
            return
        
        project = {
            "id": int(time.time()),
            "name": data['name'],
            "description": data.get('description', ''),
            "status": "created",
            "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "last_analysis": None,
            "issues_count": 0
        }
        
        self.send_json_response(project, 201)
    
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
    🚀 Enhanced AI Code Review Platform
    ===================================
    
    Features:
    ✅ Real-time code analysis
    ✅ Multi-language support (Python, JavaScript)
    ✅ Issue detection and metrics
    ✅ File analysis
    ✅ Project management
    
    Available endpoints:
    📡 GET  /health - Health check
    📡 GET  /api/status - Service status
    📡 GET  /api/v1/projects - List projects
    📡 GET  /api/v1/analysis/languages - Supported languages
    📡 POST /api/v1/analyze/code - Analyze code
    📡 POST /api/v1/analyze/file - Analyze file
    📡 POST /api/v1/projects - Create project
    
    Server running on http://localhost:{}
    """.format(PORT))

if __name__ == "__main__":
    print_banner()
    
    with socketserver.TCPServer(("", PORT), EnhancedAPIHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")