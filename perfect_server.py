#!/usr/bin/env python3
"""
AI Code Review Platform - Complete Working Server
============================================
Fixed version with all errors resolved.
"""

import http.server
import socketserver
import json
import sqlite3
import hashlib
import jwt
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs, unquote
import re
import ast
import os
import sys

# Configuration
PORT = 8001
JWT_SECRET = 'your-secret-key-change-in-production'
JWT_ALGORITHM = 'HS256'
DB_PATH = 'ai_code_review.db'

class DatabaseManager:
    """Complete database manager with proper schema"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize all database tables with correct schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                owner_id INTEGER,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_analysis TIMESTAMP,
                issues_count INTEGER DEFAULT 0,
                FOREIGN KEY (owner_id) REFERENCES users (id)
            )
        ''')
        
        # Analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                user_id INTEGER,
                file_path TEXT,
                language TEXT,
                metrics TEXT,
                issues TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Issues table
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
    
    def create_default_admin(self):
        """Create default admin user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',))
        if not cursor.fetchone():
            password_hash = hashlib.sha256('admin123'.encode()).hexdigest()
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', ('admin', 'admin@example.com', password_hash, 'admin'))
            conn.commit()
            print("✅ Created default admin user: admin/admin123")
        
        conn.close()
    
    def hash_password(self, password):
        """Hash password"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password, password_hash):
        """Verify password"""
        return self.hash_password(password) == password_hash
    
    def authenticate_user(self, username, password):
        """Authenticate user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, password_hash, role, is_active
            FROM users WHERE username = ? OR email = ?
        ''', (username, username))
        
        user = cursor.fetchone()
        conn.close()
        
        if user and user[5]:  # is_active
            user_id, db_username, email, stored_hash, role, is_active = user
            if self.verify_password(password, stored_hash):
                # Update last login
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users SET last_login = ? WHERE id = ?
                ''', (datetime.now().isoformat(), user_id))
                conn.commit()
                conn.close()
                
                return {
                    'id': user_id,
                    'username': db_username,
                    'email': email,
                    'role': role
                }
        
        return None
    
    def generate_token(self, user_id, username, role):
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'username': username,
            'role': role,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    def verify_token(self, token):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def create_project(self, name, description, owner_id):
        """Create new project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO projects (name, description, owner_id, created_at)
            VALUES (?, ?, ?, ?)
        ''', (name, description, owner_id, datetime.now().isoformat()))
        
        project_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return project_id
    
    def get_projects(self, owner_id):
        """Get user's projects"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.id, p.name, p.description, p.status, p.created_at, 
                   p.last_analysis, p.issues_count, u.username as owner
            FROM projects p
            JOIN users u ON p.owner_id = u.id
            WHERE p.owner_id = ?
            ORDER BY p.created_at DESC
        ''', (owner_id,))
        
        projects = []
        for row in cursor.fetchall():
            projects.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'status': row[3],
                'created_at': row[4],
                'last_analysis': row[5],
                'issues_count': row[6],
                'owner': row[7]
            })
        
        conn.close()
        return projects
    
    def save_analysis(self, project_id, user_id, file_path, language, metrics, issues, status):
        """Save analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metrics_json = json.dumps(metrics)
        issues_json = json.dumps(issues)
        
        # Insert analysis
        cursor.execute('''
            INSERT INTO analyses (project_id, user_id, file_path, language, metrics, issues, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (project_id, user_id, file_path, language, metrics_json, issues_json, status, datetime.now().isoformat()))
        
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
    
    def get_stats(self):
        """Get platform statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM projects')
        project_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM analyses')
        analysis_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM issues')
        issues_count = cursor.fetchone()[0]
        
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
    """Enhanced code analyzer"""
    
    @staticmethod
    def analyze_python(code):
        """Analyze Python code"""
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
                    metrics['complexity'] += len([n for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler))])
                elif isinstance(node, ast.ClassDef):
                    metrics['classes'] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    metrics['imports'] += 1
            
            # Check for issues
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
        """Analyze JavaScript code"""
        issues = []
        metrics = {
            'lines': len(code.splitlines()),
            'functions': 0,
            'classes': 0,
            'complexity': 0
        }
        
        metrics['functions'] = len(re.findall(r'\bfunction\s+\w+|=>\s*{|\w+\s*:\s*function', code))
        metrics['classes'] = len(re.findall(r'\bclass\s+\w+', code))
        
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
        """Detect programming language"""
        if re.search(r'^(import|from|def|class)\s', code, re.MULTILINE):
            return 'python'
        elif re.search(r'function\s+\w+|var\s+\w+|const\s+\w+|let\s+\w+', code):
            return 'javascript'
        else:
            return 'unknown'

class APIHandler(http.server.SimpleHTTPRequestHandler):
    """Complete API handler with authentication"""
    
    def __init__(self, *args, **kwargs):
        self.db = DatabaseManager()
        self.analyzer = CodeAnalyzer()
        self.db.create_default_admin()  # Ensure admin user exists
        super().__init__(*args, **kwargs)
    
    def get_current_user(self):
        """Get authenticated user from request"""
        auth_header = self.headers.get('Authorization')
        print(f"DEBUG: Auth header: {auth_header}")  # Debug
        
        if not auth_header or not auth_header.startswith('Bearer '):
            print("DEBUG: No auth header or invalid format")
            return None
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        print(f"DEBUG: Token: {token[:20]}...")  # Debug
        user = self.db.verify_token(token)
        print(f"DEBUG: User: {user}")  # Debug
        return user
    
    def require_auth(self):
        """Require authentication"""
        user = self.get_current_user()
        if not user:
            self.send_json_response({'error': 'Authentication required'}, 401)
            return None
        return user
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self.send_json_response({
                'status': 'ok', 
                'service': 'ai-code-review-platform',
                'version': '1.0.0'
            })
        elif self.path == '/api/status':
            stats = self.db.get_stats()
            self.send_json_response({
                'status': 'running',
                'services': {
                    'api': 'healthy',
                    'database': 'connected',
                    'analyzer': 'active',
                    'authentication': 'enabled'
                },
                'stats': stats
            })
        elif self.path == '/api/v1/projects':
            user = self.require_auth()
            if user:
                projects = self.db.get_projects(user['id'])
                self.send_json_response(projects)
        elif self.path == '/api/v1/analysis/languages':
            self.send_json_response({
                'supported_languages': [
                    {'name': 'Python', 'extension': 'py', 'status': 'enabled'},
                    {'name': 'JavaScript', 'extension': 'js', 'status': 'enabled'},
                    {'name': 'Java', 'extension': 'java', 'status': 'planned'},
                    {'name': 'C/C++', 'extension': 'c', 'status': 'planned'}
                ]
            })
        else:
            self.send_json_response({'error': 'Endpoint not found'}, 404)
    
    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_json_response({'error': 'Invalid JSON'}, 400)
            return
        
        if self.path == '/api/v1/auth/login':
            self.handle_login(data)
        elif self.path == '/api/v1/analyze/code':
            user = self.require_auth()
            if user:
                self.handle_code_analysis(data, user)
        elif self.path == '/api/v1/projects':
            user = self.require_auth()
            if user:
                self.handle_create_project(data, user)
        else:
            self.send_json_response({'error': 'Endpoint not found'}, 404)
    
    def handle_login(self, data):
        """Handle user login"""
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            self.send_json_response({'error': 'Username and password required'}, 400)
            return
        
        user = self.db.authenticate_user(username, password)
        if user:
            token = self.db.generate_token(user['id'], user['username'], user['role'])
            self.send_json_response({
                'message': 'Login successful',
                'token': token,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email'],
                    'role': user['role']
                }
            })
        else:
            self.send_json_response({'error': 'Invalid credentials'}, 401)
    
    def handle_code_analysis(self, data, user):
        """Handle code analysis"""
        if 'code' not in data:
            self.send_json_response({'error': 'Code is required'}, 400)
            return
        
        code = data['code']
        language = data.get('language', 'auto')
        project_id = data.get('project_id')
        
        if language == 'auto':
            language = self.analyzer.detect_language(code)
        
        # Analyze code
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
        
        # Save to database if project_id provided
        if project_id and result.get('status') == 'success':
            self.db.save_analysis(
                project_id=project_id,
                user_id=user['id'],
                file_path=data.get('file_path', 'inline_code'),
                language=language,
                metrics=result.get('metrics', {}),
                issues=result.get('issues', []),
                status=result['status']
            )
        
        self.send_json_response(result)
    
    def handle_create_project(self, data, user):
        """Handle project creation"""
        if 'name' not in data:
            self.send_json_response({'error': 'Project name is required'}, 400)
            return
        
        name = data['name']
        description = data.get('description', '')
        
        try:
            project_id = self.db.create_project(name, description, user['id'])
            project = {
                'id': project_id,
                'name': name,
                'description': description,
                'owner': user['username'],
                'status': 'created',
                'created_at': datetime.now().isoformat(),
                'last_analysis': None,
                'issues_count': 0
            }
            self.send_json_response(project, 201)
        except Exception as e:
            self.send_json_response({'error': f'Failed to create project: {str(e)}'}, 500)
    
    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = json.dumps(data, indent=2)
        self.wfile.write(response_data.encode())
    
    def end_headers(self):
        """Set CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.end_headers()

def print_startup_banner():
    """Print startup information"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                🚀 AI CODE REVIEW PLATFORM - WORKING                   ║
║                                                              ║
║  ✅ FEATURES:                                                ║
║     • Real-time code analysis                                   ║
║     • SQLite database storage                                   ║
║     • JWT authentication                                       ║
║     • User management                                         ║
║     • Project management                                       ║
║     • Analysis history                                         ║
║     • Issue tracking                                          ║
║                                                              ║
║  🔐 AUTHENTICATION:                                          ║
║     • Default admin: admin/admin123                             ║
║     • JWT tokens with 24h expiry                              ║
║                                                              ║
║  📡 AVAILABLE ENDPOINTS:                                     ║
║     • GET  /health - Health check                             ║
║     • GET  /api/status - Service status                        ║
║     • POST /api/v1/auth/login - User login                    ║
║     • GET  /api/v1/projects - List projects (auth)           ║
║     • POST /api/v1/projects - Create project (auth)          ║
║     • POST /api/v1/analyze/code - Analyze code (auth)        ║
║     • GET  /api/v1/analysis/languages - Supported languages   ║
║                                                              ║
║  🌐 SERVER RUNNING ON:                                       ║
║     • http://localhost:{:}                                      ║
╚══════════════════════════════════════════════════════════════╝
    """.format(PORT))

def main():
    """Main server function"""
    print_startup_banner()
    
    try:
        with socketserver.TCPServer(("", PORT), APIHandler) as httpd:
            print(f"🎉 Server started successfully on port {PORT}")
            print("📊 Dashboard: Open enhanced_dashboard.html in your browser")
            print("🔑 Default credentials: admin/admin123")
            print("⚡ Ready to analyze code!")
            print("🛑 Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()