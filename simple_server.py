#!/usr/bin/env python3
import http.server
import socketserver
import json
import urllib.request
import urllib.error
from urllib.parse import urlparse, parse_qs

class SimpleAPIHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "ok", "service": "ai-analysis-api"}
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "status": "running",
                "services": {
                    "api": "healthy",
                    "database": "connected",
                    "redis": "connected"
                }
            }
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/api/v1/projects':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = [
                {
                    "id": 1,
                    "name": "Sample Project",
                    "description": "A sample project for testing",
                    "status": "active"
                }
            ]
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"message": "AI Code Review Platform API", "version": "1.0.0"}
            self.wfile.write(json.dumps(response).encode())
        else:
            super().do_GET()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

PORT = 8001

with socketserver.TCPServer(("", PORT), SimpleAPIHandler) as httpd:
    print(f"Server running at http://localhost:{PORT}")
    print("Available endpoints:")
    print("  GET /health - Health check")
    print("  GET /api/status - Service status")
    print("  GET /api/v1/projects - Sample projects")
    httpd.serve_forever()