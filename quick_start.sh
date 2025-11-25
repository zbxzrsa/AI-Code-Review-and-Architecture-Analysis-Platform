#!/bin/bash

echo "🚀 Starting AI Code Review Platform - Quick Start"
echo "================================================"

# Start the simple backend server
echo "📡 Starting backend server on port 8001..."
python3 simple_server.py > server.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait a moment for server to start
sleep 2

# Test the backend
echo "🔍 Testing backend health..."
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Backend is healthy!"
else
    echo "❌ Backend failed to start"
    exit 1
fi

echo ""
echo "🌐 Platform is ready!"
echo "===================="
echo "📊 Dashboard: file://$(pwd)/dashboard.html"
echo "🔧 Backend API: http://localhost:8001"
echo "📖 API Docs: http://localhost:8001/docs"
echo ""
echo "Available API Endpoints:"
echo "  GET http://localhost:8001/health"
echo "  GET http://localhost:8001/api/status"
echo "  GET http://localhost:8001/api/v1/projects"
echo ""
echo "To stop the platform, run: kill $BACKEND_PID"
echo ""
echo "🎉 Open dashboard.html in your browser to get started!"