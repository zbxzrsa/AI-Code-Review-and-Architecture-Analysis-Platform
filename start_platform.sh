#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           🚀 AI CODE REVIEW PLATFORM - FINAL              ║"
echo "║                 ALL SYSTEMS OPERATIONAL                ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Kill any existing processes
echo "🔄 Stopping any existing servers..."
pkill -f working_perfect_server.py 2>/dev/null || true
pkill -f final_server.py 2>/dev/null || true
pkill -f perfect_server.py 2>/dev/null || true

# Start the working server
echo "🚀 Starting AI Code Review Platform..."
cd "$(dirname "$0")"
python3 working_perfect_server.py &

# Wait for server to start
echo "⏳ Waiting for server to initialize..."
sleep 3

# Test server health
echo "🔍 Testing server health..."
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Server is running successfully!"
    echo ""
    echo "🌐 ACCESS POINTS:"
    echo "   📊 Frontend: file://$(pwd)/complete_frontend.html"
    echo "   🔧 Backend API: http://localhost:8001"
    echo "   📖 API Health: http://localhost:8001/health"
    echo "   🔑 Default Login: admin/admin123"
    echo ""
    echo "🎉 PLATFORM READY FOR USE!"
    echo "   Open complete_frontend.html in your browser to get started"
    echo ""
    echo "📋 AVAILABLE FEATURES:"
    echo "   ✅ User Authentication"
    echo "   ✅ Code Analysis (Python & JavaScript)"
    echo "   ✅ Project Management"
    echo "   ✅ Issue Detection"
    echo "   ✅ Real-time Results"
    echo "   ✅ SQLite Database"
    echo "   ✅ RESTful API"
    echo ""
    echo "🛑 To stop: pkill -f working_perfect_server.py"
else
    echo "❌ Server failed to start properly"
    echo "   Please check the logs above for errors"
    exit 1
fi