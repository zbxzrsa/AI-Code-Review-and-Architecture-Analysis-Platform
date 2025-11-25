# 🚀 AI Code Review Platform - Quick Start

This is a **simplified, working version** of the AI Code Review and Architecture Analysis Platform that runs immediately without complex dependencies or Docker setup issues.

## ⚡ Quick Start (2 minutes)

### Option 1: Automated Startup

```bash
# Run the quick start script
./quick_start.sh
```

### Option 2: Manual Startup

```bash
# Start the backend server
python3 simple_server.py &

# Open the dashboard in your browser
# Double-click dashboard.html or open it in your browser
```

## 🌐 Access Points

Once started, you can access:

- **📊 Dashboard**: Open `dashboard.html` in your browser
- **🔧 Backend API**: http://localhost:8001
- **📖 API Endpoints**:
  - `GET /health` - Health check
  - `GET /api/status` - Service status
  - `GET /api/v1/projects` - Sample projects

## 🎯 What's Working

✅ **Backend API Server**

- Simple Python HTTP server with JSON responses
- CORS enabled for frontend communication
- Health check and status endpoints
- Sample data endpoints

✅ **Interactive Dashboard**

- Beautiful, responsive web interface
- Real-time API testing
- Service status monitoring
- One-click API endpoint testing

✅ **No Complex Dependencies**

- No Docker required
- No database setup needed
- No Redis/Neo4j configuration
- Works with Python 3.6+

## 📁 Files Created

- `simple_server.py` - Minimal backend API server
- `dashboard.html` - Interactive web dashboard
- `quick_start.sh` - Automated startup script

## 🔧 Next Steps

To expand from this working foundation:

1. **Add Real API Endpoints**: Extend `simple_server.py` with actual code analysis logic
2. **Connect Database**: Replace mock data with real PostgreSQL connections
3. **Add Authentication**: Implement JWT or OAuth2 security
4. **Build Frontend**: Replace the HTML dashboard with a React application
5. **Add AI Features**: Integrate actual code analysis AI models

## 🛠️ Development

### Adding New API Endpoints

Edit `simple_server.py` and add new methods to the `SimpleAPIHandler` class:

```python
def do_GET(self):
    if self.path == '/my-new-endpoint':
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"message": "Hello World"}
        self.wfile.write(json.dumps(response).encode())
    else:
        super().do_GET()
```

### Customizing the Dashboard

Edit `dashboard.html` to add new UI components or API tests.

## 🐛 Troubleshooting

**Port already in use?**

- Change `PORT = 8001` in `simple_server.py` to another port
- Update the `API_BASE` variable in `dashboard.html` to match

**Server not responding?**

- Check if Python 3 is installed: `python3 --version`
- Look at server logs: `cat server.log`
- Kill existing processes: `pkill -f simple_server.py`

## 📚 Original Project

This simplified version is based on the comprehensive AI Code Review and Architecture Analysis Platform. The original project includes:

- Full FastAPI backend with SQLAlchemy
- PostgreSQL, Redis, Neo4j databases
- React frontend with Ant Design
- Docker containerization
- AI/ML pipeline integration
- Prometheus monitoring
- Complete CI/CD setup

To explore the full project, navigate to the original directories and use the Docker setup when ready.

---

**🎉 Enjoy your working AI Code Review Platform!**
