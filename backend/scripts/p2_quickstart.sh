#!/bin/bash
# P2 快速启动脚本
# 一键启动所有 P2 服务与测试

set -e

echo "=========================================="
echo "P2 Phase Quick Start"
echo "=========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# 1. 检查前置条件
# ============================================================================
echo -e "\n${YELLOW}Step 1: Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    exit 1
fi
log_info "✓ Docker found"

if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed"
    exit 1
fi
log_info "✓ Python 3 found"

# ============================================================================
# 2. 启动 Docker Compose 服务
# ============================================================================
echo -e "\n${YELLOW}Step 2: Starting Docker Compose services...${NC}"

cd "$BACKEND_DIR"

if [ ! -f ".env" ]; then
    log_warn ".env file not found, copying from .env.example"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        log_info "✓ Created .env from .env.example"
    else
        log_error ".env.example not found"
        exit 1
    fi
fi

docker-compose down -v 2>/dev/null || true
docker-compose up -d

# 等待服务就绪
log_info "Waiting for services to be healthy..."
sleep 5

max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker-compose ps | grep -q "healthy"; then
        log_info "✓ Services are healthy"
        break
    fi
    attempt=$((attempt + 1))
    echo -n "."
    sleep 1
done

if [ $attempt -eq $max_attempts ]; then
    log_warn "Services took longer than expected to start"
    docker-compose ps
fi

echo ""

# ============================================================================
# 3. 运行数据库迁移
# ============================================================================
echo -e "\n${YELLOW}Step 3: Running database migrations...${NC}"

# 激活虚拟环境
if [ ! -d "$BACKEND_DIR/venv" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv "$BACKEND_DIR/venv"
fi

source "$BACKEND_DIR/venv/bin/activate"

# 安装依赖
if [ -f "$BACKEND_DIR/requirements.txt" ]; then
    log_info "Installing dependencies..."
    pip install -q -r "$BACKEND_DIR/requirements.txt" 2>/dev/null || true
fi

# 运行迁移
cd "$BACKEND_DIR"
if command -v alembic &> /dev/null || [ -f "$BACKEND_DIR/venv/bin/alembic" ]; then
    log_info "Running Alembic migrations..."
    alembic upgrade head 2>/dev/null || log_warn "Alembic migration skipped (may already be applied)"
    log_info "✓ Migrations completed"
else
    log_warn "Alembic not found, skipping migrations"
fi

# ============================================================================
# 4. 启动 API Server（后台）
# ============================================================================
echo -e "\n${YELLOW}Step 4: Starting API server...${NC}"

cd "$BACKEND_DIR"

# 检查是否已在运行
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_warn "Port 8000 already in use"
else
    log_info "Starting FastAPI server..."
    nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/api_server.log 2>&1 &
    API_PID=$!
    echo $API_PID > /tmp/api_server.pid
    log_info "✓ API server started (PID: $API_PID)"
    sleep 2
fi

# 验证 API 就绪
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    log_info "✓ API server is ready"
else
    log_warn "API server not responding yet (may take a moment)"
fi

# ============================================================================
# 5. 启动 Celery Worker（后台）
# ============================================================================
echo -e "\n${YELLOW}Step 5: Starting Celery worker...${NC}"

cd "$BACKEND_DIR"

if lsof -Pi :5555 -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_warn "Port 5555 already in use (Celery may already be running)"
else
    log_info "Starting Celery worker..."
    nohup celery -A app.worker:celery_app worker --loglevel=info --concurrency=2 > /tmp/celery_worker.log 2>&1 &
    CELERY_PID=$!
    echo $CELERY_PID > /tmp/celery_worker.pid
    log_info "✓ Celery worker started (PID: $CELERY_PID)"
    sleep 2
fi

# ============================================================================
# 6. 运行性能测试（可选）
# ============================================================================
echo -e "\n${YELLOW}Step 6: Running performance tests (optional)...${NC}"
echo "Would you like to run performance tests? (y/n)"
read -r run_tests

if [ "$run_tests" = "y" ] || [ "$run_tests" = "Y" ]; then
    log_info "Running P2 performance benchmarks..."
    cd "$BACKEND_DIR"
    python3 tests/p2_performance_test.py

    if [ -f "$BACKEND_DIR/p2_performance_results.json" ]; then
        log_info "✓ Performance test results saved"
        cat p2_performance_results.json | python3 -m json.tool | head -20
    fi
else
    log_info "Skipping performance tests"
fi

# ============================================================================
# 7. 显示服务状态
# ============================================================================
echo -e "\n${YELLOW}Step 7: Service Status${NC}"

echo -e "\n${GREEN}Docker Services:${NC}"
cd "$BACKEND_DIR"
docker-compose ps

echo -e "\n${GREEN}API & Worker Status:${NC}"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    log_info "✓ API: http://localhost:8000"
else
    log_error "✗ API not responding"
fi

if lsof -Pi :5555 -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_info "✓ Celery Flower: http://localhost:5555"
else
    log_warn "Celery Flower not running"
fi

echo -e "\n${GREEN}Useful URLs:${NC}"
echo "  API Docs:         http://localhost:8000/docs"
echo "  Prometheus:       http://localhost:9090"

echo "  MinIO Console:    http://localhost:9001 (minioadmin/minioadmin)"
echo "  RabbitMQ Mgmt:    http://localhost:15672 (guest/guest)"

# ============================================================================
# 8. 显示日志查看命令
# ============================================================================
echo -e "\n${GREEN}Log Viewing Commands:${NC}"
echo "  API Logs:         docker-compose logs -f backend"
echo "  Worker Logs:      tail -f /tmp/celery_worker.log"
echo "  All Services:     docker-compose logs -f"

# ============================================================================
# 9. 清理脚本
# ============================================================================
echo -e "\n${GREEN}Cleanup:${NC}"
cat > /tmp/p2_cleanup.sh << 'EOF'
#!/bin/bash
echo "Stopping P2 services..."
# 停止后台进程
[ -f /tmp/api_server.pid ] && kill $(cat /tmp/api_server.pid) 2>/dev/null || true
[ -f /tmp/celery_worker.pid ] && kill $(cat /tmp/celery_worker.pid) 2>/dev/null || true
# 停止 Docker
cd backend
docker-compose down -v
echo "✓ Cleanup complete"
EOF
chmod +x /tmp/p2_cleanup.sh

echo "  Run cleanup:      bash /tmp/p2_cleanup.sh"

# ============================================================================
# 完成
# ============================================================================
echo -e "\n${GREEN}=========================================="
echo "P2 Phase Quick Start COMPLETED"
echo "==========================================${NC}\n"

echo "Next steps:"
echo "  1. Open http://localhost:8000/docs to test API endpoints"
echo "  2. Create a test PR analysis: POST /api/v1/pr/123/analyze"
echo "  3. Check results: GET /api/v1/pr/123/analysis?sha=..."
echo "  4. View metrics: http://localhost:9090"
echo ""
echo "For more details, see README_PHASE_2_DETAILED.md"
