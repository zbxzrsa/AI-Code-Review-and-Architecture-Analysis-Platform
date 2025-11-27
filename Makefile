.PHONY: dev up down logs lint test help install install-dev install-frontend clean lint format type-check test test-coverage test-integration test-e2e security-scan docker-up docker-down docker-build docker-logs pre-commit setup-dev start-dev stop-dev health-check docs-serve docs-build migrate-db seed-db backup-db restore-db performance-test security-audit dependency-check update-deps clean-all

# One-command development
dev: up

up: ./scripts/bootstrap.sh
	docker compose up --remove-orphans

down:
	docker compose down -v

logs:
	docker compose logs -f

lint:
	cd backend && ruff check . || true && cd ../frontend && npx eslint . --ext .ts,.tsx

test:
	cd backend && pytest -q || true && cd ../frontend && npm test --if-present

# Default target
help:
	@echo "AI Code Review Platform - Development Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install          Install backend dependencies"
	@echo "  install-dev      Install backend dev dependencies"
	@echo "  install-frontend Install frontend dependencies"
	@echo "  setup-dev        Complete development environment setup"
	@echo "  clean            Clean Python cache and build artifacts"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run linting (ruff, flake8, eslint)"
	@echo "  format           Format code (black, isort, prettier)"
	@echo "  type-check       Run type checking (mypy, tsc)"
	@echo "  pre-commit       Run all pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run unit tests"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  test-integration Run integration tests"
	@echo "  test-e2e         Run end-to-end tests"
	@echo ""
	@echo "Security:"
	@echo "  security-scan    Run security scans (bandit, safety, npm audit)"
	@echo "  security-audit   Comprehensive security audit"
	@echo "  dependency-check Check for vulnerable dependencies"
	@echo ""
	@echo "Docker:"
	@echo "  docker-up        Start all services with Docker Compose"
	@echo "  docker-down      Stop all services"
	@echo "  docker-build     Build all Docker images"
	@echo "  docker-logs      Show service logs"
	@echo "  health-check     Check service health"
	@echo ""
	@echo "Database:"
	@echo "  migrate-db       Run database migrations"
	@echo "  seed-db          Seed database with initial data"
	@echo "  backup-db        Backup databases"
	@echo "  restore-db       Restore databases from backup"
	@echo ""
	@echo "Development:"
	@echo "  start-dev        Start development servers"
	@echo "  stop-dev         Stop development servers"
	@echo "  performance-test Run performance tests"
	@echo "  docs-serve       Serve documentation locally"
	@echo "  docs-build       Build documentation"
	@echo ""
	@echo "Maintenance:"
	@echo "  update-deps      Update dependencies"
	@echo "  clean-all        Clean everything"

# Setup & Installation
install:
	cd backend && pip install -e .

install-dev:
	cd backend && pip install -e ".[dev]"
	pip install pre-commit

install-frontend:
	cd frontend && npm ci

setup-dev: install-dev install-frontend
	pre-commit install
	cp backend/.env.example .env
	@echo "Development environment setup complete!"
	@echo "Edit .env file with your configuration"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage coverage.xml

clean-all: clean
	cd frontend && rm -rf node_modules/ build/ dist/
	docker system prune -f

# Code Quality
lint:
	@echo "Running Python linting..."
	cd backend && ruff check .
	@echo "Running frontend linting..."
	cd frontend && npm run lint --if-present

format:
	@echo "Formatting Python code..."
	cd backend && black .
	cd backend && isort .
	@echo "Formatting frontend code..."
	cd frontend && npm run format --if-present

type-check:
	@echo "Running Python type checking..."
	cd backend && mypy .
	@echo "Running frontend type checking..."
	cd frontend && npm run type-check --if-present

pre-commit:
	pre-commit run --all-files

# Testing
test:
	@echo "Running backend unit tests..."
	cd backend && pytest tests/unit/ -v
	@echo "Running frontend tests..."
	cd frontend && npm test --if-present

test-coverage:
	@echo "Running tests with coverage..."
	cd backend && pytest --cov=app --cov-report=html --cov-report=term

test-integration:
	@echo "Running integration tests..."
	cd backend && pytest tests/integration/ -v

test-e2e:
	@echo "Running end-to-end tests..."
	cd backend && pytest tests/e2e/ -v

# Security
security-scan:
	@echo "Running Python security scan..."
	cd backend && bandit -r app/ -f json -o bandit-report.json
	cd backend && safety check --json --output safety-report.json
	@echo "Running frontend security scan..."
	cd frontend && npm audit --audit-level moderate --json > npm-audit.json || true

security-audit: security-scan
	@echo "Running comprehensive security audit..."
	docker run --rm -v $(PWD):/app -w /app aquasec/trivy:latest fs --format json --output trivy-report.json .
	@echo "Security audit complete. Check reports: bandit-report.json, safety-report.json, npm-audit.json, trivy-report.json"

dependency-check:
	@echo "Checking for vulnerable dependencies..."
	cd backend && pip-audit --format=json --output=pip-audit.json || true
	cd frontend && npm audit --audit-level moderate

# Docker
docker-up:
	docker compose up -d
	@echo "Waiting for services to be healthy..."
	sleep 10
	$(MAKE) health-check

docker-down:
	docker compose down

docker-build:
	docker compose build

docker-logs:
	docker compose logs -f

health-check:
	@echo "Checking service health..."
	@curl -f http://localhost:8000/health || echo "Backend health check failed"
	@curl -f http://localhost:3000 || echo "Frontend health check failed"
	@docker compose ps

# Database
migrate-db:
	@echo "Running database migrations..."
	cd backend && alembic upgrade head

seed-db:
	@echo "Seeding database..."
	cd backend && python scripts/seed_db.py || echo "Seed script not found"

backup-db:
	@echo "Creating database backups..."
	mkdir -p backups
	docker exec postgres pg_dump -U postgres codeinsight > backups/postgres_$(shell date +%Y%m%d_%H%M%S).sql
	docker exec neo4j neo4j-admin database dump neo4j --to-path=/tmp/backup || true
	docker cp neo4j:/tmp/backup/neo4j.dump backups/neo4j_$(shell date +%Y%m%d_%H%M%S).dump || true

restore-db:
	@echo "Restoring databases from backup..."
	@read -p "Enter backup file path: " backup_path; \
	docker exec -i postgres psql -U postgres codeinsight < $$backup_path

# Development
start-dev: docker-up
	@echo "Development environment started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

stop-dev: docker-down

performance-test:
	@echo "Running performance tests..."
	cd backend && pytest tests/perf/ -v
	cd tests && k6 run perf/k6_script.js || echo "k6 not available"

# Documentation
docs-serve:
	@echo "Starting documentation server..."
	cd docs && python -m http.server 8080 || echo "Docs not found"

docs-build:
	@echo "Building documentation..."
	cd docs && make html || echo "Docs build not configured"

# Maintenance
update-deps:
	@echo "Updating Python dependencies..."
	cd backend && pip-compile requirements.in || echo "pip-compile not available"
	@echo "Updating frontend dependencies..."
	cd frontend && npm update

# CI/CD helpers
ci-test: lint type-check test-coverage security-scan
	@echo "CI tests completed successfully!"

ci-build: docker-build
	@echo "CI build completed successfully!"

deploy-staging:
	@echo "Deploying to staging..."
	ansible-playbook -i ansible/inventory/staging ansible/site.yml

deploy-prod:
	@echo "Deploying to production..."
	ansible-playbook -i ansible/inventory/prod ansible/site.yml --extra-vars "confirm_deployment=true"

prod:
	docker compose -f docker-compose.prod.yml up -d

prod-health:
	@echo "Checking production services..."
	curl -f http://localhost:8000/healthz || echo "Backend health check failed"
	curl -f http://localhost:3000 || echo "Frontend health check failed"
	curl -f http://localhost:8000/metrics || echo "Metrics endpoint failed"