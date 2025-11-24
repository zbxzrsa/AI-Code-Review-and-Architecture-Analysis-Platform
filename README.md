## Intelligent Code Review and Architecture Analysis Platform

This repository hosts the monorepo for the Intelligent Code Review and Architecture Analysis Platform. The platform delivers a full-stack workflow for AI-assisted code review, architectural health monitoring, GitHub automation, baseline management, and compliance reporting.

The entire codebase, documentation, and user interface are provided in English to simplify global collaboration and auditing.

## Capabilities

- **Code Analysis Workbench** – hybrid static and dynamic analysis pipelines with AI-driven review suggestions, inline remediation, and actionable quality metrics.
- **GitHub Connect** – OAuth-based authentication, repository synchronization, and actionable insights for every pull request or branch.
- **Project Operations Hub** – CRUD dashboards for projects, releases, milestones, and ownership, including relationships to sessions, baselines, and defects.
- **Session Management** – granular tracking for each analysis or test execution, artifact lineage, and intelligent auto-restart workflows.
- **Version Intelligence** – diff visualizations, semantic versioning guardrails, and changelog generation backed by diff2html.
- **Search & Discovery** – federated search with facet filters, AI summaries, and saved views for instant investigations.
- **Baseline Governance** – configurable KPI baselines, deviation detection, risk scoring, and mitigation templates.
- **Settings & Compliance** – tenant-aware configuration, feature toggles, provider credentials, and audit logging.
- **Help & Achievements** – embedded help center, self-serve runbooks, and gamified achievements that reinforce best practices.
- **AI Collaboration** – right-rail chat workspace supporting multiple LLM providers with routing, templating, and conversation storage.

## Architecture Overview

![Architecture Diagram](docs/architecture.svg)

The platform follows an event-driven microservice architecture:

1. API Gateway authenticates requests and emits intent events.
2. Code parsing services generate AST, CFG, and dataflow graphs.
3. AI analysis services run graph reasoning for quality, security, and architectural drift.
4. Persistence services capture snapshots for baselines, diffs, and audit logs.
5. Realtime channels and webhooks push verdicts back to users and partner systems.

## Tech Stack

- **Frontend**: React 18, TypeScript, Ant Design 5, React Router 6, CodeMirror 6, Diff2Html.
- **Backend**: Python 3.11, FastAPI, SQLAlchemy, PostgreSQL, Redis, Neo4j, Celery.
- **AI Layer**: PyTorch, Transformers, DGL, custom ensemble runners.
- **Orchestration**: Docker, Docker Compose, Terraform modules for AWS.
- **Monitoring**: Prometheus, Logstash, custom system monitor scripts.

## Getting Started

### 1. Clone and bootstrap

```bash
git clone <repository-url>
cd intelligent-code-review-and-architecture-analysis-platform
```

### 2. One-command launch (recommended)

The platform now includes a cross-platform one-command runner that builds and starts everything with health checks.

#### Prerequisites
- **Docker** 20.10+ and Docker Compose 2.0+
- **Node.js** 18+ (for the one-command runner)
- **Git** for cloning the repository

#### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd ai-code-review-and-architecture-analysis-platform

# One-command build and start (cross-platform)
npm run build
```

This single command will:
- 🏗️ Build all Docker images
- 🚀 Start all services with health checks
- 📊 Verify service availability
- 🎉 Display success banner with access URLs

#### Alternative Commands

```bash
npm run start      # Start services (if already built)
npm run down       # Stop and remove all services
npm run logs       # Show service logs
npm run health     # Check service health
npm run doctor     # System diagnostics
npm run clean      # Clean up Docker resources
```

#### Environment Profiles

```bash
# Development (default)
npm run build

# Production
PROFILE=prod npm run build

# Staging
PROFILE=staging npm run build
```

### 3. Manual dev setup

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd ../frontend
npm install
npm start
```

### 4. Useful endpoints

- Frontend: http://localhost:3000
- API Gateway: http://localhost:8000
- OpenAPI Docs: http://localhost:8000/docs
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Neo4j Browser: http://localhost:7474

## Repository Layout

```
.
├── backend/                 FastAPI services, AI pipelines, background workers
├── frontend/                React monorepo for the platform UI
├── docs/                    Architecture, APIs, UX guidelines, rollout plans
├── docker/                  Container images and compose definitions per service
├── infra/terraform/         Provisioning modules for AWS
├── tools/                   Quality gates, monitoring scripts, cleanup utilities
├── tests/                   E2E, performance, and integration suites
└── README.md                You are here
```

## Contributing

1. Create a feature branch from `main`.
2. Run `npm run test --prefix frontend` and `pytest` for backend changes.
3. Provide screenshots or terminal recordings for UI updates.
4. Submit a pull request with a concise summary and detailed testing evidence.

## License

Released under the [MIT License](LICENSE).