#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[ -f .env ] || cp .env.example .env

mkdir -p ai_models/stable ai_models/next ai_models/legacy

docker compose pull || true
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose build

bash ./scripts/models_pull.sh || true