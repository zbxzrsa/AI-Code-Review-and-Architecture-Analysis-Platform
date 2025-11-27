#!/usr/bin/env bash
set -euo pipefail

pull() {
    curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1 || true
    docker compose up -d ollama
    sleep 3
    docker exec $(docker compose ps -q ollama) ollama pull "$1"
}

pull "mistral:7b-instruct"
pull "qwen2:1.5b-instruct"
pull "qwen2:0.5b-instruct"