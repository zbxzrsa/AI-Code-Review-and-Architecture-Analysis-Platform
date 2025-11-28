FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app

# Install v2 experimental dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama and Python packages
RUN pip3 install --no-cache-dir \
    ollama==0.1.7 \
    bandit==1.7.5 \
    eslint==8.45.0 \
    fastapi==0.104.1 \
    uvicorn==0.24.0

# Download and setup llama2:7b model
RUN ollama pull llama2:7b

# Copy config
COPY config.yaml ./

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["ollama", "serve", "--host", "0.0.0.0", "--port", "8000"]