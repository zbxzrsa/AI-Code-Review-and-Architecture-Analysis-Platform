FROM python:3.11-slim

WORKDIR /app

# Install v1 stable dependencies
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    torch==2.0.1 \
    pylint==3.0.3 \
    bandit==1.7.5 \
    fastapi==0.104.1 \
    uvicorn==0.24.0

# Copy model and config
COPY model/ ./model/
COPY config.yaml ./

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]