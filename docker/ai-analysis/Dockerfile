FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092
ENV CONSUL_HOST=consul
ENV CONSUL_PORT=8500
ENV SERVICE_NAME=ai-analysis-service
ENV SERVICE_PORT=8002
ENV MODEL_PATH=/app/models

# 暴露端口
EXPOSE 8002

# 启动服务
CMD ["python", "main.py"]