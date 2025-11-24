FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092
ENV CONSUL_HOST=consul
ENV CONSUL_PORT=8500
ENV SERVICE_NAME=data-service
ENV SERVICE_PORT=8003
ENV POSTGRES_HOST=postgres
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB=codeinsight
ENV POSTGRES_USER=postgres
ENV NEO4J_HOST=neo4j
ENV NEO4J_PORT=7687
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# 暴露端口
EXPOSE 8003

# 启动服务
CMD ["python", "main.py"]