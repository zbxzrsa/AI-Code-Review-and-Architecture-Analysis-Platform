FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装语言解析器
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装Tree-sitter和语言解析器
RUN pip install tree-sitter
RUN python -c "from tree_sitter import Language; \
    Language.build_library('/app/build/languages.so', \
    ['vendor/tree-sitter-python', 'vendor/tree-sitter-java', \
    'vendor/tree-sitter-javascript', 'vendor/tree-sitter-go'])"

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092
ENV CONSUL_HOST=consul
ENV CONSUL_PORT=8500
ENV SERVICE_NAME=code-parser-service
ENV SERVICE_PORT=8001

# 暴露端口
EXPOSE 8001

# 启动服务
CMD ["python", "main.py"]