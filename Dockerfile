# 代码解析服务 Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY ./requirements.txt /app/requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装Tree-sitter及语言库
RUN pip install tree-sitter

# 克隆并构建Tree-sitter语言库
RUN mkdir -p /app/tree-sitter-libs && \
    cd /app/tree-sitter-libs && \
    git clone https://github.com/tree-sitter/tree-sitter-python.git && \
    git clone https://github.com/tree-sitter/tree-sitter-java.git && \
    git clone https://github.com/tree-sitter/tree-sitter-javascript.git && \
    git clone https://github.com/tree-sitter/tree-sitter-go.git

# 复制应用代码
COPY ./backend /app/backend

# 设置环境变量
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]