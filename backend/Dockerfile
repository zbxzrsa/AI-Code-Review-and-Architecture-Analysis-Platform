# 多阶段构建 - 基础镜像
FROM python:3.11-slim as base

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 设置工作目录
WORKDIR /app

# 安装系统依赖（优化层大小）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 复制依赖文件
COPY requirements.txt .

# 优化Python依赖安装
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# 开发阶段
FROM base as development

# 安装开发依赖
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# 复制源代码
COPY . .

# 暴露端口
EXPOSE 8000

# 开发环境启动命令
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# 生产阶段
FROM base as production

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 复制源代码
COPY --chown=appuser:appuser . .

# 切换到非root用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 生产环境启动命令
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# 边缘计算优化版本
FROM python:3.11-alpine as edge

WORKDIR /app

# 安装最小依赖
RUN apk add --no-cache gcc musl-dev

# 复制核心依赖
COPY requirements-edge.txt .
RUN pip install --no-cache-dir -r requirements-edge.txt

# 复制核心模块
COPY app/core ./app/core
COPY app/services/translation ./app/services/translation
COPY app/services/plugins ./app/services/plugins
COPY app/main.py ./app/

# 边缘环境启动命令
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]