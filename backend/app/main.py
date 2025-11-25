from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from pathlib import Path
import os

from app.api.endpoints import ai_analysis, sessions, versions, search, baselines, auth
from app.api import code_converter  # 导入代码转换API
from app.api import config_management  # 导入配置管理API
from app.core.config_manager import init_config_system, get_app_config
from app.core.dynamic_config import init_dynamic_config_system
try:
    from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore
    _PROM_AVAILABLE = True
except Exception:
    _PROM_AVAILABLE = False

import socket
try:
    import logstash
    LOGSTASH_AVAILABLE = True
except Exception:
    LOGSTASH_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if LOGSTASH_AVAILABLE:
    try:
        # 将日志发送到Logstash（ELK）
        handler = logstash.TCPLogstashHandler(
            host='logstash',
            port=5000,
            version=1
        )
        logger.addHandler(handler)
        logger.info("Logstash handler attached", extra={"service": "backend", "host": socket.gethostname()})
    except Exception as e:
        logger.warning(f"Failed to attach Logstash handler: {e}", extra={"service": "backend"})

# 创建FastAPI应用
app = FastAPI(
    title="智能代码审查与架构分析平台",
    description="提供代码向量化、缺陷检测、架构模式识别和代码相似度计算等AI分析服务",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 暴露Prometheus指标
if '_PROM_AVAILABLE' in globals() and _PROM_AVAILABLE:
    try:
        Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    except Exception as e:
        logger.warning(f"Prometheus instrumentation disabled: {e}")


# 添加请求处理时间中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 注册路由
app.include_router(ai_analysis.router)
app.include_router(sessions.router)
app.include_router(versions.router)
app.include_router(search.router)
app.include_router(baselines.router)
app.include_router(auth.router)
# 包含代码转换API路由
code_converter.include_router(app)
# 包含配置管理API路由
app.include_router(config_management.router, prefix="/api")

# 添加健康检查路由
try:
    from app.api import health
    app.include_router(health.router)
except ImportError:
    pass

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ai-analysis-api"}

# 启动事件：初始化配置系统与动态配置
@app.on_event("startup")
async def startup_event():
    try:
        # 计算配置目录（backend/config）
        config_dir = Path(__file__).resolve().parents[1].parent / "config"
        await init_config_system(str(config_dir))
        app_config = get_app_config()
        env_name = getattr(app_config.environment, "value", str(app_config.environment))
        logger.info(f"Configuration initialized: env={env_name}, app={app_config.app_name}")

        # 可选：初始化远程动态配置（通过环境变量控制）
        remote_url = os.environ.get("REMOTE_CONFIG_URL")
        instance_id = os.environ.get("INSTANCE_ID")
        auth_token = os.environ.get("CONFIG_AUTH_TOKEN")
        try:
            await init_dynamic_config_system(remote_url, instance_id, auth_token)
            logger.info("Dynamic config system started")
        except Exception as e:
            logger.warning(f"Dynamic config initialization skipped/failed: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize configuration systems: {e}")

# 异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"全局异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后重试"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)