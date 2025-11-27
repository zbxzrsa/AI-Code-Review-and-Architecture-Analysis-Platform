from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import time
import logging
import json
import asyncio
from pathlib import Path
import os
from backend.ai.router import pick
from backend.ai.metrics import structured_logger, generate_request_id, get_metrics

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
origins = [o for o in os.getenv("CORS_ALLOWED_ORIGINS","").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or [""],
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

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/ai/review")
async def ai_review(req: Request):
    """AI review endpoint with structured mode support"""
    body = await req.json()
    text = body.get("text","")
    context = body.get("context", [])
    meta = body.get("meta", {})
    
    # Determine if structured mode is enabled
    structured_mode = (
        req.headers.get("X-Structured-JSON", "").lower() == "true" or
        os.getenv("AI_STRUCTURED_MODE", "false").lower() == "true"
    )
    
    ch = req.headers.get("X-AI-Channel") or os.getenv("AI_CHANNEL","stable")
    engine = pick(ch)
    
    if structured_mode and hasattr(engine.review, '__code__') and 'text, context, meta' in engine.review.__code__.co_varnames:
        # Use new structured interface
        out = engine.review(text=text, context=context, meta=meta)
        return {"channel": ch, "structured": True, "result": out}
    else:
        # Use legacy interface
        prompt = f"Review this code change and suggest improvements:\n{text}"
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        out = engine.review(prompt)
        return {"channel": ch, "structured": False, "result": out}

async def stream_ollama_response(engine, prompt: str, structured_mode: bool = False, text: str = "", context: list = None, meta: dict = None):
    """Stream response from Ollama with heartbeat and structured mode support"""
    import requests
    
    if structured_mode and hasattr(engine.review, '__code__') and 'text, context, meta' in engine.review.__code__.co_varnames:
        # For structured mode, we need to handle differently
        # Since structured output isn't streaming-friendly, we'll run it and stream the result
        try:
            result = engine.review(text=text, context=context, meta=meta)
            yield f"data: {json.dumps({'chunk': 'Processing structured review...'})}\n\n"
            yield f"data: {json.dumps({'structured_result': result, 'done': True})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
            return
    
    # Legacy streaming mode
    payload = {
        "model": engine.CFG["model"],
        "prompt": f'{engine.CFG["system"]}\n\n{prompt}',
        "options": {
            "temperature": engine.CFG["temperature"],
            "top_p": engine.CFG["top_p"]
        },
        "stream": True
    }
    
    try:
        with requests.post("http://ollama:11434/api/generate", 
                         json=payload, 
                         stream=True, 
                         timeout=120) as r:
            r.raise_for_status()
            
            last_heartbeat = time.time()
            
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            chunk = data["response"]
                            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                            last_heartbeat = time.time()
                        
                        # Send heartbeat every 15 seconds
                        if time.time() - last_heartbeat > 15:
                            yield f": heartbeat\n\n"
                            last_heartbeat = time.time()
                            
                    except json.JSONDecodeError:
                        continue
                        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/ai/review/stream")
async def ai_review_stream(req: Request):
    """Streaming AI review endpoint with SSE and structured mode support"""
    request_id = getattr(req.state, 'request_id', generate_request_id())
    body = await req.json()
    text = body.get("text","")
    context = body.get("context", [])
    meta = body.get("meta", {})
    
    # Determine if structured mode is enabled
    structured_mode = (
        req.headers.get("X-Structured-JSON", "").lower() == "true" or
        os.getenv("AI_STRUCTURED_MODE", "false").lower() == "true"
    )
    
    ch = req.headers.get("X-AI-Channel") or os.getenv("AI_CHANNEL","stable")
    engine = pick(ch)
    
    prompt = f"Review this code change and suggest improvements:\n{text}"
    if context:
        prompt = f"Context: {context}\n\n{prompt}"
    
    return StreamingResponse(
        stream_ollama_response(engine, prompt, structured_mode, text, context, meta),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-Request-ID": request_id,
            "X-Structured-Mode": str(structured_mode).lower()
        }
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return get_metrics()

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