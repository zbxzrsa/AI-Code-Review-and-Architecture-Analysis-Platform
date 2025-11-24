from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import numpy as np
import time
from datetime import datetime
from prometheus_client import Counter, Histogram

# 导入限流和缓存依赖
from app.core.dependencies import rate_limiter, get_cache, Cache
from ai.inference.ensemble import EnsembleAggregator, ExpertOutput
from ai.inference.postprocessing import RuleEngine, ContextFilter, ConfidenceCalibrator
from app.tasks.analysis_tasks import analyze_file_with_glm, batch_glm_analysis

router = APIRouter(prefix="/api/v1", tags=["AI Analysis"])

# 请求和响应模型定义
class CodeEmbedRequest(BaseModel):
    code_structure: Dict[str, Any] = Field(..., description="解析后的代码结构")
    language: str = Field(..., description="编程语言")

class CodeEmbedResponse(BaseModel):
    vector: List[float] = Field(..., description="768维代码向量")
    processing_time: float = Field(..., description="处理时间(秒)")

class DefectAnalysisRequest(BaseModel):
    code: Optional[str] = Field(None, description="原始代码")
    vector: Optional[List[float]] = Field(None, description="代码向量")
    language: str = Field(..., description="编程语言")

class DefectLocation(BaseModel):
    start_line: int = Field(..., description="缺陷开始行")
    end_line: int = Field(..., description="缺陷结束行")
    start_column: Optional[int] = Field(None, description="缺陷开始列")
    end_column: Optional[int] = Field(None, description="缺陷结束列")

class Defect(BaseModel):
    defect_type: str = Field(..., description="缺陷类型")
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)
    location: DefectLocation = Field(..., description="位置信息")
    description: Optional[str] = Field(None, description="缺陷描述")

class DefectAnalysisResponse(BaseModel):
    defects: List[Defect] = Field(..., description="检测到的缺陷列表")
    processing_time: float = Field(..., description="处理时间(秒)")
    uncertainty: Optional[float] = Field(None, description="缺陷集合的不确定性估计(0-1)")

class ArchitectureAnalysisRequest(BaseModel):
    project_structure: Optional[Dict[str, Any]] = Field(None, description="项目文件结构")
    vectors: Optional[Dict[str, List[float]]] = Field(None, description="文件路径到代码向量的映射")

class ArchitecturePattern(BaseModel):
    pattern_name: str = Field(..., description="架构模式名称")
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)
    components: List[str] = Field(..., description="相关组件")
    description: Optional[str] = Field(None, description="模式描述")

class ArchitectureSmell(BaseModel):
    smell_type: str = Field(..., description="架构坏味类型")
    severity: str = Field(..., description="严重程度", pattern="^(low|medium|high|critical)$")
    affected_components: List[str] = Field(..., description="受影响的组件")
    description: Optional[str] = Field(None, description="坏味描述")
    recommendation: Optional[str] = Field(None, description="改进建议")

class ArchitectureAnalysisResponse(BaseModel):
    patterns: List[ArchitecturePattern] = Field(..., description="检测到的架构模式")
    smells: List[ArchitectureSmell] = Field(..., description="检测到的架构坏味")
    processing_time: float = Field(..., description="处理时间(秒)")

class SimilarityRequest(BaseModel):
    code1: str = Field(..., description="第一段代码")
    code2: str = Field(..., description="第二段代码")
    language: str = Field(..., description="编程语言")
    detailed_analysis: bool = Field(False, description="是否返回详细分析")

class SimilarSegment(BaseModel):
    code1_start_line: int = Field(..., description="代码1中相似段开始行")
    code1_end_line: int = Field(..., description="代码1中相似段结束行")
    code2_start_line: int = Field(..., description="代码2中相似段开始行")
    code2_end_line: int = Field(..., description="代码2中相似段结束行")
    similarity_score: float = Field(..., description="该段的相似度分数", ge=0.0, le=1.0)

class SimilarityResponse(BaseModel):
    similarity_score: float = Field(..., description="总体相似度分数", ge=0.0, le=1.0)
    similar_segments: Optional[List[SimilarSegment]] = Field(None, description="相似代码段")
    processing_time: float = Field(..., description="处理时间(秒)")

# 优化的AI模型服务
class AIModelService:
    _embedding_cache = {}
    _model_instances = {}
    
    @classmethod
    async def embed_code(cls, code_structure: Dict[str, Any], language: str) -> List[float]:
        """优化的代码向量化，支持缓存和批处理"""
        # 生成缓存键
        import hashlib
        cache_key = hashlib.md5(
            f"{language}:{str(sorted(code_structure.items()))}".encode()
        ).hexdigest()
        
        # 检查缓存
        if cache_key in cls._embedding_cache:
            return cls._embedding_cache[cache_key]
        
        # 模拟向量化（实际实现中会调用真实模型）
        vector = list(np.random.rand(768).astype(float))
        
        # 缓存结果（LRU策略，限制缓存大小）
        if len(cls._embedding_cache) > 1000:
            # 移除最旧的条目
            oldest_key = next(iter(cls._embedding_cache))
            del cls._embedding_cache[oldest_key]
        
        cls._embedding_cache[cache_key] = vector
        return vector
    
    @classmethod
    async def detect_defects(cls, code: Optional[str] = None, vector: Optional[List[float]] = None, language: str = None, model_version: Optional[str] = None) -> List[Defect]:
        """优化的缺陷检测，支持并行处理和智能过滤"""
        import hashlib
        from concurrent.futures import ThreadPoolExecutor
        
        # 生成缓存键
        cache_data = code or (str(vector) if vector else "")
        cache_key = hashlib.md5(f"{language}:{cache_data}:{model_version}".encode()).hexdigest()
        
        # 检查缓存
        if not hasattr(cls, '_defect_cache'):
            cls._defect_cache = {}
        if cache_key in cls._defect_cache:
            return cls._defect_cache[cache_key]
        
        # 模拟并行缺陷检测
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            if code and "password" in code.lower():
                futures.append(executor.submit(cls._detect_security_issues, code))
            if code:
                futures.append(executor.submit(cls._detect_quality_issues, code))
                futures.append(executor.submit(cls._detect_performance_issues, code))
            
            defects = []
            for future in futures:
                try:
                    defects.extend(future.result(timeout=5))
                except Exception:
                    continue
        
        # 缓存结果
        if len(cls._defect_cache) > 500:
            oldest_key = next(iter(cls._defect_cache))
            del cls._defect_cache[oldest_key]
        
        cls._defect_cache[cache_key] = defects
        return defects
    
    @staticmethod
    def _detect_security_issues(code: str) -> List[Defect]:
        """检测安全问题"""
        defects = []
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if 'password' in line.lower() and '=' in line:
                defects.append(Defect(
                    defect_type="hardcoded_password",
                    confidence=0.9,
                    location=DefectLocation(start_line=i, end_line=i),
                    description="硬编码密码存在安全风险"
                ))
        return defects
    
    @staticmethod
    def _detect_quality_issues(code: str) -> List[Defect]:
        """检测代码质量问题"""
        defects = []
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if len(line.strip()) > 120:
                defects.append(Defect(
                    defect_type="line_too_long",
                    confidence=0.7,
                    location=DefectLocation(start_line=i, end_line=i),
                    description="代码行过长，影响可读性"
                ))
        return defects
    
    @staticmethod
    def _detect_performance_issues(code: str) -> List[Defect]:
        """检测性能问题"""
        defects = []
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if 'for' in line and 'range(' in line and 'len(' in line:
                defects.append(Defect(
                    defect_type="inefficient_loop",
                    confidence=0.8,
                    location=DefectLocation(start_line=i, end_line=i),
                    description="循环中重复计算长度，影响性能"
                ))
        return defects
    
    @staticmethod
    async def analyze_architecture(project_structure: Optional[Dict[str, Any]] = None, 
                                  vectors: Optional[Dict[str, List[float]]] = None,
                                  model_version: Optional[str] = None) -> tuple:
        """模拟架构分析"""
        # 实际实现中，这里会调用AI模型进行架构分析
        # 这里返回模拟数据作为示例
        patterns = [
            ArchitecturePattern(
                pattern_name="MVC",
                confidence=0.88,
                components=["controllers/", "models/", "views/"],
                description="Model-View-Controller架构模式"
            ),
            ArchitecturePattern(
                pattern_name="Repository",
                confidence=0.76,
                components=["repositories/", "services/"],
                description="仓储模式"
            )
        ]
        
        smells = [
            ArchitectureSmell(
                smell_type="cyclic_dependency",
                severity="high",
                affected_components=["service/UserService.java", "service/AuthService.java"],
                description="服务之间存在循环依赖",
                recommendation="考虑引入中介者模式或事件驱动架构"
            )
        ]
        
        return patterns, smells


# GLM-4.6 Specific Endpoints

class GLMAnalysisRequest(BaseModel):
    code: str = Field(..., description="Code to analyze")
    language: str = Field(default="python", description="Programming language")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on")
    file_path: Optional[str] = Field(None, description="File path for context")

class GLMAnalysisResponse(BaseModel):
    success: bool = Field(..., description="Analysis completed successfully")
    file_path: str = Field(..., description="Analyzed file path")
    analysis_type: str = Field(..., description="Type of analysis performed")
    issues_found: int = Field(..., description="Number of issues found")
    overall_score: int = Field(..., description="Overall code quality score")
    review_data: Dict[str, Any] = Field(..., description="Detailed review results")
    model_used: str = Field(..., description="AI model used")
    tokens_consumed: int = Field(..., description="Tokens consumed")
    duration: float = Field(..., description="Analysis duration in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")


@router.post("/glm/analyze", response_model=GLMAnalysisResponse)
async def analyze_code_with_glm(
    request: GLMAnalysisRequest,
    background_tasks: BackgroundTasks,
    ai_service: Any = Depends(get_ai_model_service),  # Will be GLM service
    cache: Cache = Depends(get_cache),
    _: None = Depends(rate_limiter(max_requests=10, window_seconds=60))
) -> GLMAnalysisResponse:
    """
    Analyze code using GLM-4.6:cloud model
    
    This endpoint provides direct access to GLM-4.6 analysis capabilities
    """
    try:
        # Log the analysis request
        logger.info(f"GLM analysis request for {request.file_path or 'unknown file'}")
        
        # Trigger analysis in background
        task_result = await analyze_file_with_glm(
            request.file_path or "unknown",
            request.code,
            request.language,
            request.focus_areas
        )
        
        if task_result.get("status") == "completed":
            return GLMAnalysisResponse(
                success=True,
                file_path=request.file_path or "unknown",
                analysis_type=task_result.get("analysis_type", "glm_ai_review"),
                issues_found=task_result.get("issues_found", 0),
                overall_score=task_result.get("review_data", {}).get("overall_score", 75),
                review_data=task_result.get("review_data", {}),
                model_used=task_result.get("review_data", {}).get("model", "glm-4.6:cloud"),
                tokens_consumed=task_result.get("review_data", {}).get("tokens_used", 0),
                duration=task_result.get("review_data", {}).get("duration", 0)
            )
        else:
            return GLMAnalysisResponse(
                success=False,
                file_path=request.file_path or "unknown",
                analysis_type="glm_ai_review",
                issues_found=0,
                overall_score=0,
                review_data={},
                model_used="glm-4.6:cloud",
                error=task_result.get("error", "Analysis failed")
            )
            
    except Exception as e:
        logger.exception(f"GLM analysis endpoint error: {str(e)}")
        API_ERRORS.labels(endpoint="/glm/analyze").inc()
        return GLMAnalysisResponse(
            success=False,
            file_path=request.file_path or "unknown",
            analysis_type="glm_ai_review",
            issues_found=0,
            overall_score=0,
            review_data={},
            model_used="glm-4.6:cloud",
            error=str(e)
        )


@router.post("/glm/batch", response_model=Dict[str, Any])
async def batch_glm_analysis(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    ai_service: Any = Depends(get_ai_model_service),
    cache: Cache = Depends(get_cache),
    _: None = Depends(rate_limiter(max_requests=5, window_seconds=60))
) -> Dict[str, Any]:
    """
    Run batch GLM analysis for multiple files
    
    Expected request format:
    {
        "session_id": 123,
        "file_paths": ["file1.py", "file2.js"],
        "focus_areas": ["security", "performance"]
    }
    """
    try:
        session_id = request.get("session_id")
        file_paths = request.get("file_paths", [])
        focus_areas = request.get("focus_areas", [])
        
        if not session_id or not file_paths:
            return {
                "success": False,
                "error": "session_id and file_paths are required"
            }
        
        logger.info(f"GLM batch analysis request for session {session_id}, {len(file_paths)} files")
        
        # Trigger batch analysis
        task_result = await batch_glm_analysis(session_id, file_paths, focus_areas)
        
        return {
            "success": True,
            "session_id": session_id,
            "analysis_type": "glm_batch_review",
            "result": task_result
        }
        
    except Exception as e:
        logger.exception(f"GLM batch analysis endpoint error: {str(e)}")
        API_ERRORS.labels(endpoint="/glm/batch").inc()
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/glm/status", response_model=Dict[str, Any])
async def glm_status(
    ai_service: Any = Depends(get_ai_model_service)
) -> Dict[str, Any]:
    """
    Check GLM service status and availability
    """
    try:
        # Import here to avoid circular import
        from app.services.glm_service import glm_service
        
        status = glm_service.health_check()
        return {
            "success": True,
            "glm_status": status,
            "endpoint": "http://10.122.131.109:11434"
        }
        
    except Exception as e:
        logger.exception(f"GLM status check error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
    
    @staticmethod
    async def calculate_similarity(code1: str, code2: str, language: str, detailed_analysis: bool = False) -> tuple:
        """模拟代码相似度计算"""
        # 实际实现中，这里会调用AI模型进行相似度计算
        # 这里返回模拟数据作为示例
        similarity_score = 0.75
        
        similar_segments = None
        if detailed_analysis:
            similar_segments = [
                SimilarSegment(
                    code1_start_line=5,
                    code1_end_line=10,
                    code2_start_line=8,
                    code2_end_line=13,
                    similarity_score=0.92
                ),
                SimilarSegment(
                    code1_start_line=15,
                    code1_end_line=20,
                    code2_start_line=25,
                    code2_end_line=30,
                    similarity_score=0.85
                )
            ]
        
        return similarity_score, similar_segments

# 获取AI模型服务实例
def get_ai_model_service():
    return AIModelService()

# Prometheus metrics
API_REQUESTS = Counter('ai_api_requests_total', 'Total API requests', ['endpoint'])
API_ERRORS = Counter('ai_api_errors_total', 'API errors', ['endpoint'])
API_LATENCY = Histogram('ai_api_latency_seconds', 'API latency (seconds)', ['endpoint'])
INFERENCE_TOTAL = Counter('ai_inference_total', 'Inference requests', ['model_version', 'ab_group'])
INFERENCE_LATENCY = Histogram('ai_inference_latency_seconds', 'Inference latency (seconds)', ['model_version'])

# API端点实现
@router.post("/embed", response_model=CodeEmbedResponse)
async def embed_code(
    request: CodeEmbedRequest,
    background_tasks: BackgroundTasks,
    ai_service: AIModelService = Depends(get_ai_model_service),
    cache: Cache = Depends(get_cache),
    _: None = Depends(rate_limiter(max_requests=10, window_seconds=60))
):
    """
    将代码结构转换为向量表示
    
    - **code_structure**: 解析后的代码结构
    - **language**: 编程语言
    
    返回768维的代码向量
    """
    API_REQUESTS.labels(endpoint="/embed").inc()
    # 生成缓存键
    cache_key = f"embed:{hash(str(request.code_structure))}"
    
    # 检查缓存
    cached_result = await cache.get(cache_key)
    if cached_result:
        return JSONResponse(content=cached_result)
    
    # 计时开始
    start_time = time.time()
    
    # 调用AI服务进行代码向量化
    vector = await ai_service.embed_code(request.code_structure, request.language)
    
    # 计算处理时间
    processing_time = time.time() - start_time
    API_LATENCY.labels(endpoint="/embed").observe(processing_time)
    
    # 构建响应
    response = CodeEmbedResponse(
        vector=vector,
        processing_time=processing_time
    )
    
    # 异步缓存结果
    background_tasks.add_task(cache.set, cache_key, response.dict(), expire=3600)
    
    return response

@router.post("/analyze/defects", response_model=DefectAnalysisResponse)
async def analyze_defects(
    request: DefectAnalysisRequest,
    background_tasks: BackgroundTasks,
    ai_service: AIModelService = Depends(get_ai_model_service),
    cache: Cache = Depends(get_cache),
    _: None = Depends(rate_limiter(max_requests=5, window_seconds=60)),
    model_version: Optional[str] = Query(None, description="模型版本号"),
    ab_group: Optional[str] = Query(None, description="A/B测试分组"),
    use_ensemble: bool = Query(False, description="启用专家模型集成"),
    estimate_uncertainty: bool = Query(False, description="返回不确定性估计"),
    apply_postprocess: bool = Query(True, description="启用规则与上下文后处理"),
    calibrate_confidence: bool = Query(True, description="启用置信度校准")
):
    """
    分析代码中的缺陷
    
    - **code**: 原始代码 (可选)
    - **vector**: 代码向量 (可选)
    - **language**: 编程语言
    
    至少需要提供code或vector其中之一
    
    返回检测到的缺陷列表，包含缺陷类型、置信度和位置信息
    """
    if not request.code and not request.vector:
        raise HTTPException(status_code=400, detail="必须提供code或vector其中之一")
    
    API_REQUESTS.labels(endpoint="/analyze/defects").inc()
    # 自动分配A/B分组（基于输入的哈希）
    if not ab_group:
        ab_group = 'A' if hash(request.code or str(request.vector)) % 2 == 0 else 'B'

    # 生成缓存键
    cache_key = f"defects:{hash(request.code or str(request.vector))}"
    
    # 检查缓存
    cached_result = await cache.get(cache_key)
    if cached_result:
        return JSONResponse(content=cached_result)
    
    # 计时开始
    start_time = time.time()
    
    # 调用AI服务进行缺陷检测（支持集成）
    try:
        if use_ensemble:
            # 简化：多次调用同一服务模拟不同专家（真实场景替换为不同模型）
            expert_outputs = []
            for _ in range(3):
                ds = await ai_service.detect_defects(request.code, request.vector, request.language, model_version)
                expert_outputs.append(ExpertOutput(defects=[d.dict() if hasattr(d, 'dict') else d for d in ds], metadata={}))
            aggregator = EnsembleAggregator(weights=[1.0, 1.0, 1.0])
            fused, uncertainty = aggregator.fuse_defects(expert_outputs)
            defects = [Defect(**d) for d in fused]
            fused_uncertainty = uncertainty if estimate_uncertainty else None
        else:
            ds = await ai_service.detect_defects(request.code, request.vector, request.language, model_version)
            defects = ds
            fused_uncertainty = None
    except Exception:
        API_ERRORS.labels(endpoint="/analyze/defects").inc()
        raise
    
    # 计算处理时间
    processing_time = time.time() - start_time
    API_LATENCY.labels(endpoint="/analyze/defects").observe(processing_time)
    INFERENCE_TOTAL.labels(model_version or 'default', ab_group).inc()
    INFERENCE_LATENCY.labels(model_version or 'default').observe(processing_time)
    
    # 后处理（规则、上下文过滤、校准）
    if apply_postprocess or calibrate_confidence:
        re_engine = RuleEngine()
        ctx_filter = ContextFilter()
        defects_dicts = [d.dict() if hasattr(d, 'dict') else d for d in defects]
        if apply_postprocess:
            defects_dicts = re_engine.apply(defects_dicts, request.language, request.code or "")
            defects_dicts = ctx_filter.filter(defects_dicts, request.language)
        if calibrate_confidence:
            calibrator = ConfidenceCalibrator()
            defects_dicts = calibrator.calibrate(defects_dicts)
        defects = [Defect(**d) for d in defects_dicts]

    # 构建响应
    response = DefectAnalysisResponse(
        defects=defects,
        processing_time=processing_time,
        uncertainty=fused_uncertainty
    )
    
    # 异步缓存结果
    background_tasks.add_task(cache.set, cache_key, response.dict(), expire=3600)
    
    return response

@router.post("/analyze/architecture", response_model=ArchitectureAnalysisResponse)
async def analyze_architecture(
    request: ArchitectureAnalysisRequest,
    background_tasks: BackgroundTasks,
    ai_service: AIModelService = Depends(get_ai_model_service),
    cache: Cache = Depends(get_cache),
    _: None = Depends(rate_limiter(max_requests=2, window_seconds=300)),
    model_version: Optional[str] = Query(None, description="模型版本号"),
    ab_group: Optional[str] = Query(None, description="A/B测试分组"),
    calibrate_confidence: bool = Query(True, description="对架构模式置信度进行校准")
):
    """
    分析项目架构模式和架构坏味
    
    - **project_structure**: 项目文件结构 (可选)
    - **vectors**: 文件路径到代码向量的映射 (可选)
    
    至少需要提供project_structure或vectors其中之一
    
    返回检测到的架构模式和架构坏味
    """
    if not request.project_structure and not request.vectors:
        raise HTTPException(status_code=400, detail="必须提供project_structure或vectors其中之一")
    
    API_REQUESTS.labels(endpoint="/analyze/architecture").inc()
    if not ab_group:
        ab_group = 'A' if hash(str(request.project_structure or request.vectors)) % 2 == 0 else 'B'
    # 生成缓存键
    cache_key = f"architecture:{hash(str(request.project_structure or request.vectors))}"
    
    # 检查缓存
    cached_result = await cache.get(cache_key)
    if cached_result:
        return JSONResponse(content=cached_result)
    
    # 计时开始
    start_time = time.time()
    
    # 调用AI服务进行架构分析
    try:
        patterns, smells = await ai_service.analyze_architecture(
            request.project_structure, request.vectors, model_version
        )
    except Exception:
        API_ERRORS.labels(endpoint="/analyze/architecture").inc()
        raise

    # 置信度校准（仅对模式进行）
    if calibrate_confidence and patterns:
        calibrator = ConfidenceCalibrator()
        pattern_dicts = [p.dict() if hasattr(p, 'dict') else p for p in patterns]
        pattern_dicts = calibrator.calibrate(pattern_dicts)
        patterns = [ArchitecturePattern(**p) for p in pattern_dicts]

    # 计算处理时间
    processing_time = time.time() - start_time
    API_LATENCY.labels(endpoint="/analyze/architecture").observe(processing_time)
    INFERENCE_TOTAL.labels(model_version or 'default', ab_group).inc()
    INFERENCE_LATENCY.labels(model_version or 'default').observe(processing_time)
    
    # 构建响应
    response = ArchitectureAnalysisResponse(
        patterns=patterns,
        smells=smells,
        processing_time=processing_time
    )
    
    # 异步缓存结果
    background_tasks.add_task(cache.set, cache_key, response.dict(), expire=86400)  # 缓存1天
    
    return response

@router.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(
    request: SimilarityRequest,
    background_tasks: BackgroundTasks,
    ai_service: AIModelService = Depends(get_ai_model_service),
    cache: Cache = Depends(get_cache),
    _: None = Depends(rate_limiter(max_requests=10, window_seconds=60))
):
    """
    计算两段代码的相似度
    
    - **code1**: 第一段代码
    - **code2**: 第二段代码
    - **language**: 编程语言
    - **detailed_analysis**: 是否返回详细分析
    
    返回相似度分数和相似部分
    """
    # 生成缓存键
    cache_key = f"similarity:{hash(request.code1 + request.code2)}"
    
    # 检查缓存
    cached_result = await cache.get(cache_key)
    if cached_result:
        return JSONResponse(content=cached_result)
    
    # 计时开始
    start_time = time.time()
    
    # 调用AI服务进行相似度计算
    similarity_score, similar_segments = await ai_service.calculate_similarity(
        request.code1, request.code2, request.language, request.detailed_analysis
    )
    
    # 计算处理时间
    processing_time = time.time() - start_time
    
    # 构建响应
    response = SimilarityResponse(
        similarity_score=similarity_score,
        similar_segments=similar_segments,
        processing_time=processing_time
    )
    
    # 异步缓存结果
    background_tasks.add_task(cache.set, cache_key, response.dict(), expire=3600)
    
    return response