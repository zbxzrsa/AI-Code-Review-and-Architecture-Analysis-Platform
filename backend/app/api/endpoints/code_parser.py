"""
代码解析服务API接口
提供代码解析、AST生成、CFG/DFG提取和度量计算的HTTP接口
"""
from typing import List, Dict, Any, Optional
from enum import Enum
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.code_parser.parser import CodeParserService, FeatureType, Language, ParserError

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["code-parser"])

# 定义请求模型
class FeatureEnum(str, Enum):
    AST = "ast"
    CFG = "cfg"
    DFG = "dfg"
    METRICS = "metrics"

class LanguageEnum(str, Enum):
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    GO = "go"

class ParseRequest(BaseModel):
    code: str = Field(..., description="源代码")
    language: LanguageEnum = Field(..., description="编程语言")
    features: List[FeatureEnum] = Field(..., description="需要提取的特征")

# 定义响应模型
class ParseResponse(BaseModel):
    ast: Optional[Dict[str, Any]] = Field(None, description="抽象语法树")
    cfg: Optional[Dict[str, Any]] = Field(None, description="控制流图")
    dfg: Optional[Dict[str, Any]] = Field(None, description="数据流图")
    metrics: Optional[Dict[str, Any]] = Field(None, description="代码度量指标")

# 创建代码解析服务实例
code_parser_service = CodeParserService()

# 依赖项：获取代码解析服务
def get_code_parser_service():
    return code_parser_service

@router.post("/parse", response_model=ParseResponse)
async def parse_code(
    request: ParseRequest,
    parser_service: CodeParserService = Depends(get_code_parser_service)
):
    """
    解析代码并提取请求的特征
    
    Args:
        request: 解析请求，包含代码、语言和特征
        
    Returns:
        解析结果，包含请求的特征
    """
    try:
        # 转换请求参数
        language = Language[request.language.upper()]
        features = [FeatureType[feature.upper()] for feature in request.features]
        
        # 解析代码
        result = await parser_service.parse(request.code, language, features)
        
        # 构建响应
        response = ParseResponse()
        
        # 添加特征到响应
        if FeatureType.AST in features and FeatureType.AST in result:
            response.ast = result[FeatureType.AST]
        
        if FeatureType.CFG in features and FeatureType.CFG in result:
            response.cfg = result[FeatureType.CFG]
        
        if FeatureType.DFG in features and FeatureType.DFG in result:
            response.dfg = result[FeatureType.DFG]
        
        if FeatureType.METRICS in features and FeatureType.METRICS in result:
            response.metrics = result[FeatureType.METRICS]
        
        return response
    
    except ParserError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")

@router.post("/batch-parse")
async def batch_parse_code(
    requests: List[ParseRequest],
    parser_service: CodeParserService = Depends(get_code_parser_service)
):
    """
    批量解析代码并提取请求的特征
    
    Args:
        requests: 解析请求列表，每个请求包含代码、语言和特征
        
    Returns:
        解析结果列表，每个结果包含请求的特征
    """
    try:
        # 准备批量解析任务
        tasks = []
        for request in requests:
            # 转换请求参数
            language = Language[request.language.upper()]
            features = [FeatureType[feature.upper()] for feature in request.features]
            
            # 添加解析任务
            tasks.append((request.code, language, features))
        
        # 批量解析代码
        results = await parser_service.batch_parse(tasks)
        
        # 构建响应
        responses = []
        for i, result in enumerate(results):
            response = ParseResponse()
            features = [FeatureType[feature.upper()] for feature in requests[i].features]
            
            # 添加特征到响应
            if FeatureType.AST in features and FeatureType.AST in result:
                response.ast = result[FeatureType.AST]
            
            if FeatureType.CFG in features and FeatureType.CFG in result:
                response.cfg = result[FeatureType.CFG]
            
            if FeatureType.DFG in features and FeatureType.DFG in result:
                response.dfg = result[FeatureType.DFG]
            
            if FeatureType.METRICS in features and FeatureType.METRICS in result:
                response.metrics = result[FeatureType.METRICS]
            
            responses.append(response)
        
        return responses
    
    except ParserError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量解析失败: {str(e)}")