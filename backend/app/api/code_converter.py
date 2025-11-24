"""
代码转换API接口

提供控制流和内存管理转换的REST API接口，以及转换质量保证和渐进式转换功能
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from ..services.flow_converter.loops import convert_loop
from ..services.flow_converter.conditions import convert_condition
from ..services.flow_converter.exceptions import convert_exception_handling
from ..services.flow_converter.functions import convert_function_flow
from ..services.memory_converter.resources import convert_resource_management
from ..services.memory_converter.ownership import convert_ownership_system
from ..services.api_mapping.standard_lib import convert_standard_api, generate_api_migration_guide
from ..services.framework_converter.frameworks import (
    convert_frameworks,
    convert_framework_dependencies,
    generate_framework_migration_guide,
)
from ..services.quality_assurance.integration import ConversionQualityManager

router = APIRouter(prefix="/api/code-converter", tags=["code-converter"])


class ConversionRequest(BaseModel):
    """代码转换请求模型"""
    source_code: str
    source_language: str
    target_language: str
    conversion_type: str  # 'flow', 'memory', 'api', 'framework', 'all'
    flow_options: Optional[List[str]] = None  # ['loops', 'conditions', 'exceptions', 'functions']
    memory_options: Optional[List[str]] = None  # ['resources', 'ownership']
    api_options: Optional[List[str]] = None    # ['stdlib', 'collections']（扩展保留）
    framework_options: Optional[List[str]] = None  # ['mvc', 'routing']（扩展保留）
    quality_options: Optional[Dict[str, Any]] = None  # 质量保证选项
    enable_incremental: Optional[bool] = False  # 是否启用渐进式转换
    enable_error_recovery: Optional[bool] = True  # 是否启用错误恢复


class ConversionResponse(BaseModel):
    """代码转换响应模型"""
    converted_code: str
    conversion_details: Dict[str, Any]
    warnings: Optional[List[str]] = None
    quality_metrics: Optional[Dict[str, Any]] = None  # 质量评估结果
    validation_results: Optional[Dict[str, Any]] = None  # 验证结果
    recovery_suggestions: Optional[List[Dict[str, Any]]] = None  # 错误恢复建议


@router.post("/convert", response_model=ConversionResponse)
async def convert_code(request: ConversionRequest):
    """
    转换代码的API端点
    
    根据请求参数执行不同类型的代码转换，并提供质量保证和渐进式转换功能
    """
    try:
        source_code = request.source_code
        source_language = request.source_language.lower()
        target_language = request.target_language.lower()
        conversion_type = request.conversion_type.lower()
        
        # 验证语言支持
        supported_languages = ["python", "javascript", "typescript", "cpp", "rust", "java", "csharp"]
        if source_language not in supported_languages:
            raise HTTPException(status_code=400, detail=f"不支持的源语言: {source_language}")
        if target_language not in supported_languages:
            raise HTTPException(status_code=400, detail=f"不支持的目标语言: {target_language}")
        
        # 初始化转换结果
        converted_code = source_code
        conversion_details = {}
        warnings = []
        
        # 初始化质量管理器
        quality_manager = ConversionQualityManager()
        
        # 如果启用渐进式转换，先检查代码是否可转换
        if request.enable_incremental:
            incremental_result = quality_manager.incremental_converter.can_convert(source_code)
            if not incremental_result["convertible"]:
                # 如果不可完全转换，尝试部分转换
                partial_result = quality_manager.incremental_converter.partial_convert(
                    source_code, 
                    strategy='conservative' if request.quality_options and request.quality_options.get('conservative', False) else 'aggressive'
                )
                source_code = partial_result["convertible_code"]
                for warning in partial_result["warnings"]:
                    warnings.append(f"渐进式转换警告: {warning}")
        
        # 执行控制流转换
        if conversion_type in ["flow", "all"]:
            flow_options = request.flow_options or ["loops", "conditions", "exceptions", "functions"]
            
            if "loops" in flow_options:
                try:
                    converted_code = convert_loop(converted_code, source_language, target_language)
                    conversion_details["loops"] = "转换成功"
                except Exception as e:
                    warnings.append(f"循环转换失败: {str(e)}")
            
            if "conditions" in flow_options:
                try:
                    converted_code = convert_condition(converted_code, source_language, target_language)
                    conversion_details["conditions"] = "转换成功"
                except Exception as e:
                    warnings.append(f"条件语句转换失败: {str(e)}")
            
            if "exceptions" in flow_options:
                try:
                    converted_code = convert_exception_handling(converted_code, source_language, target_language)
                    conversion_details["exceptions"] = "转换成功"
                except Exception as e:
                    warnings.append(f"异常处理转换失败: {str(e)}")
            
            if "functions" in flow_options:
                try:
                    converted_code = convert_function_flow(converted_code, source_language, target_language)
                    conversion_details["functions"] = "转换成功"
                except Exception as e:
                    warnings.append(f"函数控制流转换失败: {str(e)}")
        
        # 执行内存管理转换
        if conversion_type in ["memory", "all"]:
            memory_options = request.memory_options or ["resources", "ownership"]
            
            if "resources" in memory_options:
                try:
                    converted_code = convert_resource_management(converted_code, source_language, target_language)
                    conversion_details["resources"] = "转换成功"
                except Exception as e:
                    warnings.append(f"资源管理转换失败: {str(e)}")
            
            if "ownership" in memory_options:
                try:
                    converted_code = convert_ownership_system(converted_code, source_language, target_language)
                    conversion_details["ownership"] = "转换成功"
                except Exception as e:
                    warnings.append(f"所有权系统转换失败: {str(e)}")

        # 执行标准库 API 映射
        if conversion_type in ["api", "all"]:
            api_options = request.api_options or ["stdlib"]
            if "stdlib" in api_options:
                try:
                    converted_code = convert_standard_api(converted_code, source_language, target_language)
                    conversion_details["api_stdlib"] = "转换成功"
                    # 生成简版迁移指南
                    conversion_details["api_guide"] = generate_api_migration_guide(source_code, source_language, target_language)
                except Exception as e:
                    warnings.append(f"标准库API映射失败: {str(e)}")

        # 执行第三方框架转换
        if conversion_type in ["framework", "all"]:
            framework_options = request.framework_options or ["mvc"]
            if "mvc" in framework_options:
                try:
                    converted_code = convert_frameworks(converted_code, source_language, target_language)
                    conversion_details["framework_mvc"] = "转换成功"
                except Exception as e:
                    warnings.append(f"框架 MVC 转换失败: {str(e)}")
            if "dependencies" in framework_options:
                try:
                    dep_out = convert_framework_dependencies(source_code, source_language, target_language)
                    conversion_details["dependencies"] = dep_out
                except Exception as e:
                    warnings.append(f"依赖管理转换失败: {str(e)}")
            # 统一生成迁移指南
            try:
                conversion_details["framework_guide"] = generate_framework_migration_guide(source_code, source_language, target_language)
            except Exception:
                pass
        
        # 应用质量保证和错误恢复
        quality_results = {}
        validation_results = {}
        recovery_suggestions = []
        
        # 验证语法和语义
        syntax_validation = quality_manager.validator.validate_syntax(
            converted_code, target_language
        )
        if not syntax_validation["valid"] and request.enable_error_recovery:
            # 如果语法验证失败且启用了错误恢复，尝试修复
            for error in syntax_validation["errors"]:
                recovery_result = quality_manager.error_recovery.handle_syntax_error(
                    error, {"code": converted_code, "language": target_language}
                )
                if recovery_result["fixed"]:
                    converted_code = recovery_result["fixed_code"]
                    recovery_suggestions.append({
                        "type": "syntax_fix",
                        "original_error": error,
                        "fix_applied": recovery_result["fix_description"]
                    })
                else:
                    recovery_suggestions.append({
                        "type": "syntax_suggestion",
                        "error": error,
                        "suggestions": recovery_result["suggestions"]
                    })
        
        # 重新验证修复后的代码
        validation_results["syntax"] = quality_manager.validator.validate_syntax(
            converted_code, target_language
        )
        
        # 计算质量指标
        quality_results = quality_manager.metrics.calculate_conversion_score(
            source_code, converted_code, target_language
        )
        
        return ConversionResponse(
            converted_code=converted_code,
            conversion_details=conversion_details,
            warnings=warnings if warnings else None,
            quality_metrics=quality_results,
            validation_results=validation_results,
            recovery_suggestions=recovery_suggestions if recovery_suggestions else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"代码转换失败: {str(e)}")


class QualityAssessmentRequest(BaseModel):
    """质量评估请求模型"""
    source_code: str
    converted_code: str
    source_language: str
    target_language: str
    assessment_type: Optional[List[str]] = ["syntax", "semantics", "quality", "best_practices"]


class QualityAssessmentResponse(BaseModel):
    """质量评估响应模型"""
    validation_results: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    improvement_suggestions: Optional[List[Dict[str, Any]]] = None


@router.post("/assess-quality", response_model=QualityAssessmentResponse)
async def assess_conversion_quality(request: QualityAssessmentRequest):
    """
    评估代码转换质量的API端点
    
    对已转换的代码进行质量评估，包括语法验证、语义验证和质量指标
    """
    try:
        source_code = request.source_code
        converted_code = request.converted_code
        source_language = request.source_language.lower()
        target_language = request.target_language.lower()
        assessment_type = request.assessment_type
        
        # 初始化质量管理器
        quality_manager = ConversionQualityManager()
        
        validation_results = {}
        quality_metrics = {}
        improvement_suggestions = []
        
        # 语法验证
        if "syntax" in assessment_type:
            validation_results["syntax"] = quality_manager.validator.validate_syntax(
                converted_code, target_language
            )
        
        # 语义验证
        if "semantics" in assessment_type:
            validation_results["semantics"] = quality_manager.validator.validate_semantics(
                source_code, converted_code
            )
        
        # 质量指标
        if "quality" in assessment_type:
            quality_metrics = quality_manager.metrics.calculate_conversion_score(
                source_code, converted_code
            )
        
        # 最佳实践检查
        if "best_practices" in assessment_type:
            best_practices = quality_manager.metrics.check_best_practices(
                converted_code, target_language
            )
            quality_metrics["best_practices"] = best_practices
            
            # 生成改进建议
            for issue in best_practices.get("issues", []):
                improvement_suggestions.append({
                    "type": "best_practice",
                    "issue": issue["description"],
                    "suggestion": issue["suggestion"],
                    "severity": issue["severity"]
                })
        
        return QualityAssessmentResponse(
            validation_results=validation_results,
            quality_metrics=quality_metrics,
            improvement_suggestions=improvement_suggestions if improvement_suggestions else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"质量评估失败: {str(e)}")


# 添加到主应用
def include_router(app):
    """将路由器添加到主应用"""
    app.include_router(router)