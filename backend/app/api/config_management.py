"""
配置管理API端点

提供转换配置的管理接口，包括：
- 配置的获取、更新和验证
- 规则的启用/禁用和优先级管理
- 配置预设的保存、加载和管理
- 支持的语言和特性查询
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio

from ..services.conversion_config import ConversionConfig, ConversionStrategy, get_conversion_config
from ..services.live_preview import LivePreview, get_live_preview


router = APIRouter(prefix="/config", tags=["配置管理"])


# Pydantic 模型定义
class LanguageInfo(BaseModel):
    """语言信息模型"""
    name: str
    extensions: List[str]
    features: List[str]


class RuleConfigModel(BaseModel):
    """规则配置模型"""
    name: str
    enabled: bool = True
    priority: int = 100
    settings: Dict[str, Any] = {}


class ConversionConfigModel(BaseModel):
    """转换配置模型"""
    source_language: str = ""
    target_language: str = ""
    conversion_strategy: str = "balanced"
    rules: Dict[str, RuleConfigModel] = {}
    language_features: Dict[str, Any] = {}
    advanced_options: Dict[str, Any] = {}


class PresetModel(BaseModel):
    """预设模型"""
    name: str
    description: str = ""


class RuleUpdateModel(BaseModel):
    """规则更新模型"""
    enabled: Optional[bool] = None
    priority: Optional[int] = None
    settings: Optional[Dict[str, Any]] = None


class PreviewRequest(BaseModel):
    """预览请求模型"""
    source_code: str
    force_refresh: bool = False


# 依赖注入
def get_config() -> ConversionConfig:
    """获取配置实例"""
    return get_conversion_config()


def get_preview() -> LivePreview:
    """获取预览实例"""
    return get_live_preview()


@router.get("/languages", response_model=Dict[str, LanguageInfo])
async def get_supported_languages():
    """
    获取支持的编程语言列表
    
    Returns:
        支持的语言信息字典
    """
    config = get_conversion_config()
    return {
        lang_code: LanguageInfo(**lang_info)
        for lang_code, lang_info in config.SUPPORTED_LANGUAGES.items()
    }


@router.get("/current", response_model=Dict[str, Any])
async def get_current_config(config: ConversionConfig = Depends(get_config)):
    """
    获取当前配置
    
    Returns:
        当前配置信息
    """
    return config.to_dict()


@router.post("/update")
async def update_config(
    config_data: ConversionConfigModel,
    config: ConversionConfig = Depends(get_config)
):
    """
    更新配置
    
    Args:
        config_data: 新的配置数据
        
    Returns:
        更新结果
    """
    try:
        # 更新基础配置
        config.source_language = config_data.source_language
        config.target_language = config_data.target_language
        
        # 更新转换策略
        if config_data.conversion_strategy in [s.value for s in ConversionStrategy]:
            config.conversion_strategy = ConversionStrategy(config_data.conversion_strategy)
        
        # 更新规则配置
        for rule_name, rule_data in config_data.rules.items():
            if rule_name in config.rules:
                config.rules[rule_name].enabled = rule_data.enabled
                config.rules[rule_name].priority = rule_data.priority
                config.rules[rule_name].settings.update(rule_data.settings)
        
        # 更新其他配置
        config.language_features.update(config_data.language_features)
        config.advanced_options.update(config_data.advanced_options)
        
        # 验证配置
        validation_result = config.validate_config()
        
        return {
            "success": True,
            "message": "配置更新成功",
            "validation": validation_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"配置更新失败: {str(e)}")


@router.post("/validate")
async def validate_config(config: ConversionConfig = Depends(get_config)):
    """
    验证当前配置
    
    Returns:
        验证结果
    """
    validation_result = config.validate_config()
    return {
        "validation": validation_result,
        "timestamp": "now"  # 简化时间戳
    }


@router.get("/rules", response_model=Dict[str, RuleConfigModel])
async def get_rules(config: ConversionConfig = Depends(get_config)):
    """
    获取所有规则配置
    
    Returns:
        规则配置字典
    """
    return {
        name: RuleConfigModel(
            name=rule.name,
            enabled=rule.enabled,
            priority=rule.priority,
            settings=rule.settings
        )
        for name, rule in config.rules.items()
    }


@router.get("/rules/enabled")
async def get_enabled_rules(config: ConversionConfig = Depends(get_config)):
    """
    获取已启用的规则列表
    
    Returns:
        已启用的规则列表（按优先级排序）
    """
    enabled_rules = config.get_enabled_rules()
    return {
        "enabled_rules": [
            {
                "name": rule.name,
                "priority": rule.priority,
                "settings": rule.settings
            }
            for rule in enabled_rules
        ],
        "count": len(enabled_rules)
    }


@router.put("/rules/{rule_name}")
async def update_rule(
    rule_name: str,
    rule_update: RuleUpdateModel,
    config: ConversionConfig = Depends(get_config)
):
    """
    更新特定规则配置
    
    Args:
        rule_name: 规则名称
        rule_update: 规则更新数据
        
    Returns:
        更新结果
    """
    if rule_name not in config.rules:
        raise HTTPException(status_code=404, detail=f"规则 '{rule_name}' 不存在")
    
    try:
        # 更新规则属性
        if rule_update.enabled is not None:
            config.enable_rule(rule_name, rule_update.enabled)
        
        if rule_update.priority is not None:
            config.set_rule_priority(rule_name, rule_update.priority)
        
        if rule_update.settings is not None:
            config.update_rule_settings(rule_name, rule_update.settings)
        
        return {
            "success": True,
            "message": f"规则 '{rule_name}' 更新成功",
            "rule": {
                "name": config.rules[rule_name].name,
                "enabled": config.rules[rule_name].enabled,
                "priority": config.rules[rule_name].priority,
                "settings": config.rules[rule_name].settings
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"规则更新失败: {str(e)}")


@router.post("/rules/{rule_name}/toggle")
async def toggle_rule(
    rule_name: str,
    config: ConversionConfig = Depends(get_config)
):
    """
    切换规则启用状态
    
    Args:
        rule_name: 规则名称
        
    Returns:
        切换结果
    """
    if rule_name not in config.rules:
        raise HTTPException(status_code=404, detail=f"规则 '{rule_name}' 不存在")
    
    current_state = config.rules[rule_name].enabled
    new_state = not current_state
    
    config.enable_rule(rule_name, new_state)
    
    return {
        "success": True,
        "message": f"规则 '{rule_name}' 已{'启用' if new_state else '禁用'}",
        "enabled": new_state
    }


@router.post("/rules/custom")
async def add_custom_rule(
    rule_config: RuleConfigModel,
    config: ConversionConfig = Depends(get_config)
):
    """
    添加自定义规则
    
    Args:
        rule_config: 规则配置
        
    Returns:
        添加结果
    """
    success = config.add_custom_rule(
        rule_name=rule_config.name,
        priority=rule_config.priority,
        enabled=rule_config.enabled,
        settings=rule_config.settings
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=f"规则 '{rule_config.name}' 已存在")
    
    return {
        "success": True,
        "message": f"自定义规则 '{rule_config.name}' 添加成功"
    }


@router.delete("/rules/{rule_name}")
async def remove_rule(
    rule_name: str,
    config: ConversionConfig = Depends(get_config)
):
    """
    移除自定义规则
    
    Args:
        rule_name: 规则名称
        
    Returns:
        移除结果
    """
    success = config.remove_rule(rule_name)
    
    if not success:
        raise HTTPException(
            status_code=400, 
            detail=f"无法移除规则 '{rule_name}'（可能是默认规则或不存在）"
        )
    
    return {
        "success": True,
        "message": f"规则 '{rule_name}' 移除成功"
    }


@router.get("/presets")
async def list_presets(config: ConversionConfig = Depends(get_config)):
    """
    获取所有配置预设
    
    Returns:
        预设列表
    """
    presets = config.list_presets()
    return {
        "presets": presets,
        "count": len(presets)
    }


@router.post("/presets")
async def save_preset(
    preset: PresetModel,
    config: ConversionConfig = Depends(get_config)
):
    """
    保存配置预设
    
    Args:
        preset: 预设信息
        
    Returns:
        保存结果
    """
    success = config.save_preset(preset.name, preset.description)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"预设 '{preset.name}' 保存失败")
    
    return {
        "success": True,
        "message": f"预设 '{preset.name}' 保存成功"
    }


@router.post("/presets/{preset_name}/load")
async def load_preset(
    preset_name: str,
    config: ConversionConfig = Depends(get_config)
):
    """
    加载配置预设
    
    Args:
        preset_name: 预设名称
        
    Returns:
        加载结果
    """
    success = config.load_preset(preset_name)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"预设 '{preset_name}' 不存在或加载失败")
    
    return {
        "success": True,
        "message": f"预设 '{preset_name}' 加载成功",
        "config": config.to_dict()
    }


@router.delete("/presets/{preset_name}")
async def delete_preset(
    preset_name: str,
    config: ConversionConfig = Depends(get_config)
):
    """
    删除配置预设
    
    Args:
        preset_name: 预设名称
        
    Returns:
        删除结果
    """
    success = config.delete_preset(preset_name)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"预设 '{preset_name}' 不存在或删除失败")
    
    return {
        "success": True,
        "message": f"预设 '{preset_name}' 删除成功"
    }


@router.post("/preview")
async def generate_preview(
    request: PreviewRequest,
    preview: LivePreview = Depends(get_preview)
):
    """
    生成实时预览
    
    Args:
        request: 预览请求
        
    Returns:
        预览结果
    """
    try:
        result = await preview.update_preview(
            request.source_code,
            request.force_refresh
        )
        
        return {
            "success": result.success,
            "source_code": result.source_code,
            "converted_code": result.converted_code,
            "issues": [
                {
                    "type": issue.type.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "line_number": issue.line_number,
                    "column_start": issue.column_start,
                    "column_end": issue.column_end,
                    "suggested_fix": issue.suggested_fix,
                    "rule_name": issue.rule_name
                }
                for issue in result.issues
            ],
            "quality_score": result.quality_score,
            "conversion_time": result.conversion_time,
            "error_message": result.error_message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预览生成失败: {str(e)}")


@router.get("/preview/highlights")
async def get_issue_highlights(preview: LivePreview = Depends(get_preview)):
    """
    获取问题高亮信息
    
    Returns:
        高亮信息
    """
    return preview.highlight_issues()


@router.get("/preview/side-by-side")
async def get_side_by_side_view(preview: LivePreview = Depends(get_preview)):
    """
    获取并排对比视图
    
    Returns:
        并排对比数据
    """
    return preview.show_side_by_side()


@router.get("/preview/diff")
async def get_diff_view(preview: LivePreview = Depends(get_preview)):
    """
    获取差异视图
    
    Returns:
        差异视图数据
    """
    return preview.generate_diff_view()


@router.get("/preview/stats")
async def get_preview_stats(preview: LivePreview = Depends(get_preview)):
    """
    获取预览统计信息
    
    Returns:
        统计信息
    """
    return preview.get_preview_stats()


@router.post("/preview/clear-cache")
async def clear_preview_cache(preview: LivePreview = Depends(get_preview)):
    """
    清空预览缓存
    
    Returns:
        清空结果
    """
    preview.clear_cache()
    return {
        "success": True,
        "message": "预览缓存已清空"
    }


@router.get("/defaults")
async def get_default_config():
    """
    获取默认配置
    
    Returns:
        默认配置信息
    """
    default_config = ConversionConfig()
    return {
        "default_rules": default_config.DEFAULT_RULES,
        "supported_languages": default_config.SUPPORTED_LANGUAGES,
        "conversion_strategies": [strategy.value for strategy in ConversionStrategy],
        "advanced_options": default_config.advanced_options
    }


@router.post("/reset")
async def reset_config(config: ConversionConfig = Depends(get_config)):
    """
    重置配置为默认值
    
    Returns:
        重置结果
    """
    try:
        # 重新初始化配置
        config.__init__()
        
        return {
            "success": True,
            "message": "配置已重置为默认值",
            "config": config.to_dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"配置重置失败: {str(e)}")


@router.get("/export")
async def export_config(config: ConversionConfig = Depends(get_config)):
    """
    导出当前配置
    
    Returns:
        配置数据
    """
    return {
        "config": config.to_dict(),
        "export_time": "now",  # 简化时间戳
        "version": "1.0"
    }


@router.post("/import")
async def import_config(
    config_data: Dict[str, Any],
    config: ConversionConfig = Depends(get_config)
):
    """
    导入配置
    
    Args:
        config_data: 配置数据
        
    Returns:
        导入结果
    """
    try:
        success = config.from_dict(config_data.get("config", config_data))
        
        if not success:
            raise HTTPException(status_code=400, detail="配置数据格式无效")
        
        # 验证导入的配置
        validation_result = config.validate_config()
        
        return {
            "success": True,
            "message": "配置导入成功",
            "validation": validation_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"配置导入失败: {str(e)}")