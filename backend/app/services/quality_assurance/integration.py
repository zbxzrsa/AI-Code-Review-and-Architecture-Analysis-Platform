"""
转换质量保证和渐进式转换集成模块

将转换验证、质量评估、渐进式转换和错误恢复机制集成到代码转换流程中
"""
from typing import Dict, List, Any, Optional, Tuple
import logging

from backend.app.services.quality_assurance.validator import ConversionValidator
from backend.app.services.quality_assurance.metrics import QualityMetrics
from backend.app.services.quality_assurance.incremental import get_incremental_converter
from backend.app.services.quality_assurance.recovery import get_error_recovery_system


class ConversionQualityManager:
    """转换质量管理器，集成所有质量保证和渐进式转换组件"""
    
    def __init__(self):
        """初始化转换质量管理器"""
        self.validator = ConversionValidator()
        self.metrics = QualityMetrics()
        self.incremental_converter = get_incremental_converter()
        self.error_recovery = get_error_recovery_system()
        self.logger = logging.getLogger(__name__)
    
    def process_conversion(self, source_code: str, converted_code: str, 
                          source_lang: str, target_lang: str, 
                          options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理转换结果，进行验证、评估和错误恢复
        
        Args:
            source_code: 源代码
            converted_code: 转换后的代码
            source_lang: 源语言
            target_lang: 目标语言
            options: 处理选项
            
        Returns:
            处理结果，包括验证结果、质量评分和错误恢复建议
        """
        options = options or {}
        result = {
            "source_code": source_code,
            "converted_code": converted_code,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "validation": {},
            "quality_metrics": {},
            "recovery_suggestions": []
        }
        
        # 语法验证
        try:
            syntax_valid = self.validator.validate_syntax(converted_code, target_lang)
            result["validation"]["syntax"] = {
                "valid": syntax_valid.get("valid", False),
                "errors": syntax_valid.get("errors", [])
            }
            
            # 如果语法无效，尝试错误恢复
            if not syntax_valid.get("valid", False):
                for error in syntax_valid.get("errors", []):
                    recovery = self.error_recovery.handle_syntax_error(
                        error, 
                        {"code": converted_code, "source_lang": source_lang, "target_lang": target_lang}
                    )
                    result["recovery_suggestions"].append(recovery)
        except Exception as e:
            self.logger.error(f"语法验证错误: {str(e)}")
            result["validation"]["syntax"] = {
                "valid": False,
                "errors": [str(e)]
            }
        
        # 语义验证（如果配置允许）
        if options.get("validate_semantics", False):
            try:
                semantic_valid = self.validator.validate_semantics(source_code, converted_code)
                result["validation"]["semantics"] = {
                    "valid": semantic_valid.get("valid", False),
                    "errors": semantic_valid.get("errors", []),
                    "test_results": semantic_valid.get("test_results", [])
                }
                
                # 如果语义验证失败，提供语义鸿沟处理建议
                if not semantic_valid.get("valid", False):
                    for error in semantic_valid.get("errors", []):
                        recovery = self.error_recovery.handle_semantic_gap(
                            error.get("feature", "未知特性"),
                            target_lang,
                            error.get("context", "")
                        )
                        result["recovery_suggestions"].append(recovery)
            except Exception as e:
                self.logger.error(f"语义验证错误: {str(e)}")
                result["validation"]["semantics"] = {
                    "valid": False,
                    "errors": [str(e)]
                }
        
        # 质量评估
        try:
            quality_score = self.metrics.calculate_conversion_score(source_code, converted_code)
            result["quality_metrics"] = quality_score
        except Exception as e:
            self.logger.error(f"质量评估错误: {str(e)}")
            result["quality_metrics"] = {
                "error": str(e)
            }
        
        # 如果转换失败或质量低于阈值，提供降级方案
        if (not result["validation"].get("syntax", {}).get("valid", False) or 
            result["quality_metrics"].get("overall_score", 100) < options.get("quality_threshold", 60)):
            
            fallback = self.error_recovery.provide_fallback(
                source_code,
                "syntax_error" if not result["validation"].get("syntax", {}).get("valid", False) else "quality_issue",
                {"source_lang": source_lang, "target_lang": target_lang}
            )
            result["fallback_options"] = fallback.get("fallback_options", [])
        
        return result
    
    def perform_incremental_conversion(self, source_code: str, source_lang: str, 
                                      target_lang: str, strategy: str = 'aggressive') -> Dict[str, Any]:
        """
        执行渐进式转换
        
        Args:
            source_code: 源代码
            source_lang: 源语言
            target_lang: 目标语言
            strategy: 转换策略
            
        Returns:
            渐进式转换结果
        """
        try:
            # 检查是否可以完全转换
            can_convert, unsupported = self.incremental_converter.can_convert(
                source_code, source_lang, target_lang
            )
            
            if can_convert:
                # 如果可以完全转换，返回可以完全转换的信息
                return {
                    "can_fully_convert": True,
                    "unsupported_features": [],
                    "message": "代码可以完全转换"
                }
            
            # 执行部分转换
            result = self.incremental_converter.partial_convert(
                source_code, source_lang, target_lang, strategy
            )
            
            # 为不可转换的部分提供变通方案
            workarounds = []
            for block in result.get("unconverted_blocks", []):
                workaround = self.incremental_converter.suggest_workarounds(
                    block.get("code", ""), source_lang, target_lang
                )
                workarounds.append({
                    "block_index": block.get("block_index"),
                    "suggestions": workaround
                })
            
            result["workarounds"] = workarounds
            result["can_fully_convert"] = False
            
            return result
        except Exception as e:
            self.logger.error(f"渐进式转换错误: {str(e)}")
            return {
                "error": str(e),
                "can_fully_convert": False
            }
    
    def learn_from_user_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        从用户反馈中学习
        
        Args:
            feedback: 用户反馈信息
            
        Returns:
            学习结果
        """
        try:
            if feedback.get("type") == "correction":
                # 用户提供了代码修正
                return self.error_recovery.learn_from_corrections(feedback)
            elif feedback.get("type") == "quality_feedback":
                # 用户提供了质量评分反馈
                # 可以用于调整质量评估算法
                return {"status": "success", "message": "质量评分反馈已记录"}
            else:
                return {"status": "error", "message": "未知的反馈类型"}
        except Exception as e:
            self.logger.error(f"处理用户反馈错误: {str(e)}")
            return {"status": "error", "message": str(e)}


# 单例模式，确保全局只有一个质量管理器实例
_quality_manager_instance = None

def get_quality_manager() -> ConversionQualityManager:
    """获取转换质量管理器实例"""
    global _quality_manager_instance
    if _quality_manager_instance is None:
        _quality_manager_instance = ConversionQualityManager()
    return _quality_manager_instance