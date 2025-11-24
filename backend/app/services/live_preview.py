"""
实时预览服务

提供代码转换的实时预览功能，包括：
- 实时代码转换预览
- 转换问题高亮显示
- 并排代码对比
- 差异视图生成
- 转换质量评估
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import difflib
import re
from dataclasses import dataclass
from enum import Enum

from .conversion_config import ConversionConfig, get_conversion_config
from .code_converter import CodeConverter
from .quality_assurance.metrics import QualityMetrics
from .quality_assurance.validator import ConversionValidator


class IssueType(Enum):
    """问题类型枚举"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"


class IssueSeverity(Enum):
    """问题严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConversionIssue:
    """转换问题数据类"""
    type: IssueType
    severity: IssueSeverity
    message: str
    line_number: int
    column_start: int = 0
    column_end: int = 0
    source_code: str = ""
    suggested_fix: str = ""
    rule_name: str = ""


@dataclass
class PreviewResult:
    """预览结果数据类"""
    source_code: str
    converted_code: str
    issues: List[ConversionIssue]
    quality_score: Dict[str, Any]
    conversion_time: float
    success: bool
    error_message: str = ""


class LivePreview:
    """实时预览管理器"""
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        """
        初始化实时预览管理器
        
        Args:
            config: 转换配置，如果为None则使用全局配置
        """
        self.config = config or get_conversion_config()
        self.converter = CodeConverter()
        self.quality_metrics = QualityMetrics()
        self.validator = ConversionValidator()
        
        # 预览状态
        self.current_source = ""
        self.current_target = ""
        self.current_issues: List[ConversionIssue] = []
        self.last_preview_time = 0
        
        # 缓存机制
        self._preview_cache: Dict[str, PreviewResult] = {}
        self._cache_max_size = 100
    
    async def update_preview(self, source_code: str, 
                           force_refresh: bool = False) -> PreviewResult:
        """
        更新实时预览
        
        Args:
            source_code: 源代码
            force_refresh: 是否强制刷新
            
        Returns:
            预览结果
        """
        import time
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._generate_cache_key(source_code)
        if not force_refresh and cache_key in self._preview_cache:
            cached_result = self._preview_cache[cache_key]
            cached_result.conversion_time = time.time() - start_time
            return cached_result
        
        try:
            # 验证配置
            config_validation = self.config.validate_config()
            if not config_validation["valid"]:
                return PreviewResult(
                    source_code=source_code,
                    converted_code="",
                    issues=[],
                    quality_score={},
                    conversion_time=time.time() - start_time,
                    success=False,
                    error_message=f"配置错误: {', '.join(config_validation['errors'])}"
                )
            
            # 执行代码转换
            conversion_result = await self._perform_conversion(source_code)
            
            # 分析转换问题
            issues = await self._analyze_conversion_issues(
                source_code, 
                conversion_result.get("converted_code", "")
            )
            
            # 计算质量评分
            quality_score = await self._calculate_quality_score(
                source_code, 
                conversion_result.get("converted_code", "")
            )
            
            # 创建预览结果
            result = PreviewResult(
                source_code=source_code,
                converted_code=conversion_result.get("converted_code", ""),
                issues=issues,
                quality_score=quality_score,
                conversion_time=time.time() - start_time,
                success=conversion_result.get("success", False),
                error_message=conversion_result.get("error", "")
            )
            
            # 更新缓存
            self._update_cache(cache_key, result)
            
            # 更新当前状态
            self.current_source = source_code
            self.current_target = result.converted_code
            self.current_issues = issues
            self.last_preview_time = time.time()
            
            return result
            
        except Exception as e:
            return PreviewResult(
                source_code=source_code,
                converted_code="",
                issues=[],
                quality_score={},
                conversion_time=time.time() - start_time,
                success=False,
                error_message=f"预览生成失败: {str(e)}"
            )
    
    async def _perform_conversion(self, source_code: str) -> Dict[str, Any]:
        """
        执行代码转换
        
        Args:
            source_code: 源代码
            
        Returns:
            转换结果
        """
        try:
            # 使用配置的转换器进行转换
            result = await self.converter.convert_code(
                source_code=source_code,
                source_language=self.config.source_language,
                target_language=self.config.target_language,
                conversion_rules=self.config.get_enabled_rules()
            )
            
            return {
                "success": True,
                "converted_code": result.get("converted_code", ""),
                "metadata": result.get("metadata", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "converted_code": "",
                "error": str(e)
            }
    
    async def _analyze_conversion_issues(self, source_code: str, 
                                       converted_code: str) -> List[ConversionIssue]:
        """
        分析转换问题
        
        Args:
            source_code: 源代码
            converted_code: 转换后代码
            
        Returns:
            问题列表
        """
        issues = []
        
        try:
            # 语法验证
            syntax_result = self.validator.validate_syntax(
                converted_code, 
                self.config.target_language
            )
            
            if not syntax_result.get("valid", True):
                for error in syntax_result.get("errors", []):
                    issues.append(ConversionIssue(
                        type=IssueType.ERROR,
                        severity=IssueSeverity.HIGH,
                        message=f"语法错误: {error}",
                        line_number=self._extract_line_number(error),
                        rule_name="syntax_validation"
                    ))
            
            # 转换完整性检查
            completeness_issues = await self._check_conversion_completeness(
                source_code, converted_code
            )
            issues.extend(completeness_issues)
            
            # 最佳实践检查
            best_practices_issues = await self._check_best_practices(converted_code)
            issues.extend(best_practices_issues)
            
            # 性能问题检查
            performance_issues = await self._check_performance_issues(converted_code)
            issues.extend(performance_issues)
            
        except Exception as e:
            issues.append(ConversionIssue(
                type=IssueType.ERROR,
                severity=IssueSeverity.MEDIUM,
                message=f"问题分析失败: {str(e)}",
                line_number=0,
                rule_name="analysis_error"
            ))
        
        return issues
    
    async def _check_conversion_completeness(self, source_code: str, 
                                           converted_code: str) -> List[ConversionIssue]:
        """检查转换完整性"""
        issues = []
        
        # 检查是否有未转换的代码块
        if "TODO" in converted_code or "FIXME" in converted_code:
            todo_lines = [i+1 for i, line in enumerate(converted_code.split('\n')) 
                         if "TODO" in line or "FIXME" in line]
            for line_num in todo_lines:
                issues.append(ConversionIssue(
                    type=IssueType.WARNING,
                    severity=IssueSeverity.MEDIUM,
                    message="存在未完成的转换标记",
                    line_number=line_num,
                    rule_name="completeness_check"
                ))
        
        # 检查代码长度差异
        source_lines = len(source_code.split('\n'))
        target_lines = len(converted_code.split('\n'))
        
        if abs(source_lines - target_lines) > source_lines * 0.5:
            issues.append(ConversionIssue(
                type=IssueType.INFO,
                severity=IssueSeverity.LOW,
                message=f"代码行数变化较大: {source_lines} -> {target_lines}",
                line_number=0,
                rule_name="length_check"
            ))
        
        return issues
    
    async def _check_best_practices(self, converted_code: str) -> List[ConversionIssue]:
        """检查最佳实践"""
        issues = []
        
        try:
            best_practices_result = self.quality_metrics.check_best_practices(
                converted_code, 
                self.config.target_language
            )
            
            for suggestion in best_practices_result.get("suggestions", []):
                issues.append(ConversionIssue(
                    type=IssueType.SUGGESTION,
                    severity=IssueSeverity.LOW,
                    message=suggestion,
                    line_number=0,
                    rule_name="best_practices"
                ))
                
        except Exception:
            pass
        
        return issues
    
    async def _check_performance_issues(self, converted_code: str) -> List[ConversionIssue]:
        """检查性能问题"""
        issues = []
        
        # 简单的性能问题检查
        lines = converted_code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # 检查嵌套循环
            if re.search(r'for.*for.*:', line):
                issues.append(ConversionIssue(
                    type=IssueType.WARNING,
                    severity=IssueSeverity.MEDIUM,
                    message="检测到嵌套循环，可能影响性能",
                    line_number=i,
                    rule_name="performance_check"
                ))
            
            # 检查递归调用
            if re.search(r'def\s+(\w+).*:\s*.*\1\s*\(', line):
                issues.append(ConversionIssue(
                    type=IssueType.INFO,
                    severity=IssueSeverity.LOW,
                    message="检测到递归调用，注意栈溢出风险",
                    line_number=i,
                    rule_name="recursion_check"
                ))
        
        return issues
    
    async def _calculate_quality_score(self, source_code: str, 
                                     converted_code: str) -> Dict[str, Any]:
        """计算质量评分"""
        try:
            # 计算转换评分
            conversion_score = self.quality_metrics.calculate_conversion_score(
                source_code, 
                converted_code, 
                self.config.target_language
            )
            
            # 计算可读性评分
            readability_score = self.quality_metrics.analyze_readability(converted_code)
            
            # 计算可维护性评分
            maintainability_score = self.quality_metrics.assess_maintainability(converted_code)
            
            return {
                "conversion_score": conversion_score,
                "readability_score": readability_score,
                "maintainability_score": maintainability_score,
                "overall_score": (
                    conversion_score.get("overall_score", 0) * 0.4 +
                    readability_score.get("score", 0) * 0.3 +
                    maintainability_score.get("score", 0) * 0.3
                )
            }
            
        except Exception as e:
            return {
                "error": f"质量评分计算失败: {str(e)}",
                "overall_score": 0
            }
    
    def highlight_issues(self, issues: Optional[List[ConversionIssue]] = None) -> Dict[str, Any]:
        """
        生成问题高亮信息
        
        Args:
            issues: 问题列表，如果为None则使用当前问题
            
        Returns:
            高亮信息
        """
        if issues is None:
            issues = self.current_issues
        
        # 按行号分组问题
        issues_by_line = {}
        for issue in issues:
            line_num = issue.line_number
            if line_num not in issues_by_line:
                issues_by_line[line_num] = []
            issues_by_line[line_num].append(issue)
        
        # 生成高亮标记
        highlights = []
        for line_num, line_issues in issues_by_line.items():
            severity_levels = [issue.severity.value for issue in line_issues]
            max_severity = max(severity_levels) if severity_levels else "low"
            
            highlights.append({
                "line": line_num,
                "severity": max_severity,
                "issues": [
                    {
                        "type": issue.type.value,
                        "message": issue.message,
                        "suggested_fix": issue.suggested_fix,
                        "rule_name": issue.rule_name
                    }
                    for issue in line_issues
                ],
                "column_start": min(issue.column_start for issue in line_issues),
                "column_end": max(issue.column_end for issue in line_issues)
            })
        
        return {
            "highlights": highlights,
            "summary": {
                "total_issues": len(issues),
                "errors": len([i for i in issues if i.type == IssueType.ERROR]),
                "warnings": len([i for i in issues if i.type == IssueType.WARNING]),
                "suggestions": len([i for i in issues if i.type == IssueType.SUGGESTION])
            }
        }
    
    def show_side_by_side(self, source: Optional[str] = None, 
                         target: Optional[str] = None) -> Dict[str, Any]:
        """
        生成并排显示数据
        
        Args:
            source: 源代码，如果为None则使用当前源代码
            target: 目标代码，如果为None则使用当前目标代码
            
        Returns:
            并排显示数据
        """
        source = source or self.current_source
        target = target or self.current_target
        
        source_lines = source.split('\n')
        target_lines = target.split('\n')
        
        # 对齐行数
        max_lines = max(len(source_lines), len(target_lines))
        source_lines.extend([''] * (max_lines - len(source_lines)))
        target_lines.extend([''] * (max_lines - len(target_lines)))
        
        # 生成并排数据
        side_by_side = []
        for i, (src_line, tgt_line) in enumerate(zip(source_lines, target_lines), 1):
            side_by_side.append({
                "line_number": i,
                "source_line": src_line,
                "target_line": tgt_line,
                "is_different": src_line.strip() != tgt_line.strip()
            })
        
        return {
            "side_by_side": side_by_side,
            "source_language": self.config.source_language,
            "target_language": self.config.target_language,
            "total_lines": max_lines
        }
    
    def generate_diff_view(self, source: Optional[str] = None, 
                          target: Optional[str] = None) -> Dict[str, Any]:
        """
        生成差异视图
        
        Args:
            source: 源代码，如果为None则使用当前源代码
            target: 目标代码，如果为None则使用当前目标代码
            
        Returns:
            差异视图数据
        """
        source = source or self.current_source
        target = target or self.current_target
        
        # 生成统一差异格式
        diff_lines = list(difflib.unified_diff(
            source.splitlines(keepends=True),
            target.splitlines(keepends=True),
            fromfile=f"source.{self._get_file_extension(self.config.source_language)}",
            tofile=f"target.{self._get_file_extension(self.config.target_language)}",
            lineterm=''
        ))
        
        # 解析差异
        changes = []
        current_hunk = None
        
        for line in diff_lines:
            if line.startswith('@@'):
                # 新的差异块
                if current_hunk:
                    changes.append(current_hunk)
                current_hunk = {
                    "header": line.strip(),
                    "lines": []
                }
            elif current_hunk:
                line_type = "context"
                if line.startswith('+'):
                    line_type = "addition"
                elif line.startswith('-'):
                    line_type = "deletion"
                
                current_hunk["lines"].append({
                    "type": line_type,
                    "content": line[1:] if line_type != "context" else line,
                    "line_number": len(current_hunk["lines"]) + 1
                })
        
        if current_hunk:
            changes.append(current_hunk)
        
        # 计算统计信息
        additions = sum(1 for change in changes for line in change["lines"] 
                       if line["type"] == "addition")
        deletions = sum(1 for change in changes for line in change["lines"] 
                       if line["type"] == "deletion")
        
        return {
            "diff_text": ''.join(diff_lines),
            "changes": changes,
            "statistics": {
                "additions": additions,
                "deletions": deletions,
                "modifications": len(changes)
            }
        }
    
    def _generate_cache_key(self, source_code: str) -> str:
        """生成缓存键"""
        import hashlib
        content = f"{source_code}_{self.config.source_language}_{self.config.target_language}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _update_cache(self, key: str, result: PreviewResult):
        """更新缓存"""
        if len(self._preview_cache) >= self._cache_max_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self._preview_cache))
            del self._preview_cache[oldest_key]
        
        self._preview_cache[key] = result
    
    def _extract_line_number(self, error_message: str) -> int:
        """从错误消息中提取行号"""
        import re
        match = re.search(r'line (\d+)', error_message)
        return int(match.group(1)) if match else 0
    
    def _get_file_extension(self, language: str) -> str:
        """获取语言对应的文件扩展名"""
        extensions = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "java": "java",
            "csharp": "cs",
            "cpp": "cpp",
            "rust": "rs"
        }
        return extensions.get(language, "txt")
    
    def clear_cache(self):
        """清空缓存"""
        self._preview_cache.clear()
    
    def get_preview_stats(self) -> Dict[str, Any]:
        """获取预览统计信息"""
        return {
            "cache_size": len(self._preview_cache),
            "last_preview_time": self.last_preview_time,
            "current_issues_count": len(self.current_issues),
            "source_lines": len(self.current_source.split('\n')) if self.current_source else 0,
            "target_lines": len(self.current_target.split('\n')) if self.current_target else 0
        }


# 全局预览实例
_preview_instance = None

def get_live_preview() -> LivePreview:
    """获取全局预览实例"""
    global _preview_instance
    if _preview_instance is None:
        _preview_instance = LivePreview()
    return _preview_instance