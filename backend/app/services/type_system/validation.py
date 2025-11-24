"""
类型安全验证模块

提供转换前后类型等价性、边界情况处理和运行时行为保持的验证功能
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .inference import TypeInfo, TypeCategory, TypeFactory
from .mapping import TypeMapping, TypeMappingSystem


class ValidationLevel(Enum):
    """验证级别"""
    STRICT = "strict"  # 严格验证，要求完全等价
    COMPATIBLE = "compatible"  # 兼容验证，允许安全的类型转换
    PERMISSIVE = "permissive"  # 宽松验证，允许可能的类型转换


@dataclass
class ValidationIssue:
    """验证问题"""
    issue_type: str
    source_type: TypeInfo
    target_type: TypeInfo
    message: str
    severity: str  # "error", "warning", "info"
    location: Optional[Dict[str, Any]] = None
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """添加问题"""
        self.issues.append(issue)
        if issue.severity == "error":
            self.is_valid = False
    
    def merge(self, other: 'ValidationResult') -> None:
        """合并验证结果"""
        if not other.is_valid:
            self.is_valid = False
        self.issues.extend(other.issues)


class TypeSafetyValidator:
    """类型安全验证器"""
    
    def __init__(self, mapping_system: TypeMappingSystem):
        self.mapping_system = mapping_system
    
    def validate_type_conversion(self, source_type: TypeInfo, target_type: TypeInfo, 
                                level: ValidationLevel = ValidationLevel.COMPATIBLE) -> ValidationResult:
        """验证类型转换"""
        result = ValidationResult(is_valid=True)
        
        # 根据类型分类验证
        if source_type.category == TypeCategory.PRIMITIVE:
            self._validate_primitive_conversion(source_type, target_type, level, result)
        
        elif source_type.category == TypeCategory.CONTAINER:
            self._validate_container_conversion(source_type, target_type, level, result)
        
        elif source_type.category == TypeCategory.UNION:
            self._validate_union_conversion(source_type, target_type, level, result)
        
        elif source_type.category == TypeCategory.OPTIONAL:
            self._validate_optional_conversion(source_type, target_type, level, result)
        
        elif source_type.category == TypeCategory.CLASS:
            self._validate_class_conversion(source_type, target_type, level, result)
        
        elif source_type.category == TypeCategory.FUNCTION:
            self._validate_function_conversion(source_type, target_type, level, result)
        
        elif source_type.category == TypeCategory.ANY:
            self._validate_any_conversion(source_type, target_type, level, result)
        
        return result
    
    def _validate_primitive_conversion(self, source_type: TypeInfo, target_type: TypeInfo, 
                                      level: ValidationLevel, result: ValidationResult) -> None:
        """验证原始类型转换"""
        # 如果目标类型也是原始类型
        if target_type.category == TypeCategory.PRIMITIVE:
            # 检查类型映射是否存在
            mapping = self.mapping_system.registry.get_mapping(
                source_type.source_lang, target_type.source_lang, source_type.name
            )
            
            if mapping and mapping.target_type.name == target_type.name:
                # 映射匹配，验证通过
                return
            
            # 检查数值类型的兼容性
            if level != ValidationLevel.STRICT and self._is_numeric_compatible(source_type, target_type):
                if level == ValidationLevel.COMPATIBLE:
                    result.add_issue(ValidationIssue(
                        issue_type="numeric_conversion",
                        source_type=source_type,
                        target_type=target_type,
                        message=f"数值类型转换可能导致精度损失: {source_type.name} -> {target_type.name}",
                        severity="warning",
                        suggestions=["考虑使用更精确的类型", "添加显式类型转换"]
                    ))
                return
            
            # 检查字符串类型的兼容性
            if level != ValidationLevel.STRICT and self._is_string_compatible(source_type, target_type):
                return
            
            # 检查布尔类型的兼容性
            if level != ValidationLevel.STRICT and self._is_boolean_compatible(source_type, target_type):
                return
            
            # 其他情况，添加错误
            result.add_issue(ValidationIssue(
                issue_type="incompatible_primitive_types",
                source_type=source_type,
                target_type=target_type,
                message=f"不兼容的原始类型转换: {source_type.name} -> {target_type.name}",
                severity="error",
                suggestions=["使用适当的类型转换函数", "修改目标类型"]
            ))
        
        # 如果目标类型是联合类型，检查源类型是否是联合类型的成员
        elif target_type.category == TypeCategory.UNION:
            for member_type in target_type.type_args:
                sub_result = self.validate_type_conversion(source_type, member_type, level)
                if sub_result.is_valid:
                    return
            
            result.add_issue(ValidationIssue(
                issue_type="primitive_to_union_incompatible",
                source_type=source_type,
                target_type=target_type,
                message=f"原始类型 {source_type.name} 不兼容联合类型的任何成员",
                severity="error",
                suggestions=["扩展联合类型以包含兼容类型", "修改源类型"]
            ))
        
        # 如果目标类型是可选类型，检查源类型是否与内部类型兼容
        elif target_type.category == TypeCategory.OPTIONAL:
            inner_type = target_type.type_args[0]
            sub_result = self.validate_type_conversion(source_type, inner_type, level)
            if not sub_result.is_valid:
                result.add_issue(ValidationIssue(
                    issue_type="primitive_to_optional_incompatible",
                    source_type=source_type,
                    target_type=target_type,
                    message=f"原始类型 {source_type.name} 不兼容可选类型的内部类型 {inner_type.name}",
                    severity="error",
                    suggestions=["修改可选类型的内部类型", "修改源类型"]
                ))
        
        # 如果目标类型是任意类型，总是有效
        elif target_type.category == TypeCategory.ANY:
            return
        
        # 其他情况，添加错误
        else:
            result.add_issue(ValidationIssue(
                issue_type="primitive_to_non_primitive",
                source_type=source_type,
                target_type=target_type,
                message=f"原始类型 {source_type.name} 不能转换为非原始类型 {target_type.name}",
                severity="error",
                suggestions=["使用适当的包装类型", "修改目标类型"]
            ))
    
    def _validate_container_conversion(self, source_type: TypeInfo, target_type: TypeInfo, 
                                      level: ValidationLevel, result: ValidationResult) -> None:
        """验证容器类型转换"""
        # 如果目标类型也是容器类型
        if target_type.category == TypeCategory.CONTAINER:
            # 检查容器类型是否兼容
            if not self._is_container_compatible(source_type, target_type, level):
                result.add_issue(ValidationIssue(
                    issue_type="incompatible_container_types",
                    source_type=source_type,
                    target_type=target_type,
                    message=f"不兼容的容器类型转换: {source_type.name} -> {target_type.name}",
                    severity="error",
                    suggestions=["使用适当的容器转换函数", "修改目标容器类型"]
                ))
                return
            
            # 检查容器元素类型是否兼容
            if source_type.type_args and target_type.type_args:
                for i, (source_arg, target_arg) in enumerate(zip(source_type.type_args, target_type.type_args)):
                    sub_result = self.validate_type_conversion(source_arg, target_arg, level)
                    if not sub_result.is_valid:
                        result.add_issue(ValidationIssue(
                            issue_type="incompatible_container_element_types",
                            source_type=source_arg,
                            target_type=target_arg,
                            message=f"容器元素类型不兼容: 位置 {i}, {source_arg.name} -> {target_arg.name}",
                            severity="error",
                            suggestions=["修改容器元素类型", "使用类型转换函数"]
                        ))
                        result.merge(sub_result)
        
        # 如果目标类型是任意类型，总是有效
        elif target_type.category == TypeCategory.ANY:
            return
        
        # 其他情况，添加错误
        else:
            result.add_issue(ValidationIssue(
                issue_type="container_to_non_container",
                source_type=source_type,
                target_type=target_type,
                message=f"容器类型 {source_type.name} 不能转换为非容器类型 {target_type.name}",
                severity="error",
                suggestions=["使用适当的容器转换函数", "修改目标类型"]
            ))
    
    def _validate_union_conversion(self, source_type: TypeInfo, target_type: TypeInfo, 
                                  level: ValidationLevel, result: ValidationResult) -> None:
        """验证联合类型转换"""
        # 如果目标类型是联合类型
        if target_type.category == TypeCategory.UNION:
            # 检查每个源成员是否至少与一个目标成员兼容
            all_source_members_compatible = True
            
            for source_member in source_type.type_args:
                member_compatible = False
                
                for target_member in target_type.type_args:
                    sub_result = self.validate_type_conversion(source_member, target_member, level)
                    if sub_result.is_valid:
                        member_compatible = True
                        break
                
                if not member_compatible:
                    all_source_members_compatible = False
                    result.add_issue(ValidationIssue(
                        issue_type="union_member_incompatible",
                        source_type=source_member,
                        target_type=target_type,
                        message=f"联合类型成员 {source_member.name} 不兼容目标联合类型的任何成员",
                        severity="error",
                        suggestions=["扩展目标联合类型", "修改源联合类型"]
                    ))
            
            if all_source_members_compatible:
                return
        
        # 如果目标类型是单一类型，检查每个源成员是否与目标类型兼容
        elif target_type.category != TypeCategory.UNION:
            all_source_members_compatible = True
            
            for source_member in source_type.type_args:
                sub_result = self.validate_type_conversion(source_member, target_type, level)
                if not sub_result.is_valid:
                    all_source_members_compatible = False
                    result.add_issue(ValidationIssue(
                        issue_type="union_to_single_incompatible",
                        source_type=source_member,
                        target_type=target_type,
                        message=f"联合类型成员 {source_member.name} 不兼容目标类型 {target_type.name}",
                        severity="error",
                        suggestions=["使用更通用的目标类型", "修改源联合类型"]
                    ))
                    result.merge(sub_result)
            
            if all_source_members_compatible:
                return
        
        # 如果目标类型是任意类型，总是有效
        if target_type.category == TypeCategory.ANY:
            return
    
    def _validate_optional_conversion(self, source_type: TypeInfo, target_type: TypeInfo, 
                                     level: ValidationLevel, result: ValidationResult) -> None:
        """验证可选类型转换"""
        # 如果目标类型也是可选类型
        if target_type.category == TypeCategory.OPTIONAL:
            # 验证内部类型
            source_inner = source_type.type_args[0]
            target_inner = target_type.type_args[0]
            
            sub_result = self.validate_type_conversion(source_inner, target_inner, level)
            if not sub_result.is_valid:
                result.add_issue(ValidationIssue(
                    issue_type="incompatible_optional_inner_types",
                    source_type=source_inner,
                    target_type=target_inner,
                    message=f"可选类型的内部类型不兼容: {source_inner.name} -> {target_inner.name}",
                    severity="error",
                    suggestions=["修改内部类型", "使用类型转换函数"]
                ))
                result.merge(sub_result)
        
        # 如果目标类型是联合类型，检查是否包含null/None和兼容类型
        elif target_type.category == TypeCategory.UNION:
            # 检查源内部类型是否与目标联合类型的某个成员兼容
            source_inner = source_type.type_args[0]
            inner_compatible = False
            null_compatible = False
            
            for target_member in target_type.type_args:
                # 检查是否有null/None类型
                if self._is_null_type(target_member):
                    null_compatible = True
                else:
                    # 检查内部类型兼容性
                    sub_result = self.validate_type_conversion(source_inner, target_member, level)
                    if sub_result.is_valid:
                        inner_compatible = True
            
            if not (inner_compatible and null_compatible):
                result.add_issue(ValidationIssue(
                    issue_type="optional_to_union_incompatible",
                    source_type=source_type,
                    target_type=target_type,
                    message=f"可选类型 {source_type.name} 不兼容目标联合类型 {target_type.name}",
                    severity="error",
                    suggestions=[
                        "确保联合类型包含null/None类型",
                        "确保联合类型包含与内部类型兼容的类型"
                    ]
                ))
        
        # 如果目标类型是非可选非联合类型，检查是否处理了null/None情况
        elif target_type.category not in [TypeCategory.OPTIONAL, TypeCategory.UNION, TypeCategory.ANY]:
            # 验证内部类型
            source_inner = source_type.type_args[0]
            sub_result = self.validate_type_conversion(source_inner, target_type, level)
            
            if sub_result.is_valid:
                # 内部类型兼容，但需要处理null/None情况
                result.add_issue(ValidationIssue(
                    issue_type="optional_to_non_optional",
                    source_type=source_type,
                    target_type=target_type,
                    message=f"可选类型 {source_type.name} 转换为非可选类型 {target_type.name} 需要处理null/None情况",
                    severity="warning",
                    suggestions=[
                        "添加null/None检查",
                        "提供默认值",
                        "使用可选类型作为目标类型"
                    ]
                ))
            else:
                result.add_issue(ValidationIssue(
                    issue_type="incompatible_optional_to_non_optional",
                    source_type=source_type,
                    target_type=target_type,
                    message=f"可选类型 {source_type.name} 的内部类型不兼容目标类型 {target_type.name}",
                    severity="error",
                    suggestions=["修改目标类型", "修改源可选类型的内部类型"]
                ))
                result.merge(sub_result)
        
        # 如果目标类型是任意类型，总是有效
        elif target_type.category == TypeCategory.ANY:
            return
    
    def _validate_class_conversion(self, source_type: TypeInfo, target_type: TypeInfo, 
                                  level: ValidationLevel, result: ValidationResult) -> None:
        """验证类类型转换"""
        # 如果目标类型也是类类型
        if target_type.category == TypeCategory.CLASS:
            # 检查类型映射是否存在
            mapping = self.mapping_system.registry.get_mapping(
                source_type.source_lang, target_type.source_lang, source_type.name
            )
            
            if mapping and mapping.target_type.name == target_type.name:
                # 映射匹配，验证通过
                return
            
            # 如果没有直接映射，但级别不是严格的，可以考虑结构兼容性
            if level != ValidationLevel.STRICT:
                # 这里应该检查结构兼容性，但需要更多的类型信息
                # 简化处理：添加警告
                result.add_issue(ValidationIssue(
                    issue_type="class_structural_compatibility",
                    source_type=source_type,
                    target_type=target_type,
                    message=f"类类型 {source_type.name} 到 {target_type.name} 的转换需要结构兼容性验证",
                    severity="warning",
                    suggestions=[
                        "确保目标类包含源类的所有必要属性和方法",
                        "实现适当的转换函数"
                    ]
                ))
                
                if level == ValidationLevel.PERMISSIVE:
                    return
                else:
                    result.add_issue(ValidationIssue(
                        issue_type="class_no_direct_mapping",
                        source_type=source_type,
                        target_type=target_type,
                        message=f"类类型 {source_type.name} 到 {target_type.name} 没有直接映射",
                        severity="error",
                        suggestions=[
                            "添加显式类型映射",
                            "使用适当的转换函数"
                        ]
                    ))
            else:
                result.add_issue(ValidationIssue(
                    issue_type="class_no_direct_mapping",
                    source_type=source_type,
                    target_type=target_type,
                    message=f"类类型 {source_type.name} 到 {target_type.name} 没有直接映射",
                    severity="error",
                    suggestions=[
                        "添加显式类型映射",
                        "使用适当的转换函数"
                    ]
                ))
        
        # 如果目标类型是任意类型，总是有效
        elif target_type.category == TypeCategory.ANY:
            return
        
        # 其他情况，添加错误
        else:
            result.add_issue(ValidationIssue(
                issue_type="class_to_non_class",
                source_type=source_type,
                target_type=target_type,
                message=f"类类型 {source_type.name} 不能转换为非类类型 {target_type.name}",
                severity="error",
                suggestions=["使用适当的转换函数", "修改目标类型"]
            ))
    
    def _validate_function_conversion(self, source_type: TypeInfo, target_type: TypeInfo, 
                                     level: ValidationLevel, result: ValidationResult) -> None:
        """验证函数类型转换"""
        # 如果目标类型也是函数类型
        if target_type.category == TypeCategory.FUNCTION:
            # 简化处理：添加警告
            result.add_issue(ValidationIssue(
                issue_type="function_compatibility",
                source_type=source_type,
                target_type=target_type,
                message=f"函数类型 {source_type.name} 到 {target_type.name} 的转换需要参数和返回类型兼容性验证",
                severity="warning",
                suggestions=[
                    "确保参数类型和返回类型兼容",
                    "实现适当的包装函数"
                ]
            ))
            
            if level == ValidationLevel.PERMISSIVE:
                return
        
        # 如果目标类型是任意类型，总是有效
        elif target_type.category == TypeCategory.ANY:
            return
        
        # 其他情况，添加错误
        else:
            result.add_issue(ValidationIssue(
                issue_type="function_to_non_function",
                source_type=source_type,
                target_type=target_type,
                message=f"函数类型 {source_type.name} 不能转换为非函数类型 {target_type.name}",
                severity="error",
                suggestions=["使用适当的转换函数", "修改目标类型"]
            ))
    
    def _validate_any_conversion(self, source_type: TypeInfo, target_type: TypeInfo, 
                               level: ValidationLevel, result: ValidationResult) -> None:
        """验证任意类型转换"""
        # 如果目标类型不是任意类型，添加警告
        if target_type.category != TypeCategory.ANY:
            result.add_issue(ValidationIssue(
                issue_type="any_to_specific",
                source_type=source_type,
                target_type=target_type,
                message=f"从任意类型 {source_type.name} 到具体类型 {target_type.name} 的转换需要运行时类型检查",
                severity="warning",
                suggestions=[
                    "添加运行时类型检查",
                    "使用类型断言或类型守卫"
                ]
            ))
    
    def _is_numeric_compatible(self, source_type: TypeInfo, target_type: TypeInfo) -> bool:
        """检查数值类型是否兼容"""
        numeric_types = {
            "python": ["int", "float"],
            "javascript": ["number"],
            "typescript": ["number"],
            "java": ["byte", "short", "int", "long", "float", "double"],
            "csharp": ["byte", "sbyte", "short", "ushort", "int", "uint", "long", "ulong", "float", "double", "decimal"]
        }
        
        source_lang = source_type.source_lang
        target_lang = target_type.source_lang
        
        # 检查源类型和目标类型是否都是数值类型
        if (source_lang in numeric_types and source_type.name in numeric_types[source_lang] and
            target_lang in numeric_types and target_type.name in numeric_types[target_lang]):
            return True
        
        return False
    
    def _is_string_compatible(self, source_type: TypeInfo, target_type: TypeInfo) -> bool:
        """检查字符串类型是否兼容"""
        string_types = {
            "python": ["str"],
            "javascript": ["string"],
            "typescript": ["string"],
            "java": ["String", "char"],
            "csharp": ["string", "char"]
        }
        
        source_lang = source_type.source_lang
        target_lang = target_type.source_lang
        
        # 检查源类型和目标类型是否都是字符串类型
        if (source_lang in string_types and source_type.name in string_types[source_lang] and
            target_lang in string_types and target_type.name in string_types[target_lang]):
            return True
        
        return False
    
    def _is_boolean_compatible(self, source_type: TypeInfo, target_type: TypeInfo) -> bool:
        """检查布尔类型是否兼容"""
        boolean_types = {
            "python": ["bool"],
            "javascript": ["boolean"],
            "typescript": ["boolean"],
            "java": ["boolean"],
            "csharp": ["bool"]
        }
        
        source_lang = source_type.source_lang
        target_lang = target_type.source_lang
        
        # 检查源类型和目标类型是否都是布尔类型
        if (source_lang in boolean_types and source_type.name in boolean_types[source_lang] and
            target_lang in boolean_types and target_type.name in boolean_types[target_lang]):
            return True
        
        return False
    
    def _is_container_compatible(self, source_type: TypeInfo, target_type: TypeInfo, 
                               level: ValidationLevel) -> bool:
        """检查容器类型是否兼容"""
        # 定义容器类型的兼容性映射
        container_compatibility = {
            "python": {
                "List": ["javascript:Array", "typescript:Array", "java:List", "csharp:List"],
                "Dict": ["javascript:Object", "typescript:Record", "java:Map", "csharp:Dictionary"],
                "Set": ["javascript:Set", "typescript:Set", "java:Set", "csharp:HashSet"],
                "Tuple": ["javascript:Array", "typescript:Array"]
            },
            "javascript": {
                "Array": ["python:List", "typescript:Array", "java:List", "csharp:List"],
                "Object": ["python:Dict", "typescript:Record", "java:Map", "csharp:Dictionary"],
                "Set": ["python:Set", "typescript:Set", "java:Set", "csharp:HashSet"]
            },
            "typescript": {
                "Array": ["python:List", "javascript:Array", "java:List", "csharp:List"],
                "Record": ["python:Dict", "javascript:Object", "java:Map", "csharp:Dictionary"],
                "Set": ["python:Set", "javascript:Set", "java:Set", "csharp:HashSet"]
            },
            "java": {
                "List": ["python:List", "javascript:Array", "typescript:Array", "csharp:List"],
                "Map": ["python:Dict", "javascript:Object", "typescript:Record", "csharp:Dictionary"],
                "Set": ["python:Set", "javascript:Set", "typescript:Set", "csharp:HashSet"]
            },
            "csharp": {
                "List": ["python:List", "javascript:Array", "typescript:Array", "java:List"],
                "Dictionary": ["python:Dict", "javascript:Object", "typescript:Record", "java:Map"],
                "HashSet": ["python:Set", "javascript:Set", "typescript:Set", "java:Set"]
            }
        }
        
        source_lang = source_type.source_lang
        target_lang = target_type.source_lang
        
        # 如果源语言和目标语言相同，容器类型也相同，则兼容
        if source_lang == target_lang and source_type.name == target_type.name:
            return True
        
        # 检查容器类型的兼容性映射
        if (source_lang in container_compatibility and 
            source_type.name in container_compatibility[source_lang]):
            compatible_targets = container_compatibility[source_lang][source_type.name]
            target_key = f"{target_lang}:{target_type.name}"
            
            if target_key in compatible_targets:
                return True
        
        # 如果是宽松验证，可以考虑更宽松的兼容性
        if level == ValidationLevel.PERMISSIVE:
            # 简化处理：在宽松模式下，允许任何容器类型之间的转换
            return True
        
        return False
    
    def _is_null_type(self, type_info: TypeInfo) -> bool:
        """检查是否是null/None类型"""
        null_types = {
            "python": ["None"],
            "javascript": ["null", "undefined"],
            "typescript": ["null", "undefined"],
            "java": ["null"],
            "csharp": ["null"]
        }
        
        lang = type_info.source_lang
        
        return lang in null_types and type_info.name in null_types[lang]


class TypeSafetyReport:
    """类型安全报告"""
    
    def __init__(self, validation_result: ValidationResult):
        self.validation_result = validation_result
    
    def generate_report(self) -> Dict[str, Any]:
        """生成报告"""
        issues_by_severity = {
            "error": [],
            "warning": [],
            "info": []
        }
        
        for issue in self.validation_result.issues:
            issues_by_severity[issue.severity].append({
                "issue_type": issue.issue_type,
                "source_type": self._format_type(issue.source_type),
                "target_type": self._format_type(issue.target_type),
                "message": issue.message,
                "suggestions": issue.suggestions,
                "location": issue.location
            })
        
        return {
            "is_valid": self.validation_result.is_valid,
            "issues_count": {
                "error": len(issues_by_severity["error"]),
                "warning": len(issues_by_severity["warning"]),
                "info": len(issues_by_severity["info"]),
                "total": len(self.validation_result.issues)
            },
            "issues": {
                "error": issues_by_severity["error"],
                "warning": issues_by_severity["warning"],
                "info": issues_by_severity["info"]
            }
        }
    
    def _format_type(self, type_info: TypeInfo) -> str:
        """格式化类型信息"""
        if type_info.category == TypeCategory.CONTAINER and type_info.type_args:
            args_str = ", ".join(self._format_type(arg) for arg in type_info.type_args)
            return f"{type_info.name}<{args_str}>"
        elif type_info.category == TypeCategory.UNION:
            args_str = " | ".join(self._format_type(arg) for arg in type_info.type_args)
            return f"({args_str})"
        elif type_info.category == TypeCategory.OPTIONAL:
            inner_str = self._format_type(type_info.type_args[0])
            return f"{inner_str} | null"
        else:
            return type_info.name