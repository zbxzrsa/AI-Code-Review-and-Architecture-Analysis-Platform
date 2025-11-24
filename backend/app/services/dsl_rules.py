"""
自定义DSL规则引擎
支持用户编写自定义规则来扩展代码分析能力
"""

import ast
import re
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """规则类型"""
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    CUSTOM = "custom"


class Severity(Enum):
    """严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class RuleContext:
    """规则执行上下文"""
    file_path: str
    file_content: str
    ast_tree: Optional[ast.AST]
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleViolation:
    """规则违反"""
    rule_id: str
    rule_name: str
    severity: Severity
    message: str
    line_number: int
    column_number: int
    end_line_number: Optional[int] = None
    end_column_number: Optional[int] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DSLRule:
    """DSL规则定义"""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    severity: Severity
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    dsl_code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type.value,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "tags": self.tags,
            "dsl_code": self.dsl_code,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DSLRule":
        """从字典创建规则"""
        return cls(
            rule_id=data["rule_id"],
            name=data["name"],
            description=data["description"],
            rule_type=RuleType(data["rule_type"]),
            severity=Severity(data["severity"]),
            enabled=data.get("enabled", True),
            tags=data.get("tags", []),
            dsl_code=data.get("dsl_code", ""),
            metadata=data.get("metadata", {})
        )


class DSLFunction(ABC):
    """DSL函数基类"""
    
    @abstractmethod
    def execute(self, context: RuleContext, *args, **kwargs) -> Any:
        """执行函数"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """函数名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """函数描述"""
        pass


class FileFunction(DSLFunction):
    """文件相关函数"""
    
    def __init__(self, name: str, func: Callable, description: str):
        self._name = name
        self._func = func
        self._description = description
    
    def execute(self, context: RuleContext, *args, **kwargs) -> Any:
        return self._func(context, *args, **kwargs)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description


class ASTFunction(DSLFunction):
    """AST相关函数"""
    
    def __init__(self, name: str, func: Callable, description: str):
        self._name = name
        self._func = func
        self._description = description
    
    def execute(self, context: RuleContext, *args, **kwargs) -> Any:
        if not context.ast_tree:
            return None
        return self._func(context.ast_tree, *args, **kwargs)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description


class DSLEngine:
    """DSL引擎"""
    
    def __init__(self):
        self.functions: Dict[str, DSLFunction] = {}
        self._register_builtin_functions()
    
    def _register_builtin_functions(self):
        """注册内置函数"""
        
        # 文件相关函数
        self.register_function(FileFunction(
            "file_size",
            lambda ctx: len(ctx.file_content),
            "Get file size in characters"
        ))
        
        self.register_function(FileFunction(
            "file_lines",
            lambda ctx: len(ctx.file_content.splitlines()),
            "Get number of lines in file"
        ))
        
        self.register_function(FileFunction(
            "file_extension",
            lambda ctx: Path(ctx.file_path).suffix,
            "Get file extension"
        ))
        
        self.register_function(FileFunction(
            "file_name",
            lambda ctx: Path(ctx.file_path).name,
            "Get file name"
        ))
        
        self.register_function(FileFunction(
            "contains_regex",
            lambda ctx, pattern: bool(re.search(pattern, ctx.file_content)),
            "Check if file content matches regex pattern"
        ))
        
        self.register_function(FileFunction(
            "count_regex",
            lambda ctx, pattern: len(re.findall(pattern, ctx.file_content)),
            "Count regex matches in file"
        ))
        
        # AST相关函数
        self.register_function(ASTFunction(
            "count_functions",
            lambda tree: len([node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]),
            "Count number of functions"
        ))
        
        self.register_function(ASTFunction(
            "count_classes",
            lambda tree: len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
            "Count number of classes"
        ))
        
        self.register_function(ASTFunction(
            "get_function_names",
            lambda tree: [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))],
            "Get all function names"
        ))
        
        self.register_function(ASTFunction(
            "get_class_names",
            lambda tree: [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)],
            "Get all class names"
        ))
        
        self.register_function(ASTFunction(
            "has_import",
            lambda tree, module: any(
                isinstance(node, ast.Import) and any(alias.name == module for alias in node.names) or
                isinstance(node, ast.ImportFrom) and node.module == module
                for node in ast.walk(tree)
            ),
            "Check if module is imported"
        ))
        
        self.register_function(ASTFunction(
            "count_lines_of_code",
            lambda tree: sum(
                node.end_lineno - node.lineno + 1
                for node in ast.walk(tree)
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno')
            ),
            "Count lines of code (approximate)"
        ))
    
    def register_function(self, function: DSLFunction):
        """注册函数"""
        self.functions[function.name] = function
        logger.info(f"Registered DSL function: {function.name}")
    
    def execute_rule(self, rule: DSLRule, context: RuleContext) -> List[RuleViolation]:
        """执行规则"""
        if not rule.enabled:
            return []
        
        violations = []
        
        try:
            # 编译DSL代码
            compiled_code = compile(rule.dsl_code, f"<rule:{rule.rule_id}>", "exec")
            
            # 创建执行环境
            exec_env = {
                "context": context,
                "violation": self._create_violation_function(rule),
                "info": self._create_log_function(Severity.INFO),
                "warning": self._create_log_function(Severity.WARNING),
                "error": self._create_log_function(Severity.ERROR),
                "critical": self._create_log_function(Severity.CRITICAL),
                **{func.name: func.execute for func in self.functions.values()}
            }
            
            # 执行DSL代码
            exec(compiled_code, exec_env)
            
            # 获取违规记录
            violations = exec_env.get("_violations", [])
            
        except Exception as e:
            logger.error(f"Error executing rule {rule.rule_id}: {e}")
            violations.append(RuleViolation(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=Severity.ERROR,
                message=f"Rule execution error: {str(e)}",
                line_number=1,
                column_number=1
            ))
        
        return violations
    
    def _create_violation_function(self, rule: DSLRule) -> Callable:
        """创建违规记录函数"""
        def violation_func(
            message: str,
            line_number: int = 1,
            column_number: int = 1,
            end_line_number: Optional[int] = None,
            end_column_number: Optional[int] = None,
            severity: Optional[Severity] = None,
            suggestion: Optional[str] = None,
            **metadata
        ):
            violations = globals().get("_violations", [])
            if not violations:
                globals()["_violations"] = violations
            
            violation = RuleViolation(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=severity or rule.severity,
                message=message,
                line_number=line_number,
                column_number=column_number,
                end_line_number=end_line_number,
                end_column_number=end_column_number,
                suggestion=suggestion,
                metadata=metadata
            )
            
            violations.append(violation)
        
        return violation_func
    
    def _create_log_function(self, severity: Severity) -> Callable:
        """创建日志函数"""
        def log_func(message: str, **kwargs):
            logger.info(f"[{severity.value.upper()}] {message}")
        
        return log_func
    
    def validate_dsl_code(self, dsl_code: str) -> List[str]:
        """验证DSL代码"""
        errors = []
        
        try:
            # 尝试编译代码
            compile(dsl_code, "<validation>", "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
        except Exception as e:
            errors.append(f"Compilation error: {e}")
        
        return errors
    
    def get_function_list(self) -> List[Dict[str, str]]:
        """获取可用函数列表"""
        return [
            {
                "name": func.name,
                "description": func.description,
                "type": type(func).__name__
            }
            for func in self.functions.values()
        ]


class RuleManager:
    """规则管理器"""
    
    def __init__(self, dsl_engine: DSLEngine):
        self.dsl_engine = dsl_engine
        self.rules: Dict[str, DSLRule] = {}
        self.rule_categories: Dict[RuleType, List[str]] = {
            rule_type: [] for rule_type in RuleType
        }
    
    def add_rule(self, rule: DSLRule):
        """添加规则"""
        self.rules[rule.rule_id] = rule
        self.rule_categories[rule.rule_type].append(rule.rule_id)
        logger.info(f"Added rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """删除规则"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            self.rule_categories[rule.rule_type].remove(rule_id)
            del self.rules[rule_id]
            logger.info(f"Removed rule: {rule_id}")
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[DSLRule]:
        """获取规则"""
        return self.rules.get(rule_id)
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[DSLRule]:
        """按类型获取规则"""
        return [
            self.rules[rule_id] 
            for rule_id in self.rule_categories[rule_type]
            if rule_id in self.rules
        ]
    
    def get_enabled_rules(self) -> List[DSLRule]:
        """获取启用的规则"""
        return [rule for rule in self.rules.values() if rule.enabled]
    
    def enable_rule(self, rule_id: str) -> bool:
        """启用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """禁用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            return True
        return False
    
    def execute_rules(
        self, 
        context: RuleContext,
        rule_ids: Optional[List[str]] = None,
        rule_types: Optional[List[RuleType]] = None
    ) -> List[RuleViolation]:
        """执行规则"""
        violations = []
        
        # 确定要执行的规则
        rules_to_execute = []
        
        if rule_ids:
            rules_to_execute = [
                self.rules[rule_id] for rule_id in rule_ids 
                if rule_id in self.rules
            ]
        elif rule_types:
            rules_to_execute = []
            for rule_type in rule_types:
                rules_to_execute.extend(self.get_rules_by_type(rule_type))
        else:
            rules_to_execute = self.get_enabled_rules()
        
        # 执行规则
        for rule in rules_to_execute:
            try:
                rule_violations = self.dsl_engine.execute_rule(rule, context)
                violations.extend(rule_violations)
            except Exception as e:
                logger.error(f"Error executing rule {rule.rule_id}: {e}")
        
        return violations
    
    def load_rules_from_file(self, file_path: str) -> int:
        """从文件加载规则"""
        loaded_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            if isinstance(data, list):
                rules_data = data
            elif isinstance(data, dict) and 'rules' in data:
                rules_data = data['rules']
            else:
                raise ValueError("Invalid rule file format")
            
            for rule_data in rules_data:
                rule = DSLRule.from_dict(rule_data)
                self.add_rule(rule)
                loaded_count += 1
            
            logger.info(f"Loaded {loaded_count} rules from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load rules from {file_path}: {e}")
        
        return loaded_count
    
    def save_rules_to_file(self, file_path: str, rule_ids: Optional[List[str]] = None):
        """保存规则到文件"""
        rules_to_save = []
        
        if rule_ids:
            rules_to_save = [
                self.rules[rule_id] for rule_id in rule_ids 
                if rule_id in self.rules
            ]
        else:
            rules_to_save = list(self.rules.values())
        
        rules_data = [rule.to_dict() for rule in rules_to_save]
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump({"rules": rules_data}, f, default_flow_style=False, indent=2)
                else:
                    json.dump({"rules": rules_data}, f, indent=2)
            
            logger.info(f"Saved {len(rules_to_save)} rules to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save rules to {file_path}: {e}")
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """获取规则统计"""
        stats = {
            "total_rules": len(self.rules),
            "enabled_rules": len(self.get_enabled_rules()),
            "disabled_rules": len(self.rules) - len(self.get_enabled_rules()),
            "by_type": {},
            "by_severity": {}
        }
        
        # 按类型统计
        for rule_type in RuleType:
            rules = self.get_rules_by_type(rule_type)
            stats["by_type"][rule_type.value] = {
                "total": len(rules),
                "enabled": len([r for r in rules if r.enabled])
            }
        
        # 按严重程度统计
        for severity in Severity:
            rules = [r for r in self.rules.values() if r.severity == severity]
            stats["by_severity"][severity.value] = {
                "total": len(rules),
                "enabled": len([r for r in rules if r.enabled])
            }
        
        return stats


# 全局实例
dsl_engine = DSLEngine()
rule_manager = RuleManager(dsl_engine)


# 示例规则
EXAMPLE_RULES = [
    {
        "rule_id": "custom_large_file",
        "name": "Large File Detection",
        "description": "Detect files that are too large",
        "rule_type": "maintainability",
        "severity": "warning",
        "enabled": True,
        "tags": ["size", "maintainability"],
        "dsl_code": """
# Check if file is too large
if file_lines() > 500:
    violation(
        message=f"File is too large ({file_lines()} lines). Consider splitting it.",
        line_number=1,
        severity=warning,
        suggestion="Split the file into smaller, more focused modules."
    )
""",
        "metadata": {
            "max_lines": 500,
            "author": "system"
        }
    },
    {
        "rule_id": "custom_many_functions",
        "name": "Too Many Functions",
        "description": "Detect files with too many functions",
        "rule_type": "maintainability",
        "severity": "info",
        "enabled": True,
        "tags": ["functions", "maintainability"],
        "dsl_code": """
func_count = count_functions()
if func_count > 20:
    violation(
        message=f"File has too many functions ({func_count}). Consider organizing into classes.",
        line_number=1,
        severity=info,
        suggestion="Group related functions into classes or separate modules."
    )
""",
        "metadata": {
            "max_functions": 20,
            "author": "system"
        }
    },
    {
        "rule_id": "custom_debug_imports",
        "name": "Debug Imports",
        "description": "Detect debug imports in production code",
        "rule_type": "security",
        "severity": "warning",
        "enabled": True,
        "tags": ["debug", "security"],
        "dsl_code": """
debug_modules = ["pdb", "ipdb", "pprint", "logging"]
for module in debug_modules:
    if has_import(module):
        violation(
            message=f"Debug module '{module}' imported. Remove before production.",
            line_number=1,
            severity=warning,
            suggestion="Remove debug imports before committing to production."
        )
        break
""",
        "metadata": {
            "debug_modules": ["pdb", "ipdb", "pprint", "logging"],
            "author": "system"
        }
    }
]


def load_example_rules():
    """加载示例规则"""
    for rule_data in EXAMPLE_RULES:
        rule = DSLRule.from_dict(rule_data)
        rule_manager.add_rule(rule)
    
    logger.info(f"Loaded {len(EXAMPLE_RULES)} example rules")


# 初始化时加载示例规则
load_example_rules()