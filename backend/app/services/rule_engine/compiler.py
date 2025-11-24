"""
规则编译器模块

负责将DSL规则编译为可执行代码，进行优化和验证
"""
import ast
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .dsl import Rule, PatternNode, ActionNode


class RuleCompilationError(Exception):
    """规则编译错误"""
    pass


class CompiledPattern:
    """编译后的模式"""
    def __init__(self, pattern_func: Callable, captures: List[str]):
        self.pattern_func = pattern_func
        self.captures = captures
    
    def match(self, node: Any) -> Optional[Dict[str, Any]]:
        """匹配节点，返回捕获变量"""
        return self.pattern_func(node)


class CompiledAction:
    """编译后的动作"""
    def __init__(self, action_func: Callable):
        self.action_func = action_func
    
    def apply(self, captures: Dict[str, Any]) -> Any:
        """应用转换动作"""
        return self.action_func(captures)


class CompiledRule:
    """编译后的规则"""
    def __init__(
        self, 
        rule: Rule, 
        pattern: CompiledPattern, 
        condition: Optional[Callable] = None,
        action: CompiledAction = None
    ):
        self.rule = rule
        self.pattern = pattern
        self.condition = condition
        self.action = action
    
    def apply(self, node: Any) -> Optional[Any]:
        """应用规则"""
        # 尝试匹配模式
        captures = self.pattern.match(node)
        if not captures:
            return None
        
        # 检查条件
        if self.condition and not self.condition(captures):
            return None
        
        # 应用转换动作
        return self.action.apply(captures)


class RuleCompiler:
    """规则编译器"""
    
    def compile_rule(self, rule: Rule) -> CompiledRule:
        """编译规则"""
        try:
            # 解析模式和动作
            pattern_node = rule.parse_pattern()
            action_node = rule.parse_action()
            
            # 编译模式
            compiled_pattern = self._compile_pattern(pattern_node)
            
            # 编译条件
            compiled_condition = None
            if rule.condition:
                compiled_condition = self._compile_condition(rule.condition)
            
            # 编译动作
            compiled_action = self._compile_action(action_node)
            
            return CompiledRule(
                rule=rule,
                pattern=compiled_pattern,
                condition=compiled_condition,
                action=compiled_action
            )
        except Exception as e:
            raise RuleCompilationError(f"Failed to compile rule '{rule.name}': {str(e)}")
    
    def _compile_pattern(self, pattern_node: PatternNode) -> CompiledPattern:
        """编译模式节点"""
        captures = []
        
        # 收集捕获变量
        self._collect_captures(pattern_node, captures)
        
        # 创建模式匹配函数
        def pattern_func(node: Any) -> Optional[Dict[str, Any]]:
            return self._match_pattern(pattern_node, node, {})
        
        return CompiledPattern(pattern_func, captures)
    
    def _collect_captures(self, pattern_node: PatternNode, captures: List[str]) -> None:
        """收集模式中的捕获变量"""
        if pattern_node.capture and pattern_node.capture not in captures:
            captures.append(pattern_node.capture)
        
        if pattern_node.children:
            for child in pattern_node.children:
                self._collect_captures(child, captures)
    
    def _match_pattern(
        self, 
        pattern: PatternNode, 
        node: Any, 
        captures: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """匹配模式"""
        # 检查节点类型
        if not hasattr(node, '__class__') or not hasattr(node.__class__, '__name__'):
            return None
        
        node_type = node.__class__.__name__
        if pattern.type != node_type and pattern.type != '*':
            return None
        
        # 捕获节点
        if pattern.capture:
            captures[pattern.capture] = node
        
        # 匹配子节点
        if pattern.children and hasattr(node, 'children'):
            for i, child_pattern in enumerate(pattern.children):
                if i >= len(node.children) and not child_pattern.optional:
                    return None
                
                if i < len(node.children):
                    child_captures = self._match_pattern(child_pattern, node.children[i], {})
                    if not child_captures and not child_pattern.optional:
                        return None
                    
                    if child_captures:
                        captures.update(child_captures)
        
        return captures
    
    def _compile_condition(self, condition_str: str) -> Callable:
        """编译条件表达式"""
        # 简化实现，实际应使用更安全的方法
        condition_code = f"def condition_func(captures):\n    return {condition_str}"
        
        compiled_code = compile(condition_code, '<string>', 'exec')
        namespace = {}
        exec(compiled_code, namespace)
        
        return namespace['condition_func']
    
    def _compile_action(self, action_node: ActionNode) -> CompiledAction:
        """编译动作节点"""
        # 创建动作函数
        def action_func(captures: Dict[str, Any]) -> Any:
            return self._apply_action(action_node, captures)
        
        return CompiledAction(action_func)
    
    def _apply_action(self, action: ActionNode, captures: Dict[str, Any]) -> Any:
        """应用转换动作"""
        # 创建目标节点
        # 简化实现，实际应根据目标语言创建相应的AST节点
        node_type = action.type
        
        # 创建新节点
        new_node = type(node_type, (), {})()
        
        # 设置属性
        if action.value and action.source and action.source in captures:
            source_value = getattr(captures[action.source], action.value, None)
            if source_value is not None:
                setattr(new_node, action.value, source_value)
        
        # 处理子节点
        if action.children:
            new_node.children = []
            for child_action in action.children:
                child_node = self._apply_action(child_action, captures)
                if child_node:
                    new_node.children.append(child_node)
        
        return new_node


class RuleOptimizer:
    """规则优化器"""
    
    def optimize_rules(self, rules: List[CompiledRule]) -> List[CompiledRule]:
        """优化规则集"""
        # 按优先级排序
        sorted_rules = sorted(rules, key=lambda r: r.rule.priority, reverse=True)
        
        # 分析规则依赖
        dependencies = self._analyze_dependencies(sorted_rules)
        
        # 拓扑排序
        ordered_rules = self._topological_sort(sorted_rules, dependencies)
        
        return ordered_rules
    
    def _analyze_dependencies(self, rules: List[CompiledRule]) -> Dict[int, Set[int]]:
        """分析规则依赖关系"""
        # 简化实现，实际应分析规则间的依赖
        dependencies = {i: set() for i in range(len(rules))}
        return dependencies
    
    def _topological_sort(
        self, 
        rules: List[CompiledRule], 
        dependencies: Dict[int, Set[int]]
    ) -> List[CompiledRule]:
        """拓扑排序规则"""
        # 简化实现，实际应根据依赖关系进行拓扑排序
        return rules


class RuleValidator:
    """规则验证器"""
    
    def validate_rule(self, rule: Rule) -> List[str]:
        """验证规则，返回错误列表"""
        errors = []
        
        # 验证基本属性
        if not rule.id:
            errors.append("Rule ID is required")
        
        if not rule.name:
            errors.append("Rule name is required")
        
        if not rule.source_lang:
            errors.append("Source language is required")
        
        if not rule.target_lang:
            errors.append("Target language is required")
        
        if not rule.pattern:
            errors.append("Pattern is required")
        
        if not rule.action:
            errors.append("Action is required")
        
        # 验证模式语法
        try:
            rule.parse_pattern()
        except Exception as e:
            errors.append(f"Invalid pattern syntax: {str(e)}")
        
        # 验证条件语法
        if rule.condition:
            try:
                condition_code = f"def condition_func():\n    return {rule.condition}"
                ast.parse(condition_code)
            except SyntaxError as e:
                errors.append(f"Invalid condition syntax: {str(e)}")
        
        # 验证动作语法
        try:
            rule.parse_action()
        except Exception as e:
            errors.append(f"Invalid action syntax: {str(e)}")
        
        return errors