"""
规则执行引擎模块

负责AST遍历、模式匹配和转换应用
"""
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import logging

from .dsl import Rule, RuleSet
from .compiler import CompiledRule, RuleCompiler, RuleOptimizer, RuleValidator


logger = logging.getLogger(__name__)


class TraversalStrategy(str, Enum):
    """AST遍历策略"""
    TOP_DOWN = "top_down"  # 从上到下遍历
    BOTTOM_UP = "bottom_up"  # 从下到上遍历
    HYBRID = "hybrid"  # 混合策略


class TransformationResult:
    """转换结果"""
    def __init__(self, 
                 transformed_ast: Any, 
                 applied_rules: List[Rule] = None,
                 errors: List[str] = None,
                 warnings: List[str] = None):
        self.transformed_ast = transformed_ast
        self.applied_rules = applied_rules or []
        self.errors = errors or []
        self.warnings = warnings or []
        
    @property
    def success(self) -> bool:
        """转换是否成功"""
        return len(self.errors) == 0
    
    def add_error(self, error: str) -> None:
        """添加错误"""
        self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        """添加警告"""
        self.warnings.append(warning)
    
    def add_applied_rule(self, rule: Rule) -> None:
        """添加应用的规则"""
        self.applied_rules.append(rule)


class ConflictResolutionStrategy(str, Enum):
    """冲突解决策略"""
    PRIORITY = "priority"  # 按优先级解决
    FIRST_MATCH = "first_match"  # 按首次匹配解决
    ALL_MATCHES = "all_matches"  # 应用所有匹配


class RuleExecutionEngine:
    """规则执行引擎"""
    
    def __init__(self, 
                 traversal_strategy: TraversalStrategy = TraversalStrategy.BOTTOM_UP,
                 conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.PRIORITY):
        self.compiler = RuleCompiler()
        self.optimizer = RuleOptimizer()
        self.validator = RuleValidator()
        self.traversal_strategy = traversal_strategy
        self.conflict_resolution = conflict_resolution
    
    def transform(self, 
                  ast: Any, 
                  ruleset: RuleSet, 
                  source_lang: str, 
                  target_lang: str) -> TransformationResult:
        """应用规则集转换AST"""
        # 获取适用的规则
        rules = ruleset.get_rules_for_languages(source_lang, target_lang)
        
        # 验证规则
        for rule in rules:
            errors = self.validator.validate_rule(rule)
            if errors:
                logger.warning(f"Rule '{rule.name}' has validation errors: {errors}")
        
        # 编译规则
        compiled_rules = []
        for rule in rules:
            try:
                compiled_rule = self.compiler.compile_rule(rule)
                compiled_rules.append(compiled_rule)
            except Exception as e:
                logger.error(f"Failed to compile rule '{rule.name}': {str(e)}")
        
        # 优化规则
        optimized_rules = self.optimizer.optimize_rules(compiled_rules)
        
        # 执行转换
        result = TransformationResult(ast)
        transformed_ast = self._apply_rules(ast, optimized_rules, result)
        result.transformed_ast = transformed_ast
        
        return result
    
    def _apply_rules(self, 
                    ast: Any, 
                    rules: List[CompiledRule], 
                    result: TransformationResult) -> Any:
        """应用规则到AST"""
        if self.traversal_strategy == TraversalStrategy.TOP_DOWN:
            return self._apply_rules_top_down(ast, rules, result)
        elif self.traversal_strategy == TraversalStrategy.BOTTOM_UP:
            return self._apply_rules_bottom_up(ast, rules, result)
        else:  # HYBRID
            return self._apply_rules_hybrid(ast, rules, result)
    
    def _apply_rules_top_down(self, 
                             node: Any, 
                             rules: List[CompiledRule], 
                             result: TransformationResult) -> Any:
        """从上到下应用规则"""
        # 先应用规则到当前节点
        node = self._apply_rules_to_node(node, rules, result)
        
        # 然后递归处理子节点
        if hasattr(node, 'children'):
            new_children = []
            for child in node.children:
                new_child = self._apply_rules_top_down(child, rules, result)
                new_children.append(new_child)
            node.children = new_children
        
        return node
    
    def _apply_rules_bottom_up(self, 
                              node: Any, 
                              rules: List[CompiledRule], 
                              result: TransformationResult) -> Any:
        """从下到上应用规则"""
        # 先递归处理子节点
        if hasattr(node, 'children'):
            new_children = []
            for child in node.children:
                new_child = self._apply_rules_bottom_up(child, rules, result)
                new_children.append(new_child)
            node.children = new_children
        
        # 然后应用规则到当前节点
        return self._apply_rules_to_node(node, rules, result)
    
    def _apply_rules_hybrid(self, 
                           node: Any, 
                           rules: List[CompiledRule], 
                           result: TransformationResult) -> Any:
        """混合策略应用规则"""
        # 先应用一部分规则（如声明相关规则）
        declaration_rules = [r for r in rules if self._is_declaration_rule(r)]
        node = self._apply_rules_to_node(node, declaration_rules, result)
        
        # 递归处理子节点
        if hasattr(node, 'children'):
            new_children = []
            for child in node.children:
                new_child = self._apply_rules_hybrid(child, rules, result)
                new_children.append(new_child)
            node.children = new_children
        
        # 再应用其他规则
        other_rules = [r for r in rules if not self._is_declaration_rule(r)]
        return self._apply_rules_to_node(node, other_rules, result)
    
    def _is_declaration_rule(self, rule: CompiledRule) -> bool:
        """判断是否为声明相关规则"""
        # 简化实现，实际应根据规则特性判断
        return "declaration" in rule.rule.name.lower() or "def" in rule.rule.name.lower()
    
    def _apply_rules_to_node(self, 
                            node: Any, 
                            rules: List[CompiledRule], 
                            result: TransformationResult) -> Any:
        """应用规则到单个节点"""
        if node is None:
            return None
        
        # 收集匹配的规则
        matching_rules = []
        for rule in rules:
            try:
                captures = rule.pattern.match(node)
                if captures:
                    if rule.condition is None or rule.condition(captures):
                        matching_rules.append((rule, captures))
            except Exception as e:
                result.add_error(f"Error matching rule '{rule.rule.name}': {str(e)}")
        
        # 如果没有匹配的规则，返回原节点
        if not matching_rules:
            return node
        
        # 根据冲突解决策略选择要应用的规则
        if self.conflict_resolution == ConflictResolutionStrategy.FIRST_MATCH:
            # 应用第一个匹配的规则
            rule, captures = matching_rules[0]
            try:
                transformed_node = rule.action.apply(captures)
                result.add_applied_rule(rule.rule)
                return transformed_node
            except Exception as e:
                result.add_error(f"Error applying rule '{rule.rule.name}': {str(e)}")
                return node
        
        elif self.conflict_resolution == ConflictResolutionStrategy.PRIORITY:
            # 应用优先级最高的规则
            matching_rules.sort(key=lambda x: x[0].rule.priority, reverse=True)
            rule, captures = matching_rules[0]
            try:
                transformed_node = rule.action.apply(captures)
                result.add_applied_rule(rule.rule)
                return transformed_node
            except Exception as e:
                result.add_error(f"Error applying rule '{rule.rule.name}': {str(e)}")
                return node
        
        else:  # ALL_MATCHES
            # 应用所有匹配的规则
            current_node = node
            for rule, captures in matching_rules:
                try:
                    current_node = rule.action.apply(captures)
                    result.add_applied_rule(rule.rule)
                except Exception as e:
                    result.add_error(f"Error applying rule '{rule.rule.name}': {str(e)}")
            
            return current_node