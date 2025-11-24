"""
规则定义语言(DSL)模块

提供声明式转换规则的定义、解析和表示功能
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import yaml
from pydantic import BaseModel, Field, validator


class RulePriority(int, Enum):
    """规则优先级枚举"""
    HIGHEST = 1000
    HIGH = 800
    MEDIUM = 500
    LOW = 200
    LOWEST = 100


class PatternNode(BaseModel):
    """模式匹配节点"""
    type: str
    value: Optional[str] = None
    children: Optional[List['PatternNode']] = None
    capture: Optional[str] = None
    optional: bool = False
    
    class Config:
        arbitrary_types_allowed = True


class ActionNode(BaseModel):
    """转换动作节点"""
    type: str
    value: Optional[str] = None
    children: Optional[List['ActionNode']] = None
    source: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class Rule(BaseModel):
    """转换规则定义"""
    id: str
    name: str
    description: str
    source_lang: str
    target_lang: str
    priority: int = Field(default=RulePriority.MEDIUM)
    pattern: str
    condition: Optional[str] = None
    action: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('priority')
    def validate_priority(cls, v):
        """验证优先级值"""
        if isinstance(v, int) and 0 <= v <= 1000:
            return v
        if isinstance(v, str):
            try:
                return RulePriority[v.upper()].value
            except KeyError:
                pass
        raise ValueError(f"Invalid priority value: {v}")
    
    def parse_pattern(self) -> PatternNode:
        """解析模式匹配表达式"""
        # 简化实现，实际应使用专门的解析器
        return self._parse_pattern_str(self.pattern)
    
    def parse_action(self) -> ActionNode:
        """解析转换动作表达式"""
        # 简化实现，实际应使用专门的解析器
        return self._parse_action_str(self.action)
    
    def _parse_pattern_str(self, pattern_str: str) -> PatternNode:
        """解析模式字符串为模式节点"""
        # 简化实现
        if not pattern_str.strip():
            raise ValueError("Empty pattern string")
        
        # 假设格式为 "NodeType(attr=$var)"
        node_type = pattern_str.split('(')[0].strip()
        
        return PatternNode(
            type=node_type,
            children=[]
        )
    
    def _parse_action_str(self, action_str: str) -> ActionNode:
        """解析动作字符串为动作节点"""
        # 简化实现
        if not action_str.strip():
            raise ValueError("Empty action string")
        
        # 假设格式为 "NodeType(attr=value)"
        node_type = action_str.split('(')[0].strip()
        
        return ActionNode(
            type=node_type,
            children=[]
        )


class RuleSet(BaseModel):
    """规则集合"""
    name: str
    description: Optional[str] = None
    rules: List[Rule] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_rule(self, rule: Rule) -> None:
        """添加规则"""
        self.rules.append(rule)
    
    def get_rules_for_languages(self, source_lang: str, target_lang: str) -> List[Rule]:
        """获取特定语言对的规则"""
        return [
            rule for rule in self.rules
            if rule.source_lang == source_lang and rule.target_lang == target_lang
        ]
    
    def get_sorted_rules(self) -> List[Rule]:
        """获取按优先级排序的规则列表"""
        return sorted(self.rules, key=lambda r: r.priority, reverse=True)


class DSLParser:
    """DSL解析器"""
    
    @staticmethod
    def parse_rule_from_yaml(yaml_str: str) -> Rule:
        """从YAML字符串解析规则"""
        try:
            data = yaml.safe_load(yaml_str)
            return Rule(
                id=data.get('id', data.get('rule')),
                name=data.get('rule'),
                description=data.get('description', ''),
                source_lang=data.get('source_lang'),
                target_lang=data.get('target_lang'),
                priority=data.get('priority', RulePriority.MEDIUM),
                pattern=data.get('pattern', ''),
                condition=data.get('condition'),
                action=data.get('action', ''),
                metadata=data.get('metadata', {})
            )
        except Exception as e:
            raise ValueError(f"Failed to parse rule from YAML: {str(e)}")
    
    @staticmethod
    def parse_ruleset_from_yaml(yaml_str: str) -> RuleSet:
        """从YAML字符串解析规则集"""
        try:
            data = yaml.safe_load(yaml_str)
            
            if not isinstance(data, dict):
                raise ValueError("YAML root must be a dictionary")
            
            ruleset = RuleSet(
                name=data.get('name', 'Unnamed Ruleset'),
                description=data.get('description', ''),
                metadata=data.get('metadata', {})
            )
            
            rules_data = data.get('rules', [])
            if not isinstance(rules_data, list):
                raise ValueError("Rules must be a list")
            
            for rule_data in rules_data:
                rule = Rule(
                    id=rule_data.get('id', rule_data.get('rule')),
                    name=rule_data.get('rule'),
                    description=rule_data.get('description', ''),
                    source_lang=rule_data.get('source_lang'),
                    target_lang=rule_data.get('target_lang'),
                    priority=rule_data.get('priority', RulePriority.MEDIUM),
                    pattern=rule_data.get('pattern', ''),
                    condition=rule_data.get('condition'),
                    action=rule_data.get('action', ''),
                    metadata=rule_data.get('metadata', {})
                )
                ruleset.add_rule(rule)
            
            return ruleset
        except Exception as e:
            raise ValueError(f"Failed to parse ruleset from YAML: {str(e)}")