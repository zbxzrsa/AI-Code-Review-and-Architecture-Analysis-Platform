"""
转换配置管理系统

提供代码转换的配置管理功能，包括：
- 语言选择和转换方向配置
- 转换规则的启用/禁用和优先级管理
- 配置预设的保存和加载
- 配置验证和默认值管理
"""
from typing import Dict, List, Any, Optional, Set
import json
import os
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path


class ConversionStrategy(Enum):
    """转换策略枚举"""
    SAFE = "safe"           # 安全模式：只转换确定可行的代码
    BALANCED = "balanced"   # 平衡模式：在安全性和完整性之间平衡
    AGGRESSIVE = "aggressive"  # 激进模式：尽可能转换更多代码


@dataclass
class RuleConfig:
    """规则配置数据类"""
    name: str
    enabled: bool = True
    priority: int = 100
    settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.settings is None:
            self.settings = {}


class ConversionConfig:
    """转换配置管理器"""
    
    # 支持的语言列表
    SUPPORTED_LANGUAGES = {
        "python": {"name": "Python", "extensions": [".py"], "features": ["oop", "functional", "dynamic"]},
        "javascript": {"name": "JavaScript", "extensions": [".js"], "features": ["oop", "functional", "dynamic", "async"]},
        "typescript": {"name": "TypeScript", "extensions": [".ts"], "features": ["oop", "functional", "static", "async"]},
        "java": {"name": "Java", "extensions": [".java"], "features": ["oop", "static", "multithreading"]},
        "csharp": {"name": "C#", "extensions": [".cs"], "features": ["oop", "static", "multithreading", "async"]},
        "cpp": {"name": "C++", "extensions": [".cpp", ".cc", ".cxx"], "features": ["oop", "manual_memory", "templates"]},
        "rust": {"name": "Rust", "extensions": [".rs"], "features": ["oop", "memory_safe", "functional", "static"]},
    }
    
    # 默认转换规则
    DEFAULT_RULES = {
        "syntax_conversion": {"priority": 100, "enabled": True, "description": "基础语法转换"},
        "api_mapping": {"priority": 90, "enabled": True, "description": "API映射转换"},
        "framework_conversion": {"priority": 80, "enabled": True, "description": "框架转换"},
        "memory_management": {"priority": 70, "enabled": True, "description": "内存管理转换"},
        "error_handling": {"priority": 60, "enabled": True, "description": "错误处理转换"},
        "async_patterns": {"priority": 50, "enabled": True, "description": "异步模式转换"},
        "type_annotations": {"priority": 40, "enabled": True, "description": "类型注解转换"},
        "best_practices": {"priority": 30, "enabled": True, "description": "最佳实践应用"},
        "code_style": {"priority": 20, "enabled": True, "description": "代码风格调整"},
        "documentation": {"priority": 10, "enabled": True, "description": "文档转换"},
    }
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件存储目录
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        # 基础配置
        self.source_language = ""
        self.target_language = ""
        self.conversion_strategy = ConversionStrategy.BALANCED
        
        # 规则配置
        self.rules: Dict[str, RuleConfig] = {}
        self._initialize_default_rules()
        
        # 语言特性配置
        self.language_features: Dict[str, Any] = {}
        
        # 高级配置
        self.advanced_options = {
            "preserve_comments": True,
            "maintain_formatting": True,
            "generate_documentation": True,
            "include_type_hints": True,
            "optimize_imports": True,
            "apply_linting": True,
        }
    
    def _initialize_default_rules(self):
        """初始化默认规则配置"""
        for rule_name, rule_info in self.DEFAULT_RULES.items():
            self.rules[rule_name] = RuleConfig(
                name=rule_name,
                enabled=rule_info["enabled"],
                priority=rule_info["priority"],
                settings={"description": rule_info["description"]}
            )
    
    def validate_config(self) -> Dict[str, Any]:
        """
        验证配置有效性
        
        Returns:
            验证结果，包含是否有效和错误信息
        """
        errors = []
        warnings = []
        
        # 验证语言选择
        if not self.source_language:
            errors.append("未选择源语言")
        elif self.source_language not in self.SUPPORTED_LANGUAGES:
            errors.append(f"不支持的源语言: {self.source_language}")
        
        if not self.target_language:
            errors.append("未选择目标语言")
        elif self.target_language not in self.SUPPORTED_LANGUAGES:
            errors.append(f"不支持的目标语言: {self.target_language}")
        
        # 验证语言转换对的可行性
        if self.source_language and self.target_language:
            if self.source_language == self.target_language:
                warnings.append("源语言和目标语言相同")
            
            # 检查语言特性兼容性
            source_features = set(self.SUPPORTED_LANGUAGES[self.source_language]["features"])
            target_features = set(self.SUPPORTED_LANGUAGES[self.target_language]["features"])
            
            incompatible_features = source_features - target_features
            if incompatible_features:
                warnings.append(f"目标语言不支持以下特性: {', '.join(incompatible_features)}")
        
        # 验证规则配置
        enabled_rules = [rule for rule in self.rules.values() if rule.enabled]
        if not enabled_rules:
            warnings.append("没有启用任何转换规则")
        
        # 检查规则优先级冲突
        priorities = [rule.priority for rule in enabled_rules]
        if len(priorities) != len(set(priorities)):
            warnings.append("存在相同优先级的规则")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "enabled_rules_count": len(enabled_rules),
            "total_rules_count": len(self.rules)
        }
    
    def get_rule_settings(self, rule_name: str) -> Optional[Dict[str, Any]]:
        """
        获取规则特定设置
        
        Args:
            rule_name: 规则名称
            
        Returns:
            规则设置字典，如果规则不存在则返回None
        """
        if rule_name in self.rules:
            return self.rules[rule_name].settings.copy()
        return None
    
    def update_rule_settings(self, rule_name: str, settings: Dict[str, Any]) -> bool:
        """
        更新规则设置
        
        Args:
            rule_name: 规则名称
            settings: 新的设置
            
        Returns:
            是否更新成功
        """
        if rule_name in self.rules:
            self.rules[rule_name].settings.update(settings)
            return True
        return False
    
    def enable_rule(self, rule_name: str, enabled: bool = True) -> bool:
        """
        启用或禁用规则
        
        Args:
            rule_name: 规则名称
            enabled: 是否启用
            
        Returns:
            是否操作成功
        """
        if rule_name in self.rules:
            self.rules[rule_name].enabled = enabled
            return True
        return False
    
    def set_rule_priority(self, rule_name: str, priority: int) -> bool:
        """
        设置规则优先级
        
        Args:
            rule_name: 规则名称
            priority: 优先级（数值越大优先级越高）
            
        Returns:
            是否设置成功
        """
        if rule_name in self.rules:
            self.rules[rule_name].priority = priority
            return True
        return False
    
    def add_custom_rule(self, rule_name: str, priority: int = 50, 
                       enabled: bool = True, settings: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加自定义规则
        
        Args:
            rule_name: 规则名称
            priority: 优先级
            enabled: 是否启用
            settings: 规则设置
            
        Returns:
            是否添加成功
        """
        if rule_name not in self.rules:
            self.rules[rule_name] = RuleConfig(
                name=rule_name,
                enabled=enabled,
                priority=priority,
                settings=settings or {}
            )
            return True
        return False
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        移除规则
        
        Args:
            rule_name: 规则名称
            
        Returns:
            是否移除成功
        """
        if rule_name in self.rules and rule_name not in self.DEFAULT_RULES:
            del self.rules[rule_name]
            return True
        return False
    
    def get_enabled_rules(self) -> List[RuleConfig]:
        """
        获取已启用的规则列表，按优先级排序
        
        Returns:
            已启用的规则列表
        """
        enabled_rules = [rule for rule in self.rules.values() if rule.enabled]
        return sorted(enabled_rules, key=lambda x: x.priority, reverse=True)
    
    def save_preset(self, name: str, description: str = "") -> bool:
        """
        保存配置预设
        
        Args:
            name: 预设名称
            description: 预设描述
            
        Returns:
            是否保存成功
        """
        try:
            preset_data = {
                "name": name,
                "description": description,
                "source_language": self.source_language,
                "target_language": self.target_language,
                "conversion_strategy": self.conversion_strategy.value,
                "rules": {name: asdict(rule) for name, rule in self.rules.items()},
                "language_features": self.language_features.copy(),
                "advanced_options": self.advanced_options.copy(),
                "created_at": str(Path().cwd()),  # 简化时间戳
            }
            
            preset_file = self.config_dir / f"{name}.json"
            with open(preset_file, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"保存预设失败: {e}")
            return False
    
    def load_preset(self, name: str) -> bool:
        """
        加载配置预设
        
        Args:
            name: 预设名称
            
        Returns:
            是否加载成功
        """
        try:
            preset_file = self.config_dir / f"{name}.json"
            if not preset_file.exists():
                return False
            
            with open(preset_file, 'r', encoding='utf-8') as f:
                preset_data = json.load(f)
            
            # 恢复配置
            self.source_language = preset_data.get("source_language", "")
            self.target_language = preset_data.get("target_language", "")
            self.conversion_strategy = ConversionStrategy(
                preset_data.get("conversion_strategy", ConversionStrategy.BALANCED.value)
            )
            
            # 恢复规则配置
            self.rules.clear()
            for rule_name, rule_data in preset_data.get("rules", {}).items():
                self.rules[rule_name] = RuleConfig(**rule_data)
            
            # 恢复其他配置
            self.language_features = preset_data.get("language_features", {})
            self.advanced_options.update(preset_data.get("advanced_options", {}))
            
            return True
        except Exception as e:
            print(f"加载预设失败: {e}")
            return False
    
    def list_presets(self) -> List[Dict[str, Any]]:
        """
        列出所有可用的配置预设
        
        Returns:
            预设信息列表
        """
        presets = []
        
        for preset_file in self.config_dir.glob("*.json"):
            try:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
                
                presets.append({
                    "name": preset_data.get("name", preset_file.stem),
                    "description": preset_data.get("description", ""),
                    "source_language": preset_data.get("source_language", ""),
                    "target_language": preset_data.get("target_language", ""),
                    "created_at": preset_data.get("created_at", ""),
                    "file_path": str(preset_file)
                })
            except Exception:
                continue
        
        return sorted(presets, key=lambda x: x["name"])
    
    def delete_preset(self, name: str) -> bool:
        """
        删除配置预设
        
        Args:
            name: 预设名称
            
        Returns:
            是否删除成功
        """
        try:
            preset_file = self.config_dir / f"{name}.json"
            if preset_file.exists():
                preset_file.unlink()
                return True
            return False
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典格式
        
        Returns:
            配置字典
        """
        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "conversion_strategy": self.conversion_strategy.value,
            "rules": {name: asdict(rule) for name, rule in self.rules.items()},
            "language_features": self.language_features.copy(),
            "advanced_options": self.advanced_options.copy(),
            "supported_languages": self.SUPPORTED_LANGUAGES.copy(),
        }
    
    def from_dict(self, config_data: Dict[str, Any]) -> bool:
        """
        从字典数据恢复配置
        
        Args:
            config_data: 配置数据字典
            
        Returns:
            是否恢复成功
        """
        try:
            self.source_language = config_data.get("source_language", "")
            self.target_language = config_data.get("target_language", "")
            self.conversion_strategy = ConversionStrategy(
                config_data.get("conversion_strategy", ConversionStrategy.BALANCED.value)
            )
            
            # 恢复规则配置
            if "rules" in config_data:
                self.rules.clear()
                for rule_name, rule_data in config_data["rules"].items():
                    self.rules[rule_name] = RuleConfig(**rule_data)
            
            # 恢复其他配置
            self.language_features = config_data.get("language_features", {})
            self.advanced_options.update(config_data.get("advanced_options", {}))
            
            return True
        except Exception as e:
            print(f"从字典恢复配置失败: {e}")
            return False


# 全局配置实例
_config_instance = None

def get_conversion_config() -> ConversionConfig:
    """获取全局配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConversionConfig()
    return _config_instance