"""
规则管理系统模块

负责规则加载、注册、版本控制和热重载
"""
import os
import json
import yaml
import glob
import hashlib
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .dsl import Rule, RuleSet, DSLParser
from .compiler import RuleCompiler, RuleValidator


class RuleLoadError(Exception):
    """规则加载错误"""
    pass


class RuleRegistry:
    """规则注册表"""
    
    def __init__(self):
        self.rulesets: Dict[str, RuleSet] = {}
        self.rules_by_id: Dict[str, Rule] = {}
        self.rules_by_languages: Dict[Tuple[str, str], List[Rule]] = {}
        self._lock = threading.RLock()
    
    def register_ruleset(self, ruleset: RuleSet) -> None:
        """注册规则集"""
        with self._lock:
            self.rulesets[ruleset.name] = ruleset
            
            # 注册规则集中的所有规则
            for rule in ruleset.rules:
                self.register_rule(rule, ruleset.name)
    
    def register_rule(self, rule: Rule, ruleset_name: str) -> None:
        """注册单个规则"""
        with self._lock:
            # 注册到规则ID映射
            self.rules_by_id[rule.id] = rule
            
            # 注册到语言对映射
            lang_pair = (rule.source_lang, rule.target_lang)
            if lang_pair not in self.rules_by_languages:
                self.rules_by_languages[lang_pair] = []
            self.rules_by_languages[lang_pair].append(rule)
    
    def unregister_ruleset(self, ruleset_name: str) -> None:
        """注销规则集"""
        with self._lock:
            if ruleset_name not in self.rulesets:
                return
            
            ruleset = self.rulesets[ruleset_name]
            
            # 注销规则集中的所有规则
            for rule in ruleset.rules:
                self.unregister_rule(rule.id)
            
            # 从规则集映射中移除
            del self.rulesets[ruleset_name]
    
    def unregister_rule(self, rule_id: str) -> None:
        """注销单个规则"""
        with self._lock:
            if rule_id not in self.rules_by_id:
                return
            
            rule = self.rules_by_id[rule_id]
            
            # 从语言对映射中移除
            lang_pair = (rule.source_lang, rule.target_lang)
            if lang_pair in self.rules_by_languages:
                self.rules_by_languages[lang_pair] = [
                    r for r in self.rules_by_languages[lang_pair] if r.id != rule_id
                ]
            
            # 从规则ID映射中移除
            del self.rules_by_id[rule_id]
    
    def get_ruleset(self, ruleset_name: str) -> Optional[RuleSet]:
        """获取规则集"""
        with self._lock:
            return self.rulesets.get(ruleset_name)
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """获取规则"""
        with self._lock:
            return self.rules_by_id.get(rule_id)
    
    def get_rules_for_languages(self, source_lang: str, target_lang: str) -> List[Rule]:
        """获取特定语言对的规则"""
        with self._lock:
            return self.rules_by_languages.get((source_lang, target_lang), [])
    
    def get_all_rulesets(self) -> List[RuleSet]:
        """获取所有规则集"""
        with self._lock:
            return list(self.rulesets.values())
    
    def get_all_rules(self) -> List[Rule]:
        """获取所有规则"""
        with self._lock:
            return list(self.rules_by_id.values())
    
    def clear(self) -> None:
        """清空注册表"""
        with self._lock:
            self.rulesets.clear()
            self.rules_by_id.clear()
            self.rules_by_languages.clear()


class RuleLoader:
    """规则加载器"""
    
    def __init__(self, registry: RuleRegistry):
        self.registry = registry
        self.parser = DSLParser()
        self.validator = RuleValidator()
        self.file_hashes: Dict[str, str] = {}
    
    def load_ruleset_from_file(self, file_path: str) -> RuleSet:
        """从文件加载规则集"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 计算文件哈希
            file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            self.file_hashes[file_path] = file_hash
            
            # 解析规则集
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                ruleset = self.parser.parse_ruleset_from_yaml(content)
            elif file_path.endswith('.json'):
                ruleset = self._parse_ruleset_from_json(content)
            else:
                raise RuleLoadError(f"Unsupported file format: {file_path}")
            
            # 验证规则
            for rule in ruleset.rules:
                errors = self.validator.validate_rule(rule)
                if errors:
                    print(f"Warning: Rule '{rule.name}' has validation errors: {errors}")
            
            return ruleset
        except Exception as e:
            raise RuleLoadError(f"Failed to load ruleset from {file_path}: {str(e)}")
    
    def load_rule_from_file(self, file_path: str) -> Rule:
        """从文件加载单个规则"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 计算文件哈希
            file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            self.file_hashes[file_path] = file_hash
            
            # 解析规则
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                rule = self.parser.parse_rule_from_yaml(content)
            elif file_path.endswith('.json'):
                rule = self._parse_rule_from_json(content)
            else:
                raise RuleLoadError(f"Unsupported file format: {file_path}")
            
            # 验证规则
            errors = self.validator.validate_rule(rule)
            if errors:
                print(f"Warning: Rule '{rule.name}' has validation errors: {errors}")
            
            return rule
        except Exception as e:
            raise RuleLoadError(f"Failed to load rule from {file_path}: {str(e)}")
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[RuleSet]:
        """加载目录中的所有规则文件"""
        pattern = os.path.join(directory_path, '**' if recursive else '', '*.{yaml,yml,json}')
        rule_files = glob.glob(pattern, recursive=recursive)
        
        rulesets = []
        for file_path in rule_files:
            try:
                if self._is_ruleset_file(file_path):
                    ruleset = self.load_ruleset_from_file(file_path)
                    self.registry.register_ruleset(ruleset)
                    rulesets.append(ruleset)
                else:
                    rule = self.load_rule_from_file(file_path)
                    ruleset_name = os.path.basename(os.path.dirname(file_path))
                    ruleset = self.registry.get_ruleset(ruleset_name)
                    if not ruleset:
                        ruleset = RuleSet(name=ruleset_name)
                        self.registry.register_ruleset(ruleset)
                        rulesets.append(ruleset)
                    ruleset.add_rule(rule)
                    self.registry.register_rule(rule, ruleset_name)
            except Exception as e:
                print(f"Error loading rule file {file_path}: {str(e)}")
        
        return rulesets
    
    def _is_ruleset_file(self, file_path: str) -> bool:
        """判断文件是否为规则集文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                data = yaml.safe_load(content)
            elif file_path.endswith('.json'):
                data = json.loads(content)
            else:
                return False
            
            # 如果包含rules字段，则为规则集文件
            return isinstance(data, dict) and 'rules' in data
        except Exception:
            return False
    
    def _parse_ruleset_from_json(self, json_str: str) -> RuleSet:
        """从JSON字符串解析规则集"""
        try:
            data = json.loads(json_str)
            
            if not isinstance(data, dict):
                raise ValueError("JSON root must be a dictionary")
            
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
                    priority=rule_data.get('priority', 500),
                    pattern=rule_data.get('pattern', ''),
                    condition=rule_data.get('condition'),
                    action=rule_data.get('action', ''),
                    metadata=rule_data.get('metadata', {})
                )
                ruleset.add_rule(rule)
            
            return ruleset
        except Exception as e:
            raise ValueError(f"Failed to parse ruleset from JSON: {str(e)}")
    
    def _parse_rule_from_json(self, json_str: str) -> Rule:
        """从JSON字符串解析规则"""
        try:
            data = json.loads(json_str)
            return Rule(
                id=data.get('id', data.get('rule')),
                name=data.get('rule'),
                description=data.get('description', ''),
                source_lang=data.get('source_lang'),
                target_lang=data.get('target_lang'),
                priority=data.get('priority', 500),
                pattern=data.get('pattern', ''),
                condition=data.get('condition'),
                action=data.get('action', ''),
                metadata=data.get('metadata', {})
            )
        except Exception as e:
            raise ValueError(f"Failed to parse rule from JSON: {str(e)}")
    
    def has_file_changed(self, file_path: str) -> bool:
        """检查文件是否已更改"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            old_hash = self.file_hashes.get(file_path)
            
            return old_hash != new_hash
        except Exception:
            return True  # 如果无法读取文件，假设已更改


class RuleWatcher(FileSystemEventHandler):
    """规则文件监视器"""
    
    def __init__(self, loader: RuleLoader, registry: RuleRegistry):
        self.loader = loader
        self.registry = registry
        self.observer = Observer()
        self.watched_dirs = set()
    
    def start_watching(self, directory_path: str) -> None:
        """开始监视目录"""
        if directory_path in self.watched_dirs:
            return
        
        self.observer.schedule(self, directory_path, recursive=True)
        self.watched_dirs.add(directory_path)
        
        if not self.observer.is_alive():
            self.observer.start()
    
    def stop_watching(self) -> None:
        """停止监视"""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
        self.watched_dirs.clear()
    
    def on_modified(self, event) -> None:
        """文件修改事件处理"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not (file_path.endswith('.yaml') or file_path.endswith('.yml') or file_path.endswith('.json')):
            return
        
        if not self.loader.has_file_changed(file_path):
            return
        
        try:
            if self.loader._is_ruleset_file(file_path):
                # 重新加载规则集
                ruleset = self.loader.load_ruleset_from_file(file_path)
                ruleset_name = ruleset.name
                
                # 注销旧规则集
                self.registry.unregister_ruleset(ruleset_name)
                
                # 注册新规则集
                self.registry.register_ruleset(ruleset)
                
                print(f"Reloaded ruleset '{ruleset_name}' from {file_path}")
            else:
                # 重新加载单个规则
                rule = self.loader.load_rule_from_file(file_path)
                ruleset_name = os.path.basename(os.path.dirname(file_path))
                
                # 注销旧规则
                self.registry.unregister_rule(rule.id)
                
                # 注册新规则
                self.registry.register_rule(rule, ruleset_name)
                
                print(f"Reloaded rule '{rule.name}' from {file_path}")
        except Exception as e:
            print(f"Error reloading rule file {file_path}: {str(e)}")
    
    def on_created(self, event) -> None:
        """文件创建事件处理"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not (file_path.endswith('.yaml') or file_path.endswith('.yml') or file_path.endswith('.json')):
            return
        
        try:
            if self.loader._is_ruleset_file(file_path):
                # 加载新规则集
                ruleset = self.loader.load_ruleset_from_file(file_path)
                self.registry.register_ruleset(ruleset)
                
                print(f"Loaded new ruleset '{ruleset.name}' from {file_path}")
            else:
                # 加载新规则
                rule = self.loader.load_rule_from_file(file_path)
                ruleset_name = os.path.basename(os.path.dirname(file_path))
                self.registry.register_rule(rule, ruleset_name)
                
                print(f"Loaded new rule '{rule.name}' from {file_path}")
        except Exception as e:
            print(f"Error loading new rule file {file_path}: {str(e)}")
    
    def on_deleted(self, event) -> None:
        """文件删除事件处理"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not (file_path.endswith('.yaml') or file_path.endswith('.yml') or file_path.endswith('.json')):
            return
        
        # 从文件哈希映射中移除
        if file_path in self.loader.file_hashes:
            del self.loader.file_hashes[file_path]
        
        # 注：无法直接知道删除的是哪个规则集或规则
        # 可以考虑在加载时记录文件路径到规则集/规则的映射


class RuleManager:
    """规则管理器"""
    
    def __init__(self):
        self.registry = RuleRegistry()
        self.loader = RuleLoader(self.registry)
        self.watcher = RuleWatcher(self.loader, self.registry)
        self.version_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def load_rules(self, directory_path: str, enable_hot_reload: bool = False) -> None:
        """加载规则"""
        self.loader.load_directory(directory_path)
        
        if enable_hot_reload:
            self.watcher.start_watching(directory_path)
    
    def get_ruleset(self, ruleset_name: str) -> Optional[RuleSet]:
        """获取规则集"""
        return self.registry.get_ruleset(ruleset_name)
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """获取规则"""
        return self.registry.get_rule(rule_id)
    
    def get_rules_for_languages(self, source_lang: str, target_lang: str) -> List[Rule]:
        """获取特定语言对的规则"""
        return self.registry.get_rules_for_languages(source_lang, target_lang)
    
    def create_ruleset_version(self, ruleset_name: str, version_name: str) -> None:
        """创建规则集版本"""
        ruleset = self.registry.get_ruleset(ruleset_name)
        if not ruleset:
            raise ValueError(f"Ruleset '{ruleset_name}' not found")
        
        # 创建规则集快照
        ruleset_dict = ruleset.dict()
        
        # 保存到版本历史
        if ruleset_name not in self.version_history:
            self.version_history[ruleset_name] = []
        
        self.version_history[ruleset_name].append({
            'version': version_name,
            'timestamp': time.time(),
            'ruleset': ruleset_dict
        })
    
    def get_ruleset_version(self, ruleset_name: str, version_name: str) -> Optional[RuleSet]:
        """获取规则集版本"""
        if ruleset_name not in self.version_history:
            return None
        
        for version in self.version_history[ruleset_name]:
            if version['version'] == version_name:
                ruleset_dict = version['ruleset']
                return RuleSet(**ruleset_dict)
        
        return None
    
    def stop_hot_reload(self) -> None:
        """停止热重载"""
        self.watcher.stop_watching()