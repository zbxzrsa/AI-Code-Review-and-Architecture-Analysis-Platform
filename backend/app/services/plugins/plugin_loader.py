"""
插件动态加载器 - 支持从文件系统和包中发现和加载插件
"""

import os
import sys
import importlib
import importlib.util
import inspect
from typing import Dict, List, Type, Any, Optional
from pathlib import Path
import logging
import json
import yaml
from dataclasses import asdict

from .plugin_manager import PluginInterface, PluginMetadata, PluginType

logger = logging.getLogger(__name__)


class PluginDiscovery:
    """插件发现器"""
    
    def __init__(self, plugin_directories: List[str] = None):
        self.plugin_directories = plugin_directories or []
        self.discovered_plugins = {}
        self.plugin_configs = {}
    
    def add_plugin_directory(self, directory: str):
        """添加插件目录"""
        if os.path.exists(directory) and directory not in self.plugin_directories:
            self.plugin_directories.append(directory)
            logger.info(f"Added plugin directory: {directory}")
    
    def discover_plugins(self) -> Dict[str, Dict[str, Any]]:
        """发现所有可用插件"""
        discovered = {}
        
        # 从目录中发现插件
        for directory in self.plugin_directories:
            plugins = self._discover_from_directory(directory)
            discovered.update(plugins)
        
        # 从已安装包中发现插件
        package_plugins = self._discover_from_packages()
        discovered.update(package_plugins)
        
        self.discovered_plugins = discovered
        return discovered
    
    def _discover_from_directory(self, directory: str) -> Dict[str, Dict[str, Any]]:
        """从目录中发现插件"""
        plugins = {}
        
        try:
            for root, dirs, files in os.walk(directory):
                # 查找插件配置文件
                config_files = [f for f in files if f.endswith(('.json', '.yaml', '.yml'))]
                
                for config_file in config_files:
                    config_path = os.path.join(root, config_file)
                    plugin_info = self._load_plugin_config(config_path)
                    
                    if plugin_info:
                        plugin_id = plugin_info.get('id')
                        if plugin_id:
                            plugins[plugin_id] = {
                                'config_path': config_path,
                                'directory': root,
                                'info': plugin_info,
                                'source': 'directory'
                            }
                
                # 查找Python插件文件
                python_files = [f for f in files if f.endswith('.py') and not f.startswith('__')]
                
                for py_file in python_files:
                    py_path = os.path.join(root, py_file)
                    plugin_classes = self._discover_plugin_classes(py_path)
                    
                    for plugin_class in plugin_classes:
                        try:
                            # 创建临时实例获取元数据
                            temp_instance = plugin_class()
                            metadata = temp_instance.get_metadata()
                            
                            plugins[metadata.id] = {
                                'file_path': py_path,
                                'class_name': plugin_class.__name__,
                                'directory': root,
                                'metadata': asdict(metadata),
                                'plugin_class': plugin_class,
                                'source': 'file'
                            }
                        except Exception as e:
                            logger.warning(f"Failed to get metadata from {plugin_class}: {e}")
        
        except Exception as e:
            logger.error(f"Error discovering plugins from directory {directory}: {e}")
        
        return plugins
    
    def _discover_from_packages(self) -> Dict[str, Dict[str, Any]]:
        """从已安装包中发现插件"""
        plugins = {}
        
        try:
            # 查找带有插件入口点的包
            import pkg_resources
            
            for entry_point in pkg_resources.iter_entry_points('translation_plugins'):
                try:
                    plugin_class = entry_point.load()
                    
                    if issubclass(plugin_class, PluginInterface):
                        temp_instance = plugin_class()
                        metadata = temp_instance.get_metadata()
                        
                        plugins[metadata.id] = {
                            'entry_point': entry_point.name,
                            'package': entry_point.dist.project_name,
                            'version': entry_point.dist.version,
                            'metadata': asdict(metadata),
                            'plugin_class': plugin_class,
                            'source': 'package'
                        }
                
                except Exception as e:
                    logger.warning(f"Failed to load plugin from entry point {entry_point}: {e}")
        
        except ImportError:
            logger.info("pkg_resources not available, skipping package plugin discovery")
        
        return plugins
    
    def _load_plugin_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """加载插件配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    return json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load plugin config {config_path}: {e}")
        
        return None
    
    def _discover_plugin_classes(self, file_path: str) -> List[Type[PluginInterface]]:
        """从Python文件中发现插件类"""
        plugin_classes = []
        
        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location("plugin_module", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 查找插件类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, PluginInterface) and 
                        obj != PluginInterface and 
                        not inspect.isabstract(obj)):
                        plugin_classes.append(obj)
        
        except Exception as e:
            logger.warning(f"Failed to discover plugin classes from {file_path}: {e}")
        
        return plugin_classes
    
    def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """获取插件信息"""
        return self.discovered_plugins.get(plugin_id)
    
    def list_plugins_by_type(self, plugin_type: PluginType) -> List[str]:
        """按类型列出插件"""
        plugins = []
        
        for plugin_id, plugin_info in self.discovered_plugins.items():
            metadata = plugin_info.get('metadata', {})
            if metadata.get('plugin_type') == plugin_type.value:
                plugins.append(plugin_id)
        
        return plugins


class DynamicPluginLoader:
    """动态插件加载器"""
    
    def __init__(self):
        self.loaded_modules = {}
        self.plugin_instances = {}
    
    def load_plugin_from_file(self, file_path: str, class_name: str) -> Optional[Type[PluginInterface]]:
        """从文件加载插件类"""
        try:
            # 生成模块名
            module_name = f"plugin_{Path(file_path).stem}_{hash(file_path)}"
            
            # 检查是否已加载
            if module_name in self.loaded_modules:
                module = self.loaded_modules[module_name]
            else:
                # 动态导入模块
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if not spec or not spec.loader:
                    raise ImportError(f"Cannot load spec from {file_path}")
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.loaded_modules[module_name] = module
            
            # 获取插件类
            if hasattr(module, class_name):
                plugin_class = getattr(module, class_name)
                if issubclass(plugin_class, PluginInterface):
                    return plugin_class
                else:
                    raise TypeError(f"{class_name} is not a valid plugin class")
            else:
                raise AttributeError(f"Class {class_name} not found in {file_path}")
        
        except Exception as e:
            logger.error(f"Failed to load plugin from {file_path}: {e}")
            return None
    
    def load_plugin_from_package(self, entry_point_name: str) -> Optional[Type[PluginInterface]]:
        """从包入口点加载插件"""
        try:
            import pkg_resources
            
            for entry_point in pkg_resources.iter_entry_points('translation_plugins'):
                if entry_point.name == entry_point_name:
                    plugin_class = entry_point.load()
                    
                    if issubclass(plugin_class, PluginInterface):
                        return plugin_class
                    else:
                        raise TypeError(f"{plugin_class} is not a valid plugin class")
            
            raise ValueError(f"Entry point {entry_point_name} not found")
        
        except Exception as e:
            logger.error(f"Failed to load plugin from package entry point {entry_point_name}: {e}")
            return None
    
    def create_plugin_instance(self, plugin_class: Type[PluginInterface], 
                             plugin_id: str = None) -> Optional[PluginInterface]:
        """创建插件实例"""
        try:
            instance = plugin_class()
            
            if plugin_id:
                self.plugin_instances[plugin_id] = instance
            
            return instance
        
        except Exception as e:
            logger.error(f"Failed to create plugin instance for {plugin_class}: {e}")
            return None
    
    def unload_plugin(self, plugin_id: str):
        """卸载插件"""
        if plugin_id in self.plugin_instances:
            try:
                instance = self.plugin_instances[plugin_id]
                # 如果插件有清理方法，调用它
                if hasattr(instance, 'cleanup'):
                    instance.cleanup()
                
                del self.plugin_instances[plugin_id]
                logger.info(f"Plugin {plugin_id} unloaded successfully")
            
            except Exception as e:
                logger.error(f"Error unloading plugin {plugin_id}: {e}")
    
    def reload_plugin(self, plugin_id: str, plugin_info: Dict[str, Any]) -> Optional[PluginInterface]:
        """重新加载插件"""
        # 先卸载现有插件
        self.unload_plugin(plugin_id)
        
        # 重新加载
        return self.load_plugin(plugin_id, plugin_info)
    
    def load_plugin(self, plugin_id: str, plugin_info: Dict[str, Any]) -> Optional[PluginInterface]:
        """加载插件"""
        source = plugin_info.get('source')
        
        try:
            if source == 'file':
                file_path = plugin_info.get('file_path')
                class_name = plugin_info.get('class_name')
                
                if file_path and class_name:
                    plugin_class = self.load_plugin_from_file(file_path, class_name)
                    if plugin_class:
                        return self.create_plugin_instance(plugin_class, plugin_id)
            
            elif source == 'package':
                entry_point = plugin_info.get('entry_point')
                
                if entry_point:
                    plugin_class = self.load_plugin_from_package(entry_point)
                    if plugin_class:
                        return self.create_plugin_instance(plugin_class, plugin_id)
            
            elif source == 'directory':
                # 从目录配置加载
                config_path = plugin_info.get('config_path')
                directory = plugin_info.get('directory')
                
                if config_path and directory:
                    return self._load_from_directory_config(plugin_id, config_path, directory)
        
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_id}: {e}")
        
        return None
    
    def _load_from_directory_config(self, plugin_id: str, config_path: str, 
                                  directory: str) -> Optional[PluginInterface]:
        """从目录配置加载插件"""
        try:
            # 加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            # 获取插件文件和类名
            plugin_file = config.get('plugin_file')
            class_name = config.get('class_name')
            
            if plugin_file and class_name:
                file_path = os.path.join(directory, plugin_file)
                plugin_class = self.load_plugin_from_file(file_path, class_name)
                
                if plugin_class:
                    return self.create_plugin_instance(plugin_class, plugin_id)
        
        except Exception as e:
            logger.error(f"Failed to load plugin from directory config {config_path}: {e}")
        
        return None
    
    def get_loaded_plugins(self) -> Dict[str, PluginInterface]:
        """获取已加载的插件"""
        return self.plugin_instances.copy()


class PluginValidator:
    """插件验证器"""
    
    @staticmethod
    def validate_plugin_class(plugin_class: Type[PluginInterface]) -> bool:
        """验证插件类"""
        try:
            # 检查是否继承自PluginInterface
            if not issubclass(plugin_class, PluginInterface):
                return False
            
            # 检查必需的方法
            required_methods = ['get_metadata', 'initialize', 'cleanup', 'get_health_status']
            
            for method in required_methods:
                if not hasattr(plugin_class, method):
                    logger.error(f"Plugin class {plugin_class} missing required method: {method}")
                    return False
            
            # 尝试创建实例并获取元数据
            temp_instance = plugin_class()
            metadata = temp_instance.get_metadata()
            
            # 验证元数据
            if not isinstance(metadata, PluginMetadata):
                logger.error(f"Plugin {plugin_class} metadata is not PluginMetadata instance")
                return False
            
            # 验证必需的元数据字段
            if not all([metadata.id, metadata.name, metadata.version, metadata.plugin_type]):
                logger.error(f"Plugin {plugin_class} missing required metadata fields")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating plugin class {plugin_class}: {e}")
            return False
    
    @staticmethod
    def validate_plugin_instance(instance: PluginInterface) -> bool:
        """验证插件实例"""
        try:
            # 获取并验证元数据
            metadata = instance.get_metadata()
            
            if not isinstance(metadata, PluginMetadata):
                return False
            
            # 检查健康状态方法
            health_status = instance.get_health_status()
            
            if not isinstance(health_status, dict):
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating plugin instance: {e}")
            return False


# 全局插件发现器和加载器实例
_plugin_discovery = None
_plugin_loader = None


def get_plugin_discovery() -> PluginDiscovery:
    """获取插件发现器实例"""
    global _plugin_discovery
    if _plugin_discovery is None:
        _plugin_discovery = PluginDiscovery()
    return _plugin_discovery


def get_plugin_loader() -> DynamicPluginLoader:
    """获取插件加载器实例"""
    global _plugin_loader
    if _plugin_loader is None:
        _plugin_loader = DynamicPluginLoader()
    return _plugin_loader


def initialize_plugin_system(plugin_directories: List[str] = None):
    """初始化插件系统"""
    discovery = get_plugin_discovery()
    
    if plugin_directories:
        for directory in plugin_directories:
            discovery.add_plugin_directory(directory)
    
    # 发现插件
    discovered = discovery.discover_plugins()
    logger.info(f"Discovered {len(discovered)} plugins")
    
    return discovered