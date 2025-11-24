"""
插件管理系统 - 支持动态加载和管理翻译引擎、内容处理器等插件
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import importlib
import inspect
import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """插件类型"""
    TRANSLATION_ENGINE = "translation_engine"
    CONTENT_PROCESSOR = "content_processor"
    QUALITY_CHECKER = "quality_checker"
    CACHE_PROVIDER = "cache_provider"
    MONITOR = "monitor"


class PluginStatus(Enum):
    """插件状态"""
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class PluginMetadata:
    """插件元数据"""
    id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    supported_features: List[str] = field(default_factory=list)
    min_system_version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PluginInfo:
    """插件信息"""
    metadata: PluginMetadata
    status: PluginStatus
    instance: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)
    load_time: Optional[datetime] = None
    error_message: Optional[str] = None
    usage_stats: Dict[str, Any] = field(default_factory=dict)


class PluginInterface(ABC):
    """插件基础接口"""
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """获取插件元数据"""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """清理插件资源"""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """获取插件健康状态"""
        pass


class TranslationEnginePlugin(PluginInterface):
    """翻译引擎插件接口"""
    
    @abstractmethod
    async def translate(self, text: str, source_lang: str, target_lang: str, **kwargs) -> str:
        """翻译文本"""
        pass
    
    @abstractmethod
    async def detect_language(self, text: str) -> str:
        """检测语言"""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表"""
        pass


class ContentProcessorPlugin(PluginInterface):
    """内容处理器插件接口"""
    
    @abstractmethod
    async def process_content(self, content: str, content_type: str, **kwargs) -> str:
        """处理内容"""
        pass
    
    @abstractmethod
    def get_supported_content_types(self) -> List[str]:
        """获取支持的内容类型"""
        pass


class QualityCheckerPlugin(PluginInterface):
    """质量检查器插件接口"""
    
    @abstractmethod
    async def check_quality(self, content: str, **kwargs) -> Dict[str, Any]:
        """检查内容质量"""
        pass
    
    @abstractmethod
    def get_quality_metrics(self) -> List[str]:
        """获取质量指标列表"""
        pass


class PluginLoader:
    """插件加载器"""
    
    def __init__(self, plugin_directories: List[str]):
        self.plugin_directories = plugin_directories
        self.loaded_modules: Dict[str, Any] = {}
    
    async def load_plugin_from_file(self, plugin_path: str) -> Optional[PluginInterface]:
        """从文件加载插件"""
        try:
            # 获取插件模块名
            plugin_name = Path(plugin_path).stem
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            
            if spec is None or spec.loader is None:
                logger.error(f"Cannot load plugin spec from {plugin_path}")
                return None
            
            # 加载模块
            module = importlib.util.module_from_spec(spec)
            self.loaded_modules[plugin_name] = module
            
            # 执行模块
            spec.loader.exec_module(module)
            
            # 查找插件类
            plugin_class = self._find_plugin_class(module)
            if plugin_class is None:
                logger.error(f"No plugin class found in {plugin_path}")
                return None
            
            # 创建插件实例
            plugin_instance = plugin_class()
            
            logger.info(f"Successfully loaded plugin from {plugin_path}")
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Failed to load plugin from {plugin_path}: {e}")
            return None
    
    async def load_plugin_from_package(self, package_name: str) -> Optional[PluginInterface]:
        """从包加载插件"""
        try:
            module = importlib.import_module(package_name)
            self.loaded_modules[package_name] = module
            
            plugin_class = self._find_plugin_class(module)
            if plugin_class is None:
                logger.error(f"No plugin class found in package {package_name}")
                return None
            
            plugin_instance = plugin_class()
            
            logger.info(f"Successfully loaded plugin from package {package_name}")
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Failed to load plugin from package {package_name}: {e}")
            return None
    
    def _find_plugin_class(self, module) -> Optional[Type[PluginInterface]]:
        """在模块中查找插件类"""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, PluginInterface) and 
                obj != PluginInterface):
                return obj
        return None
    
    async def discover_plugins(self) -> List[str]:
        """发现插件文件"""
        plugin_files = []
        
        for directory in self.plugin_directories:
            if not os.path.exists(directory):
                continue
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        plugin_files.append(os.path.join(root, file))
        
        return plugin_files


class PluginRegistry:
    """插件注册表"""
    
    def __init__(self):
        self.plugins: Dict[str, PluginInfo] = {}
        self.type_index: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
    
    def register_plugin(self, plugin: PluginInterface, config: Dict[str, Any] = None) -> str:
        """注册插件"""
        metadata = plugin.get_metadata()
        plugin_id = metadata.id
        
        if config is None:
            config = {}
        
        plugin_info = PluginInfo(
            metadata=metadata,
            status=PluginStatus.LOADED,
            instance=plugin,
            config=config,
            load_time=datetime.now()
        )
        
        self.plugins[plugin_id] = plugin_info
        self.type_index[metadata.plugin_type].append(plugin_id)
        
        logger.info(f"Registered plugin: {metadata.name} ({plugin_id})")
        return plugin_id
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """注销插件"""
        if plugin_id not in self.plugins:
            return False
        
        plugin_info = self.plugins[plugin_id]
        plugin_type = plugin_info.metadata.plugin_type
        
        # 从类型索引中移除
        if plugin_id in self.type_index[plugin_type]:
            self.type_index[plugin_type].remove(plugin_id)
        
        # 清理插件
        if plugin_info.instance:
            try:
                asyncio.create_task(plugin_info.instance.cleanup())
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin_id}: {e}")
        
        # 从注册表中移除
        del self.plugins[plugin_id]
        
        logger.info(f"Unregistered plugin: {plugin_id}")
        return True
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """获取插件信息"""
        return self.plugins.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """根据类型获取插件列表"""
        plugin_ids = self.type_index.get(plugin_type, [])
        return [self.plugins[pid] for pid in plugin_ids if pid in self.plugins]
    
    def get_active_plugins(self, plugin_type: Optional[PluginType] = None) -> List[PluginInfo]:
        """获取活跃插件列表"""
        plugins = []
        
        if plugin_type:
            candidates = self.get_plugins_by_type(plugin_type)
        else:
            candidates = list(self.plugins.values())
        
        for plugin_info in candidates:
            if plugin_info.status == PluginStatus.ACTIVE:
                plugins.append(plugin_info)
        
        return plugins
    
    def list_all_plugins(self) -> Dict[str, PluginInfo]:
        """列出所有插件"""
        return self.plugins.copy()


class PluginManager:
    """插件管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plugin_directories = config.get('plugin_directories', ['./plugins'])
        self.loader = PluginLoader(self.plugin_directories)
        self.registry = PluginRegistry()
        self.hooks: Dict[str, List[Callable]] = {}
        
        # 确保插件目录存在
        for directory in self.plugin_directories:
            os.makedirs(directory, exist_ok=True)
    
    async def initialize(self):
        """初始化插件管理器"""
        logger.info("Initializing plugin manager...")
        
        # 自动发现和加载插件
        if self.config.get('auto_discover', True):
            await self.discover_and_load_plugins()
        
        logger.info("Plugin manager initialized")
    
    async def discover_and_load_plugins(self):
        """发现并加载插件"""
        plugin_files = await self.loader.discover_plugins()
        
        for plugin_file in plugin_files:
            try:
                plugin = await self.loader.load_plugin_from_file(plugin_file)
                if plugin:
                    await self.register_and_activate_plugin(plugin)
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")
    
    async def register_and_activate_plugin(self, plugin: PluginInterface, config: Dict[str, Any] = None) -> str:
        """注册并激活插件"""
        # 注册插件
        plugin_id = self.registry.register_plugin(plugin, config)
        
        # 初始化插件
        try:
            plugin_config = config or {}
            success = await plugin.initialize(plugin_config)
            
            if success:
                plugin_info = self.registry.get_plugin(plugin_id)
                if plugin_info:
                    plugin_info.status = PluginStatus.ACTIVE
                logger.info(f"Activated plugin: {plugin_id}")
            else:
                plugin_info = self.registry.get_plugin(plugin_id)
                if plugin_info:
                    plugin_info.status = PluginStatus.ERROR
                    plugin_info.error_message = "Initialization failed"
                logger.error(f"Failed to activate plugin: {plugin_id}")
                
        except Exception as e:
            plugin_info = self.registry.get_plugin(plugin_id)
            if plugin_info:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = str(e)
            logger.error(f"Error activating plugin {plugin_id}: {e}")
        
        return plugin_id
    
    async def deactivate_plugin(self, plugin_id: str) -> bool:
        """停用插件"""
        plugin_info = self.registry.get_plugin(plugin_id)
        if not plugin_info or not plugin_info.instance:
            return False
        
        try:
            await plugin_info.instance.cleanup()
            plugin_info.status = PluginStatus.INACTIVE
            logger.info(f"Deactivated plugin: {plugin_id}")
            return True
        except Exception as e:
            plugin_info.status = PluginStatus.ERROR
            plugin_info.error_message = str(e)
            logger.error(f"Error deactivating plugin {plugin_id}: {e}")
            return False
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """卸载插件"""
        # 先停用插件
        await self.deactivate_plugin(plugin_id)
        
        # 从注册表中移除
        return self.registry.unregister_plugin(plugin_id)
    
    def get_translation_engines(self) -> List[TranslationEnginePlugin]:
        """获取翻译引擎插件"""
        plugins = self.registry.get_active_plugins(PluginType.TRANSLATION_ENGINE)
        return [p.instance for p in plugins if p.instance]
    
    def get_content_processors(self) -> List[ContentProcessorPlugin]:
        """获取内容处理器插件"""
        plugins = self.registry.get_active_plugins(PluginType.CONTENT_PROCESSOR)
        return [p.instance for p in plugins if p.instance]
    
    def get_quality_checkers(self) -> List[QualityCheckerPlugin]:
        """获取质量检查器插件"""
        plugins = self.registry.get_active_plugins(PluginType.QUALITY_CHECKER)
        return [p.instance for p in plugins if p.instance]
    
    def register_hook(self, event_name: str, callback: Callable):
        """注册事件钩子"""
        if event_name not in self.hooks:
            self.hooks[event_name] = []
        self.hooks[event_name].append(callback)
    
    async def trigger_hook(self, event_name: str, *args, **kwargs):
        """触发事件钩子"""
        if event_name in self.hooks:
            for callback in self.hooks[event_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in hook callback for {event_name}: {e}")
    
    async def get_plugin_health_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有插件的健康状态"""
        health_status = {}
        
        for plugin_id, plugin_info in self.registry.list_all_plugins().items():
            try:
                if plugin_info.instance and plugin_info.status == PluginStatus.ACTIVE:
                    status = plugin_info.instance.get_health_status()
                else:
                    status = {
                        'status': plugin_info.status.value,
                        'error': plugin_info.error_message
                    }
                
                health_status[plugin_id] = {
                    'name': plugin_info.metadata.name,
                    'type': plugin_info.metadata.plugin_type.value,
                    'version': plugin_info.metadata.version,
                    'status': status,
                    'load_time': plugin_info.load_time.isoformat() if plugin_info.load_time else None
                }
                
            except Exception as e:
                health_status[plugin_id] = {
                    'name': plugin_info.metadata.name,
                    'type': plugin_info.metadata.plugin_type.value,
                    'version': plugin_info.metadata.version,
                    'status': {'status': 'error', 'error': str(e)},
                    'load_time': plugin_info.load_time.isoformat() if plugin_info.load_time else None
                }
        
        return health_status
    
    async def reload_plugin(self, plugin_id: str) -> bool:
        """重新加载插件"""
        plugin_info = self.registry.get_plugin(plugin_id)
        if not plugin_info:
            return False
        
        # 保存配置
        config = plugin_info.config.copy()
        
        # 卸载插件
        await self.unload_plugin(plugin_id)
        
        # 重新加载（这里简化处理，实际应该重新从文件加载）
        # 在实际实现中，需要记录插件的来源路径
        logger.info(f"Plugin {plugin_id} reload requested (simplified implementation)")
        return True


# 全局插件管理器实例
plugin_manager: Optional[PluginManager] = None


def initialize_plugin_manager(config: Dict[str, Any]) -> PluginManager:
    """初始化插件管理器"""
    global plugin_manager
    plugin_manager = PluginManager(config)
    return plugin_manager


def get_plugin_manager() -> PluginManager:
    """获取插件管理器实例"""
    if plugin_manager is None:
        raise RuntimeError("Plugin manager not initialized")
    return plugin_manager