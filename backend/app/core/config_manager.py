"""
配置管理系统
支持多环境配置、动态更新和配置验证
"""

from typing import Dict, List, Optional, Any, Union, Type, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os
import json
import yaml
import asyncio
import logging
from pathlib import Path
from abc import ABC, abstractmethod
import hashlib
from pydantic import BaseModel, Field, validator, BaseSettings

logger = logging.getLogger(__name__)

class ConfigEnvironment(Enum):
    """配置环境"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"

class ConfigFormat(Enum):
    """配置格式"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"
    TOML = "toml"

class ConfigSource(Enum):
    """配置源"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    CONSUL = "consul"
    ETCD = "etcd"
    KUBERNETES = "kubernetes"

@dataclass
class ConfigChange:
    """配置变更记录"""
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str
    user: Optional[str] = None
    reason: Optional[str] = None

@dataclass
class ConfigValidationRule:
    """配置验证规则"""
    key: str
    validator: Callable[[Any], bool]
    error_message: str
    required: bool = True

class DatabaseConfig(BaseModel):
    """数据库配置"""
    host: str = Field(..., description="数据库主机")
    port: int = Field(5432, ge=1, le=65535, description="数据库端口")
    database: str = Field(..., description="数据库名称")
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    pool_size: int = Field(10, ge=1, le=100, description="连接池大小")
    max_overflow: int = Field(20, ge=0, le=100, description="最大溢出连接数")
    
    @validator('host')
    def validate_host(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('数据库主机不能为空')
        return v.strip()

class RedisConfig(BaseModel):
    """Redis配置"""
    host: str = Field("localhost", description="Redis主机")
    port: int = Field(6379, ge=1, le=65535, description="Redis端口")
    database: int = Field(0, ge=0, le=15, description="Redis数据库")
    password: Optional[str] = Field(None, description="Redis密码")
    max_connections: int = Field(100, ge=1, le=1000, description="最大连接数")
    timeout: int = Field(30, ge=1, le=300, description="超时时间(秒)")

class TranslationConfig(BaseModel):
    """翻译服务配置"""
    default_engine: str = Field("google", description="默认翻译引擎")
    max_text_length: int = Field(5000, ge=1, le=50000, description="最大文本长度")
    cache_ttl: int = Field(3600, ge=60, le=86400, description="缓存TTL(秒)")
    rate_limit: int = Field(100, ge=1, le=10000, description="速率限制(请求/分钟)")
    supported_languages: List[str] = Field(
        default=["en", "zh", "ja", "ko", "fr", "de", "es"],
        description="支持的语言列表"
    )
    
    @validator('supported_languages')
    def validate_languages(cls, v):
        if not v or len(v) == 0:
            raise ValueError('至少需要支持一种语言')
        return v

class MonitoringConfig(BaseModel):
    """监控配置"""
    enabled: bool = Field(True, description="是否启用监控")
    metrics_interval: int = Field(60, ge=10, le=3600, description="指标收集间隔(秒)")
    alert_threshold_cpu: float = Field(80.0, ge=0.0, le=100.0, description="CPU告警阈值(%)")
    alert_threshold_memory: float = Field(85.0, ge=0.0, le=100.0, description="内存告警阈值(%)")
    retention_days: int = Field(30, ge=1, le=365, description="数据保留天数")

class SecurityConfig(BaseModel):
    """安全配置"""
    secret_key: str = Field(..., min_length=32, description="密钥")
    jwt_expiration: int = Field(3600, ge=300, le=86400, description="JWT过期时间(秒)")
    password_min_length: int = Field(8, ge=6, le=128, description="密码最小长度")
    max_login_attempts: int = Field(5, ge=1, le=20, description="最大登录尝试次数")
    session_timeout: int = Field(1800, ge=300, le=86400, description="会话超时时间(秒)")

class ApplicationConfig(BaseSettings):
    """应用程序配置"""
    # 基础配置
    app_name: str = Field("Translation System", description="应用名称")
    app_version: str = Field("1.0.0", description="应用版本")
    environment: ConfigEnvironment = Field(ConfigEnvironment.DEVELOPMENT, description="运行环境")
    debug: bool = Field(False, description="调试模式")
    log_level: str = Field("INFO", description="日志级别")
    
    # 服务配置
    host: str = Field("0.0.0.0", description="服务主机")
    port: int = Field(8000, ge=1, le=65535, description="服务端口")
    workers: int = Field(1, ge=1, le=32, description="工作进程数")
    
    # 子配置
    database: DatabaseConfig
    redis: RedisConfig
    translation: TranslationConfig
    monitoring: MonitoringConfig
    security: SecurityConfig
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False

class ConfigProvider(ABC):
    """配置提供者抽象基类"""
    
    @abstractmethod
    async def load_config(self, key: str) -> Dict[str, Any]:
        """加载配置"""
        pass
    
    @abstractmethod
    async def save_config(self, key: str, config: Dict[str, Any]) -> bool:
        """保存配置"""
        pass
    
    @abstractmethod
    async def watch_config(self, key: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """监听配置变化"""
        pass

class FileConfigProvider(ConfigProvider):
    """文件配置提供者"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.watchers: Dict[str, List[Callable]] = {}
    
    async def load_config(self, key: str) -> Dict[str, Any]:
        """从文件加载配置"""
        try:
            config_file = self.config_dir / f"{key}.yaml"
            if not config_file.exists():
                return {}
            
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
                
        except Exception as e:
            logger.error(f"Failed to load config from file {key}: {e}")
            return {}
    
    async def save_config(self, key: str, config: Dict[str, Any]) -> bool:
        """保存配置到文件"""
        try:
            config_file = self.config_dir / f"{key}.yaml"
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # 通知监听者
            await self._notify_watchers(key, config)
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to file {key}: {e}")
            return False
    
    async def watch_config(self, key: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """监听配置文件变化"""
        if key not in self.watchers:
            self.watchers[key] = []
        
        self.watchers[key].append(callback)
        return True
    
    async def _notify_watchers(self, key: str, config: Dict[str, Any]):
        """通知配置监听者"""
        if key in self.watchers:
            for callback in self.watchers[key]:
                try:
                    await asyncio.create_task(callback(config))
                except Exception as e:
                    logger.error(f"Error in config watcher callback: {e}")

class EnvironmentConfigProvider(ConfigProvider):
    """环境变量配置提供者"""
    
    def __init__(self, prefix: str = "TRANSLATION_"):
        self.prefix = prefix
    
    async def load_config(self, key: str) -> Dict[str, Any]:
        """从环境变量加载配置"""
        config = {}
        env_key = f"{self.prefix}{key.upper()}"
        
        for env_name, env_value in os.environ.items():
            if env_name.startswith(env_key):
                # 转换环境变量名为配置键
                config_key = env_name[len(env_key):].lower().replace('_', '.')
                if config_key:
                    config[config_key] = self._parse_env_value(env_value)
        
        return config
    
    async def save_config(self, key: str, config: Dict[str, Any]) -> bool:
        """环境变量不支持保存"""
        logger.warning("Environment config provider does not support saving")
        return False
    
    async def watch_config(self, key: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """环境变量不支持监听"""
        logger.warning("Environment config provider does not support watching")
        return False
    
    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值"""
        # 尝试解析为JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # 尝试解析为布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # 尝试解析为数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # 返回字符串
        return value

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.providers: Dict[ConfigSource, ConfigProvider] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.validation_rules: Dict[str, List[ConfigValidationRule]] = {}
        self.change_history: List[ConfigChange] = []
        self.watchers: Dict[str, List[Callable]] = {}
        self.config_cache: Dict[str, tuple] = {}  # (config, hash, timestamp)
        self.cache_ttl = 300  # 5分钟缓存
    
    def register_provider(self, source: ConfigSource, provider: ConfigProvider):
        """注册配置提供者"""
        self.providers[source] = provider
        logger.info(f"Registered config provider: {source.value}")
    
    def add_validation_rule(self, config_key: str, rule: ConfigValidationRule):
        """添加配置验证规则"""
        if config_key not in self.validation_rules:
            self.validation_rules[config_key] = []
        
        self.validation_rules[config_key].append(rule)
        logger.info(f"Added validation rule for {config_key}.{rule.key}")
    
    async def load_config(self, config_key: str, sources: List[ConfigSource] = None) -> Dict[str, Any]:
        """加载配置"""
        try:
            # 检查缓存
            if config_key in self.config_cache:
                config, config_hash, timestamp = self.config_cache[config_key]
                if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                    return config
            
            if sources is None:
                sources = list(self.providers.keys())
            
            merged_config = {}
            
            # 按优先级合并配置
            for source in sources:
                if source in self.providers:
                    provider_config = await self.providers[source].load_config(config_key)
                    merged_config = self._merge_configs(merged_config, provider_config)
            
            # 验证配置
            validation_errors = self._validate_config(config_key, merged_config)
            if validation_errors:
                logger.error(f"Config validation failed for {config_key}: {validation_errors}")
                raise ValueError(f"Configuration validation failed: {validation_errors}")
            
            # 更新缓存
            config_hash = self._calculate_config_hash(merged_config)
            self.config_cache[config_key] = (merged_config, config_hash, datetime.now())
            
            self.configs[config_key] = merged_config
            logger.info(f"Loaded config: {config_key}")
            
            return merged_config
            
        except Exception as e:
            logger.error(f"Failed to load config {config_key}: {e}")
            return self.configs.get(config_key, {})
    
    async def save_config(self, config_key: str, config: Dict[str, Any], source: ConfigSource = ConfigSource.FILE) -> bool:
        """保存配置"""
        try:
            # 验证配置
            validation_errors = self._validate_config(config_key, config)
            if validation_errors:
                logger.error(f"Config validation failed: {validation_errors}")
                return False
            
            # 记录变更
            old_config = self.configs.get(config_key, {})
            changes = self._detect_changes(config_key, old_config, config)
            
            if source in self.providers:
                success = await self.providers[source].save_config(config_key, config)
                if success:
                    self.configs[config_key] = config
                    
                    # 更新缓存
                    config_hash = self._calculate_config_hash(config)
                    self.config_cache[config_key] = (config, config_hash, datetime.now())
                    
                    # 记录变更历史
                    self.change_history.extend(changes)
                    
                    # 通知监听者
                    await self._notify_watchers(config_key, config)
                    
                    logger.info(f"Saved config: {config_key}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to save config {config_key}: {e}")
            return False
    
    async def update_config(self, config_key: str, updates: Dict[str, Any], source: ConfigSource = ConfigSource.FILE) -> bool:
        """更新配置"""
        current_config = await self.load_config(config_key)
        updated_config = self._merge_configs(current_config, updates)
        return await self.save_config(config_key, updated_config, source)
    
    async def delete_config(self, config_key: str, source: ConfigSource = ConfigSource.FILE) -> bool:
        """删除配置"""
        try:
            if source in self.providers:
                success = await self.providers[source].save_config(config_key, {})
                if success:
                    if config_key in self.configs:
                        del self.configs[config_key]
                    if config_key in self.config_cache:
                        del self.config_cache[config_key]
                    
                    logger.info(f"Deleted config: {config_key}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete config {config_key}: {e}")
            return False
    
    def watch_config(self, config_key: str, callback: Callable[[Dict[str, Any]], None]):
        """监听配置变化"""
        if config_key not in self.watchers:
            self.watchers[config_key] = []
        
        self.watchers[config_key].append(callback)
        logger.info(f"Added config watcher for {config_key}")
    
    def get_config_history(self, config_key: str, limit: int = 100) -> List[ConfigChange]:
        """获取配置变更历史"""
        return [
            change for change in self.change_history[-limit:]
            if change.key.startswith(config_key)
        ]
    
    def get_config_status(self) -> Dict[str, Any]:
        """获取配置状态"""
        return {
            "loaded_configs": list(self.configs.keys()),
            "providers": [source.value for source in self.providers.keys()],
            "validation_rules": {
                key: len(rules) for key, rules in self.validation_rules.items()
            },
            "watchers": {
                key: len(callbacks) for key, callbacks in self.watchers.items()
            },
            "change_history_count": len(self.change_history),
            "cache_status": {
                key: {
                    "cached": True,
                    "timestamp": timestamp.isoformat(),
                    "hash": config_hash[:8]
                }
                for key, (_, config_hash, timestamp) in self.config_cache.items()
            }
        }
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config_key: str, config: Dict[str, Any]) -> List[str]:
        """验证配置"""
        errors = []
        
        if config_key in self.validation_rules:
            for rule in self.validation_rules[config_key]:
                value = self._get_nested_value(config, rule.key)
                
                if value is None and rule.required:
                    errors.append(f"Required config key '{rule.key}' is missing")
                elif value is not None and not rule.validator(value):
                    errors.append(rule.error_message)
        
        return errors
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """获取嵌套配置值"""
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def _detect_changes(self, config_key: str, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[ConfigChange]:
        """检测配置变更"""
        changes = []
        timestamp = datetime.now()
        
        # 检查新增和修改
        for key, new_value in new_config.items():
            old_value = old_config.get(key)
            if old_value != new_value:
                changes.append(ConfigChange(
                    key=f"{config_key}.{key}",
                    old_value=old_value,
                    new_value=new_value,
                    timestamp=timestamp,
                    source="config_manager"
                ))
        
        # 检查删除
        for key, old_value in old_config.items():
            if key not in new_config:
                changes.append(ConfigChange(
                    key=f"{config_key}.{key}",
                    old_value=old_value,
                    new_value=None,
                    timestamp=timestamp,
                    source="config_manager"
                ))
        
        return changes
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """计算配置哈希值"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    async def _notify_watchers(self, config_key: str, config: Dict[str, Any]):
        """通知配置监听者"""
        if config_key in self.watchers:
            for callback in self.watchers[config_key]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(config)
                    else:
                        callback(config)
                except Exception as e:
                    logger.error(f"Error in config watcher callback: {e}")

# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None
_app_config: Optional[ApplicationConfig] = None

def get_config_manager() -> ConfigManager:
    """获取配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_app_config() -> ApplicationConfig:
    """获取应用配置实例"""
    global _app_config
    if _app_config is None:
        _app_config = ApplicationConfig()
    return _app_config

async def init_config_system(config_dir: str = "config") -> ConfigManager:
    """初始化配置系统"""
    global _config_manager, _app_config
    
    # 创建配置管理器
    _config_manager = ConfigManager()
    
    # 注册配置提供者
    file_provider = FileConfigProvider(config_dir)
    env_provider = EnvironmentConfigProvider()
    
    _config_manager.register_provider(ConfigSource.FILE, file_provider)
    _config_manager.register_provider(ConfigSource.ENVIRONMENT, env_provider)
    
    # 添加验证规则
    _add_default_validation_rules(_config_manager)
    
    # 加载应用配置
    try:
        _app_config = ApplicationConfig()
        logger.info("Configuration system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application config: {e}")
        # 使用默认配置
        _app_config = ApplicationConfig(
            database=DatabaseConfig(host="localhost", database="translation", username="user", password="password"),
            redis=RedisConfig(),
            translation=TranslationConfig(),
            monitoring=MonitoringConfig(),
            security=SecurityConfig(secret_key="default-secret-key-change-in-production")
        )
    
    return _config_manager

def _add_default_validation_rules(config_manager: ConfigManager):
    """添加默认验证规则"""
    # 数据库配置验证
    config_manager.add_validation_rule("database", ConfigValidationRule(
        key="host",
        validator=lambda x: isinstance(x, str) and len(x.strip()) > 0,
        error_message="Database host must be a non-empty string"
    ))
    
    config_manager.add_validation_rule("database", ConfigValidationRule(
        key="port",
        validator=lambda x: isinstance(x, int) and 1 <= x <= 65535,
        error_message="Database port must be between 1 and 65535"
    ))
    
    # Redis配置验证
    config_manager.add_validation_rule("redis", ConfigValidationRule(
        key="port",
        validator=lambda x: isinstance(x, int) and 1 <= x <= 65535,
        error_message="Redis port must be between 1 and 65535"
    ))
    
    # 翻译配置验证
    config_manager.add_validation_rule("translation", ConfigValidationRule(
        key="max_text_length",
        validator=lambda x: isinstance(x, int) and x > 0,
        error_message="Max text length must be a positive integer"
    ))
    
    # 安全配置验证
    config_manager.add_validation_rule("security", ConfigValidationRule(
        key="secret_key",
        validator=lambda x: isinstance(x, str) and len(x) >= 32,
        error_message="Secret key must be at least 32 characters long"
    ))