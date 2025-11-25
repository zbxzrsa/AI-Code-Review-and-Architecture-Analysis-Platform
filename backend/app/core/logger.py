"""
统一日志模块，用于替代直接使用print语句
提供结构化日志输出和级别控制
"""

import logging
import sys
from enum import Enum
from typing import Any, Dict, Optional

# 配置基础日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Logger:
    """统一日志类"""
    
    def __init__(self, name: str):
        """
        初始化日志器
        
        Args:
            name: 日志器名称，通常是模块名
        """
        self.logger = logging.getLogger(name)
    
    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        调试级别日志
        
        Args:
            msg: 日志消息
            extra: 额外的结构化数据
        """
        self._log(self.logger.debug, msg, extra)
    
    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        信息级别日志
        
        Args:
            msg: 日志消息
            extra: 额外的结构化数据
        """
        self._log(self.logger.info, msg, extra)
    
    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        警告级别日志
        
        Args:
            msg: 日志消息
            extra: 额外的结构化数据
        """
        self._log(self.logger.warning, msg, extra)
    
    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        错误级别日志
        
        Args:
            msg: 日志消息
            extra: 额外的结构化数据
        """
        self._log(self.logger.error, msg, extra)
    
    def critical(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        严重错误级别日志
        
        Args:
            msg: 日志消息
            extra: 额外的结构化数据
        """
        self._log(self.logger.critical, msg, extra)

    def exception(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        异常级别日志
        
        Args:
            msg: 日志消息
            extra: 额外的结构化数据
        """
        self._log(self.logger.exception, msg, extra)
    
    def _log(self, log_func, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        内部日志记录方法
        
        Args:
            log_func: 日志函数
            msg: 日志消息
            extra: 额外的结构化数据
        """
        if extra:
            log_func(f"{msg} | {extra}")
        else:
            log_func(msg)


# 创建默认日志器
def get_logger(name: str) -> Logger:
    """
    获取指定名称的日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        Logger: 日志器实例
    """
    return Logger(name)

# 默认日志器实例
logger = get_logger(__name__)