"""
代码解析服务核心模块
提供多语言代码解析、AST生成、CFG/DFG提取和代码度量计算功能
"""
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from enum import Enum
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureType(str, Enum):
    """支持的特征类型"""
    AST = "ast"
    CFG = "cfg"
    DFG = "dfg"
    METRICS = "metrics"

class Language(str, Enum):
    """支持的编程语言"""
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    GO = "go"

class ParserError(Exception):
    """解析错误异常"""
    pass

class BaseParser(ABC):
    """解析器基类"""
    
    @abstractmethod
    async def parse(self, code: str, features: List[FeatureType]) -> Dict[str, Any]:
        """解析代码并提取请求的特征"""
        pass
    
    @abstractmethod
    async def generate_ast(self, code: str) -> Dict[str, Any]:
        """生成抽象语法树"""
        pass
    
    @abstractmethod
    async def generate_cfg(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """生成控制流图"""
        pass
    
    @abstractmethod
    async def generate_dfg(self, ast: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
        """生成数据流图"""
        pass
    
    @abstractmethod
    async def calculate_metrics(self, ast: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """计算代码度量指标"""
        pass

class CodeParserFactory:
    """代码解析器工厂"""
    
    @staticmethod
    def create_parser(language: Language) -> BaseParser:
        """创建特定语言的解析器"""
        if language == Language.PYTHON:
            from .parsers.python_parser import PythonParser
            return PythonParser()
        elif language == Language.JAVA:
            from .parsers.java_parser import JavaParser
            return JavaParser()
        elif language == Language.JAVASCRIPT:
            from .parsers.javascript_parser import JavaScriptParser
            return JavaScriptParser()
        elif language == Language.GO:
            from .parsers.go_parser import GoParser
            return GoParser()
        else:
            raise ValueError(f"不支持的语言: {language}")

class CodeParserService:
    """代码解析服务"""
    
    def __init__(self):
        """初始化代码解析服务"""
        self._parsers = {}
        self._semaphore = asyncio.Semaphore(10)  # 限制并发解析数量
    
    async def parse_code(self, code: str, language: Language, features: List[FeatureType]) -> Dict[str, Any]:
        """
        解析代码并提取请求的特征
        
        Args:
            code: 源代码字符串
            language: 编程语言
            features: 需要提取的特征列表
            
        Returns:
            包含请求特征的字典
        """
        async with self._semaphore:
            try:
                # 获取或创建解析器
                if language not in self._parsers:
                    self._parsers[language] = CodeParserFactory.create_parser(language)
                
                parser = self._parsers[language]
                
                # 解析代码
                result = await parser.parse(code, features)
                return result
            
            except Exception as e:
                logger.error(f"解析代码时出错: {str(e)}")
                raise ParserError(f"解析代码时出错: {str(e)}")
    
    async def batch_parse(self, batch: List[Tuple[str, Language, List[FeatureType]]]) -> List[Dict[str, Any]]:
        """
        批量解析代码
        
        Args:
            batch: 包含(代码, 语言, 特征)元组的列表
            
        Returns:
            解析结果列表
        """
        tasks = [self.parse_code(code, lang, feats) for code, lang, feats in batch]
        return await asyncio.gather(*tasks, return_exceptions=True)

# 创建全局解析服务实例
parser_service = CodeParserService()