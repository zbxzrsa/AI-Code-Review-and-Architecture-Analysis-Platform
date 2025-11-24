"""
统一的语言解析器基类

提供多语言解析的统一接口，支持增量解析和错误恢复机制
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import os
from pathlib import Path


class ParseErrorSeverity(Enum):
    """解析错误的严重程度"""
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


class SourcePosition:
    """源代码位置信息"""
    def __init__(self, line: int, column: int, offset: int):
        self.line = line  # 行号（从1开始）
        self.column = column  # 列号（从1开始）
        self.offset = offset  # 字符偏移量（从0开始）

    def __str__(self) -> str:
        return f"line {self.line}, column {self.column}"


class SourceRange:
    """源代码范围信息"""
    def __init__(self, start: SourcePosition, end: SourcePosition, file_path: Optional[str] = None):
        self.start = start
        self.end = end
        self.file_path = file_path

    def __str__(self) -> str:
        file_info = f"{os.path.basename(self.file_path)}:" if self.file_path else ""
        return f"{file_info}{self.start} to {self.end}"


class ParseError:
    """解析错误信息"""
    def __init__(
        self, 
        message: str, 
        source_range: SourceRange,
        severity: ParseErrorSeverity = ParseErrorSeverity.ERROR,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.source_range = source_range
        self.severity = severity
        self.error_code = error_code

    def __str__(self) -> str:
        severity = self.severity.name
        code_info = f" [{self.error_code}]" if self.error_code else ""
        return f"{severity}{code_info}: {self.message} at {self.source_range}"


class ParseResult:
    """解析结果"""
    def __init__(self, ast: Any, errors: List[ParseError] = None):
        self.ast = ast  # 语言特定的AST
        self.errors = errors or []
        
    @property
    def has_critical_errors(self) -> bool:
        """是否有严重错误"""
        return any(error.severity in (ParseErrorSeverity.ERROR, ParseErrorSeverity.CRITICAL) 
                  for error in self.errors)


class LanguageParser(ABC):
    """语言解析器基类"""
    
    @abstractmethod
    def parse_string(self, code: str, file_path: Optional[str] = None) -> ParseResult:
        """
        解析字符串代码
        
        Args:
            code: 源代码字符串
            file_path: 可选的文件路径，用于错误报告
            
        Returns:
            ParseResult: 包含AST和错误信息的解析结果
        """
        pass
    
    @abstractmethod
    def parse_file(self, file_path: str) -> ParseResult:
        """
        解析文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            ParseResult: 包含AST和错误信息的解析结果
        """
        pass
    
    @abstractmethod
    def parse_incremental(self, 
                         previous_result: ParseResult, 
                         edited_range: SourceRange, 
                         new_content: str) -> ParseResult:
        """
        增量解析（仅解析修改的部分）
        
        Args:
            previous_result: 之前的解析结果
            edited_range: 编辑的范围
            new_content: 新的内容
            
        Returns:
            ParseResult: 更新后的解析结果
        """
        pass
    
    @abstractmethod
    def get_language_id(self) -> str:
        """
        获取语言标识符
        
        Returns:
            str: 语言标识符，如 'python', 'javascript', 'java', 'csharp'
        """
        pass
    
    @abstractmethod
    def get_supported_file_extensions(self) -> List[str]:
        """
        获取支持的文件扩展名
        
        Returns:
            List[str]: 支持的文件扩展名列表，如 ['.py', '.pyi']
        """
        pass
    
    def can_parse_file(self, file_path: str) -> bool:
        """
        检查是否可以解析指定文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 如果可以解析则返回True
        """
        ext = Path(file_path).suffix.lower()
        return ext in self.get_supported_file_extensions()
    
    def recover_from_errors(self, code: str, errors: List[ParseError]) -> Tuple[str, List[ParseError]]:
        """
        从错误中恢复（默认实现，子类可以重写提供更好的恢复机制）
        
        Args:
            code: 原始代码
            errors: 解析错误列表
            
        Returns:
            Tuple[str, List[ParseError]]: 修复后的代码和剩余错误
        """
        # 默认实现不做任何修复，子类应该重写此方法提供更好的恢复机制
        return code, errors


class ParserRegistry:
    """解析器注册表，用于管理和获取不同语言的解析器"""
    
    _parsers: Dict[str, LanguageParser] = {}
    _extension_map: Dict[str, str] = {}
    
    @classmethod
    def register_parser(cls, parser: LanguageParser) -> None:
        """
        注册解析器
        
        Args:
            parser: 语言解析器实例
        """
        language_id = parser.get_language_id()
        cls._parsers[language_id] = parser
        
        # 注册文件扩展名映射
        for ext in parser.get_supported_file_extensions():
            cls._extension_map[ext] = language_id
    
    @classmethod
    def get_parser_for_language(cls, language_id: str) -> Optional[LanguageParser]:
        """
        获取指定语言的解析器
        
        Args:
            language_id: 语言标识符
            
        Returns:
            Optional[LanguageParser]: 语言解析器或None
        """
        return cls._parsers.get(language_id)
    
    @classmethod
    def get_parser_for_file(cls, file_path: str) -> Optional[LanguageParser]:
        """
        获取适用于指定文件的解析器
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[LanguageParser]: 语言解析器或None
        """
        ext = Path(file_path).suffix.lower()
        language_id = cls._extension_map.get(ext)
        if language_id:
            return cls._parsers.get(language_id)
        
        # 如果通过扩展名没找到，尝试所有解析器
        for parser in cls._parsers.values():
            if parser.can_parse_file(file_path):
                return parser
        
        return None
    
    @classmethod
    def get_all_parsers(cls) -> Dict[str, LanguageParser]:
        """
        获取所有注册的解析器
        
        Returns:
            Dict[str, LanguageParser]: 语言ID到解析器的映射
        """
        return cls._parsers.copy()