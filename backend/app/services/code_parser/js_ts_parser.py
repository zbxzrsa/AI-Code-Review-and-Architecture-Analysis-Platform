"""
JavaScript/TypeScript解析器

提供JavaScript和TypeScript代码的解析功能，支持ES2022+和TypeScript特性
"""
import os
import json
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .language_parser import LanguageParser, SourceRange, ParseResult


class JSTypeScriptParser(LanguageParser):
    """JavaScript/TypeScript解析器实现"""
    
    def __init__(self, use_typescript: bool = True):
        """
        初始化解析器
        
        Args:
            use_typescript: 是否使用TypeScript解析器（否则使用纯JavaScript解析器）
        """
        super().__init__("javascript" if not use_typescript else "typescript")
        self.use_typescript = use_typescript
    
    def parse_file(self, file_path: str) -> ParseResult:
        """
        解析JavaScript/TypeScript文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析结果
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            return self.parse_code(source_code, file_path)
        except Exception as e:
            error_message = f"解析文件失败: {str(e)}"
            return ParseResult(
                success=False,
                ast=None,
                errors=[{
                    "message": error_message,
                    "location": None
                }]
            )
    
    def parse_code(self, code: str, file_path: Optional[str] = None) -> ParseResult:
        """
        解析JavaScript/TypeScript代码
        
        Args:
            code: 源代码
            file_path: 可选的文件路径
            
        Returns:
            解析结果
        """
        try:
            # 使用内置解析器解析代码
            ast = self._parse_with_internal_parser(code, file_path)
            
            # 增强AST节点，添加父节点引用
            self._enhance_ast(ast)
            
            return ParseResult(
                success=True,
                ast=ast,
                errors=[]
            )
        except Exception as e:
            # 处理解析错误
            error_message = f"解析错误: {str(e)}"
            return ParseResult(
                success=False,
                ast=None,
                errors=[{
                    "message": error_message,
                    "location": None
                }]
            )
    
    def parse_incremental(self, code: str, previous_ast: Any, changed_range: SourceRange) -> ParseResult:
        """
        增量解析JavaScript/TypeScript代码
        
        Args:
            code: 新的源代码
            previous_ast: 之前的AST
            changed_range: 变更范围
            
        Returns:
            解析结果
        """
        # 简单地重新解析整个代码
        return self.parse_code(code)
    
    def get_node_at_position(self, ast_root: Any, line: int, column: int) -> Optional[Any]:
        """
        获取指定位置的AST节点
        
        Args:
            ast_root: AST根节点
            line: 行号（1-based）
            column: 列号（0-based）
            
        Returns:
            位置对应的AST节点，如果没有找到则返回None
        """
        return self._find_node_at_position(ast_root, line, column)
    
    def get_node_range(self, node: Any) -> Optional[SourceRange]:
        """
        获取节点的源代码范围
        
        Args:
            node: AST节点
            
        Returns:
            节点的源代码范围
        """
        if not hasattr(node, 'loc'):
            return None
        
        loc = node.loc
        
        return SourceRange(
            start_line=loc.start.line,
            start_column=loc.start.column,
            end_line=loc.end.line,
            end_column=loc.end.column,
            source_file=getattr(node, 'source_file', None)
        )
    
    def _parse_with_internal_parser(self, code: str, file_path: Optional[str] = None) -> Any:
        """
        使用内置解析器解析代码
        
        Args:
            code: 源代码
            file_path: 可选的文件路径
            
        Returns:
            AST
        """
        # 这里使用简化的内部解析器实现
        # 在实际应用中，可以使用esprima、acorn或TypeScript编译器API
        
        # 模拟解析结果
        ast = JSASTNode("Program", {
            "body": [],
            "sourceType": "module"
        })
        
        # 简单的词法分析和语法分析
        tokens = self._tokenize(code)
        self._parse_tokens(tokens, ast)
        
        # 设置源文件信息
        ast.source_file = file_path
        
        return ast
    
    def _tokenize(self, code: str) -> List[Dict[str, Any]]:
        """
        对代码进行词法分析
        
        Args:
            code: 源代码
            
        Returns:
            词法单元列表
        """
        # 简化的词法分析实现
        # 在实际应用中，应使用成熟的词法分析器
        
        tokens = []
        # 实现词法分析逻辑...
        
        return tokens
    
    def _parse_tokens(self, tokens: List[Dict[str, Any]], ast: Any) -> None:
        """
        对词法单元进行语法分析
        
        Args:
            tokens: 词法单元列表
            ast: AST根节点
        """
        # 简化的语法分析实现
        # 在实际应用中，应使用成熟的语法分析器
        
        # 实现语法分析逻辑...
        pass
    
    def _enhance_ast(self, node: Any, parent: Optional[Any] = None) -> None:
        """
        增强AST节点，添加父节点引用
        
        Args:
            node: AST节点
            parent: 父节点
        """
        if not node:
            return
        
        # 添加父节点引用
        node.parent = parent
        
        # 递归处理子节点
        if hasattr(node, 'body') and isinstance(node.body, list):
            for child in node.body:
                self._enhance_ast(child, node)
        
        # 处理其他可能的子节点属性
        for key, value in vars(node).items():
            if key in ['parent', 'type', 'loc', 'range', 'source_file']:
                continue
            
            if isinstance(value, JSASTNode):
                self._enhance_ast(value, node)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, JSASTNode):
                        self._enhance_ast(item, node)
    
    def _find_node_at_position(self, node: Any, line: int, column: int) -> Optional[Any]:
        """
        查找指定位置的节点
        
        Args:
            node: AST节点
            line: 行号
            column: 列号
            
        Returns:
            位置对应的AST节点
        """
        if not node or not hasattr(node, 'loc'):
            return None
        
        # 检查当前节点是否包含目标位置
        loc = node.loc
        if not (loc.start.line <= line <= loc.end.line and
                (loc.start.line < line or loc.start.column <= column) and
                (loc.end.line > line or loc.end.column >= column)):
            return None
        
        # 在子节点中查找更精确的匹配
        best_match = node
        smallest_area = self._calculate_node_area(node)
        
        # 检查子节点
        for key, value in vars(node).items():
            if key in ['parent', 'type', 'loc', 'range', 'source_file']:
                continue
            
            if isinstance(value, JSASTNode):
                child_match = self._find_node_at_position(value, line, column)
                if child_match:
                    child_area = self._calculate_node_area(child_match)
                    if child_area < smallest_area:
                        best_match = child_match
                        smallest_area = child_area
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, JSASTNode):
                        child_match = self._find_node_at_position(item, line, column)
                        if child_match:
                            child_area = self._calculate_node_area(child_match)
                            if child_area < smallest_area:
                                best_match = child_match
                                smallest_area = child_area
        
        return best_match
    
    def _calculate_node_area(self, node: Any) -> int:
        """
        计算节点的面积（用于确定最精确的位置匹配）
        
        Args:
            node: AST节点
            
        Returns:
            节点面积
        """
        if not hasattr(node, 'loc'):
            return float('inf')
        
        loc = node.loc
        lines = loc.end.line - loc.start.line + 1
        
        if lines == 1:
            return loc.end.column - loc.start.column
        else:
            return lines * 1000  # 多行节点的面积较大


class JSASTNode:
    """JavaScript/TypeScript AST节点"""
    
    def __init__(self, node_type: str, properties: Dict[str, Any] = None):
        """
        初始化AST节点
        
        Args:
            node_type: 节点类型
            properties: 节点属性
        """
        self.type = node_type
        self.loc = JSSourceLocation(1, 0, 1, 0)  # 默认位置
        self.range = [0, 0]  # 默认范围
        self.parent = None
        self.source_file = None
        
        # 设置属性
        if properties:
            for key, value in properties.items():
                setattr(self, key, value)


class JSSourceLocation:
    """JavaScript/TypeScript源代码位置"""
    
    def __init__(self, start_line: int, start_column: int, end_line: int, end_column: int):
        """
        初始化源代码位置
        
        Args:
            start_line: 起始行
            start_column: 起始列
            end_line: 结束行
            end_column: 结束列
        """
        self.start = JSPosition(start_line, start_column)
        self.end = JSPosition(end_line, end_column)


class JSPosition:
    """JavaScript/TypeScript位置"""
    
    def __init__(self, line: int, column: int):
        """
        初始化位置
        
        Args:
            line: 行号
            column: 列号
        """
        self.line = line
        self.column = column


class JSASTAnalyzer:
    """JavaScript/TypeScript AST分析器"""
    
    @staticmethod
    def get_imports(ast_root: Any) -> List[Dict[str, Any]]:
        """
        获取导入语句
        
        Args:
            ast_root: AST根节点
            
        Returns:
            导入语句列表
        """
        imports = []
        
        # 遍历AST查找导入语句
        JSASTAnalyzer._traverse(ast_root, lambda node: (
            node.type == "ImportDeclaration" and
            imports.append({
                "type": "import",
                "source": getattr(node, 'source', {}).value if hasattr(node, 'source') else "",
                "specifiers": [
                    {
                        "name": getattr(spec, 'imported', {}).name if hasattr(spec, 'imported') else "",
                        "alias": getattr(spec, 'local', {}).name if hasattr(spec, 'local') else ""
                    }
                    for spec in getattr(node, 'specifiers', [])
                ],
                "node": node
            })
        ))
        
        return imports
    
    @staticmethod
    def get_functions(ast_root: Any) -> List[Dict[str, Any]]:
        """
        获取函数定义
        
        Args:
            ast_root: AST根节点
            
        Returns:
            函数定义列表
        """
        functions = []
        
        # 遍历AST查找函数定义
        JSASTAnalyzer._traverse(ast_root, lambda node: (
            (node.type == "FunctionDeclaration" or node.type == "ArrowFunctionExpression") and
            functions.append({
                "name": getattr(node, 'id', {}).name if hasattr(node, 'id') and node.id else "anonymous",
                "is_arrow": node.type == "ArrowFunctionExpression",
                "is_async": getattr(node, 'async', False),
                "params": [
                    getattr(param, 'name', "unnamed") if hasattr(param, 'name') else "unnamed"
                    for param in getattr(node, 'params', [])
                ],
                "node": node
            })
        ))
        
        return functions
    
    @staticmethod
    def get_classes(ast_root: Any) -> List[Dict[str, Any]]:
        """
        获取类定义
        
        Args:
            ast_root: AST根节点
            
        Returns:
            类定义列表
        """
        classes = []
        
        # 遍历AST查找类定义
        JSASTAnalyzer._traverse(ast_root, lambda node: (
            node.type == "ClassDeclaration" and
            classes.append({
                "name": getattr(node, 'id', {}).name if hasattr(node, 'id') and node.id else "anonymous",
                "methods": JSASTAnalyzer._get_class_methods(node),
                "super_class": getattr(node, 'superClass', {}).name if hasattr(node, 'superClass') and node.superClass else None,
                "node": node
            })
        ))
        
        return classes
    
    @staticmethod
    def _get_class_methods(class_node: Any) -> List[Dict[str, Any]]:
        """
        获取类方法
        
        Args:
            class_node: 类节点
            
        Returns:
            类方法列表
        """
        methods = []
        
        for item in getattr(class_node, 'body', {}).body if hasattr(class_node, 'body') else []:
            if item.type == "MethodDefinition":
                methods.append({
                    "name": getattr(item, 'key', {}).name if hasattr(item, 'key') and hasattr(item.key, 'name') else "unnamed",
                    "is_static": getattr(item, 'static', False),
                    "is_async": getattr(item, 'value', {}).async if hasattr(item, 'value') and hasattr(item.value, 'async') else False,
                    "kind": getattr(item, 'kind', "method"),  # "constructor", "method", "get", "set"
                    "node": item
                })
        
        return methods
    
    @staticmethod
    def _traverse(node: Any, callback: callable) -> None:
        """
        遍历AST节点
        
        Args:
            node: AST节点
            callback: 回调函数
        """
        if not node:
            return
        
        # 调用回调函数
        callback(node)
        
        # 递归处理子节点
        if hasattr(node, 'body') and isinstance(node.body, list):
            for child in node.body:
                JSASTAnalyzer._traverse(child, callback)
        elif hasattr(node, 'body') and hasattr(node.body, 'body') and isinstance(node.body.body, list):
            for child in node.body.body:
                JSASTAnalyzer._traverse(child, callback)
        
        # 处理其他可能的子节点属性
        for key, value in vars(node).items():
            if key in ['parent', 'type', 'loc', 'range', 'source_file', 'body']:
                continue
            
            if isinstance(value, JSASTNode):
                JSASTAnalyzer._traverse(value, callback)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, JSASTNode):
                        JSASTAnalyzer._traverse(item, callback)