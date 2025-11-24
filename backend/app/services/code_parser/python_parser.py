"""
Python解析器

提供Python代码的解析功能，支持Python 3.8+语法特性
"""
import ast
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import traceback

from .language_parser import LanguageParser, SourceRange, ParseResult


class PythonParser(LanguageParser):
    """Python解析器实现"""
    
    def __init__(self):
        super().__init__("python")
    
    def parse_file(self, file_path: str) -> ParseResult:
        """
        解析Python文件
        
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
        解析Python代码
        
        Args:
            code: 源代码
            file_path: 可选的文件路径
            
        Returns:
            解析结果
        """
        try:
            # 使用Python内置的ast模块解析代码
            tree = ast.parse(code, filename=file_path or "<unknown>")
            
            # 增强AST节点，添加父节点引用
            self._enhance_ast(tree)
            
            return ParseResult(
                success=True,
                ast=tree,
                errors=[]
            )
        except SyntaxError as e:
            # 处理语法错误
            error_message = f"语法错误: {str(e)}"
            error_location = self._create_source_range_from_syntax_error(e)
            
            return ParseResult(
                success=False,
                ast=None,
                errors=[{
                    "message": error_message,
                    "location": error_location
                }]
            )
        except Exception as e:
            # 处理其他错误
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
        增量解析Python代码
        
        Args:
            code: 新的源代码
            previous_ast: 之前的AST
            changed_range: 变更范围
            
        Returns:
            解析结果
        """
        # Python的ast模块不直接支持增量解析
        # 这里简单地重新解析整个代码
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
        visitor = NodeAtPositionVisitor(line, column)
        visitor.visit(ast_root)
        return visitor.result
    
    def get_node_range(self, node: Any) -> Optional[SourceRange]:
        """
        获取节点的源代码范围
        
        Args:
            node: AST节点
            
        Returns:
            节点的源代码范围
        """
        if not hasattr(node, 'lineno'):
            return None
        
        start_line = getattr(node, 'lineno', 0)
        start_column = getattr(node, 'col_offset', 0)
        
        end_line = getattr(node, 'end_lineno', start_line)
        end_column = getattr(node, 'end_col_offset', 0)
        
        return SourceRange(
            start_line=start_line,
            start_column=start_column,
            end_line=end_line,
            end_column=end_column,
            source_file=getattr(node, 'source_file', None)
        )
    
    def _enhance_ast(self, node: Any, parent: Optional[Any] = None) -> None:
        """
        增强AST节点，添加父节点引用
        
        Args:
            node: AST节点
            parent: 父节点
        """
        # 添加父节点引用
        node.parent = parent
        
        # 递归处理子节点
        for child in ast.iter_child_nodes(node):
            self._enhance_ast(child, node)
    
    def _create_source_range_from_syntax_error(self, error: SyntaxError) -> Optional[SourceRange]:
        """
        从语法错误创建源代码范围
        
        Args:
            error: 语法错误
            
        Returns:
            源代码范围
        """
        if not hasattr(error, 'lineno') or not hasattr(error, 'offset'):
            return None
        
        return SourceRange(
            start_line=error.lineno,
            start_column=error.offset - 1 if error.offset else 0,
            end_line=error.lineno,
            end_column=error.offset if error.offset else 0,
            source_file=error.filename
        )


class NodeAtPositionVisitor(ast.NodeVisitor):
    """访问指定位置的节点的访问者"""
    
    def __init__(self, line: int, column: int):
        self.line = line
        self.column = column
        self.result = None
        self.best_match_size = float('inf')  # 最佳匹配的节点范围大小
    
    def generic_visit(self, node: Any) -> None:
        """访问节点"""
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            start_line = node.lineno
            start_column = node.col_offset
            
            end_line = getattr(node, 'end_lineno', start_line)
            end_column = getattr(node, 'end_col_offset', start_column + 1)
            
            # 检查位置是否在节点范围内
            if (start_line <= self.line <= end_line and
                (start_line < self.line or start_column <= self.column) and
                (end_line > self.line or end_column >= self.column)):
                
                # 计算节点范围大小
                size = (end_line - start_line) * 1000 + (end_column - start_column)
                
                # 更新最佳匹配
                if size < self.best_match_size:
                    self.result = node
                    self.best_match_size = size
        
        # 继续访问子节点
        super().generic_visit(node)


class PythonASTAnalyzer:
    """Python AST分析器"""
    
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
        
        for node in ast.walk(ast_root):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        "type": "import",
                        "name": name.name,
                        "alias": name.asname,
                        "node": node
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                for name in node.names:
                    imports.append({
                        "type": "import_from",
                        "module": module,
                        "name": name.name,
                        "alias": name.asname,
                        "level": node.level,
                        "node": node
                    })
        
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
        
        for node in ast.walk(ast_root):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_method = False
                
                # 检查是否是类方法
                if hasattr(node, 'parent') and isinstance(node.parent, ast.ClassDef):
                    is_method = True
                
                functions.append({
                    "name": node.name,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "is_method": is_method,
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [PythonASTAnalyzer._get_decorator_name(d) for d in node.decorator_list],
                    "node": node
                })
        
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
        
        for node in ast.walk(ast_root):
            if isinstance(node, ast.ClassDef):
                methods = []
                
                # 获取类方法
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append({
                            "name": item.name,
                            "is_async": isinstance(item, ast.AsyncFunctionDef),
                            "args": [arg.arg for arg in item.args.args],
                            "decorators": [PythonASTAnalyzer._get_decorator_name(d) for d in item.decorator_list],
                            "node": item
                        })
                
                classes.append({
                    "name": node.name,
                    "bases": [PythonASTAnalyzer._get_name(base) for base in node.bases],
                    "decorators": [PythonASTAnalyzer._get_decorator_name(d) for d in node.decorator_list],
                    "methods": methods,
                    "node": node
                })
        
        return classes
    
    @staticmethod
    def _get_decorator_name(node: Any) -> str:
        """
        获取装饰器名称
        
        Args:
            node: 装饰器节点
            
        Returns:
            装饰器名称
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{PythonASTAnalyzer._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return PythonASTAnalyzer._get_name(node.func)
        return "unknown"
    
    @staticmethod
    def _get_name(node: Any) -> str:
        """
        获取节点名称
        
        Args:
            node: AST节点
            
        Returns:
            节点名称
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{PythonASTAnalyzer._get_name(node.value)}.{node.attr}"
        return "unknown"