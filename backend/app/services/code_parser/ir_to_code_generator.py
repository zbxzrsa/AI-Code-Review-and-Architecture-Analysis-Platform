"""
IR到目标代码生成器

提供从统一IR到特定语言代码的生成功能
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, TextIO, Union
import io

from .ir_model import (
    IRNode, IRVisitor, Module, Function, Class, Variable, Parameter,
    Block, ExpressionStatement, If, Return,
    Literal, Identifier, BinaryOperation, Call,
    TypeAnnotation, Import, IRNodeType
)


class CodeFormatter:
    """代码格式化器"""
    
    def __init__(self, indent_size: int = 4, use_tabs: bool = False):
        """
        初始化格式化器
        
        Args:
            indent_size: 缩进大小
            use_tabs: 是否使用制表符缩进
        """
        self.indent_size = indent_size
        self.use_tabs = use_tabs
        self.indent_char = '\t' if use_tabs else ' '
        self.current_indent = 0
        self.output = io.StringIO()
    
    def indent(self) -> None:
        """增加缩进级别"""
        self.current_indent += 1
    
    def dedent(self) -> None:
        """减少缩进级别"""
        if self.current_indent > 0:
            self.current_indent -= 1
    
    def write(self, text: str) -> None:
        """写入文本"""
        self.output.write(text)
    
    def write_line(self, line: str = "") -> None:
        """写入一行，自动添加缩进"""
        if line:
            self.output.write(self.indent_char * (self.current_indent * self.indent_size))
            self.output.write(line)
        self.output.write('\n')
    
    def get_result(self) -> str:
        """获取格式化后的代码"""
        return self.output.getvalue()


class IRToCodeGenerator(IRVisitor, ABC):
    """IR到代码生成器基类"""
    
    def __init__(self, language_name: str, indent_size: int = 4, use_tabs: bool = False):
        """
        初始化代码生成器
        
        Args:
            language_name: 目标语言名称
            indent_size: 缩进大小
            use_tabs: 是否使用制表符缩进
        """
        self.language_name = language_name
        self.formatter = CodeFormatter(indent_size, use_tabs)
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
    
    def generate(self, ir_node: IRNode) -> str:
        """
        从IR节点生成代码
        
        Args:
            ir_node: IR节点
            
        Returns:
            生成的代码
        """
        self.errors = []
        self.warnings = []
        self.formatter = CodeFormatter(self.formatter.indent_size, self.formatter.use_tabs)
        
        try:
            self.visit(ir_node)
            return self.formatter.get_result()
        except Exception as e:
            self.errors.append({
                "message": f"代码生成错误: {str(e)}",
                "node": ir_node
            })
            return f"// 代码生成错误: {str(e)}"
    
    def add_error(self, message: str, ir_node: Optional[IRNode] = None) -> None:
        """
        添加代码生成错误
        
        Args:
            message: 错误信息
            ir_node: 相关的IR节点
        """
        self.errors.append({
            "message": message,
            "node": ir_node
        })
    
    def add_warning(self, message: str, ir_node: Optional[IRNode] = None) -> None:
        """
        添加代码生成警告
        
        Args:
            message: 警告信息
            ir_node: 相关的IR节点
        """
        self.warnings.append({
            "message": message,
            "node": ir_node
        })
    
    def has_errors(self) -> bool:
        """
        检查是否有代码生成错误
        
        Returns:
            是否有错误
        """
        return len(self.errors) > 0


class PythonCodeGenerator(IRToCodeGenerator):
    """Python代码生成器"""
    
    def __init__(self, indent_size: int = 4, use_tabs: bool = False):
        super().__init__("python", indent_size, use_tabs)
    
    def visit_module(self, node: Module) -> Any:
        """访问模块节点"""
        # 生成导入语句
        for imp in node.imports:
            self.visit(imp)
        
        # 在导入和声明之间添加空行
        if node.imports and node.declarations:
            self.formatter.write_line()
        
        # 生成声明
        for i, decl in enumerate(node.declarations):
            if i > 0:
                # 在声明之间添加空行
                self.formatter.write_line()
            self.visit(decl)
    
    def visit_function(self, node: Function) -> Any:
        """访问函数节点"""
        # 生成装饰器
        for decorator in node.decorators:
            self.formatter.write_line(f"@{self._generate_expression(decorator)}")
        
        # 函数定义
        async_prefix = "async " if node.is_async else ""
        self.formatter.write(f"{async_prefix}def {node.name}(")
        
        # 参数
        params = []
        for param in node.parameters:
            param_str = param.name
            
            # 类型注解
            if param.type_annotation:
                param_str += f": {self._generate_type_annotation(param.type_annotation)}"
            
            # 默认值
            if param.default_value:
                param_str += f" = {self._generate_expression(param.default_value)}"
            
            # 可变参数
            if param.is_rest:
                if param.get_metadata("is_kwargs", False):
                    param_str = f"**{param.name}"
                else:
                    param_str = f"*{param.name}"
            
            params.append(param_str)
        
        self.formatter.write(", ".join(params))
        
        # 返回类型
        if node.return_type:
            self.formatter.write(f") -> {self._generate_type_annotation(node.return_type)}:")
        else:
            self.formatter.write("):")
        
        # 函数体
        if node.body and node.body.statements:
            self.formatter.write_line()
            self.formatter.indent()
            self.visit(node.body)
            self.formatter.dedent()
        else:
            # 空函数体
            self.formatter.write_line()
            self.formatter.indent()
            self.formatter.write_line("pass")
            self.formatter.dedent()
    
    def visit_class(self, node: Class) -> Any:
        """访问类节点"""
        # 生成装饰器
        for decorator in node.decorators:
            self.formatter.write_line(f"@{self._generate_expression(decorator)}")
        
        # 类定义
        bases = []
        for base in node.base_classes:
            bases.append(self._generate_type_annotation(base))
        
        base_str = f"({', '.join(bases)})" if bases else ""
        self.formatter.write_line(f"class {node.name}{base_str}:")
        
        self.formatter.indent()
        
        # 类体
        if node.fields or node.methods:
            # 字段
            for field in node.fields:
                self.visit(field)
            
            # 在字段和方法之间添加空行
            if node.fields and node.methods:
                self.formatter.write_line()
            
            # 方法
            for i, method in enumerate(node.methods):
                if i > 0:
                    # 在方法之间添加空行
                    self.formatter.write_line()
                self.visit(method)
        else:
            # 空类体
            self.formatter.write_line("pass")
        
        self.formatter.dedent()
    
    def visit_variable(self, node: Variable) -> Any:
        """访问变量节点"""
        var_str = node.name
        
        # 类型注解
        if node.type_annotation:
            var_str += f": {self._generate_type_annotation(node.type_annotation)}"
        
        # 初始化器
        if node.initializer:
            var_str += f" = {self._generate_expression(node.initializer)}"
        
        self.formatter.write_line(var_str)
    
    def visit_block(self, node: Block) -> Any:
        """访问代码块节点"""
        if not node.statements:
            self.formatter.write_line("pass")
            return
        
        for stmt in node.statements:
            self.visit(stmt)
    
    def visit_expression_stmt(self, node: ExpressionStatement) -> Any:
        """访问表达式语句节点"""
        expr = self._generate_expression(node.expression)
        self.formatter.write_line(expr)
    
    def visit_if(self, node: If) -> Any:
        """访问if语句节点"""
        condition = self._generate_expression(node.condition)
        self.formatter.write_line(f"if {condition}:")
        
        # then分支
        self.formatter.indent()
        self.visit(node.then_branch)
        self.formatter.dedent()
        
        # else分支
        if node.else_branch:
            self.formatter.write_line("else:")
            self.formatter.indent()
            self.visit(node.else_branch)
            self.formatter.dedent()
    
    def visit_return(self, node: Return) -> Any:
        """访问return语句节点"""
        if node.expression:
            expr = self._generate_expression(node.expression)
            self.formatter.write_line(f"return {expr}")
        else:
            self.formatter.write_line("return")
    
    def visit_literal(self, node: Literal) -> Any:
        """访问字面量节点"""
        if node.literal_type == "string":
            # 字符串字面量
            return f'"{node.value}"'
        elif node.literal_type == "null":
            # None
            return "None"
        elif node.literal_type == "boolean":
            # 布尔值
            return "True" if node.value else "False"
        else:
            # 其他字面量
            return str(node.value)
    
    def visit_identifier(self, node: Identifier) -> Any:
        """访问标识符节点"""
        return node.name
    
    def visit_binary_op(self, node: BinaryOperation) -> Any:
        """访问二元操作节点"""
        left = self._generate_expression(node.left)
        right = self._generate_expression(node.right)
        return f"{left} {node.operator} {right}"
    
    def visit_call(self, node: Call) -> Any:
        """访问函数调用节点"""
        callee = self._generate_expression(node.callee)
        args = [self._generate_expression(arg) for arg in node.arguments]
        return f"{callee}({', '.join(args)})"
    
    def _generate_expression(self, node: IRNode) -> str:
        """生成表达式代码"""
        result = self.visit(node)
        if isinstance(result, str):
            return result
        return "..."  # 默认占位符
    
    def _generate_type_annotation(self, node: TypeAnnotation) -> str:
        """生成类型注解代码"""
        # 从元数据中获取表达式
        expr = node.get_metadata("expression")
        if expr:
            return self._generate_expression(expr)
        return "Any"  # 默认类型


class JavaScriptCodeGenerator(IRToCodeGenerator):
    """JavaScript代码生成器"""
    
    def __init__(self, indent_size: int = 2, use_tabs: bool = False):
        super().__init__("javascript", indent_size, use_tabs)
    
    def visit_module(self, node: Module) -> Any:
        """访问模块节点"""
        # 生成导入语句
        for imp in node.imports:
            self.visit(imp)
        
        # 在导入和声明之间添加空行
        if node.imports and node.declarations:
            self.formatter.write_line()
        
        # 生成声明
        for i, decl in enumerate(node.declarations):
            if i > 0:
                # 在声明之间添加空行
                self.formatter.write_line()
            self.visit(decl)
    
    def visit_function(self, node: Function) -> Any:
        """访问函数节点"""
        # 函数定义
        async_prefix = "async " if node.is_async else ""
        
        # 检查是否是箭头函数
        is_arrow = node.get_metadata("is_arrow", False)
        
        if is_arrow:
            # 箭头函数
            params = [param.name for param in node.parameters]
            param_str = ", ".join(params)
            
            if len(params) != 1:
                param_str = f"({param_str})"
            
            self.formatter.write(f"const {node.name} = {async_prefix}{param_str} => ")
            
            # 单行箭头函数
            if node.body and len(node.body.statements) == 1 and node.body.statements[0].node_type == IRNodeType.RETURN:
                return_stmt = node.body.statements[0]
                if return_stmt.expression:
                    self.formatter.write(f"{self._generate_expression(return_stmt.expression)};")
                    return
            
            # 多行箭头函数
            self.formatter.write_line("{")
            self.formatter.indent()
            if node.body:
                self.visit(node.body)
            self.formatter.dedent()
            self.formatter.write_line("};")
        else:
            # 普通函数
            self.formatter.write(f"{async_prefix}function {node.name}(")
            
            # 参数
            params = []
            for param in node.parameters:
                param_str = param.name
                
                # 默认值
                if param.default_value:
                    param_str += f" = {self._generate_expression(param.default_value)}"
                
                # 可变参数
                if param.is_rest:
                    param_str = f"...{param.name}"
                
                params.append(param_str)
            
            self.formatter.write(", ".join(params))
            self.formatter.write_line(") {")
            
            # 函数体
            self.formatter.indent()
            if node.body:
                self.visit(node.body)
            self.formatter.dedent()
            self.formatter.write_line("}")
    
    def visit_class(self, node: Class) -> Any:
        """访问类节点"""
        # 类定义
        extends_clause = ""
        if node.base_classes:
            base = self._generate_type_annotation(node.base_classes[0])
            extends_clause = f" extends {base}"
        
        self.formatter.write_line(f"class {node.name}{extends_clause} {{")
        self.formatter.indent()
        
        # 类体
        # 字段
        for field in node.fields:
            self.visit(field)
        
        # 在字段和方法之间添加空行
        if node.fields and node.methods:
            self.formatter.write_line()
        
        # 方法
        for method in node.methods:
            self.visit(method)
        
        self.formatter.dedent()
        self.formatter.write_line("}")
    
    def visit_variable(self, node: Variable) -> Any:
        """访问变量节点"""
        # 变量声明关键字
        keyword = "const" if node.is_const else "let"
        
        var_str = f"{keyword} {node.name}"
        
        # 初始化器
        if node.initializer:
            var_str += f" = {self._generate_expression(node.initializer)}"
        
        self.formatter.write_line(f"{var_str};")
    
    def visit_block(self, node: Block) -> Any:
        """访问代码块节点"""
        for stmt in node.statements:
            self.visit(stmt)
    
    def visit_expression_stmt(self, node: ExpressionStatement) -> Any:
        """访问表达式语句节点"""
        expr = self._generate_expression(node.expression)
        self.formatter.write_line(f"{expr};")
    
    def visit_if(self, node: If) -> Any:
        """访问if语句节点"""
        condition = self._generate_expression(node.condition)
        self.formatter.write_line(f"if ({condition}) {{")
        
        # then分支
        self.formatter.indent()
        self.visit(node.then_branch)
        self.formatter.dedent()
        
        # else分支
        if node.else_branch:
            self.formatter.write_line("} else {")
            self.formatter.indent()
            self.visit(node.else_branch)
            self.formatter.dedent()
        
        self.formatter.write_line("}")
    
    def visit_return(self, node: Return) -> Any:
        """访问return语句节点"""
        if node.expression:
            expr = self._generate_expression(node.expression)
            self.formatter.write_line(f"return {expr};")
        else:
            self.formatter.write_line("return;")
    
    def visit_literal(self, node: Literal) -> Any:
        """访问字面量节点"""
        if node.literal_type == "string":
            # 字符串字面量
            return f'"{node.value}"'
        elif node.literal_type == "null":
            # null
            return "null"
        elif node.literal_type == "undefined":
            # undefined
            return "undefined"
        else:
            # 其他字面量
            return str(node.value)
    
    def visit_identifier(self, node: Identifier) -> Any:
        """访问标识符节点"""
        return node.name
    
    def visit_binary_op(self, node: BinaryOperation) -> Any:
        """访问二元操作节点"""
        left = self._generate_expression(node.left)
        right = self._generate_expression(node.right)
        return f"{left} {node.operator} {right}"
    
    def visit_call(self, node: Call) -> Any:
        """访问函数调用节点"""
        callee = self._generate_expression(node.callee)
        args = [self._generate_expression(arg) for arg in node.arguments]
        return f"{callee}({', '.join(args)})"
    
    def _generate_expression(self, node: IRNode) -> str:
        """生成表达式代码"""
        result = self.visit(node)
        if isinstance(result, str):
            return result
        return "..."  # 默认占位符
    
    def _generate_type_annotation(self, node: TypeAnnotation) -> str:
        """生成类型注解代码"""
        # JavaScript不支持类型注解，除非是TypeScript
        # 从元数据中获取表达式
        expr = node.get_metadata("expression")
        if expr:
            return self._generate_expression(expr)
        return "any"  # 默认类型