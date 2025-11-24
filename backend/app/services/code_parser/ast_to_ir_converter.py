"""
AST到IR转换器

提供从语言特定AST到统一IR的转换功能
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

from .ir_model import (
    IRNode, Module, Function, Class, Variable, Parameter,
    Block, ExpressionStatement, If, Return,
    Literal, Identifier, BinaryOperation, Call,
    TypeAnnotation, Import
)
from .language_parser import SourceRange, LanguageParser


T = TypeVar('T')  # 语言特定AST节点类型


class ASTToIRConverter(Generic[T], ABC):
    """AST到IR转换器基类"""
    
    def __init__(self, language_name: str):
        """
        初始化转换器
        
        Args:
            language_name: 语言名称
        """
        self.language_name = language_name
        self.source_file: Optional[str] = None
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        
    def convert(self, ast_node: T, source_file: Optional[str] = None) -> IRNode:
        """
        将AST节点转换为IR节点
        
        Args:
            ast_node: 语言特定的AST节点
            source_file: 源文件路径
            
        Returns:
            转换后的IR节点
        """
        self.source_file = source_file
        self.errors = []
        self.warnings = []
        
        try:
            return self._convert_node(ast_node)
        except Exception as e:
            self.errors.append({
                "message": f"转换错误: {str(e)}",
                "source_file": source_file,
                "node": ast_node
            })
            # 返回一个空模块作为错误恢复
            return Module(name="error_module", metadata={"error": str(e)})
    
    @abstractmethod
    def _convert_node(self, ast_node: T) -> IRNode:
        """
        转换单个AST节点到IR节点
        
        Args:
            ast_node: 语言特定的AST节点
            
        Returns:
            转换后的IR节点
        """
        pass
    
    def _create_source_range(self, ast_node: T) -> Optional[SourceRange]:
        """
        从AST节点创建源代码位置信息
        
        Args:
            ast_node: 语言特定的AST节点
            
        Returns:
            源代码位置信息
        """
        # 子类需要实现具体逻辑
        return None
    
    def add_error(self, message: str, ast_node: Optional[T] = None) -> None:
        """
        添加转换错误
        
        Args:
            message: 错误信息
            ast_node: 相关的AST节点
        """
        self.errors.append({
            "message": message,
            "source_file": self.source_file,
            "node": ast_node
        })
    
    def add_warning(self, message: str, ast_node: Optional[T] = None) -> None:
        """
        添加转换警告
        
        Args:
            message: 警告信息
            ast_node: 相关的AST节点
        """
        self.warnings.append({
            "message": message,
            "source_file": self.source_file,
            "node": ast_node
        })
    
    def has_errors(self) -> bool:
        """
        检查是否有转换错误
        
        Returns:
            是否有错误
        """
        return len(self.errors) > 0


class PythonASTToIRConverter(ASTToIRConverter):
    """Python AST到IR转换器"""
    
    def __init__(self):
        super().__init__("python")
        
    def _convert_node(self, ast_node: Any) -> IRNode:
        """
        转换Python AST节点到IR节点
        
        Args:
            ast_node: Python AST节点
            
        Returns:
            转换后的IR节点
        """
        import ast
        
        # 根据节点类型分发到具体的转换方法
        if isinstance(ast_node, ast.Module):
            return self._convert_module(ast_node)
        elif isinstance(ast_node, ast.FunctionDef):
            return self._convert_function_def(ast_node)
        elif isinstance(ast_node, ast.AsyncFunctionDef):
            return self._convert_async_function_def(ast_node)
        elif isinstance(ast_node, ast.ClassDef):
            return self._convert_class_def(ast_node)
        elif isinstance(ast_node, ast.Assign):
            return self._convert_assign(ast_node)
        elif isinstance(ast_node, ast.AnnAssign):
            return self._convert_ann_assign(ast_node)
        elif isinstance(ast_node, ast.If):
            return self._convert_if(ast_node)
        elif isinstance(ast_node, ast.Return):
            return self._convert_return(ast_node)
        elif isinstance(ast_node, ast.Expr):
            return self._convert_expr(ast_node)
        elif isinstance(ast_node, ast.Call):
            return self._convert_call(ast_node)
        elif isinstance(ast_node, ast.Name):
            return self._convert_name(ast_node)
        elif isinstance(ast_node, ast.Constant):
            return self._convert_constant(ast_node)
        elif isinstance(ast_node, ast.BinOp):
            return self._convert_bin_op(ast_node)
        elif isinstance(ast_node, ast.Import):
            return self._convert_import(ast_node)
        elif isinstance(ast_node, ast.ImportFrom):
            return self._convert_import_from(ast_node)
        else:
            self.add_warning(f"未实现的Python AST节点类型: {type(ast_node).__name__}", ast_node)
            # 返回一个标识符作为占位符
            return Identifier(
                name=f"Unsupported_{type(ast_node).__name__}",
                source_range=self._create_source_range(ast_node),
                metadata={"original_type": type(ast_node).__name__}
            )
    
    def _convert_module(self, node: Any) -> Module:
        """转换Python模块"""
        import ast
        
        declarations = []
        imports = []
        
        for item in node.body:
            ir_node = self._convert_node(item)
            
            if ir_node.node_type.name == "IMPORT":
                imports.append(ir_node)
            else:
                declarations.append(ir_node)
        
        return Module(
            name=self.source_file or "unknown_module",
            declarations=declarations,
            imports=imports,
            source_range=self._create_source_range(node)
        )
    
    def _convert_function_def(self, node: Any) -> Function:
        """转换Python函数定义"""
        import ast
        
        parameters = []
        for arg in node.args.args:
            param = Parameter(
                name=arg.arg,
                type_annotation=self._convert_annotation(arg.annotation) if hasattr(arg, 'annotation') and arg.annotation else None,
                source_range=self._create_source_range(arg)
            )
            parameters.append(param)
        
        # 处理默认参数
        defaults_offset = len(node.args.args) - len(node.args.defaults)
        for i, default in enumerate(node.args.defaults):
            arg_index = defaults_offset + i
            if arg_index < len(parameters):
                parameters[arg_index].default_value = self._convert_node(default)
        
        # 处理可变参数
        if node.args.vararg:
            param = Parameter(
                name=node.args.vararg.arg,
                type_annotation=self._convert_annotation(node.args.vararg.annotation) if hasattr(node.args.vararg, 'annotation') and node.args.vararg.annotation else None,
                is_rest=True,
                source_range=self._create_source_range(node.args.vararg)
            )
            parameters.append(param)
        
        # 处理关键字参数
        if node.args.kwarg:
            param = Parameter(
                name=node.args.kwarg.arg,
                type_annotation=self._convert_annotation(node.args.kwarg.annotation) if hasattr(node.args.kwarg, 'annotation') and node.args.kwarg.annotation else None,
                is_rest=True,
                source_range=self._create_source_range(node.args.kwarg),
                metadata={"is_kwargs": True}
            )
            parameters.append(param)
        
        # 处理函数体
        body_statements = []
        for stmt in node.body:
            ir_stmt = self._convert_node(stmt)
            if isinstance(ir_stmt, list):
                body_statements.extend(ir_stmt)
            else:
                body_statements.append(ir_stmt)
        
        body = Block(
            statements=body_statements,
            source_range=self._create_source_range(node.body)
        )
        
        # 处理装饰器
        decorators = []
        for decorator in node.decorator_list:
            decorators.append(self._convert_node(decorator))
        
        return Function(
            name=node.name,
            parameters=parameters,
            return_type=self._convert_annotation(node.returns) if hasattr(node, 'returns') and node.returns else None,
            body=body,
            is_async=False,
            is_generator=self._is_generator(node),
            decorators=decorators,
            source_range=self._create_source_range(node)
        )
    
    def _convert_async_function_def(self, node: Any) -> Function:
        """转换Python异步函数定义"""
        func = self._convert_function_def(node)
        func.is_async = True
        return func
    
    def _convert_class_def(self, node: Any) -> Class:
        """转换Python类定义"""
        import ast
        
        methods = []
        fields = []
        
        for item in node.body:
            ir_node = self._convert_node(item)
            
            if ir_node.node_type.name == "FUNCTION":
                # 将函数转换为方法
                ir_node.node_type = "METHOD"
                methods.append(ir_node)
            elif ir_node.node_type.name == "VARIABLE":
                fields.append(ir_node)
            else:
                # 其他类成员（如嵌套类）作为字段处理
                fields.append(ir_node)
        
        # 处理基类
        base_classes = []
        for base in node.bases:
            base_type = TypeAnnotation(
                node_type="TYPE_ANNOTATION",
                source_range=self._create_source_range(base)
            )
            base_type.metadata["expression"] = self._convert_node(base)
            base_classes.append(base_type)
        
        # 处理装饰器
        decorators = []
        for decorator in node.decorator_list:
            decorators.append(self._convert_node(decorator))
        
        return Class(
            name=node.name,
            methods=methods,
            fields=fields,
            base_classes=base_classes,
            decorators=decorators,
            source_range=self._create_source_range(node)
        )
    
    def _convert_annotation(self, node: Any) -> Optional[TypeAnnotation]:
        """转换类型注解"""
        if node is None:
            return None
        
        type_annotation = TypeAnnotation(
            node_type="TYPE_ANNOTATION",
            source_range=self._create_source_range(node)
        )
        
        # 存储原始表达式
        type_annotation.metadata["expression"] = self._convert_node(node)
        
        return type_annotation
    
    def _convert_assign(self, node: Any) -> List[Variable]:
        """转换赋值语句"""
        import ast
        
        variables = []
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                var = Variable(
                    name=target.id,
                    initializer=self._convert_node(node.value),
                    source_range=self._create_source_range(node)
                )
                variables.append(var)
            else:
                # 复杂赋值（如元组解包）转换为表达式语句
                expr = ExpressionStatement(
                    expression=BinaryOperation(
                        operator="=",
                        left=self._convert_node(target),
                        right=self._convert_node(node.value),
                        source_range=self._create_source_range(node)
                    ),
                    source_range=self._create_source_range(node)
                )
                variables.append(expr)
        
        return variables
    
    def _convert_ann_assign(self, node: Any) -> Variable:
        """转换带注解的赋值语句"""
        import ast
        
        if isinstance(node.target, ast.Name):
            return Variable(
                name=node.target.id,
                type_annotation=self._convert_annotation(node.annotation),
                initializer=self._convert_node(node.value) if node.value else None,
                source_range=self._create_source_range(node)
            )
        else:
            # 复杂目标的带注解赋值
            self.add_warning("不支持的复杂目标带注解赋值", node)
            return Variable(
                name=f"complex_annotated_target",
                source_range=self._create_source_range(node),
                metadata={"unsupported": True}
            )
    
    def _convert_if(self, node: Any) -> If:
        """转换if语句"""
        import ast
        
        # 处理then分支
        then_statements = []
        for stmt in node.body:
            ir_stmt = self._convert_node(stmt)
            if isinstance(ir_stmt, list):
                then_statements.extend(ir_stmt)
            else:
                then_statements.append(ir_stmt)
        
        then_branch = Block(
            statements=then_statements,
            source_range=self._create_source_range(node.body)
        )
        
        # 处理else分支
        else_branch = None
        if node.orelse:
            else_statements = []
            for stmt in node.orelse:
                ir_stmt = self._convert_node(stmt)
                if isinstance(ir_stmt, list):
                    else_statements.extend(ir_stmt)
                else:
                    else_statements.append(ir_stmt)
            
            else_branch = Block(
                statements=else_statements,
                source_range=self._create_source_range(node.orelse)
            )
        
        return If(
            condition=self._convert_node(node.test),
            then_branch=then_branch,
            else_branch=else_branch,
            source_range=self._create_source_range(node)
        )
    
    def _convert_return(self, node: Any) -> Return:
        """转换return语句"""
        import ast
        
        return Return(
            expression=self._convert_node(node.value) if node.value else None,
            source_range=self._create_source_range(node)
        )
    
    def _convert_expr(self, node: Any) -> ExpressionStatement:
        """转换表达式语句"""
        import ast
        
        return ExpressionStatement(
            expression=self._convert_node(node.value),
            source_range=self._create_source_range(node)
        )
    
    def _convert_call(self, node: Any) -> Call:
        """转换函数调用"""
        import ast
        
        arguments = []
        for arg in node.args:
            arguments.append(self._convert_node(arg))
        
        # TODO: 处理关键字参数
        
        return Call(
            callee=self._convert_node(node.func),
            arguments=arguments,
            source_range=self._create_source_range(node)
        )
    
    def _convert_name(self, node: Any) -> Identifier:
        """转换标识符"""
        import ast
        
        return Identifier(
            name=node.id,
            source_range=self._create_source_range(node)
        )
    
    def _convert_constant(self, node: Any) -> Literal:
        """转换常量"""
        import ast
        
        value = node.value
        literal_type = type(value).__name__
        
        if value is None:
            literal_type = "null"
        elif isinstance(value, bool):
            literal_type = "boolean"
        elif isinstance(value, (int, float)):
            literal_type = "number"
        elif isinstance(value, str):
            literal_type = "string"
        
        return Literal(
            value=value,
            literal_type=literal_type,
            source_range=self._create_source_range(node)
        )
    
    def _convert_bin_op(self, node: Any) -> BinaryOperation:
        """转换二元操作"""
        import ast
        
        # 映射Python操作符到通用操作符
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**",
            ast.LShift: "<<",
            ast.RShift: ">>",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.BitAnd: "&",
            ast.MatMult: "@"
        }
        
        operator = op_map.get(type(node.op), "unknown")
        
        return BinaryOperation(
            operator=operator,
            left=self._convert_node(node.left),
            right=self._convert_node(node.right),
            source_range=self._create_source_range(node)
        )
    
    def _convert_import(self, node: Any) -> Import:
        """转换import语句"""
        import ast
        
        # 处理多个导入
        if len(node.names) > 1:
            imports = []
            for name in node.names:
                imports.append(Import(
                    source=name.name,
                    specifiers=[{"name": name.name, "alias": name.asname or name.name}],
                    source_range=self._create_source_range(node)
                ))
            return imports[0]  # 简化处理，只返回第一个
        
        # 单个导入
        name = node.names[0]
        return Import(
            source=name.name,
            specifiers=[{"name": name.name, "alias": name.asname or name.name}],
            source_range=self._create_source_range(node)
        )
    
    def _convert_import_from(self, node: Any) -> Import:
        """转换from import语句"""
        import ast
        
        specifiers = []
        for name in node.names:
            specifiers.append({
                "name": name.name,
                "alias": name.asname or name.name
            })
        
        return Import(
            source=node.module,
            specifiers=specifiers,
            source_range=self._create_source_range(node),
            metadata={"is_from_import": True, "level": node.level}
        )
    
    def _create_source_range(self, ast_node: Any) -> Optional[SourceRange]:
        """从Python AST节点创建源代码位置信息"""
        if not hasattr(ast_node, 'lineno'):
            return None
        
        # Python AST提供行号，但不总是提供列号
        start_line = getattr(ast_node, 'lineno', 0)
        start_column = getattr(ast_node, 'col_offset', 0)
        
        # 结束位置可能不存在
        end_line = getattr(ast_node, 'end_lineno', start_line)
        end_column = getattr(ast_node, 'end_col_offset', 0)
        
        return SourceRange(
            start_line=start_line,
            start_column=start_column,
            end_line=end_line,
            end_column=end_column,
            source_file=self.source_file
        )
    
    def _is_generator(self, node: Any) -> bool:
        """检查函数是否是生成器"""
        import ast
        
        # 简单检查：函数体中是否有yield语句
        for stmt in ast.walk(node):
            if isinstance(stmt, (ast.Yield, ast.YieldFrom)):
                return True
        
        return False


class JavaScriptASTToIRConverter(ASTToIRConverter):
    """JavaScript/TypeScript AST到IR转换器"""
    
    def __init__(self):
        super().__init__("javascript")
    
    def _convert_node(self, ast_node: Any) -> IRNode:
        """
        转换JavaScript/TypeScript AST节点到IR节点
        
        Args:
            ast_node: JavaScript/TypeScript AST节点
            
        Returns:
            转换后的IR节点
        """
        # 这里需要根据使用的JavaScript/TypeScript解析器实现具体的转换逻辑
        # 例如，如果使用esprima、acorn或TypeScript编译器API
        
        # 示例实现（假设使用esprima风格的AST）
        node_type = getattr(ast_node, 'type', None)
        
        if node_type == 'Program':
            return self._convert_program(ast_node)
        elif node_type == 'FunctionDeclaration':
            return self._convert_function_declaration(ast_node)
        elif node_type == 'ClassDeclaration':
            return self._convert_class_declaration(ast_node)
        elif node_type == 'VariableDeclaration':
            return self._convert_variable_declaration(ast_node)
        elif node_type == 'IfStatement':
            return self._convert_if_statement(ast_node)
        elif node_type == 'ReturnStatement':
            return self._convert_return_statement(ast_node)
        elif node_type == 'ExpressionStatement':
            return self._convert_expression_statement(ast_node)
        elif node_type == 'CallExpression':
            return self._convert_call_expression(ast_node)
        elif node_type == 'Identifier':
            return self._convert_identifier(ast_node)
        elif node_type == 'Literal':
            return self._convert_literal(ast_node)
        elif node_type == 'BinaryExpression':
            return self._convert_binary_expression(ast_node)
        elif node_type == 'ImportDeclaration':
            return self._convert_import_declaration(ast_node)
        else:
            self.add_warning(f"未实现的JavaScript/TypeScript AST节点类型: {node_type}", ast_node)
            # 返回一个标识符作为占位符
            return Identifier(
                name=f"Unsupported_{node_type or 'Unknown'}",
                source_range=self._create_source_range(ast_node),
                metadata={"original_type": node_type}
            )
    
    def _convert_program(self, node: Any) -> Module:
        """转换JavaScript/TypeScript程序"""
        declarations = []
        imports = []
        
        for item in node.body:
            ir_node = self._convert_node(item)
            
            if ir_node.node_type.name == "IMPORT":
                imports.append(ir_node)
            else:
                declarations.append(ir_node)
        
        return Module(
            name=self.source_file or "unknown_module",
            declarations=declarations,
            imports=imports,
            source_range=self._create_source_range(node)
        )
    
    # 其他转换方法的实现将类似于Python转换器，但需要适应JavaScript/TypeScript AST的结构
    # 这里省略具体实现，实际使用时需要根据所选的JavaScript/TypeScript解析器完成
    
    def _create_source_range(self, ast_node: Any) -> Optional[SourceRange]:
        """从JavaScript/TypeScript AST节点创建源代码位置信息"""
        # 大多数JavaScript/TypeScript解析器提供start和end属性，包含行和列信息
        if not hasattr(ast_node, 'loc'):
            return None
        
        loc = ast_node.loc
        
        return SourceRange(
            start_line=loc.start.line,
            start_column=loc.start.column,
            end_line=loc.end.line,
            end_column=loc.end.column,
            source_file=self.source_file
        )