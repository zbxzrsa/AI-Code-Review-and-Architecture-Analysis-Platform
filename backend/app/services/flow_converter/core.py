"""
控制流结构跨语言转换核心模块

提供控制流结构（循环、条件、异常处理、函数控制流）的跨语言转换功能
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import ast
import re


class LanguageType(Enum):
    """支持的编程语言类型"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    RUST = "rust"


class ControlFlowType(Enum):
    """控制流类型"""
    LOOP = "loop"  # 循环结构
    CONDITION = "condition"  # 条件语句
    EXCEPTION = "exception"  # 异常处理
    FUNCTION = "function"  # 函数控制流


@dataclass
class ControlFlowNode:
    """控制流节点基类"""
    node_type: ControlFlowType
    source_language: LanguageType
    source_code: str
    ast_node: Any = None
    parent: Optional['ControlFlowNode'] = None
    children: List['ControlFlowNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopNode(ControlFlowNode):
    """循环结构节点"""
    loop_type: str = ""  # for, while, foreach, etc.
    iterator: Optional[str] = None
    iterable: Optional[str] = None
    condition: Optional[str] = None
    body: Optional[str] = None
    
    def __post_init__(self):
        if not self.node_type:
            self.node_type = ControlFlowType.LOOP


@dataclass
class ConditionNode(ControlFlowNode):
    """条件语句节点"""
    condition_type: str = ""  # if, elif, else, switch, etc.
    condition: Optional[str] = None
    true_branch: Optional[str] = None
    false_branch: Optional[str] = None
    cases: List[Dict[str, str]] = field(default_factory=list)  # for switch-case
    
    def __post_init__(self):
        if not self.node_type:
            self.node_type = ControlFlowType.CONDITION


@dataclass
class ExceptionNode(ControlFlowNode):
    """异常处理节点"""
    try_block: Optional[str] = None
    except_blocks: List[Dict[str, str]] = field(default_factory=list)
    finally_block: Optional[str] = None
    resources: List[str] = field(default_factory=list)  # for with/using statements
    
    def __post_init__(self):
        if not self.node_type:
            self.node_type = ControlFlowType.EXCEPTION


@dataclass
class FunctionFlowNode(ControlFlowNode):
    """函数控制流节点"""
    is_async: bool = False
    return_statements: List[str] = field(default_factory=list)
    has_tail_call: bool = False
    await_expressions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.node_type:
            self.node_type = ControlFlowType.FUNCTION


class ControlFlowGraph:
    """控制流图"""
    
    def __init__(self, source_language: LanguageType, source_code: str):
        self.source_language = source_language
        self.source_code = source_code
        self.root: Optional[ControlFlowNode] = None
        self.nodes: List[ControlFlowNode] = []
    
    def add_node(self, node: ControlFlowNode, parent: Optional[ControlFlowNode] = None) -> ControlFlowNode:
        """添加节点到控制流图"""
        self.nodes.append(node)
        
        if parent:
            node.parent = parent
            parent.children.append(node)
        elif not self.root:
            self.root = node
        
        return node
    
    def get_nodes_by_type(self, node_type: ControlFlowType) -> List[ControlFlowNode]:
        """获取指定类型的所有节点"""
        return [node for node in self.nodes if node.node_type == node_type]


class ControlFlowParser:
    """控制流解析器基类"""
    
    def __init__(self, language: LanguageType):
        self.language = language
    
    def parse(self, source_code: str) -> ControlFlowGraph:
        """解析源代码，构建控制流图"""
        raise NotImplementedError("子类必须实现此方法")


class PythonControlFlowParser(ControlFlowParser):
    """Python控制流解析器"""
    
    def __init__(self):
        super().__init__(LanguageType.PYTHON)
    
    def parse(self, source_code: str) -> ControlFlowGraph:
        """解析Python源代码，构建控制流图"""
        graph = ControlFlowGraph(self.language, source_code)
        
        try:
            tree = ast.parse(source_code)
            root_node = ControlFlowNode(
                node_type=ControlFlowType.FUNCTION,  # 使用函数作为根节点类型
                source_language=self.language,
                source_code=source_code,
                ast_node=tree
            )
            graph.add_node(root_node)
            
            # 遍历AST，构建控制流图
            self._process_node(tree, root_node, graph)
            
            return graph
        except SyntaxError as e:
            # 处理语法错误
            print(f"语法错误: {e}")
            return graph
    
    def _process_node(self, node: ast.AST, parent: ControlFlowNode, graph: ControlFlowGraph) -> None:
        """处理AST节点，构建控制流图"""
        # 处理循环结构
        if isinstance(node, ast.For):
            loop_node = self._process_for_loop(node, parent.source_code)
            graph.add_node(loop_node, parent)
            
            # 递归处理循环体
            for child in node.body:
                self._process_node(child, loop_node, graph)
        
        elif isinstance(node, ast.While):
            loop_node = self._process_while_loop(node, parent.source_code)
            graph.add_node(loop_node, parent)
            
            # 递归处理循环体
            for child in node.body:
                self._process_node(child, loop_node, graph)
        
        # 处理条件语句
        elif isinstance(node, ast.If):
            condition_node = self._process_if_statement(node, parent.source_code)
            graph.add_node(condition_node, parent)
            
            # 递归处理条件分支
            for child in node.body:
                self._process_node(child, condition_node, graph)
            
            for child in node.orelse:
                self._process_node(child, condition_node, graph)
        
        # 处理异常处理
        elif isinstance(node, ast.Try):
            exception_node = self._process_try_except(node, parent.source_code)
            graph.add_node(exception_node, parent)
            
            # 递归处理try块
            for child in node.body:
                self._process_node(child, exception_node, graph)
            
            # 递归处理except块
            for handler in node.handlers:
                for child in handler.body:
                    self._process_node(child, exception_node, graph)
            
            # 递归处理finally块
            for child in node.finalbody:
                self._process_node(child, exception_node, graph)
        
        # 处理with语句（上下文管理）
        elif isinstance(node, ast.With):
            exception_node = self._process_with_statement(node, parent.source_code)
            graph.add_node(exception_node, parent)
            
            # 递归处理with块
            for child in node.body:
                self._process_node(child, exception_node, graph)
        
        # 处理函数定义
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            function_node = self._process_function(node, parent.source_code)
            graph.add_node(function_node, parent)
            
            # 递归处理函数体
            for child in node.body:
                self._process_node(child, function_node, graph)
        
        # 处理复合语句
        elif isinstance(node, ast.Module):
            for child in node.body:
                self._process_node(child, parent, graph)
    
    def _process_for_loop(self, node: ast.For, source_code: str) -> LoopNode:
        """处理for循环"""
        # 提取迭代器和可迭代对象
        iterator = ast.unparse(node.target)
        iterable = ast.unparse(node.iter)
        
        # 提取循环体
        body_lines = []
        for child in node.body:
            body_lines.append(ast.unparse(child))
        body = "\n".join(body_lines)
        
        return LoopNode(
            node_type=ControlFlowType.LOOP,
            source_language=self.language,
            source_code=ast.unparse(node),
            ast_node=node,
            loop_type="for",
            iterator=iterator,
            iterable=iterable,
            body=body
        )
    
    def _process_while_loop(self, node: ast.While, source_code: str) -> LoopNode:
        """处理while循环"""
        # 提取条件
        condition = ast.unparse(node.test)
        
        # 提取循环体
        body_lines = []
        for child in node.body:
            body_lines.append(ast.unparse(child))
        body = "\n".join(body_lines)
        
        return LoopNode(
            node_type=ControlFlowType.LOOP,
            source_language=self.language,
            source_code=ast.unparse(node),
            ast_node=node,
            loop_type="while",
            condition=condition,
            body=body
        )
    
    def _process_if_statement(self, node: ast.If, source_code: str) -> ConditionNode:
        """处理if语句"""
        # 提取条件
        condition = ast.unparse(node.test)
        
        # 提取true分支
        true_branch_lines = []
        for child in node.body:
            true_branch_lines.append(ast.unparse(child))
        true_branch = "\n".join(true_branch_lines)
        
        # 提取false分支
        false_branch_lines = []
        for child in node.orelse:
            false_branch_lines.append(ast.unparse(child))
        false_branch = "\n".join(false_branch_lines) if false_branch_lines else None
        
        return ConditionNode(
            node_type=ControlFlowType.CONDITION,
            source_language=self.language,
            source_code=ast.unparse(node),
            ast_node=node,
            condition_type="if",
            condition=condition,
            true_branch=true_branch,
            false_branch=false_branch
        )
    
    def _process_try_except(self, node: ast.Try, source_code: str) -> ExceptionNode:
        """处理try-except语句"""
        # 提取try块
        try_block_lines = []
        for child in node.body:
            try_block_lines.append(ast.unparse(child))
        try_block = "\n".join(try_block_lines)
        
        # 提取except块
        except_blocks = []
        for handler in node.handlers:
            except_type = ast.unparse(handler.type) if handler.type else None
            except_name = handler.name
            
            except_body_lines = []
            for child in handler.body:
                except_body_lines.append(ast.unparse(child))
            except_body = "\n".join(except_body_lines)
            
            except_blocks.append({
                "type": except_type,
                "name": except_name,
                "body": except_body
            })
        
        # 提取finally块
        finally_block_lines = []
        for child in node.finalbody:
            finally_block_lines.append(ast.unparse(child))
        finally_block = "\n".join(finally_block_lines) if finally_block_lines else None
        
        return ExceptionNode(
            node_type=ControlFlowType.EXCEPTION,
            source_language=self.language,
            source_code=ast.unparse(node),
            ast_node=node,
            try_block=try_block,
            except_blocks=except_blocks,
            finally_block=finally_block
        )
    
    def _process_with_statement(self, node: ast.With, source_code: str) -> ExceptionNode:
        """处理with语句"""
        # 提取资源
        resources = []
        for item in node.items:
            context_expr = ast.unparse(item.context_expr)
            optional_vars = ast.unparse(item.optional_vars) if item.optional_vars else None
            
            if optional_vars:
                resources.append(f"{context_expr} as {optional_vars}")
            else:
                resources.append(context_expr)
        
        # 提取with块
        body_lines = []
        for child in node.body:
            body_lines.append(ast.unparse(child))
        body = "\n".join(body_lines)
        
        return ExceptionNode(
            node_type=ControlFlowType.EXCEPTION,
            source_language=self.language,
            source_code=ast.unparse(node),
            ast_node=node,
            try_block=body,
            resources=resources
        )
    
    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], source_code: str) -> FunctionFlowNode:
        """处理函数定义"""
        # 判断是否为异步函数
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        # 提取return语句
        return_statements = []
        await_expressions = []
        
        # 递归查找return语句和await表达式
        def find_returns_and_awaits(node):
            if isinstance(node, ast.Return):
                return_statements.append(ast.unparse(node))
            elif isinstance(node, ast.Await):
                await_expressions.append(ast.unparse(node))
            
            for child in ast.iter_child_nodes(node):
                find_returns_and_awaits(child)
        
        for child in node.body:
            find_returns_and_awaits(child)
        
        # 检查是否有尾调用
        has_tail_call = False
        if return_statements:
            for stmt in return_statements:
                # 简单检查：如果return语句中包含函数调用，则可能是尾调用
                if re.search(r'return\s+\w+\(', stmt):
                    has_tail_call = True
                    break
        
        return FunctionFlowNode(
            node_type=ControlFlowType.FUNCTION,
            source_language=self.language,
            source_code=ast.unparse(node),
            ast_node=node,
            is_async=is_async,
            return_statements=return_statements,
            has_tail_call=has_tail_call,
            await_expressions=await_expressions
        )


class JavaScriptControlFlowParser(ControlFlowParser):
    """JavaScript控制流解析器"""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT)
    
    def parse(self, source_code: str) -> ControlFlowGraph:
        """解析JavaScript源代码，构建控制流图"""
        # 注意：这里需要使用JavaScript解析器，如esprima或acorn
        # 由于Python环境中可能没有这些库，这里提供一个简化的实现
        graph = ControlFlowGraph(self.language, source_code)
        
        # 创建根节点
        root_node = ControlFlowNode(
            node_type=ControlFlowType.FUNCTION,
            source_language=self.language,
            source_code=source_code
        )
        graph.add_node(root_node)
        
        # 使用正则表达式简单解析JavaScript代码
        # 注意：这只是一个简化的示例，实际应用中应使用专业的JavaScript解析器
        
        # 解析for循环
        for_loops = re.finditer(r'for\s*\((.*?)\)\s*{(.*?)}', source_code, re.DOTALL)
        for match in for_loops:
            header = match.group(1)
            body = match.group(2)
            
            # 尝试解析for循环头部
            parts = header.split(';')
            if len(parts) == 3:
                # 标准for循环: for (init; condition; update)
                init, condition, update = parts
                
                loop_node = LoopNode(
                    node_type=ControlFlowType.LOOP,
                    source_language=self.language,
                    source_code=match.group(0),
                    loop_type="for",
                    condition=condition.strip(),
                    body=body.strip()
                )
                graph.add_node(loop_node, root_node)
        
        # 解析while循环
        while_loops = re.finditer(r'while\s*\((.*?)\)\s*{(.*?)}', source_code, re.DOTALL)
        for match in while_loops:
            condition = match.group(1)
            body = match.group(2)
            
            loop_node = LoopNode(
                node_type=ControlFlowType.LOOP,
                source_language=self.language,
                source_code=match.group(0),
                loop_type="while",
                condition=condition.strip(),
                body=body.strip()
            )
            graph.add_node(loop_node, root_node)
        
        # 解析if语句
        if_statements = re.finditer(r'if\s*\((.*?)\)\s*{(.*?)}(?:\s*else\s*{(.*?)})?', source_code, re.DOTALL)
        for match in if_statements:
            condition = match.group(1)
            true_branch = match.group(2)
            false_branch = match.group(3) if match.lastindex >= 3 else None
            
            condition_node = ConditionNode(
                node_type=ControlFlowType.CONDITION,
                source_language=self.language,
                source_code=match.group(0),
                condition_type="if",
                condition=condition.strip(),
                true_branch=true_branch.strip(),
                false_branch=false_branch.strip() if false_branch else None
            )
            graph.add_node(condition_node, root_node)
        
        # 解析try-catch语句
        try_catch = re.finditer(r'try\s*{(.*?)}(?:\s*catch\s*\((.*?)\)\s*{(.*?)})(?:\s*finally\s*{(.*?)})?', source_code, re.DOTALL)
        for match in try_catch:
            try_block = match.group(1)
            catch_param = match.group(2) if match.lastindex >= 2 else None
            catch_block = match.group(3) if match.lastindex >= 3 else None
            finally_block = match.group(4) if match.lastindex >= 4 else None
            
            except_blocks = []
            if catch_param and catch_block:
                except_blocks.append({
                    "type": catch_param.strip(),
                    "body": catch_block.strip()
                })
            
            exception_node = ExceptionNode(
                node_type=ControlFlowType.EXCEPTION,
                source_language=self.language,
                source_code=match.group(0),
                try_block=try_block.strip(),
                except_blocks=except_blocks,
                finally_block=finally_block.strip() if finally_block else None
            )
            graph.add_node(exception_node, root_node)
        
        # 解析函数定义
        function_defs = re.finditer(r'(async\s+)?function\s+(\w+)\s*\((.*?)\)\s*{(.*?)}', source_code, re.DOTALL)
        for match in function_defs:
            is_async = bool(match.group(1))
            func_name = match.group(2)
            params = match.group(3)
            body = match.group(4)
            
            # 查找return语句
            return_statements = []
            for return_match in re.finditer(r'return\s+(.*?);', body):
                return_statements.append(return_match.group(0))
            
            # 查找await表达式
            await_expressions = []
            if is_async:
                for await_match in re.finditer(r'await\s+(.*?);', body):
                    await_expressions.append(await_match.group(0))
            
            # 检查是否有尾调用
            has_tail_call = False
            for stmt in return_statements:
                if re.search(r'return\s+\w+\(', stmt):
                    has_tail_call = True
                    break
            
            function_node = FunctionFlowNode(
                node_type=ControlFlowType.FUNCTION,
                source_language=self.language,
                source_code=match.group(0),
                is_async=is_async,
                return_statements=return_statements,
                has_tail_call=has_tail_call,
                await_expressions=await_expressions
            )
            graph.add_node(function_node, root_node)
        
        return graph


class ControlFlowConverter:
    """控制流转换器基类"""
    
    def __init__(self, source_language: LanguageType, target_language: LanguageType):
        self.source_language = source_language
        self.target_language = target_language
        
        # 初始化解析器
        self.parsers = {
            LanguageType.PYTHON: PythonControlFlowParser(),
            LanguageType.JAVASCRIPT: JavaScriptControlFlowParser(),
            # 其他语言的解析器...
        }
    
    def convert(self, source_code: str) -> str:
        """转换源代码到目标语言"""
        # 1. 解析源代码，构建控制流图
        parser = self.parsers.get(self.source_language)
        if not parser:
            raise ValueError(f"不支持的源语言: {self.source_language}")
        
        graph = parser.parse(source_code)
        
        # 2. 转换控制流图到目标语言
        target_code = self._convert_graph(graph)
        
        return target_code
    
    def _convert_graph(self, graph: ControlFlowGraph) -> str:
        """转换控制流图到目标语言代码"""
        raise NotImplementedError("子类必须实现此方法")
    
    def _convert_node(self, node: ControlFlowNode) -> str:
        """转换单个控制流节点"""
        if node.node_type == ControlFlowType.LOOP:
            return self._convert_loop(node)
        elif node.node_type == ControlFlowType.CONDITION:
            return self._convert_condition(node)
        elif node.node_type == ControlFlowType.EXCEPTION:
            return self._convert_exception(node)
        elif node.node_type == ControlFlowType.FUNCTION:
            return self._convert_function(node)
        else:
            return ""
    
    def _convert_loop(self, node: LoopNode) -> str:
        """转换循环结构"""
        raise NotImplementedError("子类必须实现此方法")
    
    def _convert_condition(self, node: ConditionNode) -> str:
        """转换条件语句"""
        raise NotImplementedError("子类必须实现此方法")
    
    def _convert_exception(self, node: ExceptionNode) -> str:
        """转换异常处理"""
        raise NotImplementedError("子类必须实现此方法")
    
    def _convert_function(self, node: FunctionFlowNode) -> str:
        """转换函数控制流"""
        raise NotImplementedError("子类必须实现此方法")


class PythonToJavaScriptConverter(ControlFlowConverter):
    """Python到JavaScript的控制流转换器"""
    
    def __init__(self):
        super().__init__(LanguageType.PYTHON, LanguageType.JAVASCRIPT)
    
    def _convert_graph(self, graph: ControlFlowGraph) -> str:
        """转换控制流图到JavaScript代码"""
        # 简化实现：直接转换每个节点，然后拼接
        result = []
        
        # 按节点类型分组处理
        functions = graph.get_nodes_by_type(ControlFlowType.FUNCTION)
        loops = graph.get_nodes_by_type(ControlFlowType.LOOP)
        conditions = graph.get_nodes_by_type(ControlFlowType.CONDITION)
        exceptions = graph.get_nodes_by_type(ControlFlowType.EXCEPTION)
        
        # 处理函数
        for node in functions:
            result.append(self._convert_function(node))
        
        # 处理循环
        for node in loops:
            result.append(self._convert_loop(node))
        
        # 处理条件
        for node in conditions:
            result.append(self._convert_condition(node))
        
        # 处理异常
        for node in exceptions:
            result.append(self._convert_exception(node))
        
        return "\n\n".join(result)
    
    def _convert_loop(self, node: LoopNode) -> str:
        """转换循环结构到JavaScript"""
        if not isinstance(node, LoopNode):
            return ""
        
        if node.loop_type == "for":
            # 处理Python的for循环
            if node.iterable and "range(" in node.iterable:
                # 处理range循环
                range_args = node.iterable.replace("range(", "").replace(")", "").split(",")
                
                if len(range_args) == 1:
                    # range(stop)
                    start = "0"
                    stop = range_args[0].strip()
                    step = "1"
                elif len(range_args) == 2:
                    # range(start, stop)
                    start = range_args[0].strip()
                    stop = range_args[1].strip()
                    step = "1"
                else:
                    # range(start, stop, step)
                    start = range_args[0].strip()
                    stop = range_args[1].strip()
                    step = range_args[2].strip()
                
                # 生成JavaScript的for循环
                js_loop = f"for (let {node.iterator} = {start}; {node.iterator} < {stop}; {node.iterator} += {step}) {{\n"
                js_loop += f"{node.body}\n"
                js_loop += "}"
                
                return js_loop
            else:
                # 处理一般的for-in/for-of循环
                js_loop = f"for (const {node.iterator} of {node.iterable}) {{\n"
                js_loop += f"{node.body}\n"
                js_loop += "}"
                
                return js_loop
        
        elif node.loop_type == "while":
            # 处理while循环
            js_loop = f"while ({node.condition}) {{\n"
            js_loop += f"{node.body}\n"
            js_loop += "}"
            
            return js_loop
        
        return ""
    
    def _convert_condition(self, node: ConditionNode) -> str:
        """转换条件语句到JavaScript"""
        if not isinstance(node, ConditionNode):
            return ""
        
        # 替换Python的比较运算符
        condition = node.condition
        if condition:
            condition = condition.replace(" == ", " === ")
            condition = condition.replace(" != ", " !== ")
        
        if node.condition_type == "if":
            js_condition = f"if ({condition}) {{\n"
            js_condition += f"{node.true_branch}\n"
            js_condition += "}"
            
            if node.false_branch:
                js_condition += f" else {{\n"
                js_condition += f"{node.false_branch}\n"
                js_condition += "}"
            
            return js_condition
        
        return ""
    
    def _convert_exception(self, node: ExceptionNode) -> str:
        """转换异常处理到JavaScript"""
        if not isinstance(node, ExceptionNode):
            return ""
        
        # 处理try-except
        if node.try_block:
            js_exception = f"try {{\n"
            js_exception += f"{node.try_block}\n"
            js_exception += "}"
            
            # 处理except块
            for except_block in node.except_blocks:
                except_type = except_block.get("type")
                except_body = except_block.get("body")
                
                if except_type:
                    js_exception += f" catch (error) {{\n"
                    js_exception += f"if (error instanceof {except_type}) {{\n"
                    js_exception += f"{except_body}\n"
                    js_exception += "} else {\n"
                    js_exception += "throw error;\n"
                    js_exception += "}\n"
                    js_exception += "}"
                else:
                    js_exception += f" catch (error) {{\n"
                    js_exception += f"{except_body}\n"
                    js_exception += "}"
            
            # 处理finally块
            if node.finally_block:
                js_exception += f" finally {{\n"
                js_exception += f"{node.finally_block}\n"
                js_exception += "}"
            
            return js_exception
        
        # 处理with语句
        elif node.resources:
            # JavaScript没有直接对应的with语句，转换为try-finally
            resource_setup = []
            resource_cleanup = []
            
            for resource in node.resources:
                if " as " in resource:
                    expr, var = resource.split(" as ")
                    resource_setup.append(f"const {var} = {expr};")
                    # 假设资源有close方法
                    resource_cleanup.append(f"if ({var} && typeof {var}.close === 'function') {var}.close();")
                else:
                    # 没有变量名的情况，生成一个临时变量
                    temp_var = f"_resource{len(resource_setup)}"
                    resource_setup.append(f"const {temp_var} = {resource};")
                    resource_cleanup.append(f"if ({temp_var} && typeof {temp_var}.close === 'function') {temp_var}.close();")
            
            js_with = "\n".join(resource_setup) + "\n"
            js_with += f"try {{\n"
            js_with += f"{node.try_block}\n"
            js_with += "} finally {\n"
            js_with += "\n".join(resource_cleanup) + "\n"
            js_with += "}"
            
            return js_with
        
        return ""
    
    def _convert_function(self, node: FunctionFlowNode) -> str:
        """转换函数控制流到JavaScript"""
        if not isinstance(node, FunctionFlowNode):
            return ""
        
        # 提取函数名和参数
        source_code = node.source_code
        
        # 简单解析函数定义
        match = re.search(r'(async\s+)?def\s+(\w+)\s*\((.*?)\):', source_code, re.DOTALL)
        if not match:
            return source_code  # 无法解析，返回原始代码
        
        is_async_str = match.group(1)
        func_name = match.group(2)
        params = match.group(3)
        
        # 转换参数
        js_params = params
        # 处理默认参数
        js_params = re.sub(r'(\w+)=([^,]+)', r'\1=\2', js_params)
        
        # 构建JavaScript函数
        js_func = ""
        if is_async_str:
            js_func += "async "
        
        js_func += f"function {func_name}({js_params}) {{\n"
        
        # 提取函数体
        body_match = re.search(r'def\s+\w+\s*\(.*?\):(.*?)(?=\n\S|$)', source_code, re.DOTALL)
        if body_match:
            body = body_match.group(1).strip()
            
            # 转换函数体中的Python特定语法
            # 1. print -> console.log
            body = re.sub(r'print\((.*?)\)', r'console.log(\1)', body)
            
            # 2. 缩进转换
            lines = body.split('\n')
            js_body_lines = []
            
            for line in lines:
                # 移除前导空格
                stripped = line.lstrip()
                if stripped:
                    js_body_lines.append(stripped)
            
            js_body = "\n".join(js_body_lines)
            js_func += js_body + "\n"
        
        js_func += "}"
        
        return js_func


# 工厂函数，创建适合的转换器
def create_converter(source_language: LanguageType, target_language: LanguageType) -> ControlFlowConverter:
    """创建适合的控制流转换器"""
    if source_language == LanguageType.PYTHON and target_language == LanguageType.JAVASCRIPT:
        return PythonToJavaScriptConverter()
    # 添加其他语言对的转换器...
    else:
        raise ValueError(f"不支持从 {source_language} 到 {target_language} 的转换")


# 示例使用
def convert_code(source_code: str, source_language: str, target_language: str) -> str:
    """转换源代码到目标语言"""
    try:
        source_lang = LanguageType(source_language.lower())
        target_lang = LanguageType(target_language.lower())
        
        converter = create_converter(source_lang, target_lang)
        return converter.convert(source_code)
    except ValueError as e:
        return f"错误: {str(e)}"
    except Exception as e:
        return f"转换失败: {str(e)}"