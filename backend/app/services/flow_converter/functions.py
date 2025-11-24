"""
函数控制流跨语言转换模块

提供各种编程语言间函数控制流的转换功能，包括：
- 返回值处理
- 尾调用优化
- 协程和异步转换
"""
from typing import Dict, List, Optional, Any, Tuple, Set
import re
import ast
from .core import (
    LanguageType, ControlFlowType, ControlFlowNode, FunctionFlowNode,
    ControlFlowGraph, ControlFlowConverter
)


class FunctionFlowConverter:
    """函数控制流转换器基类"""
    
    def __init__(self, source_language: LanguageType, target_language: LanguageType):
        self.source_language = source_language
        self.target_language = target_language
    
    def convert_function_flow(self, function_node: FunctionFlowNode) -> str:
        """转换函数控制流"""
        if function_node.function_type == "async":
            return self.convert_async_function(function_node)
        elif function_node.function_type == "generator":
            return self.convert_generator_function(function_node)
        elif function_node.function_type == "tail_recursive":
            return self.convert_tail_recursive_function(function_node)
        else:
            return self.convert_regular_function(function_node)
    
    def convert_regular_function(self, function_node: FunctionFlowNode) -> str:
        """转换普通函数"""
        raise NotImplementedError("子类必须实现此方法")
    
    def convert_async_function(self, function_node: FunctionFlowNode) -> str:
        """转换异步函数"""
        raise NotImplementedError("子类必须实现此方法")
    
    def convert_generator_function(self, function_node: FunctionFlowNode) -> str:
        """转换生成器函数"""
        raise NotImplementedError("子类必须实现此方法")
    
    def convert_tail_recursive_function(self, function_node: FunctionFlowNode) -> str:
        """转换尾递归函数"""
        raise NotImplementedError("子类必须实现此方法")
    
    def optimize_return_values(self, function_body: str) -> str:
        """优化返回值处理"""
        return function_body  # 默认不做优化
    
    def add_tail_call_optimization(self, function_body: str, function_name: str) -> str:
        """添加尾调用优化"""
        return function_body  # 默认不做优化


class PythonToJavaScriptFunctionConverter(FunctionFlowConverter):
    """Python到JavaScript的函数控制流转换器"""
    
    def __init__(self):
        super().__init__(LanguageType.PYTHON, LanguageType.JAVASCRIPT)
    
    def convert_regular_function(self, function_node: FunctionFlowNode) -> str:
        """转换普通函数"""
        if not isinstance(function_node, FunctionFlowNode):
            return ""
        
        # 提取函数信息
        function_name = function_node.function_name or "unnamed_function"
        params = function_node.parameters or []
        body = function_node.function_body or ""
        return_type = function_node.return_type
        
        # 转换参数
        js_params = []
        for param in params:
            param_name = param.get("name", "")
            param_default = param.get("default", None)
            
            if param_default is not None:
                # 转换默认值
                js_default = self._convert_python_value_to_js(param_default)
                js_params.append(f"{param_name} = {js_default}")
            else:
                js_params.append(param_name)
        
        # 构建JavaScript函数
        js_function = f"function {function_name}({', '.join(js_params)}) {{\n"
        
        # 转换函数体
        js_body = self._convert_function_body(body)
        
        # 优化返回值处理
        js_body = self.optimize_return_values(js_body)
        
        # 添加函数体
        js_function += js_body
        
        # 闭合函数
        js_function += "}"
        
        return js_function
    
    def convert_async_function(self, function_node: FunctionFlowNode) -> str:
        """转换异步函数"""
        if not isinstance(function_node, FunctionFlowNode):
            return ""
        
        # 提取函数信息
        function_name = function_node.function_name or "unnamed_async_function"
        params = function_node.parameters or []
        body = function_node.function_body or ""
        
        # 转换参数
        js_params = []
        for param in params:
            param_name = param.get("name", "")
            param_default = param.get("default", None)
            
            if param_default is not None:
                # 转换默认值
                js_default = self._convert_python_value_to_js(param_default)
                js_params.append(f"{param_name} = {js_default}")
            else:
                js_params.append(param_name)
        
        # 构建JavaScript异步函数
        js_function = f"async function {function_name}({', '.join(js_params)}) {{\n"
        
        # 转换函数体，处理异步特性
        js_body = self._convert_async_function_body(body)
        
        # 优化返回值处理
        js_body = self.optimize_return_values(js_body)
        
        # 添加函数体
        js_function += js_body
        
        # 闭合函数
        js_function += "}"
        
        return js_function
    
    def convert_generator_function(self, function_node: FunctionFlowNode) -> str:
        """转换生成器函数"""
        if not isinstance(function_node, FunctionFlowNode):
            return ""
        
        # 提取函数信息
        function_name = function_node.function_name or "unnamed_generator"
        params = function_node.parameters or []
        body = function_node.function_body or ""
        
        # 转换参数
        js_params = []
        for param in params:
            param_name = param.get("name", "")
            param_default = param.get("default", None)
            
            if param_default is not None:
                # 转换默认值
                js_default = self._convert_python_value_to_js(param_default)
                js_params.append(f"{param_name} = {js_default}")
            else:
                js_params.append(param_name)
        
        # 构建JavaScript生成器函数
        js_function = f"function* {function_name}({', '.join(js_params)}) {{\n"
        
        # 转换函数体，处理生成器特性
        js_body = self._convert_generator_function_body(body)
        
        # 添加函数体
        js_function += js_body
        
        # 闭合函数
        js_function += "}"
        
        return js_function
    
    def convert_tail_recursive_function(self, function_node: FunctionFlowNode) -> str:
        """转换尾递归函数"""
        if not isinstance(function_node, FunctionFlowNode):
            return ""
        
        # 提取函数信息
        function_name = function_node.function_name or "unnamed_recursive_function"
        params = function_node.parameters or []
        body = function_node.function_body or ""
        
        # 转换参数
        js_params = []
        for param in params:
            param_name = param.get("name", "")
            param_default = param.get("default", None)
            
            if param_default is not None:
                # 转换默认值
                js_default = self._convert_python_value_to_js(param_default)
                js_params.append(f"{param_name} = {js_default}")
            else:
                js_params.append(param_name)
        
        # 构建JavaScript函数
        js_function = f"function {function_name}({', '.join(js_params)}) {{\n"
        
        # 转换函数体
        js_body = self._convert_function_body(body)
        
        # 添加尾调用优化
        js_body = self.add_tail_call_optimization(js_body, function_name)
        
        # 添加函数体
        js_function += js_body
        
        # 闭合函数
        js_function += "}"
        
        # 添加尾调用优化的注释
        js_function += "\n// Note: JavaScript engines may not optimize tail calls. Consider using iteration for large recursion depths."
        
        return js_function
    
    def _convert_function_body(self, body: str) -> str:
        """转换函数体"""
        if not body:
            return "  // Empty function body\n  return undefined;"
        
        # 分行处理
        lines = body.split('\n')
        js_lines = []
        
        for line in lines:
            # 跳过空行
            if not line.strip():
                js_lines.append("")
                continue
            
            # 转换print语句
            if "print(" in line:
                js_line = re.sub(r'print\((.*)\)', r'console.log(\1)', line)
                js_lines.append("  " + js_line)
                continue
            
            # 转换Python的比较运算符
            js_line = line
            js_line = js_line.replace(" == ", " === ")
            js_line = js_line.replace(" != ", " !== ")
            js_line = js_line.replace("True", "true")
            js_line = js_line.replace("False", "false")
            js_line = js_line.replace("None", "null")
            
            # 转换逻辑运算符
            js_line = js_line.replace(" and ", " && ")
            js_line = js_line.replace(" or ", " || ")
            js_line = js_line.replace("not ", "!")
            
            # 转换列表推导式
            list_comp_match = re.search(r'\[(.*?) for (.*?) in (.*?)(?: if (.*?))?\]', js_line)
            if list_comp_match:
                expr = list_comp_match.group(1)
                var = list_comp_match.group(2)
                iterable = list_comp_match.group(3)
                condition = list_comp_match.group(4)
                
                if condition:
                    js_line = js_line.replace(
                        list_comp_match.group(0),
                        f"Array.from({iterable}).filter({var} => {condition}).map({var} => {expr})"
                    )
                else:
                    js_line = js_line.replace(
                        list_comp_match.group(0),
                        f"Array.from({iterable}).map({var} => {expr})"
                    )
            
            # 添加缩进
            js_lines.append("  " + js_line)
        
        return "\n".join(js_lines)
    
    def _convert_async_function_body(self, body: str) -> str:
        """转换异步函数体"""
        if not body:
            return "  // Empty async function body\n  return undefined;"
        
        # 分行处理
        lines = body.split('\n')
        js_lines = []
        
        for line in lines:
            # 跳过空行
            if not line.strip():
                js_lines.append("")
                continue
            
            # 转换异步上下文管理器
            if "async with" in line:
                # 提取上下文管理器表达式和变量
                async_with_match = re.search(r'async with (.*?) as (.*?):', line)
                if async_with_match:
                    expr = async_with_match.group(1)
                    var = async_with_match.group(2)
                    
                    # 转换为JavaScript的try-finally模式
                    if "aiohttp.ClientSession" in expr:
                        js_lines.append(f"  // Converted from 'async with {expr} as {var}:'")
                        js_lines.append(f"  const {var} = await fetch;")
                    elif "session.get" in expr or "session.post" in expr:
                        js_lines.append(f"  // Converted from 'async with {expr} as {var}:'")
                        js_lines.append(f"  const {var} = await {expr.replace('session.get', 'fetch').replace('session.post', 'fetch')};")
                    else:
                        js_lines.append(f"  // Converted from 'async with {expr} as {var}:'")
                        js_lines.append(f"  let {var};")
                        js_lines.append(f"  try {{")
                        js_lines.append(f"    {var} = await {expr};")
                    continue
            
            # 转换await表达式
            if "await" in line:
                # 保留await关键字，JavaScript也使用它
                js_line = line
            else:
                # 使用普通函数体转换逻辑
                js_line = line
                
                # 转换print语句
                if "print(" in js_line:
                    js_line = re.sub(r'print\((.*)\)', r'console.log(\1)', js_line)
                
                # 转换Python的比较运算符
                js_line = js_line.replace(" == ", " === ")
                js_line = js_line.replace(" != ", " !== ")
                js_line = js_line.replace("True", "true")
                js_line = js_line.replace("False", "false")
                js_line = js_line.replace("None", "null")
                
                # 转换逻辑运算符
                js_line = js_line.replace(" and ", " && ")
                js_line = js_line.replace(" or ", " || ")
                js_line = js_line.replace("not ", "!")
            
            # 添加缩进
            js_lines.append("  " + js_line)
        
        return "\n".join(js_lines)
    
    def _convert_generator_function_body(self, body: str) -> str:
        """转换生成器函数体"""
        if not body:
            return "  // Empty generator function body\n  yield null;"
        
        # 分行处理
        lines = body.split('\n')
        js_lines = []
        
        for line in lines:
            # 跳过空行
            if not line.strip():
                js_lines.append("")
                continue
            
            # 转换yield表达式
            if "yield" in line:
                # JavaScript也使用yield关键字，但语法略有不同
                js_line = line
                
                # 转换yield from为yield*
                js_line = js_line.replace("yield from", "yield*")
            else:
                # 使用普通函数体转换逻辑
                js_line = line
                
                # 转换print语句
                if "print(" in js_line:
                    js_line = re.sub(r'print\((.*)\)', r'console.log(\1)', js_line)
                
                # 转换Python的比较运算符
                js_line = js_line.replace(" == ", " === ")
                js_line = js_line.replace(" != ", " !== ")
                js_line = js_line.replace("True", "true")
                js_line = js_line.replace("False", "false")
                js_line = js_line.replace("None", "null")
                
                # 转换逻辑运算符
                js_line = js_line.replace(" and ", " && ")
                js_line = js_line.replace(" or ", " || ")
                js_line = js_line.replace("not ", "!")
            
            # 添加缩进
            js_lines.append("  " + js_line)
        
        return "\n".join(js_lines)
    
    def optimize_return_values(self, function_body: str) -> str:
        """优化返回值处理"""
        # 检查是否有多个返回点
        return_count = function_body.count("return ")
        
        if return_count <= 1:
            # 只有0或1个返回点，不需要优化
            return function_body
        
        # 多个返回点，考虑使用变量存储返回值
        lines = function_body.split('\n')
        optimized_lines = []
        has_added_result_var = False
        
        for line in lines:
            if "return " in line and not has_added_result_var:
                # 添加结果变量声明
                optimized_lines.append("  let _result;")
                has_added_result_var = True
            
            if "return " in line and "return undefined" not in line and "return null" not in line:
                # 替换返回语句为结果赋值和返回
                return_value = line.split("return ")[1].strip().rstrip(";")
                optimized_lines.append(f"  _result = {return_value};")
                optimized_lines.append("  return _result;")
            else:
                optimized_lines.append(line)
        
        return "\n".join(optimized_lines)
    
    def add_tail_call_optimization(self, function_body: str, function_name: str) -> str:
        """添加尾调用优化"""
        # 检查是否有尾递归调用
        tail_recursive_pattern = re.compile(rf"return\s+{re.escape(function_name)}\s*\(")
        if not tail_recursive_pattern.search(function_body):
            # 没有尾递归调用，不需要优化
            return function_body
        
        # 有尾递归调用，转换为迭代形式
        lines = function_body.split('\n')
        optimized_lines = []
        
        # 添加优化注释
        optimized_lines.append("  // Tail call optimization: converting recursion to iteration")
        optimized_lines.append("  let _args = Array.from(arguments);")
        optimized_lines.append("  let _result;")
        optimized_lines.append("  while (true) {")
        
        # 处理函数体
        for line in lines:
            if tail_recursive_pattern.search(line):
                # 提取递归调用的参数
                call_match = re.search(rf"return\s+{re.escape(function_name)}\s*\((.*?)\)", line)
                if call_match:
                    args = call_match.group(1)
                    optimized_lines.append(f"    // Original recursive call: return {function_name}({args})")
                    optimized_lines.append(f"    _args = [{args}];")
                    optimized_lines.append("    continue;")
            elif "return " in line:
                # 处理其他返回语句
                return_value = line.split("return ")[1].strip().rstrip(";")
                optimized_lines.append(f"    _result = {return_value};")
                optimized_lines.append("    break;")
            else:
                # 保留其他行，增加缩进
                optimized_lines.append("  " + line)
        
        # 添加循环结束和最终返回
        optimized_lines.append("  }")
        optimized_lines.append("  return _result;")
        
        return "\n".join(optimized_lines)
    
    def _convert_python_value_to_js(self, value: str) -> str:
        """转换Python值到JavaScript值"""
        if value == "None":
            return "null"
        elif value == "True":
            return "true"
        elif value == "False":
            return "false"
        elif value.startswith("[") and value.endswith("]"):
            # 列表转换
            return value
        elif value.startswith("{") and value.endswith("}"):
            # 字典转换
            return value
        elif value.startswith("'") and value.endswith("'"):
            # 字符串转换
            return value
        elif value.startswith('"') and value.endswith('"'):
            # 字符串转换
            return value
        else:
            # 默认不变
            return value


class PythonToTypeScriptFunctionConverter(PythonToJavaScriptFunctionConverter):
    """Python到TypeScript的函数控制流转换器"""
    
    def __init__(self):
        super().__init__()
        self.target_language = LanguageType.TYPESCRIPT
    
    def convert_regular_function(self, function_node: FunctionFlowNode) -> str:
        """转换普通函数"""
        if not isinstance(function_node, FunctionFlowNode):
            return ""
        
        # 提取函数信息
        function_name = function_node.function_name or "unnamed_function"
        params = function_node.parameters or []
        body = function_node.function_body or ""
        return_type = function_node.return_type
        
        # 转换参数
        ts_params = []
        for param in params:
            param_name = param.get("name", "")
            param_type = param.get("type", "")
            param_default = param.get("default", None)
            
            # 转换参数类型
            ts_type = self._convert_python_type_to_ts(param_type) if param_type else "any"
            
            if param_default is not None:
                # 转换默认值
                ts_default = self._convert_python_value_to_js(param_default)
                ts_params.append(f"{param_name}: {ts_type} = {ts_default}")
            else:
                ts_params.append(f"{param_name}: {ts_type}")
        
        # 转换返回类型
        ts_return_type = self._convert_python_type_to_ts(return_type) if return_type else "any"
        
        # 构建TypeScript函数
        ts_function = f"function {function_name}({', '.join(ts_params)}): {ts_return_type} {{\n"
        
        # 转换函数体
        ts_body = self._convert_function_body(body)
        
        # 优化返回值处理
        ts_body = self.optimize_return_values(ts_body)
        
        # 添加函数体
        ts_function += ts_body
        
        # 闭合函数
        ts_function += "}"
        
        return ts_function
    
    def convert_async_function(self, function_node: FunctionFlowNode) -> str:
        """转换异步函数"""
        if not isinstance(function_node, FunctionFlowNode):
            return ""
        
        # 提取函数信息
        function_name = function_node.function_name or "unnamed_async_function"
        params = function_node.parameters or []
        body = function_node.function_body or ""
        return_type = function_node.return_type
        
        # 转换参数
        ts_params = []
        for param in params:
            param_name = param.get("name", "")
            param_type = param.get("type", "")
            param_default = param.get("default", None)
            
            # 转换参数类型
            ts_type = self._convert_python_type_to_ts(param_type) if param_type else "any"
            
            if param_default is not None:
                # 转换默认值
                ts_default = self._convert_python_value_to_js(param_default)
                ts_params.append(f"{param_name}: {ts_type} = {ts_default}")
            else:
                ts_params.append(f"{param_name}: {ts_type}")
        
        # 转换返回类型
        ts_return_type = self._convert_python_type_to_ts(return_type) if return_type else "any"
        
        # 构建TypeScript异步函数
        ts_function = f"async function {function_name}({', '.join(ts_params)}): Promise<{ts_return_type}> {{\n"
        
        # 转换函数体，处理异步特性
        ts_body = self._convert_async_function_body(body)
        
        # 优化返回值处理
        ts_body = self.optimize_return_values(ts_body)
        
        # 添加函数体
        ts_function += ts_body
        
        # 闭合函数
        ts_function += "}"
        
        return ts_function
    
    def _convert_python_type_to_ts(self, python_type: str) -> str:
        """转换Python类型到TypeScript类型"""
        type_mapping = {
            "str": "string",
            "int": "number",
            "float": "number",
            "bool": "boolean",
            "list": "Array<any>",
            "dict": "Record<string, any>",
            "tuple": "Array<any>",
            "set": "Set<any>",
            "None": "null",
            "Any": "any",
            "Optional": "any | null",
        }
        
        # 处理泛型类型
        if "[" in python_type and "]" in python_type:
            base_type = python_type.split("[")[0]
            inner_type = python_type.split("[")[1].split("]")[0]
            
            if base_type == "List" or base_type == "list":
                return f"Array<{self._convert_python_type_to_ts(inner_type)}>"
            elif base_type == "Dict" or base_type == "dict":
                if "," in inner_type:
                    key_type, value_type = inner_type.split(",", 1)
                    return f"Record<{self._convert_python_type_to_ts(key_type.strip())}, {self._convert_python_type_to_ts(value_type.strip())}>"
                else:
                    return "Record<string, any>"
            elif base_type == "Set" or base_type == "set":
                return f"Set<{self._convert_python_type_to_ts(inner_type)}>"
            elif base_type == "Tuple" or base_type == "tuple":
                if "," in inner_type:
                    types = [self._convert_python_type_to_ts(t.strip()) for t in inner_type.split(",")]
                    return f"[{', '.join(types)}]"
                else:
                    return f"[{self._convert_python_type_to_ts(inner_type)}]"
            elif base_type == "Optional":
                return f"{self._convert_python_type_to_ts(inner_type)} | null"
            else:
                return "any"
        
        # 处理联合类型
        if "|" in python_type:
            types = [self._convert_python_type_to_ts(t.strip()) for t in python_type.split("|")]
            return " | ".join(types)
        
        # 基本类型映射
        return type_mapping.get(python_type, "any")


class JavaScriptToPythonFunctionConverter(FunctionFlowConverter):
    """JavaScript到Python的函数控制流转换器"""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT, LanguageType.PYTHON)
    
    def convert_regular_function(self, function_node: FunctionFlowNode) -> str:
        """转换普通函数"""
        if not isinstance(function_node, FunctionFlowNode):
            return ""
        
        # 提取函数信息
        function_name = function_node.function_name or "unnamed_function"
        params = function_node.parameters or []
        body = function_node.function_body or ""
        
        # 转换参数
        py_params = []
        for param in params:
            param_name = param.get("name", "")
            param_default = param.get("default", None)
            
            if param_default is not None:
                # 转换默认值
                py_default = self._convert_js_value_to_python(param_default)
                py_params.append(f"{param_name}={py_default}")
            else:
                py_params.append(param_name)
        
        # 构建Python函数
        py_function = f"def {function_name}({', '.join(py_params)}):\n"
        
        # 转换函数体
        py_body = self._convert_function_body(body)
        
        # 添加函数体
        py_function += py_body
        
        return py_function
    
    def convert_async_function(self, function_node: FunctionFlowNode) -> str:
        """转换异步函数"""
        if not isinstance(function_node, FunctionFlowNode):
            return ""
        
        # 提取函数信息
        function_name = function_node.function_name or "unnamed_async_function"
        params = function_node.parameters or []
        body = function_node.function_body or ""
        
        # 转换参数
        py_params = []
        for param in params:
            param_name = param.get("name", "")
            param_default = param.get("default", None)
            
            if param_default is not None:
                # 转换默认值
                py_default = self._convert_js_value_to_python(param_default)
                py_params.append(f"{param_name}={py_default}")
            else:
                py_params.append(param_name)
        
        # 构建Python异步函数
        py_function = f"async def {function_name}({', '.join(py_params)}):\n"
        
        # 转换函数体，处理异步特性
        py_body = self._convert_async_function_body(body)
        
        # 添加函数体
        py_function += py_body
        
        return py_function
    
    def convert_generator_function(self, function_node: FunctionFlowNode) -> str:
        """转换生成器函数"""
        if not isinstance(function_node, FunctionFlowNode):
            return ""
        
        # 提取函数信息
        function_name = function_node.function_name or "unnamed_generator"
        params = function_node.parameters or []
        body = function_node.function_body or ""
        
        # 转换参数
        py_params = []
        for param in params:
            param_name = param.get("name", "")
            param_default = param.get("default", None)
            
            if param_default is not None:
                # 转换默认值
                py_default = self._convert_js_value_to_python(param_default)
                py_params.append(f"{param_name}={py_default}")
            else:
                py_params.append(param_name)
        
        # 构建Python生成器函数
        py_function = f"def {function_name}({', '.join(py_params)}):\n"
        
        # 转换函数体，处理生成器特性
        py_body = self._convert_generator_function_body(body)
        
        # 添加函数体
        py_function += py_body
        
        return py_function
    
    def convert_tail_recursive_function(self, function_node: FunctionFlowNode) -> str:
        """转换尾递归函数"""
        if not isinstance(function_node, FunctionFlowNode):
            return ""
        
        # 提取函数信息
        function_name = function_node.function_name or "unnamed_recursive_function"
        params = function_node.parameters or []
        body = function_node.function_body or ""
        
        # 转换参数
        py_params = []
        for param in params:
            param_name = param.get("name", "")
            param_default = param.get("default", None)
            
            if param_default is not None:
                # 转换默认值
                py_default = self._convert_js_value_to_python(param_default)
                py_params.append(f"{param_name}={py_default}")
            else:
                py_params.append(param_name)
        
        # 构建Python函数
        py_function = f"def {function_name}({', '.join(py_params)}):\n"
        
        # 添加尾递归优化装饰器的注释
        py_function = f"# 使用functools.lru_cache或自定义尾递归优化装饰器可以提高性能\n" + py_function
        
        # 转换函数体
        py_body = self._convert_function_body(body)
        
        # 添加函数体
        py_function += py_body
        
        return py_function
    
    def _convert_function_body(self, body: str) -> str:
        """转换函数体"""
        if not body:
            return "    # Empty function body\n    pass"
        
        # 分行处理
        lines = body.split('\n')
        py_lines = []
        
        for line in lines:
            # 跳过空行
            if not line.strip():
                py_lines.append("")
                continue
            
            # 移除行尾分号
            line = line.rstrip(';')
            
            # 转换console.log语句
            if "console.log(" in line:
                py_line = re.sub(r'console\.log\((.*?)\)', r'print(\1)', line)
                py_lines.append("    " + py_line)
                continue
            
            # 转换JavaScript的比较运算符
            py_line = line
            py_line = py_line.replace("===", "==")
            py_line = py_line.replace("!==", "!=")
            py_line = py_line.replace("true", "True")
            py_line = py_line.replace("false", "False")
            py_line = py_line.replace("null", "None")
            py_line = py_line.replace("undefined", "None")
            
            # 转换逻辑运算符
            py_line = py_line.replace("&&", "and")
            py_line = py_line.replace("||", "or")
            py_line = py_line.replace("!", "not ")
            
            # 转换数组方法
            array_method_match = re.search(r'(.*?)\.map\((.*?) => (.*?)\)', py_line)
            if array_method_match:
                array = array_method_match.group(1)
                var = array_method_match.group(2)
                expr = array_method_match.group(3)
                py_line = py_line.replace(
                    array_method_match.group(0),
                    f"[{expr} for {var} in {array}]"
                )
            
            filter_method_match = re.search(r'(.*?)\.filter\((.*?) => (.*?)\)', py_line)
            if filter_method_match:
                array = filter_method_match.group(1)
                var = filter_method_match.group(2)
                condition = filter_method_match.group(3)
                py_line = py_line.replace(
                    filter_method_match.group(0),
                    f"[{var} for {var} in {array} if {condition}]"
                )
            
            # 添加缩进
            py_lines.append("    " + py_line)
        
        return "\n".join(py_lines)
    
    def _convert_async_function_body(self, body: str) -> str:
        """转换异步函数体"""
        if not body:
            return "    # Empty async function body\n    pass"
        
        # 分行处理
        lines = body.split('\n')
        py_lines = []
        
        for line in lines:
            # 跳过空行
            if not line.strip():
                py_lines.append("")
                continue
            
            # 移除行尾分号
            line = line.rstrip(';')
            
            # 转换fetch调用
            if "fetch(" in line:
                # 转换为aiohttp或httpx调用
                py_lines.append("    # 需要导入: import aiohttp")
                py_lines.append("    async with aiohttp.ClientSession() as session:")
                
                fetch_match = re.search(r'(?:const|let|var)?\s*(\w+)\s*=\s*await\s+fetch\((.*?)\)', line)
                if fetch_match:
                    var_name = fetch_match.group(1)
                    url = fetch_match.group(2)
                    py_lines.append(f"        async with session.get({url}) as {var_name}:")
                    continue
            
            # 转换response.json()
            if ".json()" in line:
                json_match = re.search(r'(?:const|let|var)?\s*(\w+)\s*=\s*await\s+(.*?)\.json\(\)', line)
                if json_match:
                    var_name = json_match.group(1)
                    response = json_match.group(2)
                    py_lines.append(f"            {var_name} = await {response}.json()")
                    continue
            
            # 转换await表达式
            if "await" in line:
                # Python也使用await关键字
                py_line = line
            else:
                # 使用普通函数体转换逻辑
                py_line = line
                
                # 转换console.log语句
                if "console.log(" in py_line:
                    py_line = re.sub(r'console\.log\((.*?)\)', r'print(\1)', py_line)
                
                # 转换JavaScript的比较运算符
                py_line = py_line.replace("===", "==")
                py_line = py_line.replace("!==", "!=")
                py_line = py_line.replace("true", "True")
                py_line = py_line.replace("false", "False")
                py_line = py_line.replace("null", "None")
                py_line = py_line.replace("undefined", "None")
                
                # 转换逻辑运算符
                py_line = py_line.replace("&&", "and")
                py_line = py_line.replace("||", "or")
                py_line = py_line.replace("!", "not ")
            
            # 添加缩进
            py_lines.append("    " + py_line)
        
        return "\n".join(py_lines)
    
    def _convert_generator_function_body(self, body: str) -> str:
        """转换生成器函数体"""
        if not body:
            return "    # Empty generator function body\n    yield None"
        
        # 分行处理
        lines = body.split('\n')
        py_lines = []
        
        for line in lines:
            # 跳过空行
            if not line.strip():
                py_lines.append("")
                continue
            
            # 移除行尾分号
            line = line.rstrip(';')
            
            # 转换yield表达式
            if "yield" in line:
                # Python也使用yield关键字
                py_line = line
                
                # 转换yield*为yield from
                py_line = py_line.replace("yield*", "yield from")
            else:
                # 使用普通函数体转换逻辑
                py_line = line
                
                # 转换console.log语句
                if "console.log(" in py_line:
                    py_line = re.sub(r'console\.log\((.*?)\)', r'print(\1)', py_line)
                
                # 转换JavaScript的比较运算符
                py_line = py_line.replace("===", "==")
                py_line = py_line.replace("!==", "!=")
                py_line = py_line.replace("true", "True")
                py_line = py_line.replace("false", "False")
                py_line = py_line.replace("null", "None")
                py_line = py_line.replace("undefined", "None")
                
                # 转换逻辑运算符
                py_line = py_line.replace("&&", "and")
                py_line = py_line.replace("||", "or")
                py_line = py_line.replace("!", "not ")
            
            # 添加缩进
            py_lines.append("    " + py_line)
        
        return "\n".join(py_lines)
    
    def _convert_js_value_to_python(self, value: str) -> str:
        """转换JavaScript值到Python值"""
        if value == "null" or value == "undefined":
            return "None"
        elif value == "true":
            return "True"
        elif value == "false":
            return "False"
        elif value.startswith("[") and value.endswith("]"):
            # 数组转换
            return value
        elif value.startswith("{") and value.endswith("}"):
            # 对象转换
            return value
        elif value.startswith("'") and value.endswith("'"):
            # 字符串转换
            return value
        elif value.startswith('"') and value.endswith('"'):
            # 字符串转换
            return value
        else:
            # 默认不变
            return value


# 工厂函数，创建适合的函数控制流转换器
def create_function_converter(source_language: LanguageType, target_language: LanguageType) -> FunctionFlowConverter:
    """创建适合的函数控制流转换器"""
    if source_language == LanguageType.PYTHON and target_language == LanguageType.JAVASCRIPT:
        return PythonToJavaScriptFunctionConverter()
    elif source_language == LanguageType.PYTHON and target_language == LanguageType.TYPESCRIPT:
        return PythonToTypeScriptFunctionConverter()
    elif source_language == LanguageType.JAVASCRIPT and target_language == LanguageType.PYTHON:
        return JavaScriptToPythonFunctionConverter()
    # 添加其他语言对的转换器...
    else:
        raise ValueError(f"不支持从 {source_language} 到 {target_language} 的函数控制流转换")


# 示例使用
def convert_function_flow(source_code: str, source_language: str, target_language: str) -> str:
    """转换函数控制流代码"""
    try:
        source_lang = LanguageType(source_language.lower())
        target_lang = LanguageType(target_language.lower())
        
        # 创建解析器
        from .core import PythonControlFlowParser, JavaScriptControlFlowParser
        
        parsers = {
            LanguageType.PYTHON: PythonControlFlowParser(),
            LanguageType.JAVASCRIPT: JavaScriptControlFlowParser(),
            # 其他语言的解析器...
        }
        
        parser = parsers.get(source_lang)
        if not parser:
            raise ValueError(f"不支持的源语言: {source_lang}")
        
        # 解析源代码
        graph = parser.parse(source_code)
        
        # 获取所有函数控制流节点
        function_nodes = graph.get_nodes_by_type(ControlFlowType.FUNCTION)
        
        # 创建转换器
        converter = create_function_converter(source_lang, target_lang)
        
        # 转换所有函数控制流节点
        result = []
        for node in function_nodes:
            if isinstance(node, FunctionFlowNode):
                converted = converter.convert_function_flow(node)
                result.append(converted)
        
        # 如果没有找到函数控制流节点，尝试直接转换整个代码
        if not result and "function" in source_code or "def" in source_code:
            # 创建一个模拟的函数控制流节点
            function_node = FunctionFlowNode(
                node_type=ControlFlowType.FUNCTION,
                source_language=source_lang,
                source_code=source_code
            )
            
            # 尝试解析函数定义
            if source_lang == LanguageType.PYTHON:
                # 解析Python函数
                func_match = re.search(r'(async\s+)?def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(.*?))?\s*:(.*?)(?:$|def\s+)', source_code, re.DOTALL)
                if func_match:
                    is_async = func_match.group(1) is not None
                    function_name = func_match.group(2)
                    params_str = func_match.group(3)
                    return_type = func_match.group(4)
                    function_body = func_match.group(5)
                    
                    function_node.function_name = function_name
                    function_node.function_type = "async" if is_async else "regular"
                    function_node.return_type = return_type
                    function_node.function_body = function_body
                    
                    # 解析参数
                    params = []
                    if params_str:
                        param_list = params_str.split(",")
                        for param in param_list:
                            param = param.strip()
                            if "=" in param:
                                name, default = param.split("=", 1)
                                params.append({"name": name.strip(), "default": default.strip()})
                            elif ":" in param:
                                name, type_hint = param.split(":", 1)
                                params.append({"name": name.strip(), "type": type_hint.strip()})
                            else:
                                params.append({"name": param})
                    
                    function_node.parameters = params
                    
                    # 检查是否为生成器函数
                    if "yield" in function_body:
                        function_node.function_type = "generator"
                    
                    # 检查是否为尾递归函数
                    if f"return {function_name}(" in function_body:
                        function_node.function_type = "tail_recursive"
            
            elif source_lang == LanguageType.JAVASCRIPT:
                # 解析JavaScript函数
                func_match = re.search(r'(async\s+)?function(\*?)\s+(\w+)\s*\((.*?)\)\s*{(.*?)}', source_code, re.DOTALL)
                if func_match:
                    is_async = func_match.group(1) is not None
                    is_generator = func_match.group(2) == "*"
                    function_name = func_match.group(3)
                    params_str = func_match.group(4)
                    function_body = func_match.group(5)
                    
                    function_node.function_name = function_name
                    if is_async:
                        function_node.function_type = "async"
                    elif is_generator:
                        function_node.function_type = "generator"
                    else:
                        function_node.function_type = "regular"
                    
                    function_node.function_body = function_body
                    
                    # 解析参数
                    params = []
                    if params_str:
                        param_list = params_str.split(",")
                        for param in param_list:
                            param = param.strip()
                            if "=" in param:
                                name, default = param.split("=", 1)
                                params.append({"name": name.strip(), "default": default.strip()})
                            else:
                                params.append({"name": param})
                    
                    function_node.parameters = params
                    
                    # 检查是否为尾递归函数
                    if f"return {function_name}(" in function_body:
                        function_node.function_type = "tail_recursive"
            
            converted = converter.convert_function_flow(function_node)
            result.append(converted)
        
        return "\n\n".join(result)
    except ValueError as e:
        return f"错误: {str(e)}"
    except Exception as e:
        return f"转换失败: {str(e)}"