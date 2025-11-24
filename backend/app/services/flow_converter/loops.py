"""
循环结构跨语言转换模块

提供各种编程语言间循环结构的转换功能，包括：
- for循环转换
- while循环转换
- 迭代器和生成器转换
"""
from typing import Dict, List, Optional, Any, Tuple
import re
import ast
from .core import (
    LanguageType, ControlFlowType, ControlFlowNode, LoopNode,
    ControlFlowGraph, ControlFlowConverter
)


class LoopConverter:
    """循环结构转换器基类"""
    
    def __init__(self, source_language: LanguageType, target_language: LanguageType):
        self.source_language = source_language
        self.target_language = target_language
    
    def convert_loop(self, loop_node: LoopNode) -> str:
        """转换循环结构"""
        if loop_node.loop_type == "for":
            return self.convert_for_loop(loop_node)
        elif loop_node.loop_type == "while":
            return self.convert_while_loop(loop_node)
        elif loop_node.loop_type == "foreach":
            return self.convert_foreach_loop(loop_node)
        elif loop_node.loop_type == "do_while":
            return self.convert_do_while_loop(loop_node)
        else:
            return loop_node.source_code  # 默认返回原始代码
    
    def convert_for_loop(self, loop_node: LoopNode) -> str:
        """转换for循环"""
        raise NotImplementedError("子类必须实现此方法")
    
    def convert_while_loop(self, loop_node: LoopNode) -> str:
        """转换while循环"""
        raise NotImplementedError("子类必须实现此方法")
    
    def convert_foreach_loop(self, loop_node: LoopNode) -> str:
        """转换foreach循环"""
        raise NotImplementedError("子类必须实现此方法")
    
    def convert_do_while_loop(self, loop_node: LoopNode) -> str:
        """转换do-while循环"""
        raise NotImplementedError("子类必须实现此方法")


class PythonToJavaScriptLoopConverter(LoopConverter):
    """Python到JavaScript的循环结构转换器"""
    
    def __init__(self):
        super().__init__(LanguageType.PYTHON, LanguageType.JAVASCRIPT)
    
    def convert_for_loop(self, loop_node: LoopNode) -> str:
        """转换Python的for循环到JavaScript"""
        # 检查是否是range循环
        if loop_node.iterable and "range(" in loop_node.iterable:
            return self._convert_range_loop(loop_node)
        else:
            return self._convert_for_in_loop(loop_node)
    
    def _convert_range_loop(self, loop_node: LoopNode) -> str:
        """转换Python的range循环到JavaScript的for循环"""
        # 解析range参数
        range_match = re.search(r'range\((.*?)\)', loop_node.iterable)
        if not range_match:
            return self._convert_for_in_loop(loop_node)  # 回退到for-in转换
        
        range_args = range_match.group(1).split(',')
        range_args = [arg.strip() for arg in range_args]
        
        # 根据参数数量确定起始值、结束值和步长
        if len(range_args) == 1:
            # range(stop)
            start = "0"
            stop = range_args[0]
            step = "1"
        elif len(range_args) == 2:
            # range(start, stop)
            start = range_args[0]
            stop = range_args[1]
            step = "1"
        else:
            # range(start, stop, step)
            start = range_args[0]
            stop = range_args[1]
            step = range_args[2]
        
        # 处理负步长的情况
        comparison_op = "<" if step == "1" or (step.startswith("-") is False and step != "0") else ">"
        update_op = "+=" if comparison_op == "<" else "-="
        
        # 构建JavaScript的for循环
        js_for = f"for (let {loop_node.iterator} = {start}; {loop_node.iterator} {comparison_op} {stop}; {loop_node.iterator} {update_op} {step.lstrip('-')}) {{\n"
        
        # 转换循环体
        body = self._convert_loop_body(loop_node.body)
        js_for += body
        
        js_for += "}"
        return js_for
    
    def _convert_for_in_loop(self, loop_node: LoopNode) -> str:
        """转换Python的for-in循环到JavaScript的for-of循环"""
        # 检查是否是字典迭代
        if loop_node.iterable and any(x in loop_node.iterable for x in [".items()", ".keys()", ".values()"]):
            return self._convert_dict_iteration(loop_node)
        
        # 普通的可迭代对象转换为for-of
        js_for = f"for (const {loop_node.iterator} of {loop_node.iterable}) {{\n"
        
        # 转换循环体
        body = self._convert_loop_body(loop_node.body)
        js_for += body
        
        js_for += "}"
        return js_for
    
    def _convert_dict_iteration(self, loop_node: LoopNode) -> str:
        """转换Python的字典迭代到JavaScript"""
        iterable = loop_node.iterable
        
        if ".items()" in iterable:
            # 字典的items()方法
            dict_name = iterable.replace(".items()", "")
            
            # 检查迭代器是否是元组解包
            if "," in loop_node.iterator:
                # 例如: for key, value in dict.items()
                key, value = [i.strip() for i in loop_node.iterator.split(",", 1)]
                js_for = f"for (const [{key}, {value}] of Object.entries({dict_name})) {{\n"
            else:
                # 例如: for item in dict.items()
                js_for = f"for (const {loop_node.iterator} of Object.entries({dict_name})) {{\n"
        
        elif ".keys()" in iterable:
            # 字典的keys()方法
            dict_name = iterable.replace(".keys()", "")
            js_for = f"for (const {loop_node.iterator} of Object.keys({dict_name})) {{\n"
        
        elif ".values()" in iterable:
            # 字典的values()方法
            dict_name = iterable.replace(".values()", "")
            js_for = f"for (const {loop_node.iterator} of Object.values({dict_name})) {{\n"
        
        else:
            # 默认情况，回退到普通for-of
            return self._convert_for_in_loop(loop_node)
        
        # 转换循环体
        body = self._convert_loop_body(loop_node.body)
        js_for += body
        
        js_for += "}"
        return js_for
    
    def convert_while_loop(self, loop_node: LoopNode) -> str:
        """转换Python的while循环到JavaScript"""
        # 转换条件表达式
        condition = self._convert_condition(loop_node.condition)
        
        # 构建JavaScript的while循环
        js_while = f"while ({condition}) {{\n"
        
        # 转换循环体
        body = self._convert_loop_body(loop_node.body)
        js_while += body
        
        js_while += "}"
        return js_while
    
    def convert_foreach_loop(self, loop_node: LoopNode) -> str:
        """Python没有显式的foreach循环，但可以处理类似的for-in循环"""
        return self._convert_for_in_loop(loop_node)
    
    def convert_do_while_loop(self, loop_node: LoopNode) -> str:
        """Python没有do-while循环，但可以处理转换为JavaScript的do-while"""
        # 由于Python没有do-while，这个方法通常不会被直接调用
        # 但为了完整性，我们提供一个实现
        
        # 转换条件表达式
        condition = self._convert_condition(loop_node.condition)
        
        # 构建JavaScript的do-while循环
        js_do_while = f"do {{\n"
        
        # 转换循环体
        body = self._convert_loop_body(loop_node.body)
        js_do_while += body
        
        js_do_while += f"}} while ({condition});"
        return js_do_while
    
    def _convert_loop_body(self, body: str) -> str:
        """转换循环体中的Python代码到JavaScript"""
        if not body:
            return "  // Empty loop body\n"
        
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
            
            # 转换列表推导式
            list_comp_match = re.search(r'\[(.*) for (.*) in (.*) if (.*)\]', line)
            if list_comp_match:
                expr, var, iterable, condition = list_comp_match.groups()
                js_line = f"{iterable}.filter({var} => {condition}).map({var} => {expr})"
                js_lines.append("  " + line.replace(list_comp_match.group(0), js_line))
                continue
            
            # 转换Python的比较运算符
            js_line = line
            js_line = js_line.replace(" == ", " === ")
            js_line = js_line.replace(" != ", " !== ")
            js_line = js_line.replace("True", "true")
            js_line = js_line.replace("False", "false")
            js_line = js_line.replace("None", "null")
            
            # 添加缩进
            js_lines.append("  " + js_line)
        
        return "\n".join(js_lines) + "\n"
    
    def _convert_condition(self, condition: str) -> str:
        """转换Python的条件表达式到JavaScript"""
        if not condition:
            return "true"
        
        # 转换比较运算符
        js_condition = condition
        js_condition = js_condition.replace(" == ", " === ")
        js_condition = js_condition.replace(" != ", " !== ")
        js_condition = js_condition.replace("True", "true")
        js_condition = js_condition.replace("False", "false")
        js_condition = js_condition.replace("None", "null")
        
        # 转换逻辑运算符
        js_condition = js_condition.replace(" and ", " && ")
        js_condition = js_condition.replace(" or ", " || ")
        js_condition = js_condition.replace("not ", "!")
        
        return js_condition


class PythonToTypeScriptLoopConverter(PythonToJavaScriptLoopConverter):
    """Python到TypeScript的循环结构转换器"""
    
    def __init__(self):
        super().__init__()
        self.target_language = LanguageType.TYPESCRIPT
    
    def _convert_for_in_loop(self, loop_node: LoopNode) -> str:
        """转换Python的for-in循环到TypeScript的for-of循环，添加类型注解"""
        # 基本实现与JavaScript相同，但可以添加类型注解
        js_for = super()._convert_for_in_loop(loop_node)
        
        # 将JavaScript的"const"替换为带类型注解的形式
        # 注意：这是一个简化的实现，实际应用中应该进行类型推断
        if "const " in js_for:
            # 简单替换，假设迭代的是number[]类型
            js_for = js_for.replace(f"const {loop_node.iterator}", f"const {loop_node.iterator}: any")
        
        return js_for


class PythonToCppLoopConverter(LoopConverter):
    """Python到C++的循环结构转换器"""
    
    def __init__(self):
        super().__init__(LanguageType.PYTHON, LanguageType.CPP)
    
    def convert_for_loop(self, loop_node: LoopNode) -> str:
        """转换Python的for循环到C++"""
        # 检查是否是range循环
        if loop_node.iterable and "range(" in loop_node.iterable:
            return self._convert_range_loop(loop_node)
        else:
            return self._convert_for_in_loop(loop_node)
    
    def _convert_range_loop(self, loop_node: LoopNode) -> str:
        """转换Python的range循环到C++的for循环"""
        # 解析range参数
        range_match = re.search(r'range\((.*?)\)', loop_node.iterable)
        if not range_match:
            return self._convert_for_in_loop(loop_node)  # 回退到for-in转换
        
        range_args = range_match.group(1).split(',')
        range_args = [arg.strip() for arg in range_args]
        
        # 根据参数数量确定起始值、结束值和步长
        if len(range_args) == 1:
            # range(stop)
            start = "0"
            stop = range_args[0]
            step = "1"
        elif len(range_args) == 2:
            # range(start, stop)
            start = range_args[0]
            stop = range_args[1]
            step = "1"
        else:
            # range(start, stop, step)
            start = range_args[0]
            stop = range_args[1]
            step = range_args[2]
        
        # 处理负步长的情况
        comparison_op = "<" if step == "1" or (step.startswith("-") is False and step != "0") else ">"
        update_op = "+=" if comparison_op == "<" else "-="
        
        # 构建C++的for循环
        cpp_for = f"for (int {loop_node.iterator} = {start}; {loop_node.iterator} {comparison_op} {stop}; {loop_node.iterator} {update_op} {step.lstrip('-')}) {{\n"
        
        # 转换循环体
        body = self._convert_loop_body(loop_node.body)
        cpp_for += body
        
        cpp_for += "}"
        return cpp_for
    
    def _convert_for_in_loop(self, loop_node: LoopNode) -> str:
        """转换Python的for-in循环到C++的范围for循环"""
        # 检查是否是字典迭代
        if loop_node.iterable and any(x in loop_node.iterable for x in [".items()", ".keys()", ".values()"]):
            return self._convert_dict_iteration(loop_node)
        
        # 普通的可迭代对象转换为范围for
        # 注意：这是一个简化的实现，实际应用中需要更复杂的类型推断
        cpp_for = f"for (const auto& {loop_node.iterator} : {loop_node.iterable}) {{\n"
        
        # 转换循环体
        body = self._convert_loop_body(loop_node.body)
        cpp_for += body
        
        cpp_for += "}"
        return cpp_for
    
    def _convert_dict_iteration(self, loop_node: LoopNode) -> str:
        """转换Python的字典迭代到C++"""
        iterable = loop_node.iterable
        
        if ".items()" in iterable:
            # 字典的items()方法
            dict_name = iterable.replace(".items()", "")
            
            # 检查迭代器是否是元组解包
            if "," in loop_node.iterator:
                # 例如: for key, value in dict.items()
                key, value = [i.strip() for i in loop_node.iterator.split(",", 1)]
                cpp_for = f"for (const auto& [{key}, {value}] : {dict_name}) {{\n"
            else:
                # 例如: for item in dict.items()
                cpp_for = f"for (const auto& {loop_node.iterator} : {dict_name}) {{\n"
        
        elif ".keys()" in iterable:
            # 字典的keys()方法
            dict_name = iterable.replace(".keys()", "")
            cpp_for = f"for (const auto& {loop_node.iterator} : {dict_name}) {{\n"
        
        elif ".values()" in iterable:
            # 字典的values()方法
            dict_name = iterable.replace(".values()", "")
            cpp_for = f"for (const auto& {loop_node.iterator} : {dict_name}) {{\n"
        
        else:
            # 默认情况，回退到普通范围for
            return self._convert_for_in_loop(loop_node)
        
        # 转换循环体
        body = self._convert_loop_body(loop_node.body)
        cpp_for += body
        
        cpp_for += "}"
        return cpp_for
    
    def convert_while_loop(self, loop_node: LoopNode) -> str:
        """转换Python的while循环到C++"""
        # 转换条件表达式
        condition = self._convert_condition(loop_node.condition)
        
        # 构建C++的while循环
        cpp_while = f"while ({condition}) {{\n"
        
        # 转换循环体
        body = self._convert_loop_body(loop_node.body)
        cpp_while += body
        
        cpp_while += "}"
        return cpp_while
    
    def convert_foreach_loop(self, loop_node: LoopNode) -> str:
        """Python没有显式的foreach循环，但可以处理类似的for-in循环"""
        return self._convert_for_in_loop(loop_node)
    
    def convert_do_while_loop(self, loop_node: LoopNode) -> str:
        """Python没有do-while循环，但可以处理转换为C++的do-while"""
        # 转换条件表达式
        condition = self._convert_condition(loop_node.condition)
        
        # 构建C++的do-while循环
        cpp_do_while = f"do {{\n"
        
        # 转换循环体
        body = self._convert_loop_body(loop_node.body)
        cpp_do_while += body
        
        cpp_do_while += f"}} while ({condition});"
        return cpp_do_while
    
    def _convert_loop_body(self, body: str) -> str:
        """转换循环体中的Python代码到C++"""
        if not body:
            return "  // Empty loop body\n"
        
        # 分行处理
        lines = body.split('\n')
        cpp_lines = []
        
        for line in lines:
            # 跳过空行
            if not line.strip():
                cpp_lines.append("")
                continue
            
            # 转换print语句
            if "print(" in line:
                cpp_line = re.sub(r'print\((.*)\)', r'std::cout << \1 << std::endl', line)
                cpp_lines.append("  " + cpp_line)
                continue
            
            # 转换列表推导式
            list_comp_match = re.search(r'\[(.*) for (.*) in (.*) if (.*)\]', line)
            if list_comp_match:
                # 列表推导式在C++中需要使用算法库，这里简化处理
                cpp_lines.append("  // TODO: Convert list comprehension to C++ algorithm")
                cpp_lines.append("  " + line + " // Original Python code")
                continue
            
            # 转换Python的比较运算符
            cpp_line = line
            cpp_line = cpp_line.replace("True", "true")
            cpp_line = cpp_line.replace("False", "false")
            cpp_line = cpp_line.replace("None", "nullptr")
            
            # 转换逻辑运算符
            cpp_line = cpp_line.replace(" and ", " && ")
            cpp_line = cpp_line.replace(" or ", " || ")
            cpp_line = cpp_line.replace("not ", "!")
            
            # 添加缩进
            cpp_lines.append("  " + cpp_line)
        
        return "\n".join(cpp_lines) + "\n"
    
    def _convert_condition(self, condition: str) -> str:
        """转换Python的条件表达式到C++"""
        if not condition:
            return "true"
        
        # 转换比较运算符
        cpp_condition = condition
        cpp_condition = cpp_condition.replace("True", "true")
        cpp_condition = cpp_condition.replace("False", "false")
        cpp_condition = cpp_condition.replace("None", "nullptr")
        
        # 转换逻辑运算符
        cpp_condition = cpp_condition.replace(" and ", " && ")
        cpp_condition = cpp_condition.replace(" or ", " || ")
        cpp_condition = cpp_condition.replace("not ", "!")
        
        return cpp_condition


class JavaScriptToPythonLoopConverter(LoopConverter):
    """JavaScript到Python的循环结构转换器"""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT, LanguageType.PYTHON)
    
    def convert_for_loop(self, loop_node: LoopNode) -> str:
        """转换JavaScript的for循环到Python"""
        # 检查是否是标准for循环
        if loop_node.source_code and "for (" in loop_node.source_code:
            # 尝试解析标准for循环的三个部分
            for_parts_match = re.search(r'for\s*\(\s*(.*?);\s*(.*?);\s*(.*?)\s*\)', loop_node.source_code)
            if for_parts_match:
                init, condition, update = for_parts_match.groups()
                return self._convert_standard_for_loop(init, condition, update, loop_node.body)
        
        # 检查是否是for-of循环
        if loop_node.source_code and "for (" in loop_node.source_code and " of " in loop_node.source_code:
            for_of_match = re.search(r'for\s*\(\s*(const|let|var)?\s*(\w+)\s+of\s+(.*?)\s*\)', loop_node.source_code)
            if for_of_match:
                _, var_name, iterable = for_of_match.groups()
                return self._convert_for_of_loop(var_name, iterable, loop_node.body)
        
        # 检查是否是for-in循环
        if loop_node.source_code and "for (" in loop_node.source_code and " in " in loop_node.source_code:
            for_in_match = re.search(r'for\s*\(\s*(const|let|var)?\s*(\w+)\s+in\s+(.*?)\s*\)', loop_node.source_code)
            if for_in_match:
                _, var_name, iterable = for_in_match.groups()
                return self._convert_for_in_loop(var_name, iterable, loop_node.body)
        
        # 默认情况，保留原始代码并添加注释
        return f"# TODO: Convert JavaScript loop to Python\n# Original: {loop_node.source_code}\n"
    
    def _convert_standard_for_loop(self, init: str, condition: str, update: str, body: str) -> str:
        """转换JavaScript的标准for循环到Python的range循环"""
        # 尝试解析初始化、条件和更新部分
        init_match = re.search(r'(let|var|const)?\s*(\w+)\s*=\s*(\d+)', init)
        condition_match = re.search(r'(\w+)\s*([<>]=?)\s*(\w+|\d+)', condition)
        update_match = re.search(r'(\w+)\s*([\+\-]=)\s*(\d+)', update)
        
        if init_match and condition_match and update_match:
            # 提取循环变量和范围
            _, var_name, start = init_match.groups()
            cond_var, cond_op, end = condition_match.groups()
            update_var, update_op, step = update_match.groups()
            
            # 确保所有变量名一致
            if var_name == cond_var == update_var:
                # 确定步长
                step_value = int(step)
                if update_op == "-=":
                    step_value = -step_value
                
                # 确定结束值
                if cond_op == "<":
                    # 不包含结束值
                    range_end = end
                elif cond_op == "<=":
                    # 包含结束值，需要+1
                    try:
                        range_end = str(int(end) + 1)
                    except ValueError:
                        range_end = f"({end} + 1)"
                elif cond_op == ">":
                    # 反向循环，不包含结束值
                    range_end = end
                elif cond_op == ">=":
                    # 反向循环，包含结束值，需要-1
                    try:
                        range_end = str(int(end) - 1)
                    except ValueError:
                        range_end = f"({end} - 1)"
                else:
                    # 无法确定范围，使用while循环
                    return self._fallback_to_while(init, condition, update, body)
                
                # 构建Python的range循环
                if step_value == 1:
                    py_for = f"for {var_name} in range({start}, {range_end}):\n"
                else:
                    py_for = f"for {var_name} in range({start}, {range_end}, {step_value}):\n"
                
                # 转换循环体
                body_lines = body.split('\n')
                indented_body = "\n".join(f"    {line}" for line in body_lines)
                py_for += indented_body
                
                return py_for
        
        # 无法解析为range循环，回退到while循环
        return self._fallback_to_while(init, condition, update, body)
    
    def _fallback_to_while(self, init: str, condition: str, update: str, body: str) -> str:
        """当无法转换为range循环时，回退到while循环"""
        # 转换初始化部分
        py_init = init.replace("let ", "").replace("var ", "").replace("const ", "")
        
        # 转换条件部分
        py_condition = self._convert_js_condition(condition)
        
        # 转换更新部分
        py_update = update.replace("++", " += 1").replace("--", " -= 1")
        
        # 构建Python的while循环
        py_while = f"{py_init}\n"
        py_while += f"while {py_condition}:\n"
        
        # 转换循环体并添加更新语句
        body_lines = body.split('\n')
        indented_body = "\n".join(f"    {line}" for line in body_lines)
        py_while += indented_body
        py_while += f"\n    {py_update}"
        
        return py_while
    
    def _convert_for_of_loop(self, var_name: str, iterable: str, body: str) -> str:
        """转换JavaScript的for-of循环到Python的for-in循环"""
        # 处理特殊的迭代器方法
        if "Object.entries" in iterable:
            # 转换Object.entries为items()
            dict_name = re.search(r'Object\.entries\((.*?)\)', iterable)
            if dict_name:
                py_iterable = f"{dict_name.group(1)}.items()"
                
                # 检查变量是否是解构赋值
                if "[" in var_name and "]" in var_name:
                    # 例如: for (const [key, value] of Object.entries(obj))
                    key_value_match = re.search(r'\[(.*?),\s*(.*?)\]', var_name)
                    if key_value_match:
                        key, value = key_value_match.groups()
                        py_for = f"for {key}, {value} in {py_iterable}:\n"
                    else:
                        py_for = f"for {var_name} in {py_iterable}:\n"
                else:
                    py_for = f"for {var_name} in {py_iterable}:\n"
            else:
                py_for = f"for {var_name} in {iterable}:\n"
        
        elif "Object.keys" in iterable:
            # 转换Object.keys为keys()
            dict_name = re.search(r'Object\.keys\((.*?)\)', iterable)
            if dict_name:
                py_iterable = f"{dict_name.group(1)}.keys()"
                py_for = f"for {var_name} in {py_iterable}:\n"
            else:
                py_for = f"for {var_name} in {iterable}:\n"
        
        elif "Object.values" in iterable:
            # 转换Object.values为values()
            dict_name = re.search(r'Object\.values\((.*?)\)', iterable)
            if dict_name:
                py_iterable = f"{dict_name.group(1)}.values()"
                py_for = f"for {var_name} in {py_iterable}:\n"
            else:
                py_for = f"for {var_name} in {iterable}:\n"
        
        else:
            # 普通的for-of循环
            py_for = f"for {var_name} in {iterable}:\n"
        
        # 转换循环体
        body_lines = body.split('\n')
        indented_body = "\n".join(f"    {line}" for line in body_lines)
        py_for += indented_body
        
        return py_for
    
    def _convert_for_in_loop(self, var_name: str, iterable: str, body: str) -> str:
        """转换JavaScript的for-in循环到Python"""
        # JavaScript的for-in循环遍历对象的属性名，Python中最接近的是dict.keys()
        py_for = f"# JavaScript for-in loops iterate over property names\n"
        py_for += f"for {var_name} in {iterable}.keys():\n"
        
        # 转换循环体
        body_lines = body.split('\n')
        indented_body = "\n".join(f"    {line}" for line in body_lines)
        py_for += indented_body
        
        return py_for
    
    def convert_while_loop(self, loop_node: LoopNode) -> str:
        """转换JavaScript的while循环到Python"""
        # 转换条件表达式
        condition = self._convert_js_condition(loop_node.condition)
        
        # 构建Python的while循环
        py_while = f"while {condition}:\n"
        
        # 转换循环体
        body_lines = loop_node.body.split('\n')
        indented_body = "\n".join(f"    {line}" for line in body_lines)
        py_while += indented_body
        
        return py_while
    
    def convert_foreach_loop(self, loop_node: LoopNode) -> str:
        """JavaScript没有显式的foreach循环，但可以处理类似的forEach方法"""
        # 检查是否是forEach方法调用
        if loop_node.source_code and ".forEach(" in loop_node.source_code:
            foreach_match = re.search(r'(.*?)\.forEach\(\s*(?:function\s*\((.*?)\)|(?:\((.*?)\)|(\w+))\s*=>)\s*{(.*?)}', loop_node.source_code, re.DOTALL)
            if foreach_match:
                array, func_param1, func_param2, arrow_param, body = foreach_match.groups()
                param = func_param1 or func_param2 or arrow_param or "item"
                
                # 构建Python的for循环
                py_for = f"for {param} in {array}:\n"
                
                # 转换循环体
                body_lines = body.split('\n')
                indented_body = "\n".join(f"    {line.strip()}" for line in body_lines if line.strip())
                py_for += indented_body
                
                return py_for
        
        # 默认情况，保留原始代码并添加注释
        return f"# TODO: Convert JavaScript forEach to Python\n# Original: {loop_node.source_code}\n"
    
    def convert_do_while_loop(self, loop_node: LoopNode) -> str:
        """转换JavaScript的do-while循环到Python"""
        # Python没有do-while循环，需要模拟实现
        # 转换条件表达式
        condition = self._convert_js_condition(loop_node.condition)
        
        # 构建Python的do-while模拟
        py_do_while = "# Python doesn't have do-while loops, simulating with while True\n"
        py_do_while += "while True:\n"
        
        # 转换循环体
        body_lines = loop_node.body.split('\n')
        indented_body = "\n".join(f"    {line}" for line in body_lines)
        py_do_while += indented_body
        
        # 添加条件检查和break
        py_do_while += f"\n    if not ({condition}):\n"
        py_do_while += "        break\n"
        
        return py_do_while
    
    def _convert_js_condition(self, condition: str) -> str:
        """转换JavaScript的条件表达式到Python"""
        if not condition:
            return "True"
        
        # 转换比较运算符
        py_condition = condition
        py_condition = py_condition.replace("===", "==")
        py_condition = py_condition.replace("!==", "!=")
        py_condition = py_condition.replace("true", "True")
        py_condition = py_condition.replace("false", "False")
        py_condition = py_condition.replace("null", "None")
        
        # 转换逻辑运算符
        py_condition = py_condition.replace("&&", "and")
        py_condition = py_condition.replace("||", "or")
        py_condition = py_condition.replace("!", "not ")
        
        return py_condition


# 工厂函数，创建适合的循环转换器
def create_loop_converter(source_language: LanguageType, target_language: LanguageType) -> LoopConverter:
    """创建适合的循环结构转换器"""
    if source_language == LanguageType.PYTHON and target_language == LanguageType.JAVASCRIPT:
        return PythonToJavaScriptLoopConverter()
    elif source_language == LanguageType.PYTHON and target_language == LanguageType.TYPESCRIPT:
        return PythonToTypeScriptLoopConverter()
    elif source_language == LanguageType.PYTHON and target_language == LanguageType.CPP:
        return PythonToCppLoopConverter()
    elif source_language == LanguageType.JAVASCRIPT and target_language == LanguageType.PYTHON:
        return JavaScriptToPythonLoopConverter()
    # 添加其他语言对的转换器...
    else:
        raise ValueError(f"不支持从 {source_language} 到 {target_language} 的循环结构转换")


# 示例使用
def convert_loop(source_code: str, source_language: str, target_language: str) -> str:
    """转换循环结构"""
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
        
        # 获取所有循环节点
        loop_nodes = graph.get_nodes_by_type(ControlFlowType.LOOP)
        
        # 创建转换器
        converter = create_loop_converter(source_lang, target_lang)
        
        # 转换所有循环节点
        result = []
        for node in loop_nodes:
            if isinstance(node, LoopNode):
                converted = converter.convert_loop(node)
                result.append(converted)
        
        # 如果没有找到循环节点，尝试直接转换整个代码
        if not result:
            # 创建一个模拟的循环节点
            loop_node = LoopNode(
                node_type=ControlFlowType.LOOP,
                source_language=source_lang,
                source_code=source_code,
                loop_type="unknown"
            )
            
            # 尝试根据代码内容判断循环类型
            if "for " in source_code:
                loop_node.loop_type = "for"
            elif "while " in source_code:
                loop_node.loop_type = "while"
            
            converted = converter.convert_loop(loop_node)
            result.append(converted)
        
        return "\n\n".join(result)
    except ValueError as e:
        return f"错误: {str(e)}"
    except Exception as e:
        return f"转换失败: {str(e)}"