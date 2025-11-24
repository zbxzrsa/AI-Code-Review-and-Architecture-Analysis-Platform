"""
条件语句跨语言转换模块

提供各种编程语言间条件语句的转换功能，包括：
- if-elif-else链转换
- 模式匹配转换
- 条件表达式转换
"""
from typing import Dict, List, Optional, Any, Tuple
import re
import ast
from .core import (
    LanguageType, ControlFlowType, ControlFlowNode, ConditionNode,
    ControlFlowGraph, ControlFlowConverter
)


class ConditionConverter:
    """条件语句转换器基类"""
    
    def __init__(self, source_language: LanguageType, target_language: LanguageType):
        self.source_language = source_language
        self.target_language = target_language
    
    def convert_condition(self, condition_node: ConditionNode) -> str:
        """转换条件语句"""
        if condition_node.condition_type == "if":
            return self.convert_if_statement(condition_node)
        elif condition_node.condition_type == "switch":
            return self.convert_switch_statement(condition_node)
        elif condition_node.condition_type == "match":
            return self.convert_match_statement(condition_node)
        elif condition_node.condition_type == "ternary":
            return self.convert_ternary_operator(condition_node)
        else:
            return condition_node.source_code  # 默认返回原始代码
    
    def convert_if_statement(self, condition_node: ConditionNode) -> str:
        """转换if语句"""
        raise NotImplementedError("子类必须实现此方法")
    
    def convert_switch_statement(self, condition_node: ConditionNode) -> str:
        """转换switch语句"""
        raise NotImplementedError("子类必须实现此方法")
    
    def convert_match_statement(self, condition_node: ConditionNode) -> str:
        """转换match语句（模式匹配）"""
        raise NotImplementedError("子类必须实现此方法")
    
    def convert_ternary_operator(self, condition_node: ConditionNode) -> str:
        """转换三元运算符"""
        raise NotImplementedError("子类必须实现此方法")


class PythonToJavaScriptConditionConverter(ConditionConverter):
    """Python到JavaScript的条件语句转换器"""
    
    def __init__(self):
        super().__init__(LanguageType.PYTHON, LanguageType.JAVASCRIPT)
    
    def convert_if_statement(self, condition_node: ConditionNode) -> str:
        """转换Python的if语句到JavaScript"""
        if not isinstance(condition_node, ConditionNode):
            return ""
        
        # 转换条件表达式
        condition = self._convert_condition(condition_node.condition)
        
        # 构建JavaScript的if语句
        js_if = f"if ({condition}) {{\n"
        
        # 转换true分支
        if condition_node.true_branch:
            js_if += self._convert_branch(condition_node.true_branch)
        
        js_if += "}"
        
        # 处理false分支（可能是elif或else）
        if condition_node.false_branch:
            # 检查false分支是否是另一个if语句（elif的情况）
            if condition_node.false_branch.strip().startswith("if "):
                js_if += f" else {{\n"
                js_if += self._convert_branch(condition_node.false_branch)
                js_if += "}"
            else:
                js_if += f" else {{\n"
                js_if += self._convert_branch(condition_node.false_branch)
                js_if += "}"
        
        return js_if
    
    def convert_switch_statement(self, condition_node: ConditionNode) -> str:
        """Python没有switch语句，但可以处理转换为JavaScript的switch"""
        # 由于Python没有switch语句，这个方法通常不会被直接调用
        # 但为了完整性，我们提供一个实现
        
        # 检查是否有cases属性
        if not condition_node.cases:
            return f"// Cannot convert to switch: no cases defined\n{condition_node.source_code}"
        
        # 构建JavaScript的switch语句
        js_switch = f"switch ({condition_node.condition}) {{\n"
        
        # 添加case分支
        for case in condition_node.cases:
            case_value = case.get("value", "default")
            case_body = case.get("body", "")
            
            if case_value == "default":
                js_switch += f"  default:\n"
            else:
                js_switch += f"  case {case_value}:\n"
            
            # 添加case体
            case_body_lines = case_body.split('\n')
            for line in case_body_lines:
                js_switch += f"    {line}\n"
            
            # 添加break语句（如果没有）
            if not case_body.strip().endswith("break;"):
                js_switch += "    break;\n"
        
        js_switch += "}"
        return js_switch
    
    def convert_match_statement(self, condition_node: ConditionNode) -> str:
        """转换Python 3.10+的match语句到JavaScript"""
        # 由于match语句是Python 3.10+的特性，需要特殊处理
        
        # 检查源代码是否包含match语句
        if "match " not in condition_node.source_code:
            return f"// Cannot convert match statement: not found in source\n{condition_node.source_code}"
        
        # 尝试解析match语句
        match_pattern = r'match\s+(.*?):\s*\n((?:\s+case.*?\n(?:(?:\s+).*?\n)*)+)'
        match_match = re.search(match_pattern, condition_node.source_code, re.DOTALL)
        
        if not match_match:
            return f"// Cannot parse match statement\n{condition_node.source_code}"
        
        subject = match_match.group(1)
        cases_block = match_match.group(2)
        
        # 解析case语句
        case_pattern = r'case\s+(.*?):\s*\n((?:\s+.*?\n)*)'
        case_matches = re.finditer(case_pattern, cases_block, re.DOTALL)
        
        # 构建JavaScript的switch语句
        js_switch = f"switch ({subject}) {{\n"
        
        for case_match in case_matches:
            pattern = case_match.group(1)
            body = case_match.group(2)
            
            # 处理特殊模式
            if "_" in pattern:
                # 通配符模式转为default
                js_switch += f"  default:\n"
            elif "|" in pattern:
                # 多模式转为多个case
                patterns = [p.strip() for p in pattern.split("|")]
                for p in patterns:
                    js_switch += f"  case {p}:\n"
            else:
                js_switch += f"  case {pattern}:\n"
            
            # 添加case体
            body_lines = body.split('\n')
            for line in body_lines:
                if line.strip():
                    js_switch += f"    {line.strip()}\n"
            
            # 添加break语句（如果没有）
            if not body.strip().endswith("break;"):
                js_switch += "    break;\n"
        
        js_switch += "}"
        return js_switch
    
    def convert_ternary_operator(self, condition_node: ConditionNode) -> str:
        """转换Python的条件表达式到JavaScript的三元运算符"""
        # 检查是否是条件表达式
        if not (condition_node.condition and condition_node.true_branch and condition_node.false_branch):
            return f"// Cannot convert to ternary: missing components\n{condition_node.source_code}"
        
        # 转换条件表达式
        condition = self._convert_condition(condition_node.condition)
        true_value = condition_node.true_branch.strip()
        false_value = condition_node.false_branch.strip()
        
        # 构建JavaScript的三元运算符
        js_ternary = f"({condition}) ? {true_value} : {false_value}"
        
        return js_ternary
    
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
        
        # 转换成员检查
        in_match = re.search(r'(.*?)\s+in\s+(.*)', js_condition)
        if in_match:
            item, container = in_match.groups()
            js_condition = f"{container}.includes({item})"
        
        # 转换身份检查
        js_condition = js_condition.replace(" is None", " === null")
        js_condition = js_condition.replace(" is not None", " !== null")
        
        return js_condition
    
    def _convert_branch(self, branch: str) -> str:
        """转换分支代码"""
        if not branch:
            return "  // Empty branch\n"
        
        # 分行处理
        lines = branch.split('\n')
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
            
            # 添加缩进
            js_lines.append("  " + js_line)
        
        return "\n".join(js_lines) + "\n"


class PythonToTypeScriptConditionConverter(PythonToJavaScriptConditionConverter):
    """Python到TypeScript的条件语句转换器"""
    
    def __init__(self):
        super().__init__()
        self.target_language = LanguageType.TYPESCRIPT
    
    def convert_match_statement(self, condition_node: ConditionNode) -> str:
        """转换Python 3.10+的match语句到TypeScript"""
        # TypeScript支持更高级的模式匹配（通过类型系统）
        
        # 首先使用基本的JavaScript转换
        js_switch = super().convert_match_statement(condition_node)
        
        # 然后添加TypeScript特定的类型检查
        # 这里只是一个简单的示例，实际应用中需要更复杂的类型推断
        ts_switch = js_switch.replace("switch (", "switch (")
        
        return ts_switch


class PythonToCppConditionConverter(ConditionConverter):
    """Python到C++的条件语句转换器"""
    
    def __init__(self):
        super().__init__(LanguageType.PYTHON, LanguageType.CPP)
    
    def convert_if_statement(self, condition_node: ConditionNode) -> str:
        """转换Python的if语句到C++"""
        if not isinstance(condition_node, ConditionNode):
            return ""
        
        # 转换条件表达式
        condition = self._convert_condition(condition_node.condition)
        
        # 构建C++的if语句
        cpp_if = f"if ({condition}) {{\n"
        
        # 转换true分支
        if condition_node.true_branch:
            cpp_if += self._convert_branch(condition_node.true_branch)
        
        cpp_if += "}"
        
        # 处理false分支（可能是elif或else）
        if condition_node.false_branch:
            # 检查false分支是否是另一个if语句（elif的情况）
            if condition_node.false_branch.strip().startswith("if "):
                cpp_if += f" else {{\n"
                cpp_if += self._convert_branch(condition_node.false_branch)
                cpp_if += "}"
            else:
                cpp_if += f" else {{\n"
                cpp_if += self._convert_branch(condition_node.false_branch)
                cpp_if += "}"
        
        return cpp_if
    
    def convert_switch_statement(self, condition_node: ConditionNode) -> str:
        """Python没有switch语句，但可以处理转换为C++的switch"""
        # 由于Python没有switch语句，这个方法通常不会被直接调用
        # 但为了完整性，我们提供一个实现
        
        # 检查是否有cases属性
        if not condition_node.cases:
            return f"// Cannot convert to switch: no cases defined\n{condition_node.source_code}"
        
        # 构建C++的switch语句
        cpp_switch = f"switch ({condition_node.condition}) {{\n"
        
        # 添加case分支
        for case in condition_node.cases:
            case_value = case.get("value", "default")
            case_body = case.get("body", "")
            
            if case_value == "default":
                cpp_switch += f"  default:\n"
            else:
                cpp_switch += f"  case {case_value}:\n"
            
            # 添加case体
            case_body_lines = case_body.split('\n')
            for line in case_body_lines:
                cpp_switch += f"    {line}\n"
            
            # 添加break语句（如果没有）
            if not case_body.strip().endswith("break;"):
                cpp_switch += "    break;\n"
        
        cpp_switch += "}"
        return cpp_switch
    
    def convert_match_statement(self, condition_node: ConditionNode) -> str:
        """转换Python 3.10+的match语句到C++"""
        # C++17及以前没有直接对应的模式匹配，需要转换为if-else或switch
        
        # 尝试解析match语句
        match_pattern = r'match\s+(.*?):\s*\n((?:\s+case.*?\n(?:(?:\s+).*?\n)*)+)'
        match_match = re.search(match_pattern, condition_node.source_code, re.DOTALL)
        
        if not match_match:
            return f"// Cannot parse match statement\n{condition_node.source_code}"
        
        subject = match_match.group(1)
        cases_block = match_match.group(2)
        
        # 解析case语句
        case_pattern = r'case\s+(.*?):\s*\n((?:\s+.*?\n)*)'
        case_matches = re.finditer(case_pattern, cases_block, re.DOTALL)
        
        # 检查是否可以转换为switch语句
        can_use_switch = True
        for case_match in re.finditer(case_pattern, cases_block, re.DOTALL):
            pattern = case_match.group(1)
            # 如果模式不是简单的常量，就不能使用switch
            if "|" in pattern or "_" in pattern or "(" in pattern:
                can_use_switch = False
                break
        
        if can_use_switch:
            # 构建C++的switch语句
            cpp_switch = f"switch ({subject}) {{\n"
            
            for case_match in re.finditer(case_pattern, cases_block, re.DOTALL):
                pattern = case_match.group(1)
                body = case_match.group(2)
                
                if "_" in pattern:
                    # 通配符模式转为default
                    cpp_switch += f"  default:\n"
                elif "|" in pattern:
                    # 多模式转为多个case
                    patterns = [p.strip() for p in pattern.split("|")]
                    for p in patterns:
                        cpp_switch += f"  case {p}:\n"
                else:
                    cpp_switch += f"  case {pattern}:\n"
                
                # 添加case体
                body_lines = body.split('\n')
                for line in body_lines:
                    if line.strip():
                        cpp_switch += f"    {line.strip()}\n"
                
                # 添加break语句（如果没有）
                if not body.strip().endswith("break;"):
                    cpp_switch += "    break;\n"
            
            cpp_switch += "}"
            return cpp_switch
        else:
            # 构建C++的if-else链
            cpp_if_else = f"// Converted from Python match statement\n"
            cpp_if_else += f"auto&& __match_subject = {subject};\n"
            
            is_first = True
            for case_match in re.finditer(case_pattern, cases_block, re.DOTALL):
                pattern = case_match.group(1)
                body = case_match.group(2)
                
                if is_first:
                    cpp_if_else += "if "
                    is_first = False
                else:
                    cpp_if_else += "else if "
                
                if "_" in pattern:
                    # 通配符模式转为else
                    cpp_if_else = cpp_if_else.rsplit("else if ", 1)[0]
                    cpp_if_else += "else {\n"
                elif "|" in pattern:
                    # 多模式转为多个条件
                    patterns = [p.strip() for p in pattern.split("|")]
                    conditions = [f"__match_subject == {p}" for p in patterns]
                    cpp_if_else += f"({' || '.join(conditions)}) {{\n"
                else:
                    cpp_if_else += f"(__match_subject == {pattern}) {{\n"
                
                # 添加case体
                body_lines = body.split('\n')
                for line in body_lines:
                    if line.strip():
                        cpp_if_else += f"  {line.strip()}\n"
                
                cpp_if_else += "}\n"
            
            return cpp_if_else
    
    def convert_ternary_operator(self, condition_node: ConditionNode) -> str:
        """转换Python的条件表达式到C++的三元运算符"""
        # 检查是否是条件表达式
        if not (condition_node.condition and condition_node.true_branch and condition_node.false_branch):
            return f"// Cannot convert to ternary: missing components\n{condition_node.source_code}"
        
        # 转换条件表达式
        condition = self._convert_condition(condition_node.condition)
        true_value = condition_node.true_branch.strip()
        false_value = condition_node.false_branch.strip()
        
        # 构建C++的三元运算符
        cpp_ternary = f"({condition}) ? {true_value} : {false_value}"
        
        return cpp_ternary
    
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
        
        # 转换成员检查
        in_match = re.search(r'(.*?)\s+in\s+(.*)', cpp_condition)
        if in_match:
            item, container = in_match.groups()
            cpp_condition = f"std::find({container}.begin(), {container}.end(), {item}) != {container}.end()"
        
        # 转换身份检查
        cpp_condition = cpp_condition.replace(" is None", " == nullptr")
        cpp_condition = cpp_condition.replace(" is not None", " != nullptr")
        
        return cpp_condition
    
    def _convert_branch(self, branch: str) -> str:
        """转换分支代码"""
        if not branch:
            return "  // Empty branch\n"
        
        # 分行处理
        lines = branch.split('\n')
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


class JavaScriptToPythonConditionConverter(ConditionConverter):
    """JavaScript到Python的条件语句转换器"""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT, LanguageType.PYTHON)
    
    def convert_if_statement(self, condition_node: ConditionNode) -> str:
        """转换JavaScript的if语句到Python"""
        if not isinstance(condition_node, ConditionNode):
            return ""
        
        # 转换条件表达式
        condition = self._convert_js_condition(condition_node.condition)
        
        # 构建Python的if语句
        py_if = f"if {condition}:\n"
        
        # 转换true分支
        if condition_node.true_branch:
            py_if += self._convert_branch(condition_node.true_branch)
        else:
            py_if += "    pass\n"
        
        # 处理false分支（可能是else if或else）
        if condition_node.false_branch:
            # 检查false分支是否是另一个if语句（else if的情况）
            if condition_node.false_branch.strip().startswith("if "):
                # 转换为elif
                false_branch = condition_node.false_branch.strip()
                if_match = re.match(r'if\s*\((.*?)\)\s*{(.*)}', false_branch, re.DOTALL)
                if if_match:
                    elif_condition = self._convert_js_condition(if_match.group(1))
                    elif_body = if_match.group(2)
                    
                    py_if += f"elif {elif_condition}:\n"
                    py_if += self._convert_branch(elif_body)
                else:
                    # 无法解析，保留原始代码
                    py_if += f"else:  # Original: {false_branch}\n"
                    py_if += "    pass\n"
            else:
                py_if += "else:\n"
                py_if += self._convert_branch(condition_node.false_branch)
        
        return py_if
    
    def convert_switch_statement(self, condition_node: ConditionNode) -> str:
        """转换JavaScript的switch语句到Python"""
        # Python 3.10+支持match语句，可以直接转换
        # 对于早期版本，需要转换为if-elif-else链
        
        # 检查是否有cases属性
        if not condition_node.cases:
            return f"# Cannot convert switch: no cases defined\n# {condition_node.source_code}\n"
        
        # 尝试解析switch语句
        switch_match = re.search(r'switch\s*\((.*?)\)\s*{(.*)}', condition_node.source_code, re.DOTALL)
        if not switch_match:
            return f"# Cannot parse switch statement\n# {condition_node.source_code}\n"
        
        subject = switch_match.group(1)
        cases_block = switch_match.group(2)
        
        # 解析case语句
        case_pattern = r'case\s+(.*?):\s*(.*?)(?:break;|(?=case|default)|$)'
        case_matches = re.finditer(case_pattern, cases_block, re.DOTALL)
        
        # 构建Python 3.10+的match语句
        py_match = f"match {subject}:\n"
        
        has_default = False
        for case_match in case_matches:
            case_value = case_match.group(1).strip()
            case_body = case_match.group(2).strip()
            
            if case_value == "default":
                has_default = True
                py_match += f"    case _:\n"
            else:
                py_match += f"    case {case_value}:\n"
            
            # 添加case体
            if case_body:
                body_lines = case_body.split('\n')
                for line in body_lines:
                    if line.strip():
                        py_match += f"        {line.strip()}\n"
            else:
                py_match += "        pass\n"
        
        # 如果没有default，添加一个空的default
        if not has_default:
            py_match += "    case _:\n"
            py_match += "        pass\n"
        
        # 添加注释，说明这需要Python 3.10+
        py_match = f"# Requires Python 3.10+\n{py_match}\n"
        
        # 同时提供if-elif-else版本
        py_if_else = f"# Alternative for Python < 3.10\n"
        py_if_else += f"__switch_value = {subject}\n"
        
        is_first = True
        for case_match in re.finditer(case_pattern, cases_block, re.DOTALL):
            case_value = case_match.group(1).strip()
            case_body = case_match.group(2).strip()
            
            if case_value == "default":
                py_if_else += "else:\n"
            else:
                if is_first:
                    py_if_else += f"if __switch_value == {case_value}:\n"
                    is_first = False
                else:
                    py_if_else += f"elif __switch_value == {case_value}:\n"
            
            # 添加case体
            if case_body:
                body_lines = case_body.split('\n')
                for line in body_lines:
                    if line.strip():
                        py_if_else += f"    {line.strip()}\n"
            else:
                py_if_else += "    pass\n"
        
        return py_match + "\n" + py_if_else
    
    def convert_match_statement(self, condition_node: ConditionNode) -> str:
        """JavaScript没有match语句，但可以处理类似的模式匹配"""
        # 由于JavaScript没有内置的模式匹配，这个方法通常不会被直接调用
        # 但为了完整性，我们提供一个实现
        
        return f"# Cannot convert match statement: not supported in JavaScript\n# {condition_node.source_code}\n"
    
    def convert_ternary_operator(self, condition_node: ConditionNode) -> str:
        """转换JavaScript的三元运算符到Python的条件表达式"""
        # 检查是否是三元运算符
        ternary_match = re.search(r'(.*?)\s*\?\s*(.*?)\s*:\s*(.*)', condition_node.source_code)
        if not ternary_match:
            return f"# Cannot parse ternary operator\n# {condition_node.source_code}\n"
        
        condition = ternary_match.group(1)
        true_value = ternary_match.group(2)
        false_value = ternary_match.group(3)
        
        # 转换条件表达式
        py_condition = self._convert_js_condition(condition)
        py_true_value = true_value.strip()
        py_false_value = false_value.strip()
        
        # 构建Python的条件表达式
        py_ternary = f"{py_true_value} if {py_condition} else {py_false_value}"
        
        return py_ternary
    
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
        
        # 转换成员检查
        includes_match = re.search(r'(.*?)\.includes\((.*?)\)', py_condition)
        if includes_match:
            container, item = includes_match.groups()
            py_condition = py_condition.replace(f"{container}.includes({item})", f"{item} in {container}")
        
        return py_condition
    
    def _convert_branch(self, branch: str) -> str:
        """转换分支代码"""
        if not branch:
            return "    # Empty branch\n"
        
        # 分行处理
        lines = branch.split('\n')
        py_lines = []
        
        for line in lines:
            # 跳过空行
            if not line.strip():
                py_lines.append("")
                continue
            
            # 转换console.log语句
            if "console.log(" in line:
                py_line = re.sub(r'console\.log\((.*)\);?', r'print(\1)', line)
                py_lines.append("    " + py_line)
                continue
            
            # 转换JavaScript的比较运算符
            py_line = line
            py_line = py_line.replace("===", "==")
            py_line = py_line.replace("!==", "!=")
            py_line = py_line.replace("true", "True")
            py_line = py_line.replace("false", "False")
            py_line = py_line.replace("null", "None")
            
            # 转换逻辑运算符
            py_line = py_line.replace("&&", "and")
            py_line = py_line.replace("||", "or")
            py_line = py_line.replace("!", "not ")
            
            # 移除分号
            py_line = py_line.rstrip(";")
            
            # 添加缩进
            py_lines.append("    " + py_line)
        
        return "\n".join(py_lines) + "\n"


# 工厂函数，创建适合的条件语句转换器
def create_condition_converter(source_language: LanguageType, target_language: LanguageType) -> ConditionConverter:
    """创建适合的条件语句转换器"""
    if source_language == LanguageType.PYTHON and target_language == LanguageType.JAVASCRIPT:
        return PythonToJavaScriptConditionConverter()
    elif source_language == LanguageType.PYTHON and target_language == LanguageType.TYPESCRIPT:
        return PythonToTypeScriptConditionConverter()
    elif source_language == LanguageType.PYTHON and target_language == LanguageType.CPP:
        return PythonToCppConditionConverter()
    elif source_language == LanguageType.JAVASCRIPT and target_language == LanguageType.PYTHON:
        return JavaScriptToPythonConditionConverter()
    # 添加其他语言对的转换器...
    else:
        raise ValueError(f"不支持从 {source_language} 到 {target_language} 的条件语句转换")


# 示例使用
def convert_condition(source_code: str, source_language: str, target_language: str) -> str:
    """转换条件语句"""
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
        
        # 获取所有条件节点
        condition_nodes = graph.get_nodes_by_type(ControlFlowType.CONDITION)
        
        # 创建转换器
        converter = create_condition_converter(source_lang, target_lang)
        
        # 转换所有条件节点
        result = []
        for node in condition_nodes:
            if isinstance(node, ConditionNode):
                converted = converter.convert_condition(node)
                result.append(converted)
        
        # 如果没有找到条件节点，尝试直接转换整个代码
        if not result:
            # 创建一个模拟的条件节点
            condition_node = ConditionNode(
                node_type=ControlFlowType.CONDITION,
                source_language=source_lang,
                source_code=source_code,
                condition_type="unknown"
            )
            
            # 尝试根据代码内容判断条件类型
            if "if " in source_code:
                condition_node.condition_type = "if"
            elif "switch " in source_code:
                condition_node.condition_type = "switch"
            elif "match " in source_code:
                condition_node.condition_type = "match"
            elif "?" in source_code and ":" in source_code:
                condition_node.condition_type = "ternary"
            
            converted = converter.convert_condition(condition_node)
            result.append(converted)
        
        return "\n\n".join(result)
    except ValueError as e:
        return f"错误: {str(e)}"
    except Exception as e:
        return f"转换失败: {str(e)}"