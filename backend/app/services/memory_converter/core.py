"""
内存管理跨语言转换模块的核心架构

提供各种编程语言间内存管理和资源处理的转换功能，包括：
- 垃圾回收到手动内存管理
- 资源管理转换
- 所有权系统转换
"""
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import re
import ast


class LanguageType(Enum):
    """支持的编程语言类型"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CPP = "cpp"
    JAVA = "java"
    CSHARP = "csharp"
    RUST = "rust"
    GO = "go"


class MemoryManagementType(Enum):
    """内存管理类型"""
    GARBAGE_COLLECTION = auto()  # 垃圾回收
    MANUAL = auto()              # 手动内存管理
    REFERENCE_COUNTING = auto()  # 引用计数
    RAII = auto()                # 资源获取即初始化
    OWNERSHIP = auto()           # 所有权系统


class ResourceType(Enum):
    """资源类型"""
    FILE = auto()                # 文件资源
    NETWORK = auto()             # 网络资源
    DATABASE = auto()            # 数据库资源
    LOCK = auto()                # 锁资源
    MEMORY = auto()              # 内存资源
    GENERIC = auto()             # 通用资源


class MemoryNode:
    """内存管理节点基类"""
    
    def __init__(self, node_type: MemoryManagementType, source_language: LanguageType, source_code: str):
        self.node_type = node_type
        self.source_language = source_language
        self.source_code = source_code


class ResourceNode(MemoryNode):
    """资源管理节点"""
    
    def __init__(self, 
                 source_language: LanguageType, 
                 source_code: str,
                 resource_type: ResourceType = ResourceType.GENERIC,
                 context_type: str = "",
                 resource_expr: str = "",
                 resource_var: str = "",
                 resource_body: str = ""):
        super().__init__(MemoryManagementType.RAII, source_language, source_code)
        self.resource_type = resource_type
        self.context_type = context_type  # with, using, try-with-resources等
        self.resource_expr = resource_expr  # 资源表达式
        self.resource_var = resource_var  # 资源变量名
        self.resource_body = resource_body  # 资源使用代码块


class OwnershipNode(MemoryNode):
    """所有权系统节点"""
    
    def __init__(self,
                 source_language: LanguageType,
                 source_code: str,
                 ownership_type: str = "",  # 所有权类型：move, borrow, reference等
                 variable_name: str = "",
                 is_mutable: bool = False,
                 lifetime: str = ""):
        super().__init__(MemoryManagementType.OWNERSHIP, source_language, source_code)
        self.ownership_type = ownership_type
        self.variable_name = variable_name
        self.is_mutable = is_mutable
        self.lifetime = lifetime


class MemoryAllocationNode(MemoryNode):
    """内存分配节点"""
    
    def __init__(self,
                 source_language: LanguageType,
                 source_code: str,
                 allocation_type: str = "",  # new, malloc, alloc等
                 variable_name: str = "",
                 data_type: str = "",
                 size: str = "",
                 is_array: bool = False):
        super().__init__(MemoryManagementType.MANUAL, source_language, source_code)
        self.allocation_type = allocation_type
        self.variable_name = variable_name
        self.data_type = data_type
        self.size = size
        self.is_array = is_array


class MemoryGraph:
    """内存管理图，用于表示代码中的内存管理结构"""
    
    def __init__(self):
        self.nodes: List[MemoryNode] = []
    
    def add_node(self, node: MemoryNode) -> None:
        """添加节点"""
        self.nodes.append(node)
    
    def get_nodes_by_type(self, node_type: MemoryManagementType) -> List[MemoryNode]:
        """获取指定类型的节点"""
        return [node for node in self.nodes if node.node_type == node_type]
    
    def get_resource_nodes(self) -> List[ResourceNode]:
        """获取所有资源管理节点"""
        return [node for node in self.nodes if isinstance(node, ResourceNode)]
    
    def get_ownership_nodes(self) -> List[OwnershipNode]:
        """获取所有所有权系统节点"""
        return [node for node in self.nodes if isinstance(node, OwnershipNode)]
    
    def get_memory_allocation_nodes(self) -> List[MemoryAllocationNode]:
        """获取所有内存分配节点"""
        return [node for node in self.nodes if isinstance(node, MemoryAllocationNode)]


class MemoryParser:
    """内存管理解析器基类"""
    
    def __init__(self, language: LanguageType):
        self.language = language
    
    def parse(self, code: str) -> MemoryGraph:
        """解析代码，构建内存管理图"""
        raise NotImplementedError("子类必须实现此方法")


class PythonMemoryParser(MemoryParser):
    """Python内存管理解析器"""
    
    def __init__(self):
        super().__init__(LanguageType.PYTHON)
    
    def parse(self, code: str) -> MemoryGraph:
        """解析Python代码，构建内存管理图"""
        graph = MemoryGraph()
        
        try:
            # 解析Python代码为AST
            tree = ast.parse(code)
            
            # 遍历AST，查找内存管理相关节点
            for node in ast.walk(tree):
                # 查找with语句（资源管理）
                if isinstance(node, ast.With):
                    for item in node.items:
                        context_expr = ast.unparse(item.context_expr)
                        optional_vars = ast.unparse(item.optional_vars) if item.optional_vars else ""
                        
                        # 确定资源类型
                        resource_type = ResourceType.GENERIC
                        if "open(" in context_expr:
                            resource_type = ResourceType.FILE
                        elif "socket" in context_expr or "connect" in context_expr:
                            resource_type = ResourceType.NETWORK
                        elif "lock" in context_expr.lower() or "mutex" in context_expr.lower():
                            resource_type = ResourceType.LOCK
                        elif "cursor" in context_expr.lower() or "connection" in context_expr.lower():
                            resource_type = ResourceType.DATABASE
                        
                        # 创建资源节点
                        resource_node = ResourceNode(
                            source_language=self.language,
                            source_code=ast.unparse(node),
                            resource_type=resource_type,
                            context_type="with",
                            resource_expr=context_expr,
                            resource_var=optional_vars,
                            resource_body=ast.unparse(node.body)
                        )
                        
                        graph.add_node(resource_node)
        
        except SyntaxError:
            # 如果AST解析失败，尝试使用正则表达式
            # 查找with语句
            with_matches = re.finditer(r'with\s+(.*?)\s+as\s+(.*?):(.*?)(?=\n\S|\Z)', code, re.DOTALL)
            for match in with_matches:
                expr = match.group(1)
                var = match.group(2)
                body = match.group(3)
                
                # 确定资源类型
                resource_type = ResourceType.GENERIC
                if "open(" in expr:
                    resource_type = ResourceType.FILE
                elif "socket" in expr or "connect" in expr:
                    resource_type = ResourceType.NETWORK
                elif "lock" in expr.lower() or "mutex" in expr.lower():
                    resource_type = ResourceType.LOCK
                elif "cursor" in expr.lower() or "connection" in expr.lower():
                    resource_type = ResourceType.DATABASE
                
                # 创建资源节点
                resource_node = ResourceNode(
                    source_language=self.language,
                    source_code=match.group(0),
                    resource_type=resource_type,
                    context_type="with",
                    resource_expr=expr,
                    resource_var=var,
                    resource_body=body
                )
                
                graph.add_node(resource_node)
        
        return graph


class JavaScriptMemoryParser(MemoryParser):
    """JavaScript内存管理解析器"""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT)
    
    def parse(self, code: str) -> MemoryGraph:
        """解析JavaScript代码，构建内存管理图"""
        graph = MemoryGraph()
        
        # JavaScript没有明确的资源管理语法，但可以识别常见模式
        
        # 查找try-finally模式（常用于资源管理）
        try_finally_matches = re.finditer(
            r'try\s*{(.*?)}(?:\s*catch\s*\((.*?)\)\s*{(.*?)})?'
            r'\s*finally\s*{(.*?)}',
            code, re.DOTALL
        )
        
        for match in try_finally_matches:
            try_block = match.group(1)
            finally_block = match.group(4)
            
            # 检查是否包含资源清理模式
            if re.search(r'\.close\(\)|\.dispose\(\)|\.end\(\)', finally_block):
                # 尝试提取资源变量
                var_match = re.search(r'(?:const|let|var)\s+(\w+)\s*=\s*(.*?);', try_block)
                if var_match:
                    var_name = var_match.group(1)
                    expr = var_match.group(2)
                    
                    # 确定资源类型
                    resource_type = ResourceType.GENERIC
                    if "createReadStream" in expr or "createWriteStream" in expr or "fs." in expr:
                        resource_type = ResourceType.FILE
                    elif "socket" in expr or "http" in expr or "fetch" in expr:
                        resource_type = ResourceType.NETWORK
                    elif "connection" in expr or "cursor" in expr:
                        resource_type = ResourceType.DATABASE
                    
                    # 创建资源节点
                    resource_node = ResourceNode(
                        source_language=self.language,
                        source_code=match.group(0),
                        resource_type=resource_type,
                        context_type="try-finally",
                        resource_expr=expr,
                        resource_var=var_name,
                        resource_body=try_block
                    )
                    
                    graph.add_node(resource_node)
        
        return graph


class CppMemoryParser(MemoryParser):
    """C++内存管理解析器"""
    
    def __init__(self):
        super().__init__(LanguageType.CPP)
    
    def parse(self, code: str) -> MemoryGraph:
        """解析C++代码，构建内存管理图"""
        graph = MemoryGraph()
        
        # 查找内存分配模式
        new_matches = re.finditer(r'(\w+(?:<.*?>)?)\s*\*\s*(\w+)\s*=\s*new\s+(\w+(?:<.*?>)?)(?:\[(.*?)\])?', code)
        for match in new_matches:
            data_type = match.group(1)
            var_name = match.group(2)
            alloc_type = match.group(3)
            size = match.group(4) or ""
            
            # 创建内存分配节点
            alloc_node = MemoryAllocationNode(
                source_language=self.language,
                source_code=match.group(0),
                allocation_type="new",
                variable_name=var_name,
                data_type=data_type,
                size=size,
                is_array=bool(size)
            )
            
            graph.add_node(alloc_node)
        
        # 查找RAII模式（作用域资源管理）
        # 例如：std::ifstream file("filename.txt");
        raii_matches = re.finditer(
            r'std::(\w+)(?:<.*?>)?\s+(\w+)\s*\((.*?)\)',
            code
        )
        
        for match in raii_matches:
            raii_type = match.group(1)
            var_name = match.group(2)
            args = match.group(3)
            
            # 确定资源类型
            resource_type = ResourceType.GENERIC
            if raii_type in ["ifstream", "ofstream", "fstream"]:
                resource_type = ResourceType.FILE
            elif raii_type in ["mutex", "lock_guard", "unique_lock"]:
                resource_type = ResourceType.LOCK
            elif "socket" in raii_type:
                resource_type = ResourceType.NETWORK
            
            # 创建资源节点
            resource_node = ResourceNode(
                source_language=self.language,
                source_code=match.group(0),
                resource_type=resource_type,
                context_type="raii",
                resource_expr=f"std::{raii_type}({args})",
                resource_var=var_name,
                resource_body=""  # 无法确定作用域，留空
            )
            
            graph.add_node(resource_node)
        
        return graph


class RustMemoryParser(MemoryParser):
    """Rust内存管理解析器"""
    
    def __init__(self):
        super().__init__(LanguageType.RUST)
    
    def parse(self, code: str) -> MemoryGraph:
        """解析Rust代码，构建内存管理图"""
        graph = MemoryGraph()
        
        # 查找所有权模式
        # 例如：let mut x = String::from("hello");
        ownership_matches = re.finditer(
            r'let\s+(mut\s+)?(\w+)(?::\s*([^=]+))?\s*=\s*(.*?);',
            code
        )
        
        for match in ownership_matches:
            is_mutable = match.group(1) is not None
            var_name = match.group(2)
            type_annotation = match.group(3) or ""
            expr = match.group(4)
            
            # 确定所有权类型
            ownership_type = "move"  # 默认为移动语义
            if "&mut " in expr:
                ownership_type = "mutable_borrow"
            elif "&" in expr:
                ownership_type = "immutable_borrow"
            elif "Box::new" in expr or "Rc::new" in expr or "Arc::new" in expr:
                ownership_type = "heap_allocation"
            
            # 创建所有权节点
            ownership_node = OwnershipNode(
                source_language=self.language,
                source_code=match.group(0),
                ownership_type=ownership_type,
                variable_name=var_name,
                is_mutable=is_mutable,
                lifetime=""  # 无法从简单正则中提取生命周期
            )
            
            graph.add_node(ownership_node)
        
        return graph


class MemoryConverter:
    """内存管理转换器基类"""
    
    def __init__(self, source_language: LanguageType, target_language: LanguageType):
        self.source_language = source_language
        self.target_language = target_language
    
    def convert(self, node: MemoryNode) -> str:
        """转换内存管理节点"""
        if isinstance(node, ResourceNode):
            return self.convert_resource(node)
        elif isinstance(node, OwnershipNode):
            return self.convert_ownership(node)
        elif isinstance(node, MemoryAllocationNode):
            return self.convert_memory_allocation(node)
        else:
            return node.source_code  # 默认返回原始代码
    
    def convert_resource(self, node: ResourceNode) -> str:
        """转换资源管理节点"""
        raise NotImplementedError("子类必须实现此方法")
    
    def convert_ownership(self, node: OwnershipNode) -> str:
        """转换所有权系统节点"""
        raise NotImplementedError("子类必须实现此方法")
    
    def convert_memory_allocation(self, node: MemoryAllocationNode) -> str:
        """转换内存分配节点"""
        raise NotImplementedError("子类必须实现此方法")


class PythonToCppConverter(MemoryConverter):
    """Python到C++的内存管理转换器"""
    
    def __init__(self):
        super().__init__(LanguageType.PYTHON, LanguageType.CPP)
    
    def convert_resource(self, node: ResourceNode) -> str:
        """转换Python的with语句到C++的RAII模式"""
        if not isinstance(node, ResourceNode) or node.context_type != "with":
            return node.source_code
        
        # 处理文件资源
        if node.resource_type == ResourceType.FILE:
            # 检查是否是文件操作
            if "open(" in node.resource_expr:
                # 提取文件名和模式
                file_match = re.search(r'open\((.*?)(?:,\s*[\'"]([rwxa+]+)[\'"])?(,\s*encoding=[\'"](.*?)[\'"])?\)', node.resource_expr)
                if file_match:
                    file_path = file_match.group(1).strip()
                    mode = file_match.group(2) or "r"
                    encoding = file_match.group(4) or "utf8"
                    
                    # 确定C++文件流类型
                    stream_type = "std::ifstream"
                    if "w" in mode or "a" in mode:
                        stream_type = "std::ofstream"
                    elif "+" in mode:
                        stream_type = "std::fstream"
                    
                    # 构建C++代码
                    cpp_code = f"// 转换自Python的with语句\n"
                    cpp_code += f"#include <fstream>\n"
                    cpp_code += f"#include <string>\n\n"
                    cpp_code += f"{{\n"
                    cpp_code += f"    {stream_type} {node.resource_var}({file_path});\n"
                    
                    # 添加文件内容读取
                    if "r" in mode:
                        cpp_code += f"    std::string content;\n"
                        cpp_code += f"    std::string line;\n"
                        cpp_code += f"    while (std::getline({node.resource_var}, line)) {{\n"
                        cpp_code += f"        content += line + '\\n';\n"
                        cpp_code += f"    }}\n"
                    
                    # 添加原始代码块的注释
                    cpp_code += f"\n    // 原始Python代码块:\n"
                    cpp_code += f"    // {node.resource_body.replace(chr(10), chr(10) + '    // ')}\n"
                    
                    # 闭合作用域
                    cpp_code += f"}} // {node.resource_var}自动关闭\n"
                    
                    return cpp_code
        
        # 处理锁资源
        elif node.resource_type == ResourceType.LOCK:
            cpp_code = f"// 转换自Python的with语句(锁)\n"
            cpp_code += f"#include <mutex>\n\n"
            cpp_code += f"{{\n"
            cpp_code += f"    std::lock_guard<std::mutex> lock({node.resource_expr.replace('lock', 'mutex')});\n"
            
            # 添加原始代码块的注释
            cpp_code += f"\n    // 原始Python代码块:\n"
            cpp_code += f"    // {node.resource_body.replace(chr(10), chr(10) + '    // ')}\n"
            
            # 闭合作用域
            cpp_code += f"}} // 锁自动释放\n"
            
            return cpp_code
        
        # 默认情况
        return f"// 无法转换的Python with语句\n// {node.source_code}\n"
    
    def convert_ownership(self, node: OwnershipNode) -> str:
        """Python没有显式的所有权系统，此方法通常不会被调用"""
        return f"// Python没有显式的所有权系统\n// {node.source_code}\n"
    
    def convert_memory_allocation(self, node: MemoryAllocationNode) -> str:
        """Python没有显式的内存分配，此方法通常不会被调用"""
        return f"// Python没有显式的内存分配\n// {node.source_code}\n"


class JavaScriptToRustConverter(MemoryConverter):
    """JavaScript到Rust的内存管理转换器"""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT, LanguageType.RUST)
    
    def convert_resource(self, node: ResourceNode) -> str:
        """转换JavaScript的try-finally模式到Rust"""
        if not isinstance(node, ResourceNode) or node.context_type != "try-finally":
            return node.source_code
        
        # 处理文件资源
        if node.resource_type == ResourceType.FILE:
            rust_code = f"// 转换自JavaScript的try-finally模式\n"
            rust_code += f"use std::fs::File;\n"
            rust_code += f"use std::io::{{Read, Write}};\n\n"
            
            if "createReadStream" in node.resource_expr or "readFile" in node.resource_expr:
                # 读取文件
                file_path = re.search(r'[\'"]([^\'"]+)[\'"]', node.resource_expr)
                if file_path:
                    rust_code += f"let mut {node.resource_var} = File::open({file_path.group(0)})?;\n"
                    rust_code += f"let mut content = String::new();\n"
                    rust_code += f"{node.resource_var}.read_to_string(&mut content)?;\n"
            elif "createWriteStream" in node.resource_expr or "writeFile" in node.resource_expr:
                # 写入文件
                file_path = re.search(r'[\'"]([^\'"]+)[\'"]', node.resource_expr)
                if file_path:
                    rust_code += f"let mut {node.resource_var} = File::create({file_path.group(0)})?;\n"
            
            # 添加原始代码块的注释
            rust_code += f"\n// 原始JavaScript代码块:\n"
            rust_code += f"// {node.resource_body.replace(chr(10), chr(10) + '// ')}\n"
            
            return rust_code
        
        # 默认情况
        return f"// 无法转换的JavaScript资源管理模式\n// {node.source_code}\n"
    
    def convert_ownership(self, node: OwnershipNode) -> str:
        """JavaScript没有显式的所有权系统，此方法通常不会被调用"""
        return f"// JavaScript没有显式的所有权系统\n// {node.source_code}\n"
    
    def convert_memory_allocation(self, node: MemoryAllocationNode) -> str:
        """转换JavaScript的数组和对象创建到Rust"""
        # 这里我们假设node是从JavaScript代码中提取的内存分配模式
        js_code = node.source_code
        
        # 处理数组创建
        if "[" in js_code and "]" in js_code:
            array_match = re.search(r'(?:const|let|var)\s+(\w+)\s*=\s*\[(.*?)\]', js_code)
            if array_match:
                var_name = array_match.group(1)
                elements = array_match.group(2)
                
                rust_code = f"// 转换自JavaScript数组\n"
                rust_code += f"let {var_name} = vec![{elements}];\n"
                
                return rust_code
        
        # 处理对象创建
        if "{" in js_code and "}" in js_code:
            obj_match = re.search(r'(?:const|let|var)\s+(\w+)\s*=\s*{(.*?)}', js_code, re.DOTALL)
            if obj_match:
                var_name = obj_match.group(1)
                properties = obj_match.group(2)
                
                rust_code = f"// 转换自JavaScript对象\n"
                rust_code += f"// 使用结构体或HashMap\n"
                rust_code += f"use std::collections::HashMap;\n"
                rust_code += f"let mut {var_name} = HashMap::new();\n"
                
                # 处理属性
                for prop in re.finditer(r'(\w+):\s*(.*?)(?:,|$)', properties):
                    key = prop.group(1)
                    value = prop.group(2)
                    rust_code += f"{var_name}.insert(\"{key}\".to_string(), {value});\n"
                
                return rust_code
        
        # 默认情况
        return f"// 无法转换的JavaScript内存分配\n// {js_code}\n"


class RustToJavaScriptConverter(MemoryConverter):
    """Rust到JavaScript的内存管理转换器"""
    
    def __init__(self):
        super().__init__(LanguageType.RUST, LanguageType.JAVASCRIPT)
    
    def convert_resource(self, node: ResourceNode) -> str:
        """转换Rust的资源管理到JavaScript"""
        # Rust通常使用作用域和Drop trait进行资源管理
        return f"// 转换自Rust的资源管理\n"
    
    def convert_ownership(self, node: OwnershipNode) -> str:
        """转换Rust的所有权系统到JavaScript"""
        if not isinstance(node, OwnershipNode):
            return node.source_code
        
        js_code = f"// 转换自Rust的所有权系统\n"
        js_code += f"// 注意: JavaScript没有所有权概念，所有变量都是引用\n"
        
        # 根据所有权类型进行转换
        if node.ownership_type == "move":
            js_code += f"const {node.variable_name} = {node.source_code.split('=')[1].strip().rstrip(';')};\n"
        elif node.ownership_type == "mutable_borrow":
            js_code += f"// 在JavaScript中，所有对象都是可变的\n"
            js_code += f"let {node.variable_name} = {node.source_code.split('=')[1].strip().rstrip(';').replace('&mut ', '')};\n"
        elif node.ownership_type == "immutable_borrow":
            js_code += f"// 在JavaScript中模拟不可变引用\n"
            js_code += f"const {node.variable_name} = Object.freeze({node.source_code.split('=')[1].strip().rstrip(';').replace('&', '')});\n"
        elif node.ownership_type == "heap_allocation":
            js_code += f"// JavaScript自动管理内存，不需要显式分配\n"
            js_code += f"const {node.variable_name} = {node.source_code.split('=')[1].strip().rstrip(';').replace('Box::new', '').replace('Rc::new', '').replace('Arc::new', '')};\n"
        
        return js_code
    
    def convert_memory_allocation(self, node: MemoryAllocationNode) -> str:
        """Rust的内存分配转换到JavaScript"""
        return f"// Rust的内存分配转换到JavaScript\n// JavaScript自动管理内存\n"


# 工厂函数，创建适合的内存管理转换器
def create_memory_converter(source_language: LanguageType, target_language: LanguageType) -> MemoryConverter:
    """创建适合的内存管理转换器"""
    if source_language == LanguageType.PYTHON and target_language == LanguageType.CPP:
        return PythonToCppConverter()
    elif source_language == LanguageType.JAVASCRIPT and target_language == LanguageType.RUST:
        return JavaScriptToRustConverter()
    elif source_language == LanguageType.RUST and target_language == LanguageType.JAVASCRIPT:
        return RustToJavaScriptConverter()
    # 添加其他语言对的转换器...
    else:
        raise ValueError(f"不支持从 {source_language} 到 {target_language} 的内存管理转换")


# 工厂函数，创建适合的内存管理解析器
def create_memory_parser(language: LanguageType) -> MemoryParser:
    """创建适合的内存管理解析器"""
    if language == LanguageType.PYTHON:
        return PythonMemoryParser()
    elif language == LanguageType.JAVASCRIPT:
        return JavaScriptMemoryParser()
    elif language == LanguageType.CPP:
        return CppMemoryParser()
    elif language == LanguageType.RUST:
        return RustMemoryParser()
    # 添加其他语言的解析器...
    else:
        raise ValueError(f"不支持的语言: {language}")


# 示例使用
def convert_memory_management(source_code: str, source_language: str, target_language: str) -> str:
    """转换内存管理代码"""
    try:
        source_lang = LanguageType(source_language.lower())
        target_lang = LanguageType(target_language.lower())
        
        # 创建解析器
        parser = create_memory_parser(source_lang)
        
        # 解析源代码
        graph = parser.parse(source_code)
        
        # 创建转换器
        converter = create_memory_converter(source_lang, target_lang)
        
        # 转换所有节点
        result = []
        for node in graph.nodes:
            converted = converter.convert(node)
            result.append(converted)
        
        # 如果没有找到任何节点，返回原始代码
        if not result:
            return f"// 未找到可转换的内存管理模式\n{source_code}"
        
        return "\n\n".join(result)
    except ValueError as e:
        return f"错误: {str(e)}"
    except Exception as e:
        return f"转换失败: {str(e)}"