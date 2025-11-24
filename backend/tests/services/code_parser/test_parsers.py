"""
解析器单元测试
"""
import os
import unittest
from typing import Any, Dict, List, Optional

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from app.services.code_parser.language_parser import LanguageParser, ParseResult, SourceRange
from app.services.code_parser.python_parser import PythonParser
from app.services.code_parser.js_ts_parser import JSTypeScriptParser
from app.services.code_parser.ir_model import Module, Function, Class, Statement, Expression
from app.services.code_parser.ast_to_ir_converter import ASTToIRConverter, PythonASTToIRConverter
from app.services.code_parser.ir_to_code_generator import IRToCodeGenerator, PythonCodeGenerator, JavaScriptCodeGenerator


class TestPythonParser(unittest.TestCase):
    """Python解析器测试"""
    
    def setUp(self):
        self.parser = PythonParser()
    
    def test_parse_code_success(self):
        """测试成功解析Python代码"""
        code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"

class Person:
    def __init__(self, name: str):
        self.name = name
        
    def greet(self) -> str:
        return hello(self.name)
"""
        result = self.parser.parse_code(code)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.ast)
        self.assertEqual(len(result.errors), 0)
    
    def test_parse_code_error(self):
        """测试解析错误的Python代码"""
        code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
    
# 语法错误
class Person:
    def __init__(self, name: str)
        self.name = name
"""
        result = self.parser.parse_code(code)
        
        self.assertFalse(result.success)
        self.assertIsNone(result.ast)
        self.assertGreater(len(result.errors), 0)
    
    def test_get_node_at_position(self):
        """测试获取指定位置的节点"""
        code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
"""
        result = self.parser.parse_code(code)
        
        # 获取函数定义节点
        node = self.parser.get_node_at_position(result.ast, 2, 5)
        self.assertIsNotNone(node)
        self.assertEqual(getattr(node, 'name', None), 'hello')
    
    def test_python_ast_analyzer(self):
        """测试Python AST分析器"""
        code = """
import os
from typing import List, Dict

def hello(name: str) -> str:
    return f"Hello, {name}!"

class Person:
    def __init__(self, name: str):
        self.name = name
        
    def greet(self) -> str:
        return hello(self.name)
"""
        result = self.parser.parse_code(code)
        
        # 获取导入语句
        imports = self.parser.analyzer.get_imports(result.ast)
        self.assertEqual(len(imports), 2)
        self.assertEqual(imports[0]['module'], 'os')
        
        # 获取函数定义
        functions = self.parser.analyzer.get_functions(result.ast)
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0]['name'], 'hello')
        
        # 获取类定义
        classes = self.parser.analyzer.get_classes(result.ast)
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]['name'], 'Person')
        self.assertEqual(len(classes[0]['methods']), 2)


class TestJSTypeScriptParser(unittest.TestCase):
    """JavaScript/TypeScript解析器测试"""
    
    def setUp(self):
        self.js_parser = JSTypeScriptParser(use_typescript=False)
        self.ts_parser = JSTypeScriptParser(use_typescript=True)
    
    def test_parse_js_code(self):
        """测试解析JavaScript代码"""
        code = """
function hello(name) {
    return `Hello, ${name}!`;
}

class Person {
    constructor(name) {
        this.name = name;
    }
    
    greet() {
        return hello(this.name);
    }
}
"""
        result = self.js_parser.parse_code(code)
        
        # 由于我们使用的是简化实现，这里只测试基本功能
        self.assertTrue(result.success)
        self.assertIsNotNone(result.ast)
        self.assertEqual(len(result.errors), 0)
    
    def test_parse_ts_code(self):
        """测试解析TypeScript代码"""
        code = """
function hello(name: string): string {
    return `Hello, ${name}!`;
}

interface Person {
    name: string;
    greet(): string;
}

class PersonImpl implements Person {
    constructor(public name: string) {}
    
    greet(): string {
        return hello(this.name);
    }
}
"""
        result = self.ts_parser.parse_code(code)
        
        # 由于我们使用的是简化实现，这里只测试基本功能
        self.assertTrue(result.success)
        self.assertIsNotNone(result.ast)
        self.assertEqual(len(result.errors), 0)


class TestASTToIRConverter(unittest.TestCase):
    """AST到IR转换器测试"""
    
    def setUp(self):
        self.python_parser = PythonParser()
        self.python_converter = PythonASTToIRConverter()
    
    def test_python_to_ir_conversion(self):
        """测试Python AST到IR的转换"""
        code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"

class Person:
    def __init__(self, name: str):
        self.name = name
        
    def greet(self) -> str:
        return hello(self.name)
"""
        parse_result = self.python_parser.parse_code(code)
        
        # 转换到IR
        ir_module = self.python_converter.convert(parse_result.ast)
        
        # 验证IR结构
        self.assertIsInstance(ir_module, Module)
        self.assertEqual(len(ir_module.functions), 1)
        self.assertEqual(ir_module.functions[0].name, 'hello')
        self.assertEqual(len(ir_module.classes), 1)
        self.assertEqual(ir_module.classes[0].name, 'Person')
        self.assertEqual(len(ir_module.classes[0].methods), 2)


class TestIRToCodeGenerator(unittest.TestCase):
    """IR到代码生成器测试"""
    
    def setUp(self):
        self.python_parser = PythonParser()
        self.python_converter = PythonASTToIRConverter()
        self.python_generator = PythonCodeGenerator()
        self.js_generator = JavaScriptCodeGenerator()
    
    def test_ir_to_python_code(self):
        """测试IR到Python代码的生成"""
        code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
"""
        parse_result = self.python_parser.parse_code(code)
        
        # 转换到IR
        ir_module = self.python_converter.convert(parse_result.ast)
        
        # 生成Python代码
        generated_code = self.python_generator.generate(ir_module)
        
        # 验证生成的代码
        self.assertIn('def hello', generated_code)
        self.assertIn('return f"Hello, {name}!"', generated_code)
    
    def test_ir_to_js_code(self):
        """测试IR到JavaScript代码的生成"""
        code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
"""
        parse_result = self.python_parser.parse_code(code)
        
        # 转换到IR
        ir_module = self.python_converter.convert(parse_result.ast)
        
        # 生成JavaScript代码
        generated_code = self.js_generator.generate(ir_module)
        
        # 验证生成的代码
        self.assertIn('function hello', generated_code)
        self.assertIn('return `Hello, ${name}!`', generated_code)


if __name__ == '__main__':
    unittest.main()