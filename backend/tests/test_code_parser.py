import unittest
import asyncio
from unittest.mock import patch, MagicMock

from app.services.code_parser.parser import CodeParserService, FeatureType, Language
from app.services.code_parser.parsers.python_parser import PythonParser
from app.services.code_parser.parsers.java_parser import JavaParser
from app.services.code_parser.parsers.javascript_parser import JavaScriptParser
from app.services.code_parser.parsers.go_parser import GoParser


class TestCodeParserService(unittest.TestCase):
    """代码解析服务测试类"""

    def setUp(self):
        """设置测试环境"""
        self.parser_service = CodeParserService()
        self.python_code = """
def hello_world():
    print("Hello, World!")
    
    if True:
        print("Condition is true")
    
    for i in range(5):
        print(i)
        
hello_world()
"""
        self.java_code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        
        if (true) {
            System.out.println("Condition is true");
        }
        
        for (int i = 0; i < 5; i++) {
            System.out.println(i);
        }
    }
}
"""
        self.js_code = """
function helloWorld() {
    console.log("Hello, World!");
    
    if (true) {
        console.log("Condition is true");
    }
    
    for (let i = 0; i < 5; i++) {
        console.log(i);
    }
}

helloWorld();
"""
        self.go_code = """
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
    
    if true {
        fmt.Println("Condition is true")
    }
    
    for i := 0; i < 5; i++ {
        fmt.Println(i)
    }
}
"""

    def test_parser_factory(self):
        """测试解析器工厂"""
        python_parser = self.parser_service.parser_factory.get_parser(Language.PYTHON)
        java_parser = self.parser_service.parser_factory.get_parser(Language.JAVA)
        js_parser = self.parser_service.parser_factory.get_parser(Language.JAVASCRIPT)
        go_parser = self.parser_service.parser_factory.get_parser(Language.GO)
        
        self.assertIsInstance(python_parser, PythonParser)
        self.assertIsInstance(java_parser, JavaParser)
        self.assertIsInstance(js_parser, JavaScriptParser)
        self.assertIsInstance(go_parser, GoParser)
    
    @patch.object(PythonParser, 'parse')
    def test_parse_python(self, mock_parse):
        """测试Python代码解析"""
        # 设置模拟返回值
        mock_parse.return_value = asyncio.Future()
        mock_parse.return_value.set_result({
            FeatureType.AST: {"type": "module"},
            FeatureType.METRICS: {"cyclomatic_complexity": 3}
        })
        
        # 执行测试
        result = asyncio.run(self.parser_service.parse(
            self.python_code,
            Language.PYTHON,
            [FeatureType.AST, FeatureType.METRICS]
        ))
        
        # 验证结果
        self.assertIn(FeatureType.AST, result)
        self.assertIn(FeatureType.METRICS, result)
        self.assertEqual(result[FeatureType.METRICS]["cyclomatic_complexity"], 3)
        
        # 验证调用
        mock_parse.assert_called_once_with(
            self.python_code,
            [FeatureType.AST, FeatureType.METRICS]
        )
    
    @patch.object(JavaParser, 'parse')
    def test_parse_java(self, mock_parse):
        """测试Java代码解析"""
        # 设置模拟返回值
        mock_parse.return_value = asyncio.Future()
        mock_parse.return_value.set_result({
            FeatureType.AST: {"type": "compilation_unit"},
            FeatureType.METRICS: {"cyclomatic_complexity": 3}
        })
        
        # 执行测试
        result = asyncio.run(self.parser_service.parse(
            self.java_code,
            Language.JAVA,
            [FeatureType.AST, FeatureType.METRICS]
        ))
        
        # 验证结果
        self.assertIn(FeatureType.AST, result)
        self.assertIn(FeatureType.METRICS, result)
        self.assertEqual(result[FeatureType.METRICS]["cyclomatic_complexity"], 3)
        
        # 验证调用
        mock_parse.assert_called_once_with(
            self.java_code,
            [FeatureType.AST, FeatureType.METRICS]
        )
    
    @patch.object(CodeParserService, 'parse')
    def test_batch_parse(self, mock_parse):
        """测试批量解析"""
        # 设置模拟返回值
        async def mock_parse_side_effect(code, language, features):
            if language == Language.PYTHON:
                return {FeatureType.AST: {"type": "module"}}
            elif language == Language.JAVA:
                return {FeatureType.AST: {"type": "compilation_unit"}}
            return {}
        
        mock_parse.side_effect = mock_parse_side_effect
        
        # 执行测试
        batch_requests = [
            {"code": self.python_code, "language": Language.PYTHON, "features": [FeatureType.AST]},
            {"code": self.java_code, "language": Language.JAVA, "features": [FeatureType.AST]}
        ]
        
        results = asyncio.run(self.parser_service.batch_parse(batch_requests))
        
        # 验证结果
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_parse.call_count, 2)
    
    @patch.object(CodeParserService, 'parse')
    def test_concurrent_parse(self, mock_parse):
        """测试并发解析"""
        # 设置模拟返回值
        async def mock_parse_side_effect(code, language, features):
            await asyncio.sleep(0.1)  # 模拟解析延迟
            if language == Language.PYTHON:
                return {FeatureType.AST: {"type": "module"}}
            elif language == Language.JAVA:
                return {FeatureType.AST: {"type": "compilation_unit"}}
            return {}
        
        mock_parse.side_effect = mock_parse_side_effect
        
        # 执行测试
        batch_requests = [
            {"code": self.python_code, "language": Language.PYTHON, "features": [FeatureType.AST]},
            {"code": self.java_code, "language": Language.JAVA, "features": [FeatureType.AST]},
            {"code": self.js_code, "language": Language.JAVASCRIPT, "features": [FeatureType.AST]},
            {"code": self.go_code, "language": Language.GO, "features": [FeatureType.AST]}
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = asyncio.run(self.parser_service.batch_parse(batch_requests))
        end_time = asyncio.get_event_loop().time()
        
        # 验证结果
        self.assertEqual(len(results), 4)
        self.assertEqual(mock_parse.call_count, 4)
        
        # 验证并发执行（总时间应该小于串行执行的时间）
        # 如果是串行执行，总时间应该约为0.4秒（4个请求 * 0.1秒）
        # 如果是并发执行，总时间应该约为0.1秒
        self.assertLess(end_time - start_time, 0.3)


if __name__ == '__main__':
    unittest.main()