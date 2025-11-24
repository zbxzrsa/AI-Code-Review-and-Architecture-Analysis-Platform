"""
核心算法单元测试
测试代码解析器、复杂度计算、AST生成等核心算法功能
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from app.services.code_parser.parsers.python_parser import PythonParser
from app.services.code_parser.parsers.javascript_parser import JavaScriptParser
from app.services.code_parser.parsers.go_parser import GoParser
from app.services.code_parser.parser import CodeParserService, FeatureType, Language


class TestPythonParser:
    """Python解析器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.parser = PythonParser()
        self.simple_code = """
def hello_world():
    print("Hello, World!")
    return True
"""
        self.complex_code = """
def complex_function(x, y):
    if x > 0:
        if y > 0:
            return x + y
        elif y < 0:
            return x - y
        else:
            return x
    else:
        for i in range(10):
            if i % 2 == 0:
                print(i)
        return 0
"""

    @pytest.mark.asyncio
    async def test_generate_ast_simple(self):
        """测试简单代码的AST生成"""
        with patch('app.services.code_parser.parsers.python_parser.ts_parser') as mock_ts:
            mock_ts.parse_code.return_value = {
                "type": "module",
                "children": [{"type": "function_definition", "name": "hello_world"}]
            }
            
            with patch('app.services.code_parser.parsers.python_parser.standardize_ast') as mock_std:
                mock_std.return_value = {"type": "module", "functions": ["hello_world"]}
                
                result = await self.parser.generate_ast(self.simple_code)
                
                assert result is not None
                assert "type" in result
                mock_ts.parse_code.assert_called_once_with(self.simple_code, "python")

    @pytest.mark.asyncio
    async def test_generate_ast_invalid_code(self):
        """测试无效代码的AST生成"""
        invalid_code = "def invalid_syntax("
        
        with patch('app.services.code_parser.parsers.python_parser.ts_parser') as mock_ts:
            mock_ts.parse_code.side_effect = Exception("Syntax error")
            
            with pytest.raises(ValueError, match="生成AST失败"):
                await self.parser.generate_ast(invalid_code)

    @pytest.mark.asyncio
    async def test_calculate_complexity_simple(self):
        """测试简单代码的复杂度计算"""
        ast = {
            "type": "module",
            "children": [
                {
                    "type": "function_definition",
                    "children": [
                        {"type": "expression_statement"}
                    ]
                }
            ]
        }
        
        complexity = await self.parser._calculate_complexity_recursive(ast, 1)
        assert complexity == 1  # 基础复杂度

    @pytest.mark.asyncio
    async def test_calculate_complexity_with_conditions(self):
        """测试包含条件语句的复杂度计算"""
        ast = {
            "type": "module",
            "children": [
                {
                    "type": "if_statement",
                    "children": [
                        {
                            "type": "for_statement",
                            "children": [
                                {"type": "if_statement"}
                            ]
                        }
                    ]
                }
            ]
        }
        
        complexity = await self.parser._calculate_complexity_recursive(ast, 1)
        assert complexity == 4  # 1 + 1(if) + 1(for) + 1(nested if)

    @pytest.mark.asyncio
    async def test_generate_cfg(self):
        """测试控制流图生成"""
        ast = {
            "type": "module",
            "children": [
                {"type": "function_definition", "name": "test_func"}
            ]
        }
        
        cfg = await self.parser.generate_cfg(ast)
        
        assert "nodes" in cfg
        assert "edges" in cfg
        assert isinstance(cfg["nodes"], list)
        assert isinstance(cfg["edges"], list)

    @pytest.mark.asyncio
    async def test_parse_with_all_features(self):
        """测试解析所有特征"""
        features = [FeatureType.AST, FeatureType.CFG, FeatureType.DFG, FeatureType.METRICS]
        
        with patch.object(self.parser, 'generate_ast') as mock_ast:
            mock_ast.return_value = {"type": "module"}
            
            with patch.object(self.parser, 'generate_cfg') as mock_cfg:
                mock_cfg.return_value = {"nodes": [], "edges": []}
                
                with patch.object(self.parser, 'generate_dfg') as mock_dfg:
                    mock_dfg.return_value = {"variables": []}
                    
                    with patch.object(self.parser, 'calculate_metrics') as mock_metrics:
                        mock_metrics.return_value = {"complexity": 1}
                        
                        result = await self.parser.parse(self.simple_code, features)
                        
                        assert FeatureType.AST in result
                        assert FeatureType.CFG in result
                        assert FeatureType.DFG in result
                        assert FeatureType.METRICS in result


class TestJavaScriptParser:
    """JavaScript解析器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.parser = JavaScriptParser()
        self.js_code = """
function testFunction(x) {
    if (x > 0) {
        return x * 2;
    } else {
        return 0;
    }
}
"""

    @pytest.mark.asyncio
    async def test_complexity_calculation(self):
        """测试JavaScript复杂度计算"""
        ast = {
            "type": "program",
            "children": [
                {
                    "type": "if_statement",
                    "children": [
                        {"type": "else_clause"}
                    ]
                }
            ]
        }
        
        complexity = await self.parser._calculate_complexity_recursive(ast, 1)
        assert complexity == 3  # 1 + 1(if) + 1(else)


class TestGoParser:
    """Go解析器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.parser = GoParser()
        self.go_code = """
func testFunction(x int) int {
    if x > 0 {
        return x * 2
    }
    return 0
}
"""

    @pytest.mark.asyncio
    async def test_complexity_calculation(self):
        """测试Go复杂度计算"""
        ast = {
            "type": "source_file",
            "children": [
                {
                    "type": "if_statement",
                    "children": [
                        {
                            "type": "for_statement",
                            "children": []
                        }
                    ]
                }
            ]
        }
        
        complexity = await self.parser._calculate_complexity_recursive(ast, 1)
        assert complexity == 3  # 1 + 1(if) + 1(for)


class TestCodeParserService:
    """代码解析服务测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.service = CodeParserService()

    def test_get_parser_python(self):
        """测试获取Python解析器"""
        parser = self.service._get_parser(Language.PYTHON)
        assert isinstance(parser, PythonParser)

    def test_get_parser_javascript(self):
        """测试获取JavaScript解析器"""
        parser = self.service._get_parser(Language.JAVASCRIPT)
        assert isinstance(parser, JavaScriptParser)

    def test_get_parser_go(self):
        """测试获取Go解析器"""
        parser = self.service._get_parser(Language.GO)
        assert isinstance(parser, GoParser)

    def test_get_parser_unsupported(self):
        """测试不支持的语言"""
        with pytest.raises(ValueError, match="不支持的编程语言"):
            self.service._get_parser("unsupported")

    @pytest.mark.asyncio
    async def test_parse_success(self):
        """测试成功解析"""
        code = "def test(): pass"
        language = Language.PYTHON
        features = [FeatureType.AST]
        
        with patch.object(self.service, '_get_parser') as mock_get_parser:
            mock_parser = AsyncMock()
            mock_parser.parse.return_value = {FeatureType.AST: {"type": "module"}}
            mock_get_parser.return_value = mock_parser
            
            result = await self.service.parse(code, language, features)
            
            assert FeatureType.AST in result
            mock_parser.parse.assert_called_once_with(code, features)

    @pytest.mark.asyncio
    async def test_batch_parse(self):
        """测试批量解析"""
        requests = [
            {"code": "def test1(): pass", "language": Language.PYTHON, "features": [FeatureType.AST]},
            {"code": "function test2() {}", "language": Language.JAVASCRIPT, "features": [FeatureType.AST]}
        ]
        
        with patch.object(self.service, 'parse') as mock_parse:
            mock_parse.side_effect = [
                {FeatureType.AST: {"type": "module"}},
                {FeatureType.AST: {"type": "program"}}
            ]
            
            results = await self.service.batch_parse(requests)
            
            assert len(results) == 2
            assert mock_parse.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_parse_with_errors(self):
        """测试批量解析包含错误"""
        requests = [
            {"code": "def test1(): pass", "language": Language.PYTHON, "features": [FeatureType.AST]},
            {"code": "invalid syntax", "language": Language.PYTHON, "features": [FeatureType.AST]}
        ]
        
        with patch.object(self.service, 'parse') as mock_parse:
            mock_parse.side_effect = [
                {FeatureType.AST: {"type": "module"}},
                Exception("Parse error")
            ]
            
            results = await self.service.batch_parse(requests)
            
            assert len(results) == 2
            assert "error" not in results[0]
            assert "error" in results[1]


if __name__ == "__main__":
    pytest.main([__file__])