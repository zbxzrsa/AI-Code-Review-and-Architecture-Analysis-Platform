"""
转换质量保证和渐进式转换的单元测试
"""
import unittest
from unittest.mock import patch, MagicMock

from backend.app.services.quality_assurance.validator import ConversionValidator
from backend.app.services.quality_assurance.metrics import QualityMetrics
from backend.app.services.quality_assurance.incremental import IncrementalConverter
from backend.app.services.quality_assurance.recovery import ErrorRecoverySystem
from backend.app.services.quality_assurance.integration import ConversionQualityManager


class TestConversionValidator(unittest.TestCase):
    """测试转换验证器"""
    
    def setUp(self):
        self.validator = ConversionValidator()
    
    def test_validate_python_syntax(self):
        """测试Python语法验证"""
        # 有效的Python代码
        valid_code = "def hello():\n    print('Hello, World!')\n"
        result = self.validator.validate_syntax(valid_code, "python")
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)
        
        # 无效的Python代码
        invalid_code = "def hello()\n    print('Hello, World!')\n"
        result = self.validator.validate_syntax(invalid_code, "python")
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["errors"]), 0)
    
    def test_validate_javascript_syntax(self):
        """测试JavaScript语法验证"""
        # 使用模拟替代实际的Node.js调用
        with patch('subprocess.run') as mock_run:
            # 模拟成功的语法检查
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            
            valid_code = "function hello() {\n  console.log('Hello, World!');\n}\n"
            result = self.validator.validate_syntax(valid_code, "javascript")
            self.assertTrue(result["valid"])
            
            # 模拟失败的语法检查
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="SyntaxError: Unexpected token")
            
            invalid_code = "function hello() {\n  console.log('Hello, World!')\n"
            result = self.validator.validate_syntax(invalid_code, "javascript")
            self.assertFalse(result["valid"])
    
    def test_generate_test_cases(self):
        """测试测试用例生成"""
        # 简单的Python函数
        code = """
def add(a, b):
    return a + b
        """
        test_cases = self.validator.generate_test_cases(code)
        self.assertIsInstance(test_cases, list)
        self.assertGreater(len(test_cases), 0)
        
        # 检查测试用例格式
        for test_case in test_cases:
            self.assertIn("inputs", test_case)
            self.assertIn("expected_output", test_case)


class TestQualityMetrics(unittest.TestCase):
    """测试质量评分系统"""
    
    def setUp(self):
        self.metrics = QualityMetrics()
    
    def test_analyze_readability(self):
        """测试可读性分析"""
        # 高可读性代码
        readable_code = """
def calculate_average(numbers):
    '''计算平均值'''
    if not numbers:
        return 0
    
    total = sum(numbers)
    return total / len(numbers)
"""
        result = self.metrics.analyze_readability(readable_code)
        self.assertGreaterEqual(result["score"], 70)
        
        # 低可读性代码
        unreadable_code = """
def x(a):
 b=0
 for i in a:b+=i
 return b/len(a)if a else 0
"""
        result = self.metrics.analyze_readability(unreadable_code)
        self.assertLess(result["score"], 70)
    
    def test_check_best_practices(self):
        """测试最佳实践检查"""
        # Python代码
        python_code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result
"""
        result = self.metrics.check_best_practices(python_code, "python")
        self.assertIn("suggestions", result)
        
        # JavaScript代码
        js_code = """
function processData(data) {
    var result = [];
    for (var i = 0; i < data.length; i++) {
        result.push(data[i] * 2);
    }
    return result;
}
"""
        result = self.metrics.check_best_practices(js_code, "javascript")
        self.assertIn("suggestions", result)
    
    def test_calculate_conversion_score(self):
        """测试转换质量综合评分"""
        source_code = """
def process_list(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result
"""
        target_code = """
function processList(items) {
    const result = [];
    for (const item of items) {
        if (item > 0) {
            result.push(item * 2);
        }
    }
    return result;
}
"""
        result = self.metrics.calculate_conversion_score(source_code, target_code)
        self.assertIn("overall_score", result)
        self.assertIn("readability", result)
        self.assertIn("maintainability", result)
        self.assertIn("best_practices", result)


class TestIncrementalConverter(unittest.TestCase):
    """测试渐进式转换引擎"""
    
    def setUp(self):
        self.converter = IncrementalConverter()
    
    def test_can_convert(self):
        """测试可转换性检查"""
        # 可转换的Python到JavaScript代码
        python_code = """
def greet(name):
    return "Hello, " + name
"""
        can_convert, issues = self.converter.can_convert(python_code, "python", "javascript")
        self.assertTrue(can_convert)
        self.assertEqual(len(issues), 0)
        
        # 不支持的语言对
        can_convert, issues = self.converter.can_convert(python_code, "python", "unknown")
        self.assertFalse(can_convert)
        self.assertGreater(len(issues), 0)
    
    def test_partial_convert(self):
        """测试部分转换"""
        # 包含可转换和不可转换部分的Python代码
        python_code = """
def simple_function(x):
    return x * 2

# 使用Python特有的列表推导式
def complex_function(items):
    return [x**2 for x in items if x > 0]
"""
        result = self.converter.partial_convert(python_code, "python", "javascript")
        self.assertIn("converted_code", result)
        self.assertIn("converted_blocks", result)
        self.assertIn("unconverted_blocks", result)
        self.assertIn("completion_percentage", result)
    
    def test_suggest_workarounds(self):
        """测试变通方案建议"""
        # Django视图函数
        django_code = """
from django.http import JsonResponse

def api_view(request):
    data = {'message': 'Hello'}
    return JsonResponse(data)
"""
        suggestions = self.converter.suggest_workarounds(django_code, "python", "javascript")
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)


class TestErrorRecoverySystem(unittest.TestCase):
    """测试错误恢复系统"""
    
    def setUp(self):
        self.recovery = ErrorRecoverySystem()
    
    def test_handle_syntax_error(self):
        """测试语法错误处理"""
        # JavaScript语法错误
        error = "SyntaxError: Unexpected token }"
        context = {
            "code": "function test() {\n  console.log('test');\n}}\n",
            "source_lang": "javascript",
            "target_lang": "javascript"
        }
        
        result = self.recovery.handle_syntax_error(error, context)
        self.assertIn("fixed_code", result)
        self.assertIn("description", result)
        self.assertIn("confidence", result)
    
    def test_handle_semantic_gap(self):
        """测试语义鸿沟处理"""
        # Python列表推导式
        feature = "list comprehension"
        target_lang = "javascript"
        code_context = "[x*2 for x in range(10)]"
        
        result = self.recovery.handle_semantic_gap(feature, target_lang, code_context)
        self.assertIn("suggestions", result)
    
    def test_provide_fallback(self):
        """测试降级方案提供"""
        original_code = "def test():\n    yield from range(10)\n"
        error_type = "semantic_gap"
        context = {"source_lang": "python", "target_lang": "javascript"}
        
        result = self.recovery.provide_fallback(original_code, error_type, context)
        self.assertIn("fallback_options", result)
        self.assertGreater(len(result["fallback_options"]), 0)


class TestConversionQualityManager(unittest.TestCase):
    """测试转换质量管理器"""
    
    def setUp(self):
        self.manager = ConversionQualityManager()
    
    @patch('backend.app.services.quality_assurance.validator.ConversionValidator.validate_syntax')
    @patch('backend.app.services.quality_assurance.metrics.QualityMetrics.calculate_conversion_score')
    def test_process_conversion(self, mock_calculate_score, mock_validate_syntax):
        """测试转换处理"""
        # 模拟验证和评分结果
        mock_validate_syntax.return_value = {"valid": True, "errors": []}
        mock_calculate_score.return_value = {
            "overall_score": 85,
            "readability": {"score": 80},
            "maintainability": {"score": 90},
            "best_practices": {"score": 85}
        }
        
        source_code = "def test(): pass"
        converted_code = "function test() {}"
        
        result = self.manager.process_conversion(
            source_code, converted_code, "python", "javascript"
        )
        
        self.assertIn("validation", result)
        self.assertIn("quality_metrics", result)
        self.assertTrue(result["validation"]["syntax"]["valid"])
    
    @patch('backend.app.services.quality_assurance.incremental.IncrementalConverter.can_convert')
    @patch('backend.app.services.quality_assurance.incremental.IncrementalConverter.partial_convert')
    def test_perform_incremental_conversion(self, mock_partial_convert, mock_can_convert):
        """测试渐进式转换执行"""
        # 模拟不可完全转换的情况
        mock_can_convert.return_value = (False, ["不支持的特性"])
        mock_partial_convert.return_value = {
            "converted_code": "// 部分转换的代码",
            "converted_blocks": [{"block_index": 0, "converted": "// 转换的块"}],
            "unconverted_blocks": [{"block_index": 1, "code": "# 未转换的块", "reason": "不支持的特性"}],
            "completion_percentage": 50
        }
        
        result = self.manager.perform_incremental_conversion(
            "def test(): pass", "python", "javascript"
        )
        
        self.assertIn("can_fully_convert", result)
        self.assertFalse(result["can_fully_convert"])
        self.assertIn("workarounds", result)


if __name__ == '__main__':
    unittest.main()