"""
AI模型推理单元测试
测试模型加载、推理性能、准确性等功能
"""
import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

from app.api.endpoints.ai_analysis import AIModelService, Defect, DefectLocation


class TestAIModelService:
    """AI模型服务测试"""
    
    def setup_method(self):
        """测试前设置"""
        # AIModelService is now a static class, no instantiation needed
        pass

    @pytest.mark.asyncio
    async def test_embed_code_success(self):
        """测试代码向量化成功"""
        code_structure = {
            "type": "module",
            "children": [
                {"type": "function_definition", "name": "test_func"}
            ]
        }
        language = "python"
        
        # 模拟向量化结果
        expected_vector = [0.1] * 768
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_embedding_model') as mock_load:
            mock_model = Mock()
            mock_model.encode.return_value = np.array(expected_vector)
            mock_load.return_value = mock_model
            
            result = await AIModelService.embed_code(code_structure, language)
            
            assert len(result) == 768
            assert isinstance(result, list)
            assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_code_empty_structure(self):
        """测试空代码结构的向量化"""
        code_structure = {}
        language = "python"
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_embedding_model') as mock_load:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.0] * 768)
            mock_load.return_value = mock_model
            
            result = await AIModelService.embed_code(code_structure, language)
            
            assert len(result) == 768
            assert all(x == 0.0 for x in result)

    @pytest.mark.asyncio
    async def test_detect_defects_with_code(self):
        """测试基于代码的缺陷检测"""
        code = """
def unsafe_function():
    password = "hardcoded_password"
    return password
"""
        language = "python"
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_defect_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = {
                'defects': [
                    {
                        'type': 'security_vulnerability',
                        'confidence': 0.95,
                        'location': {'start_line': 2, 'end_line': 2}
                    }
                ]
            }
            mock_load.return_value = mock_model
            
            result = await AIModelService.detect_defects(code=code, language=language)
            
            assert len(result) == 1
            assert isinstance(result[0], Defect)
            assert result[0].defect_type == "security_vulnerability"
            assert result[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_detect_defects_with_vector(self):
        """测试基于向量的缺陷检测"""
        vector = [0.1] * 768
        language = "python"
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_defect_model') as mock_load:
            mock_model = Mock()
            mock_model.predict_from_vector.return_value = {
                'defects': [
                    {
                        'type': 'code_smell',
                        'confidence': 0.75,
                        'location': {'start_line': 1, 'end_line': 5}
                    }
                ]
            }
            mock_load.return_value = mock_model
            
            result = await AIModelService.detect_defects(vector=vector, language=language)
            
            assert len(result) == 1
            assert result[0].defect_type == "code_smell"
            assert result[0].confidence == 0.75

    @pytest.mark.asyncio
    async def test_detect_defects_no_defects(self):
        """测试无缺陷代码的检测"""
        code = """
def clean_function(x):
    return x * 2
"""
        language = "python"
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_defect_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = {'defects': []}
            mock_load.return_value = mock_model
            
            result = await AIModelService.detect_defects(code=code, language=language)
            
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_analyze_architecture_success(self):
        """测试架构分析成功"""
        project_structure = {
            "files": ["main.py", "utils.py", "models.py"],
            "dependencies": {"main.py": ["utils.py", "models.py"]}
        }
        vectors = {
            "main.py": [0.1] * 768,
            "utils.py": [0.2] * 768,
            "models.py": [0.3] * 768
        }
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_architecture_model') as mock_load:
            mock_model = Mock()
            mock_model.analyze.return_value = (
                [  # patterns
                    {
                        'pattern_name': 'MVC',
                        'confidence': 0.85,
                        'components': ['main.py', 'models.py']
                    }
                ],
                [  # smells
                    {
                        'smell_type': 'god_class',
                        'severity': 'medium',
                        'affected_components': ['main.py']
                    }
                ]
            )
            mock_load.return_value = mock_model
            
            patterns, smells = await AIModelService.analyze_architecture(
                project_structure=project_structure,
                vectors=vectors
            )
            
            assert len(patterns) == 1
            assert patterns[0]['pattern_name'] == 'MVC'
            assert len(smells) == 1
            assert smells[0]['smell_type'] == 'god_class'

    @pytest.mark.asyncio
    async def test_calculate_similarity_high(self):
        """测试高相似度代码计算"""
        code1 = """
def add(a, b):
    return a + b
"""
        code2 = """
def sum_numbers(x, y):
    return x + y
"""
        language = "python"
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_similarity_model') as mock_load:
            mock_model = Mock()
            mock_model.calculate_similarity.return_value = (
                0.92,  # similarity_score
                [  # similar_segments
                    {
                        'code1_start_line': 1,
                        'code1_end_line': 2,
                        'code2_start_line': 1,
                        'code2_end_line': 2,
                        'similarity_score': 0.92
                    }
                ]
            )
            mock_load.return_value = mock_model
            
            similarity_score, segments = await AIModelService.calculate_similarity(
                code1, code2, language, detailed_analysis=True
            )
            
            assert similarity_score == 0.92
            assert len(segments) == 1
            assert segments[0]['similarity_score'] == 0.92

    @pytest.mark.asyncio
    async def test_calculate_similarity_low(self):
        """测试低相似度代码计算"""
        code1 = """
def add(a, b):
    return a + b
"""
        code2 = """
class DatabaseConnection:
    def __init__(self, host):
        self.host = host
"""
        language = "python"
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_similarity_model') as mock_load:
            mock_model = Mock()
            mock_model.calculate_similarity.return_value = (0.15, [])
            mock_load.return_value = mock_model
            
            similarity_score, segments = await AIModelService.calculate_similarity(
                code1, code2, language, detailed_analysis=False
            )
            
            assert similarity_score == 0.15
            assert len(segments) == 0


class TestModelPerformance:
    """模型性能测试"""
    
    @pytest.mark.asyncio
    async def test_embedding_performance(self):
        """测试向量化性能"""
        service = AIModelService()
        code_structure = {
            "type": "module",
            "children": [{"type": "function_definition"}] * 100  # 大型代码结构
        }
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_embedding_model') as mock_load:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.1] * 768)
            mock_load.return_value = mock_model
            
            start_time = time.time()
            result = await service.embed_code(code_structure, "python")
            end_time = time.time()
            
            # 性能要求：处理时间应小于1秒
            processing_time = end_time - start_time
            assert processing_time < 1.0
            assert len(result) == 768

    @pytest.mark.asyncio
    async def test_defect_detection_performance(self):
        """测试缺陷检测性能"""
        service = AIModelService()
        # 模拟大型代码文件
        large_code = "def function():\n    pass\n" * 1000
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_defect_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = {'defects': []}
            mock_load.return_value = mock_model
            
            start_time = time.time()
            result = await service.detect_defects(code=large_code, language="python")
            end_time = time.time()
            
            # 性能要求：处理时间应小于5秒
            processing_time = end_time - start_time
            assert processing_time < 5.0

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self):
        """测试批量处理性能"""
        service = AIModelService()
        
        # 模拟批量向量化请求
        batch_requests = [
            {"type": "module", "children": [{"type": "function_definition"}]}
            for _ in range(10)
        ]
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_embedding_model') as mock_load:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.1] * 768)
            mock_load.return_value = mock_model
            
            start_time = time.time()
            results = []
            for request in batch_requests:
                result = await service.embed_code(request, "python")
                results.append(result)
            end_time = time.time()
            
            # 性能要求：批量处理平均每个请求应小于0.5秒
            avg_time_per_request = (end_time - start_time) / len(batch_requests)
            assert avg_time_per_request < 0.5
            assert len(results) == 10


class TestModelAccuracy:
    """模型准确性测试"""
    
    @pytest.mark.asyncio
    async def test_defect_detection_accuracy(self):
        """测试缺陷检测准确性"""
        service = AIModelService()
        
        # 已知包含安全漏洞的代码
        vulnerable_code = """
import os
password = "admin123"
os.system("rm -rf " + user_input)
"""
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_defect_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = {
                'defects': [
                    {
                        'type': 'security_vulnerability',
                        'confidence': 0.95,
                        'location': {'start_line': 2, 'end_line': 2}
                    },
                    {
                        'type': 'command_injection',
                        'confidence': 0.88,
                        'location': {'start_line': 3, 'end_line': 3}
                    }
                ]
            }
            mock_load.return_value = mock_model
            
            result = await service.detect_defects(code=vulnerable_code, language="python")
            
            # 应该检测到安全漏洞
            assert len(result) >= 1
            defect_types = [defect.defect_type for defect in result]
            assert 'security_vulnerability' in defect_types or 'command_injection' in defect_types
            
            # 置信度应该较高
            high_confidence_defects = [d for d in result if d.confidence > 0.8]
            assert len(high_confidence_defects) >= 1

    @pytest.mark.asyncio
    async def test_clean_code_detection(self):
        """测试干净代码的检测"""
        service = AIModelService()
        
        # 干净的代码示例
        clean_code = """
def calculate_area(radius: float) -> float:
    \"\"\"Calculate the area of a circle.\"\"\"
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return 3.14159 * radius ** 2
"""
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_defect_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = {'defects': []}
            mock_load.return_value = mock_model
            
            result = await service.detect_defects(code=clean_code, language="python")
            
            # 干净代码应该检测不到缺陷或只有低置信度的缺陷
            high_confidence_defects = [d for d in result if d.confidence > 0.7]
            assert len(high_confidence_defects) == 0

    @pytest.mark.asyncio
    async def test_similarity_accuracy(self):
        """测试相似度计算准确性"""
        service = AIModelService()
        
        # 相同的代码
        identical_code1 = "def add(a, b): return a + b"
        identical_code2 = "def add(a, b): return a + b"
        
        # 完全不同的代码
        different_code1 = "def add(a, b): return a + b"
        different_code2 = "class Car: def __init__(self): pass"
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_similarity_model') as mock_load:
            mock_model = Mock()
            
            # 模拟相同代码的高相似度
            mock_model.calculate_similarity.return_value = (1.0, [])
            similarity_identical, _ = await service.calculate_similarity(
                identical_code1, identical_code2, "python"
            )
            
            # 模拟不同代码的低相似度
            mock_model.calculate_similarity.return_value = (0.1, [])
            similarity_different, _ = await service.calculate_similarity(
                different_code1, different_code2, "python"
            )
            
            # 相同代码相似度应该很高
            assert similarity_identical > 0.9
            
            # 不同代码相似度应该很低
            assert similarity_different < 0.3


class TestModelRobustness:
    """模型鲁棒性测试"""
    
    @pytest.mark.asyncio
    async def test_malformed_input_handling(self):
        """测试畸形输入处理"""
        service = AIModelService()
        
        # 畸形代码结构
        malformed_structure = {
            "invalid_key": "invalid_value",
            "nested": {"deeply": {"nested": {"structure": None}}}
        }
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_embedding_model') as mock_load:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.0] * 768)
            mock_load.return_value = mock_model
            
            # 应该能处理畸形输入而不崩溃
            result = await service.embed_code(malformed_structure, "python")
            assert len(result) == 768

    @pytest.mark.asyncio
    async def test_large_input_handling(self):
        """测试大输入处理"""
        service = AIModelService()
        
        # 非常大的代码结构
        large_structure = {
            "type": "module",
            "children": [
                {"type": "function_definition", "name": f"func_{i}"}
                for i in range(10000)
            ]
        }
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_embedding_model') as mock_load:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.1] * 768)
            mock_load.return_value = mock_model
            
            # 应该能处理大输入
            result = await service.embed_code(large_structure, "python")
            assert len(result) == 768

    @pytest.mark.asyncio
    async def test_unicode_handling(self):
        """测试Unicode字符处理"""
        service = AIModelService()
        
        # 包含Unicode字符的代码
        unicode_code = """
def 计算面积(半径):
    \"\"\"计算圆的面积\"\"\"
    return 3.14159 * 半径 ** 2

# 测试中文注释和变量名
结果 = 计算面积(5.0)
"""
        
        with patch('app.api.endpoints.ai_analysis.AIModelService._load_defect_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = {'defects': []}
            mock_load.return_value = mock_model
            
            # 应该能处理Unicode字符
            result = await service.detect_defects(code=unicode_code, language="python")
            assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__])