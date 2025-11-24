"""
AI模型性能测试
测试模型推理速度、内存使用、并发处理等性能指标
"""
import pytest
import asyncio
import time
import psutil
import threading
from typing import List
from unittest.mock import patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor

from app.services.ai_model_service import AIModelService


class TestInferencePerformance:
    """推理性能测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_embedding_inference_speed(self, ai_service):
        """测试嵌入推理速度"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            # 模拟快速响应
            mock_embed.return_value = [0.1] * 768  # 典型的嵌入维度
            
            code_samples = [
                "def hello(): return 'world'",
                "class User: pass",
                "for i in range(10): print(i)",
                "import numpy as np",
                "x = [1, 2, 3, 4, 5]"
            ]
            
            start_time = time.time()
            
            # 批量处理
            tasks = [ai_service.embed_code(code) for code in code_samples]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 验证结果
            assert len(results) == len(code_samples)
            assert all(len(result) == 768 for result in results)
            
            # 性能要求：每个样本平均处理时间不超过100ms
            avg_time_per_sample = total_time / len(code_samples)
            assert avg_time_per_sample < 0.1, f"Average inference time {avg_time_per_sample:.3f}s exceeds 0.1s threshold"
    
    @pytest.mark.asyncio
    async def test_defect_detection_speed(self, ai_service):
        """测试缺陷检测速度"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            mock_detect.return_value = {
                "defects": [
                    {
                        "type": "potential_bug",
                        "severity": "medium",
                        "confidence": 0.85,
                        "line": 2,
                        "message": "Test defect"
                    }
                ],
                "summary": {"total_defects": 1}
            }
            
            # 测试不同大小的代码
            code_samples = [
                "def small(): pass",  # 小代码
                "\n".join([f"def func_{i}(): pass" for i in range(10)]),  # 中等代码
                "\n".join([f"def func_{i}(): pass" for i in range(100)])  # 大代码
            ]
            
            for i, code in enumerate(code_samples):
                start_time = time.time()
                result = await ai_service.detect_defects(code)
                end_time = time.time()
                
                inference_time = end_time - start_time
                
                # 性能要求：根据代码大小设置不同的时间阈值
                if i == 0:  # 小代码
                    assert inference_time < 0.05, f"Small code inference time {inference_time:.3f}s exceeds 0.05s"
                elif i == 1:  # 中等代码
                    assert inference_time < 0.2, f"Medium code inference time {inference_time:.3f}s exceeds 0.2s"
                else:  # 大代码
                    assert inference_time < 1.0, f"Large code inference time {inference_time:.3f}s exceeds 1.0s"
    
    @pytest.mark.asyncio
    async def test_architecture_analysis_speed(self, ai_service):
        """测试架构分析速度"""
        with patch.object(ai_service, 'analyze_architecture') as mock_analyze:
            mock_analyze.return_value = {
                "components": [
                    {"name": "TestClass", "type": "class", "complexity": 2}
                ],
                "dependencies": [],
                "metrics": {
                    "coupling": 0.3,
                    "cohesion": 0.8,
                    "complexity": 2.0
                }
            }
            
            # 复杂的代码结构
            complex_code = """
class UserService:
    def __init__(self, db_service, email_service):
        self.db_service = db_service
        self.email_service = email_service
    
    def create_user(self, user_data):
        user = self.db_service.save(user_data)
        self.email_service.send_welcome_email(user.email)
        return user
    
    def update_user(self, user_id, updates):
        user = self.db_service.get(user_id)
        if user:
            updated_user = self.db_service.update(user_id, updates)
            return updated_user
        return None

class DatabaseService:
    def save(self, data): pass
    def get(self, id): pass
    def update(self, id, data): pass

class EmailService:
    def send_welcome_email(self, email): pass
            """
            
            start_time = time.time()
            result = await ai_service.analyze_architecture(complex_code)
            end_time = time.time()
            
            inference_time = end_time - start_time
            
            # 验证结果
            assert "components" in result
            assert "dependencies" in result
            assert "metrics" in result
            
            # 性能要求：复杂架构分析不超过500ms
            assert inference_time < 0.5, f"Architecture analysis time {inference_time:.3f}s exceeds 0.5s threshold"


class TestConcurrentPerformance:
    """并发性能测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding_requests(self, ai_service):
        """测试并发嵌入请求"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            # 模拟并发请求
            concurrent_requests = 20
            code_samples = [f"def func_{i}(): pass" for i in range(concurrent_requests)]
            
            start_time = time.time()
            
            # 并发执行
            tasks = [ai_service.embed_code(code) for code in code_samples]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 验证结果
            assert len(results) == concurrent_requests
            assert all(len(result) == 768 for result in results)
            
            # 性能要求：并发处理应该比串行处理更快
            # 假设串行处理每个请求需要10ms，并发应该显著减少总时间
            expected_serial_time = concurrent_requests * 0.01  # 10ms per request
            assert total_time < expected_serial_time * 0.5, f"Concurrent processing not efficient enough"
    
    @pytest.mark.asyncio
    async def test_concurrent_defect_detection(self, ai_service):
        """测试并发缺陷检测"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            mock_detect.return_value = {
                "defects": [],
                "summary": {"total_defects": 0}
            }
            
            # 模拟并发缺陷检测请求
            concurrent_requests = 10
            code_samples = [
                f"def risky_func_{i}(x): return x / 0" 
                for i in range(concurrent_requests)
            ]
            
            start_time = time.time()
            
            # 并发执行
            tasks = [ai_service.detect_defects(code) for code in code_samples]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 验证结果
            assert len(results) == concurrent_requests
            assert all("defects" in result for result in results)
            
            # 性能要求：并发处理时间应该合理
            avg_time_per_request = total_time / concurrent_requests
            assert avg_time_per_request < 0.1, f"Average concurrent request time {avg_time_per_request:.3f}s too high"
    
    @pytest.mark.asyncio
    async def test_mixed_concurrent_requests(self, ai_service):
        """测试混合并发请求"""
        with patch.object(ai_service, 'embed_code') as mock_embed, \
             patch.object(ai_service, 'detect_defects') as mock_detect, \
             patch.object(ai_service, 'analyze_architecture') as mock_analyze:
            
            # 设置模拟返回值
            mock_embed.return_value = [0.1] * 768
            mock_detect.return_value = {"defects": [], "summary": {"total_defects": 0}}
            mock_analyze.return_value = {
                "components": [],
                "dependencies": [],
                "metrics": {"coupling": 0.0, "cohesion": 1.0, "complexity": 1.0}
            }
            
            code = "def test(): pass"
            
            # 混合不同类型的请求
            tasks = []
            tasks.extend([ai_service.embed_code(code) for _ in range(5)])
            tasks.extend([ai_service.detect_defects(code) for _ in range(5)])
            tasks.extend([ai_service.analyze_architecture(code) for _ in range(5)])
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # 验证结果
            assert len(results) == 15
            
            # 性能要求：混合请求处理时间应该合理
            assert total_time < 1.0, f"Mixed concurrent requests time {total_time:.3f}s exceeds 1.0s"


class TestMemoryPerformance:
    """内存性能测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_inference(self, ai_service):
        """测试推理过程中的内存使用"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            # 记录初始内存使用
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 处理大量请求
            large_code_samples = [f"def func_{i}(): pass" for i in range(100)]
            
            tasks = [ai_service.embed_code(code) for code in large_code_samples]
            results = await asyncio.gather(*tasks)
            
            # 记录峰值内存使用
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # 验证结果
            assert len(results) == 100
            
            # 内存要求：内存增长不应该超过100MB
            assert memory_increase < 100, f"Memory increase {memory_increase:.2f}MB exceeds 100MB threshold"
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, ai_service):
        """测试内存泄漏检测"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            process = psutil.Process()
            memory_readings = []
            
            # 多轮处理，检测内存是否持续增长
            for round_num in range(5):
                # 每轮处理一些请求
                code_samples = [f"def func_{round_num}_{i}(): pass" for i in range(20)]
                tasks = [ai_service.embed_code(code) for code in code_samples]
                await asyncio.gather(*tasks)
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                # 记录内存使用
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_readings.append(current_memory)
                
                # 短暂等待
                await asyncio.sleep(0.1)
            
            # 检查内存是否持续增长（可能的内存泄漏）
            if len(memory_readings) >= 3:
                # 计算内存增长趋势
                memory_growth = memory_readings[-1] - memory_readings[0]
                
                # 允许一定的内存增长，但不应该过多
                assert memory_growth < 50, f"Potential memory leak detected: {memory_growth:.2f}MB growth"


class TestThroughputPerformance:
    """吞吐量性能测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_embedding_throughput(self, ai_service):
        """测试嵌入吞吐量"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            # 测试1分钟内能处理多少请求
            test_duration = 5  # 5秒测试（实际应用中可能是60秒）
            start_time = time.time()
            processed_requests = 0
            
            while time.time() - start_time < test_duration:
                # 批量处理
                batch_size = 10
                code_samples = [f"def func_{processed_requests + i}(): pass" for i in range(batch_size)]
                tasks = [ai_service.embed_code(code) for code in code_samples]
                await asyncio.gather(*tasks)
                
                processed_requests += batch_size
            
            actual_duration = time.time() - start_time
            throughput = processed_requests / actual_duration  # 请求/秒
            
            # 性能要求：至少每秒处理50个嵌入请求
            assert throughput >= 50, f"Embedding throughput {throughput:.2f} req/s below 50 req/s threshold"
    
    @pytest.mark.asyncio
    async def test_defect_detection_throughput(self, ai_service):
        """测试缺陷检测吞吐量"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            mock_detect.return_value = {
                "defects": [],
                "summary": {"total_defects": 0}
            }
            
            # 测试吞吐量
            test_duration = 5  # 5秒测试
            start_time = time.time()
            processed_requests = 0
            
            while time.time() - start_time < test_duration:
                # 批量处理
                batch_size = 5
                code_samples = [f"def risky_func_{processed_requests + i}(): pass" for i in range(batch_size)]
                tasks = [ai_service.detect_defects(code) for code in code_samples]
                await asyncio.gather(*tasks)
                
                processed_requests += batch_size
            
            actual_duration = time.time() - start_time
            throughput = processed_requests / actual_duration  # 请求/秒
            
            # 性能要求：至少每秒处理20个缺陷检测请求
            assert throughput >= 20, f"Defect detection throughput {throughput:.2f} req/s below 20 req/s threshold"


class TestScalabilityPerformance:
    """可扩展性性能测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_load_scaling(self, ai_service):
        """测试负载扩展性"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            # 测试不同负载级别的性能
            load_levels = [10, 50, 100, 200]
            performance_results = []
            
            for load in load_levels:
                code_samples = [f"def func_{i}(): pass" for i in range(load)]
                
                start_time = time.time()
                tasks = [ai_service.embed_code(code) for code in code_samples]
                results = await asyncio.gather(*tasks)
                end_time = time.time()
                
                total_time = end_time - start_time
                throughput = load / total_time
                
                performance_results.append({
                    "load": load,
                    "time": total_time,
                    "throughput": throughput
                })
                
                # 验证结果
                assert len(results) == load
            
            # 分析性能扩展性
            # 理想情况下，吞吐量应该随负载线性增长（在资源限制内）
            for i in range(1, len(performance_results)):
                current = performance_results[i]
                previous = performance_results[i-1]
                
                # 检查性能不应该急剧下降
                throughput_ratio = current["throughput"] / previous["throughput"]
                load_ratio = current["load"] / previous["load"]
                
                # 吞吐量下降不应该超过负载增长的比例
                efficiency = throughput_ratio / load_ratio
                assert efficiency > 0.5, f"Performance degradation too severe at load {current['load']}: efficiency {efficiency:.3f}"
    
    @pytest.mark.asyncio
    async def test_concurrent_user_simulation(self, ai_service):
        """测试并发用户模拟"""
        with patch.object(ai_service, 'embed_code') as mock_embed, \
             patch.object(ai_service, 'detect_defects') as mock_detect:
            
            mock_embed.return_value = [0.1] * 768
            mock_detect.return_value = {"defects": [], "summary": {"total_defects": 0}}
            
            # 模拟多个并发用户
            num_users = 10
            requests_per_user = 5
            
            async def simulate_user(user_id: int):
                """模拟单个用户的行为"""
                user_results = []
                for i in range(requests_per_user):
                    # 随机选择操作类型
                    if i % 2 == 0:
                        result = await ai_service.embed_code(f"def user_{user_id}_func_{i}(): pass")
                    else:
                        result = await ai_service.detect_defects(f"def user_{user_id}_func_{i}(): pass")
                    user_results.append(result)
                return user_results
            
            start_time = time.time()
            
            # 并发执行所有用户
            user_tasks = [simulate_user(user_id) for user_id in range(num_users)]
            all_results = await asyncio.gather(*user_tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 验证结果
            assert len(all_results) == num_users
            total_requests = num_users * requests_per_user
            
            # 性能要求：并发用户处理时间应该合理
            avg_time_per_request = total_time / total_requests
            assert avg_time_per_request < 0.1, f"Average time per request {avg_time_per_request:.3f}s too high for concurrent users"


class TestResourceUtilization:
    """资源利用率测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_cpu_utilization(self, ai_service):
        """测试CPU利用率"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            # 监控CPU使用率
            process = psutil.Process()
            cpu_percentages = []
            
            # 在高负载下监控CPU
            async def monitor_cpu():
                for _ in range(10):
                    cpu_percent = process.cpu_percent()
                    cpu_percentages.append(cpu_percent)
                    await asyncio.sleep(0.1)
            
            async def generate_load():
                # 生成计算负载
                code_samples = [f"def func_{i}(): pass" for i in range(100)]
                tasks = [ai_service.embed_code(code) for code in code_samples]
                await asyncio.gather(*tasks)
            
            # 并发执行监控和负载生成
            await asyncio.gather(monitor_cpu(), generate_load())
            
            # 分析CPU使用率
            if cpu_percentages:
                avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
                max_cpu = max(cpu_percentages)
                
                # CPU使用率应该在合理范围内
                assert avg_cpu < 80, f"Average CPU usage {avg_cpu:.2f}% too high"
                assert max_cpu < 95, f"Peak CPU usage {max_cpu:.2f}% too high"
    
    @pytest.mark.asyncio
    async def test_response_time_under_load(self, ai_service):
        """测试负载下的响应时间"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            # 测试不同负载下的响应时间
            response_times = []
            
            # 轻负载
            start_time = time.time()
            await ai_service.embed_code("def light_load(): pass")
            light_load_time = time.time() - start_time
            response_times.append(("light", light_load_time))
            
            # 中等负载
            start_time = time.time()
            medium_tasks = [ai_service.embed_code(f"def medium_{i}(): pass") for i in range(10)]
            await asyncio.gather(*medium_tasks)
            medium_load_time = (time.time() - start_time) / 10  # 平均时间
            response_times.append(("medium", medium_load_time))
            
            # 高负载
            start_time = time.time()
            heavy_tasks = [ai_service.embed_code(f"def heavy_{i}(): pass") for i in range(50)]
            await asyncio.gather(*heavy_tasks)
            heavy_load_time = (time.time() - start_time) / 50  # 平均时间
            response_times.append(("heavy", heavy_load_time))
            
            # 分析响应时间变化
            for load_type, response_time in response_times:
                if load_type == "light":
                    assert response_time < 0.05, f"Light load response time {response_time:.3f}s too high"
                elif load_type == "medium":
                    assert response_time < 0.1, f"Medium load response time {response_time:.3f}s too high"
                else:  # heavy
                    assert response_time < 0.2, f"Heavy load response time {response_time:.3f}s too high"