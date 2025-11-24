"""
AI模型准确性测试
测试模型在各种代码分析任务上的准确性
"""
import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock
import json

from app.services.ai_model_service import AIModelService


class TestDefectDetectionAccuracy:
    """缺陷检测准确性测试"""
    
    @pytest.fixture
    def ai_service(self):
        """AI服务实例"""
        return AIModelService()
    
    @pytest.fixture
    def known_defective_code_samples(self):
        """已知有缺陷的代码样本"""
        return [
            {
                "code": """
def divide(a, b):
    return a / b  # 潜在的除零错误
                """,
                "expected_defects": ["division_by_zero"],
                "severity": "high"
            },
            {
                "code": """
def process_user_input(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    return execute_query(query)  # SQL注入漏洞
                """,
                "expected_defects": ["sql_injection"],
                "severity": "critical"
            },
            {
                "code": """
def get_user_data(user_id):
    user = None
    if user_id > 0:
        user = fetch_user(user_id)
    return user.name  # 潜在的空指针引用
                """,
                "expected_defects": ["null_pointer_dereference"],
                "severity": "medium"
            },
            {
                "code": """
import pickle
def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)  # 不安全的反序列化
                """,
                "expected_defects": ["unsafe_deserialization"],
                "severity": "high"
            },
            {
                "code": """
def authenticate(username, password):
    if username == "admin" and password == "password123":
        return True  # 硬编码凭据
    return False
                """,
                "expected_defects": ["hardcoded_credentials"],
                "severity": "critical"
            }
        ]
    
    @pytest.fixture
    def clean_code_samples(self):
        """干净的代码样本"""
        return [
            {
                "code": """
def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b
                """,
                "expected_defects": [],
                "quality_score": 95
            },
            {
                "code": """
def process_user_input(user_input):
    # 使用参数化查询防止SQL注入
    query = "SELECT * FROM users WHERE name = %s"
    return execute_query(query, (user_input,))
                """,
                "expected_defects": [],
                "quality_score": 90
            },
            {
                "code": """
def get_user_data(user_id):
    if user_id <= 0:
        raise ValueError("Invalid user ID")
    
    user = fetch_user(user_id)
    if user is None:
        raise ValueError("User not found")
    
    return user.name
                """,
                "expected_defects": [],
                "quality_score": 92
            }
        ]
    
    @pytest.mark.asyncio
    async def test_defect_detection_accuracy(self, ai_service, known_defective_code_samples):
        """测试缺陷检测准确性"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            total_samples = len(known_defective_code_samples)
            correct_detections = 0
            
            for sample in known_defective_code_samples:
                # 模拟检测结果
                mock_detect.return_value = {
                    "defects": [
                        {
                            "type": sample["expected_defects"][0],
                            "severity": sample["severity"],
                            "confidence": 0.85,
                            "line": 2,
                            "message": f"Detected {sample['expected_defects'][0]}"
                        }
                    ],
                    "summary": {
                        "total_defects": 1,
                        "critical": 1 if sample["severity"] == "critical" else 0,
                        "high": 1 if sample["severity"] == "high" else 0,
                        "medium": 1 if sample["severity"] == "medium" else 0
                    }
                }
                
                result = await ai_service.detect_defects(sample["code"])
                
                # 检查是否正确检测到预期的缺陷类型
                detected_types = [defect["type"] for defect in result["defects"]]
                if any(expected in detected_types for expected in sample["expected_defects"]):
                    correct_detections += 1
            
            # 计算准确率
            accuracy = correct_detections / total_samples
            assert accuracy >= 0.8, f"Defect detection accuracy {accuracy:.2f} is below threshold 0.8"
    
    @pytest.mark.asyncio
    async def test_false_positive_rate(self, ai_service, clean_code_samples):
        """测试误报率"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            total_samples = len(clean_code_samples)
            false_positives = 0
            
            for sample in clean_code_samples:
                # 模拟检测结果（干净代码应该很少或没有缺陷）
                mock_detect.return_value = {
                    "defects": [],  # 干净代码不应该有缺陷
                    "summary": {
                        "total_defects": 0,
                        "critical": 0,
                        "high": 0,
                        "medium": 0
                    }
                }
                
                result = await ai_service.detect_defects(sample["code"])
                
                # 如果检测到缺陷，则为误报
                if result["summary"]["total_defects"] > 0:
                    false_positives += 1
            
            # 计算误报率
            false_positive_rate = false_positives / total_samples
            assert false_positive_rate <= 0.1, f"False positive rate {false_positive_rate:.2f} is above threshold 0.1"
    
    @pytest.mark.asyncio
    async def test_severity_classification_accuracy(self, ai_service, known_defective_code_samples):
        """测试严重性分类准确性"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            correct_severity_classifications = 0
            total_samples = len(known_defective_code_samples)
            
            for sample in known_defective_code_samples:
                mock_detect.return_value = {
                    "defects": [
                        {
                            "type": sample["expected_defects"][0],
                            "severity": sample["severity"],  # 正确的严重性
                            "confidence": 0.85,
                            "line": 2,
                            "message": f"Detected {sample['expected_defects'][0]}"
                        }
                    ],
                    "summary": {
                        "total_defects": 1,
                        "critical": 1 if sample["severity"] == "critical" else 0,
                        "high": 1 if sample["severity"] == "high" else 0,
                        "medium": 1 if sample["severity"] == "medium" else 0
                    }
                }
                
                result = await ai_service.detect_defects(sample["code"])
                
                # 检查严重性分类是否正确
                if result["defects"] and result["defects"][0]["severity"] == sample["severity"]:
                    correct_severity_classifications += 1
            
            # 计算严重性分类准确率
            severity_accuracy = correct_severity_classifications / total_samples
            assert severity_accuracy >= 0.75, f"Severity classification accuracy {severity_accuracy:.2f} is below threshold 0.75"


class TestCodeEmbeddingAccuracy:
    """代码嵌入准确性测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.fixture
    def similar_code_pairs(self):
        """相似代码对"""
        return [
            {
                "code1": "def add(a, b): return a + b",
                "code2": "def sum_numbers(x, y): return x + y",
                "expected_similarity": 0.9
            },
            {
                "code1": "for i in range(10): print(i)",
                "code2": "for j in range(10): print(j)",
                "expected_similarity": 0.95
            },
            {
                "code1": "class User: def __init__(self, name): self.name = name",
                "code2": "class Person: def __init__(self, name): self.name = name",
                "expected_similarity": 0.85
            }
        ]
    
    @pytest.fixture
    def dissimilar_code_pairs(self):
        """不相似代码对"""
        return [
            {
                "code1": "def add(a, b): return a + b",
                "code2": "import requests; response = requests.get('http://api.com')",
                "expected_similarity": 0.2
            },
            {
                "code1": "for i in range(10): print(i)",
                "code2": "class DatabaseConnection: def connect(self): pass",
                "expected_similarity": 0.15
            }
        ]
    
    @pytest.mark.asyncio
    async def test_embedding_similarity_accuracy(self, ai_service, similar_code_pairs):
        """测试嵌入相似性准确性"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            correct_similarities = 0
            total_pairs = len(similar_code_pairs)
            
            for pair in similar_code_pairs:
                # 模拟相似的嵌入向量
                mock_embed.side_effect = [
                    [0.1, 0.2, 0.3, 0.4, 0.5],  # code1的嵌入
                    [0.12, 0.22, 0.32, 0.42, 0.52]  # code2的嵌入（相似）
                ]
                
                embedding1 = await ai_service.embed_code(pair["code1"])
                embedding2 = await ai_service.embed_code(pair["code2"])
                
                # 计算余弦相似度
                similarity = self._cosine_similarity(embedding1, embedding2)
                
                # 检查相似度是否符合预期
                if similarity >= pair["expected_similarity"] - 0.1:
                    correct_similarities += 1
            
            accuracy = correct_similarities / total_pairs
            assert accuracy >= 0.8, f"Embedding similarity accuracy {accuracy:.2f} is below threshold 0.8"
    
    @pytest.mark.asyncio
    async def test_embedding_dissimilarity_accuracy(self, ai_service, dissimilar_code_pairs):
        """测试嵌入不相似性准确性"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            correct_dissimilarities = 0
            total_pairs = len(dissimilar_code_pairs)
            
            for pair in dissimilar_code_pairs:
                # 模拟不相似的嵌入向量
                mock_embed.side_effect = [
                    [0.1, 0.2, 0.3, 0.4, 0.5],  # code1的嵌入
                    [0.9, 0.8, 0.7, 0.6, 0.1]   # code2的嵌入（不相似）
                ]
                
                embedding1 = await ai_service.embed_code(pair["code1"])
                embedding2 = await ai_service.embed_code(pair["code2"])
                
                # 计算余弦相似度
                similarity = self._cosine_similarity(embedding1, embedding2)
                
                # 检查相似度是否符合预期（应该很低）
                if similarity <= pair["expected_similarity"] + 0.1:
                    correct_dissimilarities += 1
            
            accuracy = correct_dissimilarities / total_pairs
            assert accuracy >= 0.8, f"Embedding dissimilarity accuracy {accuracy:.2f} is below threshold 0.8"
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class TestArchitectureAnalysisAccuracy:
    """架构分析准确性测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.fixture
    def architecture_samples(self):
        """架构分析样本"""
        return [
            {
                "code": """
class UserService:
    def __init__(self, db_service):
        self.db_service = db_service
    
    def create_user(self, user_data):
        return self.db_service.save(user_data)

class DatabaseService:
    def save(self, data):
        pass
                """,
                "expected_components": ["UserService", "DatabaseService"],
                "expected_dependencies": [("UserService", "DatabaseService")],
                "expected_complexity": "low"
            },
            {
                "code": """
class OrderService:
    def __init__(self, user_service, payment_service, inventory_service):
        self.user_service = user_service
        self.payment_service = payment_service
        self.inventory_service = inventory_service
    
    def process_order(self, order):
        user = self.user_service.get_user(order.user_id)
        payment = self.payment_service.process_payment(order.amount)
        inventory = self.inventory_service.reserve_items(order.items)
        return self._create_order(user, payment, inventory)
                """,
                "expected_components": ["OrderService", "UserService", "PaymentService", "InventoryService"],
                "expected_dependencies": [
                    ("OrderService", "UserService"),
                    ("OrderService", "PaymentService"),
                    ("OrderService", "InventoryService")
                ],
                "expected_complexity": "medium"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_component_identification_accuracy(self, ai_service, architecture_samples):
        """测试组件识别准确性"""
        with patch.object(ai_service, 'analyze_architecture') as mock_analyze:
            correct_identifications = 0
            total_samples = len(architecture_samples)
            
            for sample in architecture_samples:
                mock_analyze.return_value = {
                    "components": [
                        {"name": comp, "type": "class", "complexity": 2}
                        for comp in sample["expected_components"]
                    ],
                    "dependencies": [
                        {"from": dep[0], "to": dep[1], "type": "uses"}
                        for dep in sample["expected_dependencies"]
                    ],
                    "metrics": {
                        "coupling": 0.3,
                        "cohesion": 0.8,
                        "complexity": 2.0
                    }
                }
                
                result = await ai_service.analyze_architecture(sample["code"])
                
                # 检查组件识别准确性
                identified_components = [comp["name"] for comp in result["components"]]
                expected_components = sample["expected_components"]
                
                # 计算识别准确率
                correct_count = sum(1 for comp in expected_components if comp in identified_components)
                accuracy = correct_count / len(expected_components)
                
                if accuracy >= 0.8:
                    correct_identifications += 1
            
            overall_accuracy = correct_identifications / total_samples
            assert overall_accuracy >= 0.8, f"Component identification accuracy {overall_accuracy:.2f} is below threshold 0.8"
    
    @pytest.mark.asyncio
    async def test_dependency_analysis_accuracy(self, ai_service, architecture_samples):
        """测试依赖分析准确性"""
        with patch.object(ai_service, 'analyze_architecture') as mock_analyze:
            correct_dependencies = 0
            total_samples = len(architecture_samples)
            
            for sample in architecture_samples:
                mock_analyze.return_value = {
                    "components": [
                        {"name": comp, "type": "class", "complexity": 2}
                        for comp in sample["expected_components"]
                    ],
                    "dependencies": [
                        {"from": dep[0], "to": dep[1], "type": "uses"}
                        for dep in sample["expected_dependencies"]
                    ],
                    "metrics": {
                        "coupling": 0.3,
                        "cohesion": 0.8,
                        "complexity": 2.0
                    }
                }
                
                result = await ai_service.analyze_architecture(sample["code"])
                
                # 检查依赖关系识别准确性
                identified_deps = [(dep["from"], dep["to"]) for dep in result["dependencies"]]
                expected_deps = sample["expected_dependencies"]
                
                # 计算依赖识别准确率
                correct_count = sum(1 for dep in expected_deps if dep in identified_deps)
                accuracy = correct_count / len(expected_deps) if expected_deps else 1.0
                
                if accuracy >= 0.8:
                    correct_dependencies += 1
            
            overall_accuracy = correct_dependencies / total_samples
            assert overall_accuracy >= 0.8, f"Dependency analysis accuracy {overall_accuracy:.2f} is below threshold 0.8"


class TestModelConsistency:
    """模型一致性测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_embedding_consistency(self, ai_service):
        """测试嵌入一致性"""
        with patch.object(ai_service, 'embed_code') as mock_embed:
            # 模拟一致的嵌入结果
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            code = "def hello(): return 'world'"
            
            # 多次调用应该返回相同的结果
            embeddings = []
            for _ in range(5):
                embedding = await ai_service.embed_code(code)
                embeddings.append(embedding)
            
            # 检查一致性
            first_embedding = embeddings[0]
            for embedding in embeddings[1:]:
                similarity = self._cosine_similarity(first_embedding, embedding)
                assert similarity > 0.99, f"Embedding consistency failed: similarity {similarity:.3f}"
    
    @pytest.mark.asyncio
    async def test_defect_detection_consistency(self, ai_service):
        """测试缺陷检测一致性"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            # 模拟一致的检测结果
            mock_detect.return_value = {
                "defects": [
                    {
                        "type": "division_by_zero",
                        "severity": "high",
                        "confidence": 0.85,
                        "line": 2,
                        "message": "Potential division by zero"
                    }
                ],
                "summary": {
                    "total_defects": 1,
                    "critical": 0,
                    "high": 1,
                    "medium": 0
                }
            }
            
            code = "def divide(a, b): return a / b"
            
            # 多次调用应该返回相同的结果
            results = []
            for _ in range(3):
                result = await ai_service.detect_defects(code)
                results.append(result)
            
            # 检查一致性
            first_result = results[0]
            for result in results[1:]:
                assert result["summary"]["total_defects"] == first_result["summary"]["total_defects"]
                assert len(result["defects"]) == len(first_result["defects"])
                if result["defects"]:
                    assert result["defects"][0]["type"] == first_result["defects"][0]["type"]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class TestModelCalibration:
    """模型校准测试"""
    
    @pytest.fixture
    def ai_service(self):
        return AIModelService()
    
    @pytest.mark.asyncio
    async def test_confidence_calibration(self, ai_service):
        """测试置信度校准"""
        with patch.object(ai_service, 'detect_defects') as mock_detect:
            # 测试不同置信度的预测
            test_cases = [
                {"confidence": 0.9, "should_be_correct": True},
                {"confidence": 0.7, "should_be_correct": True},
                {"confidence": 0.5, "should_be_correct": False},
                {"confidence": 0.3, "should_be_correct": False}
            ]
            
            for case in test_cases:
                mock_detect.return_value = {
                    "defects": [
                        {
                            "type": "potential_bug",
                            "severity": "medium",
                            "confidence": case["confidence"],
                            "line": 1,
                            "message": "Test defect"
                        }
                    ],
                    "summary": {"total_defects": 1}
                }
                
                result = await ai_service.detect_defects("test code")
                confidence = result["defects"][0]["confidence"]
                
                # 高置信度的预测应该更准确
                if confidence >= 0.8:
                    assert case["should_be_correct"], f"High confidence prediction should be correct"
                elif confidence <= 0.4:
                    assert not case["should_be_correct"], f"Low confidence prediction should be incorrect"