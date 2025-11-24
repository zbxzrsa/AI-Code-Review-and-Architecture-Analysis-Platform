"""
类型系统集成测试

测试类型推断、映射、验证和注解生成的完整流程
"""
import unittest
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from backend.app.services.type_system.inference import (
    TypeInferenceEngine, TypeInfo, TypeCategory, TypeFactory
)
from backend.app.services.type_system.mapping import TypeMappingSystem
from backend.app.services.type_system.validation import TypeSafetyValidator, ValidationLevel
from backend.app.services.type_system.annotation import (
    AnnotationGenerator, AnnotationOptions, CodeAnnotator
)


class TypeSystemIntegrationTest(unittest.TestCase):
    """类型系统集成测试"""

    def setUp(self):
        """设置测试环境"""
        self.inference_engine = TypeInferenceEngine()
        self.mapping_system = TypeMappingSystem()
        self.validator = TypeSafetyValidator()
        self.type_factory = TypeFactory()

    def test_python_to_typescript_conversion(self):
        """测试Python到TypeScript的类型转换"""
        # 1. Python代码
        python_code = """
def calculate_total(items, tax_rate=0.1):
    total = 0
    for item in items:
        total += item['price'] * item['quantity']
    
    return total * (1 + tax_rate)

user_data = {
    'name': 'John',
    'age': 30,
    'is_active': True,
    'items': [
        {'id': 1, 'price': 10.5, 'quantity': 2},
        {'id': 2, 'price': 5.0, 'quantity': 3}
    ]
}

result = calculate_total(user_data['items'])
"""

        # 2. 类型推断
        type_infos = self.inference_engine.infer_types(python_code, "python")
        
        # 验证推断结果
        self.assertIn('calculate_total', type_infos)
        self.assertEqual(type_infos['calculate_total'].category, TypeCategory.FUNCTION)
        self.assertIn('user_data', type_infos)
        self.assertEqual(type_infos['user_data'].category, TypeCategory.CONTAINER)
        
        # 3. 类型映射到TypeScript
        ts_type_infos = {}
        for name, type_info in type_infos.items():
            ts_type_info = self.mapping_system.map_type(type_info, "python", "typescript")
            ts_type_infos[name] = ts_type_info
        
        # 验证映射结果
        self.assertIn('calculate_total', ts_type_infos)
        self.assertEqual(ts_type_infos['calculate_total'].category, TypeCategory.FUNCTION)
        
        # 4. 类型安全验证
        validation_results = {}
        for name, type_info in type_infos.items():
            ts_type_info = ts_type_infos[name]
            result = self.validator.validate_conversion(
                type_info, ts_type_info, ValidationLevel.COMPATIBLE
            )
            validation_results[name] = result
            self.assertTrue(result.is_valid, f"类型 {name} 转换验证失败: {result.issues}")
        
        # 5. 生成TypeScript代码和类型注解
        ts_options = AnnotationOptions(
            include_docstrings=True,
            add_imports=True,
            include_return_types=True,
            include_parameter_types=True,
            include_variable_types=True
        )
        
        ts_annotator = CodeAnnotator("typescript", ts_options)
        
        # 转换后的TypeScript代码
        expected_ts_code = """
function calculateTotal(items: Array<Record<string, any>>, taxRate: number = 0.1): number {
    let total = 0;
    for (const item of items) {
        total += item['price'] * item['quantity'];
    }
    
    return total * (1 + taxRate);
}

const userData: {
    name: string;
    age: number;
    isActive: boolean;
    items: Array<Record<string, any>>;
} = {
    name: 'John',
    age: 30,
    isActive: true,
    items: [
        {id: 1, price: 10.5, quantity: 2},
        {id: 2, price: 5.0, quantity: 3}
    ]
};

const result: number = calculateTotal(userData.items);
"""
        
        # 验证生成的TypeScript代码包含类型注解
        ts_code = ts_annotator.annotate_code(expected_ts_code, ts_type_infos)
        self.assertIn("function calculateTotal(items: Array<", ts_code)
        self.assertIn("): number {", ts_code)
        self.assertIn("const userData: {", ts_code)
        self.assertIn("const result: number", ts_code)

    def test_javascript_to_python_conversion(self):
        """测试JavaScript到Python的类型转换"""
        # 1. JavaScript代码
        js_code = """
function processData(data, callback) {
    const result = [];
    
    for (let i = 0; i < data.length; i++) {
        const item = data[i];
        const processed = {
            id: item.id,
            value: item.value * 2,
            valid: item.value > 0
        };
        result.push(processed);
    }
    
    callback(result);
    return result;
}

const items = [
    {id: 'a1', value: 10},
    {id: 'a2', value: -5},
    {id: 'a3', value: 20}
];

const printResults = function(data) {
    console.log(data);
};

const results = processData(items, printResults);
"""

        # 2. 类型推断
        type_infos = self.inference_engine.infer_types(js_code, "javascript")
        
        # 验证推断结果
        self.assertIn('processData', type_infos)
        self.assertEqual(type_infos['processData'].category, TypeCategory.FUNCTION)
        self.assertIn('items', type_infos)
        self.assertEqual(type_infos['items'].category, TypeCategory.CONTAINER)
        
        # 3. 类型映射到Python
        py_type_infos = {}
        for name, type_info in type_infos.items():
            py_type_info = self.mapping_system.map_type(type_info, "javascript", "python")
            py_type_infos[name] = py_type_info
        
        # 验证映射结果
        self.assertIn('processData', py_type_infos)
        self.assertEqual(py_type_infos['processData'].category, TypeCategory.FUNCTION)
        
        # 4. 类型安全验证
        validation_results = {}
        for name, type_info in type_infos.items():
            py_type_info = py_type_infos[name]
            result = self.validator.validate_conversion(
                type_info, py_type_info, ValidationLevel.COMPATIBLE
            )
            validation_results[name] = result
            self.assertTrue(result.is_valid, f"类型 {name} 转换验证失败: {result.issues}")
        
        # 5. 生成Python代码和类型注解
        py_options = AnnotationOptions(
            include_docstrings=True,
            add_imports=True,
            include_return_types=True,
            include_parameter_types=True,
            include_variable_types=True
        )
        
        py_annotator = CodeAnnotator("python", py_options)
        
        # 转换后的Python代码
        expected_py_code = """
def process_data(data, callback):
    result = []
    
    for i in range(len(data)):
        item = data[i]
        processed = {
            'id': item['id'],
            'value': item['value'] * 2,
            'valid': item['value'] > 0
        }
        result.append(processed)
    
    callback(result)
    return result

items = [
    {'id': 'a1', 'value': 10},
    {'id': 'a2', 'value': -5},
    {'id': 'a3', 'value': 20}
]

def print_results(data):
    print(data)

results = process_data(items, print_results)
"""
        
        # 验证生成的Python代码包含类型注解
        py_code = py_annotator.annotate_code(expected_py_code, py_type_infos)
        self.assertIn("def process_data(data: List[Dict[str, Any]], callback: Callable", py_code)
        self.assertIn(") -> List[Dict[str, Any]]:", py_code)
        self.assertIn("items: List[Dict[str, Any]] = [", py_code)
        self.assertIn("results: List[Dict[str, Any]] = process_data", py_code)

    def test_java_to_csharp_conversion(self):
        """测试Java到C#的类型转换模拟"""
        # 创建Java类型信息
        java_class = self.type_factory.create_class_type(
            "User", "java", 
            attributes={
                "id": self.type_factory.create_primitive_type("long", "java"),
                "name": self.type_factory.create_primitive_type("String", "java"),
                "active": self.type_factory.create_primitive_type("boolean", "java"),
                "scores": self.type_factory.create_container_type(
                    "List", "java", 
                    [self.type_factory.create_primitive_type("Integer", "java")]
                )
            }
        )
        
        # 映射到C#
        csharp_class = self.mapping_system.map_type(java_class, "java", "csharp")
        
        # 验证映射结果
        self.assertEqual(csharp_class.category, TypeCategory.CLASS)
        self.assertEqual(csharp_class.name, "User")
        self.assertEqual(csharp_class.source_language, "csharp")
        
        # 验证属性映射
        self.assertEqual(csharp_class.attributes["id"].name, "long")
        self.assertEqual(csharp_class.attributes["name"].name, "string")
        self.assertEqual(csharp_class.attributes["active"].name, "bool")
        self.assertEqual(csharp_class.attributes["scores"].name, "List")
        self.assertEqual(csharp_class.attributes["scores"].type_args[0].name, "int")
        
        # 验证类型安全
        result = self.validator.validate_conversion(
            java_class, csharp_class, ValidationLevel.STRICT
        )
        self.assertTrue(result.is_valid, f"Java到C#类型转换验证失败: {result.issues}")

    def test_end_to_end_conversion(self):
        """测试端到端的类型转换流程"""
        # 1. 源代码
        source_code = """
class DataProcessor:
    def __init__(self, config=None):
        self.config = config or {}
        self.processed_count = 0
    
    def process_item(self, item):
        if not isinstance(item, dict):
            return None
        
        result = {
            "id": item.get("id", "unknown"),
            "value": float(item.get("value", 0)) * 1.5,
            "tags": item.get("tags", []),
            "metadata": {
                "processed_at": "2023-01-01",
                "version": 1.0
            }
        }
        
        self.processed_count += 1
        return result
    
    def get_stats(self):
        return {
            "count": self.processed_count,
            "config": self.config
        }

processor = DataProcessor({"mode": "fast"})
items = [
    {"id": "item1", "value": "10", "tags": ["a", "b"]},
    {"id": "item2", "value": "20", "tags": ["c"]}
]

results = []
for item in items:
    result = processor.process_item(item)
    results.append(result)

stats = processor.get_stats()
"""
        
        # 2. 类型推断
        source_lang = "python"
        target_lang = "typescript"
        type_infos = self.inference_engine.infer_types(source_code, source_lang)
        
        # 3. 类型映射
        target_type_infos = {}
        for name, type_info in type_infos.items():
            target_type_info = self.mapping_system.map_type(type_info, source_lang, target_lang)
            target_type_infos[name] = target_type_info
        
        # 4. 类型安全验证
        for name, source_type in type_infos.items():
            target_type = target_type_infos.get(name)
            if target_type:
                result = self.validator.validate_conversion(
                    source_type, target_type, ValidationLevel.COMPATIBLE
                )
                self.assertTrue(
                    result.is_valid, 
                    f"类型 {name} 从 {source_lang} 到 {target_lang} 的转换验证失败: {result.issues}"
                )
        
        # 5. 生成目标代码
        # 这里我们模拟已转换的TypeScript代码
        target_code = """
class DataProcessor {
    config: any;
    processedCount: number;
    
    constructor(config = null) {
        this.config = config || {};
        this.processedCount = 0;
    }
    
    processItem(item) {
        if (typeof item !== 'object' || item === null) {
            return null;
        }
        
        const result = {
            id: item.id || "unknown",
            value: parseFloat(item.value || 0) * 1.5,
            tags: item.tags || [],
            metadata: {
                processedAt: "2023-01-01",
                version: 1.0
            }
        };
        
        this.processedCount += 1;
        return result;
    }
    
    getStats() {
        return {
            count: this.processedCount,
            config: this.config
        };
    }
}

const processor = new DataProcessor({mode: "fast"});
const items = [
    {id: "item1", value: "10", tags: ["a", "b"]},
    {id: "item2", value: "20", tags: ["c"]}
];

const results = [];
for (const item of items) {
    const result = processor.processItem(item);
    results.push(result);
}

const stats = processor.getStats();
"""
        
        # 6. 添加类型注解
        options = AnnotationOptions(
            include_docstrings=True,
            add_imports=True,
            include_return_types=True,
            include_parameter_types=True,
            include_variable_types=True
        )
        
        annotator = CodeAnnotator(target_lang, options)
        annotated_code = annotator.annotate_code(target_code, target_type_infos)
        
        # 验证生成的代码包含类型注解
        self.assertIn("config: Record<string, any>;", annotated_code)
        self.assertIn("processedCount: number;", annotated_code)
        self.assertIn("processItem(item: Record<string, any>): Record<string, any> | null {", annotated_code)
        self.assertIn("getStats(): Record<string, any> {", annotated_code)
        self.assertIn("const processor: DataProcessor = new DataProcessor", annotated_code)
        self.assertIn("const items: Array<Record<string, any>> = [", annotated_code)
        self.assertIn("const results: Array<Record<string, any> | null> = [];", annotated_code)
        self.assertIn("const stats: Record<string, any> = processor.getStats();", annotated_code)


if __name__ == "__main__":
    unittest.main()