"""
控制流和内存管理转换的集成测试

测试不同语言之间的代码转换功能，包括：
- 循环结构转换
- 条件语句转换
- 异常处理转换
- 函数控制流转换
- 资源管理转换
- 所有权系统转换
"""
import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.app.services.flow_converter.loops import convert_loop
from backend.app.services.flow_converter.conditions import convert_condition
from backend.app.services.flow_converter.exceptions import convert_exception_handling
from backend.app.services.flow_converter.functions import convert_function_flow
from backend.app.services.memory_converter.resources import convert_resource_management
from backend.app.services.memory_converter.ownership import convert_ownership_system


class TestFlowConverters(unittest.TestCase):
    """测试控制流转换功能"""

    def test_python_to_javascript_loop_conversion(self):
        """测试Python到JavaScript的循环转换"""
        # 测试range循环转换
        python_code = """
for i in range(10):
    if i % 2 == 0:
        print(i)
"""
        expected_js = """
for (let i = 0; i < 10; i++) {
    if (i % 2 === 0) {
        console.log(i);
    }
}
"""
        result = convert_loop(python_code, "python", "javascript")
        self.assertIn("for (let i = 0; i < 10; i++)", result)
        self.assertIn("if (i % 2 === 0)", result)
        self.assertIn("console.log(i);", result)

        # 测试列表迭代转换
        python_code = """
for item in items:
    print(item)
"""
        expected_js = """
for (const item of items) {
    console.log(item);
}
"""
        result = convert_loop(python_code, "python", "javascript")
        self.assertIn("for (const item of items)", result)
        self.assertIn("console.log(item);", result)

    def test_python_to_javascript_condition_conversion(self):
        """测试Python到JavaScript的条件语句转换"""
        # 测试if-elif-else转换
        python_code = """
if x > 10:
    print("Greater than 10")
elif x > 5:
    print("Between 5 and 10")
else:
    print("Less than or equal to 5")
"""
        result = convert_condition(python_code, "python", "javascript")
        self.assertIn("if (x > 10)", result)
        self.assertIn("else if (x > 5)", result)
        self.assertIn("console.log(\"Greater than 10\");", result)
        self.assertIn("console.log(\"Between 5 and 10\");", result)
        self.assertIn("console.log(\"Less than or equal to 5\");", result)

        # 测试三元运算符转换
        python_code = """
result = "Even" if num % 2 == 0 else "Odd"
"""
        result = convert_condition(python_code, "python", "javascript")
        self.assertIn("const result = num % 2 === 0 ? \"Even\" : \"Odd\";", result)

    def test_python_to_javascript_exception_conversion(self):
        """测试Python到JavaScript的异常处理转换"""
        # 测试try-except转换
        python_code = """
try:
    result = process_data(data)
except ValueError as e:
    print("Value error:", e)
except Exception as e:
    print("Error:", e)
finally:
    cleanup()
"""
        result = convert_exception_handling(python_code, "python", "javascript")
        self.assertIn("try {", result)
        self.assertIn("result = processData(data);", result)
        self.assertIn("catch (e) {", result)
        self.assertIn("if (e instanceof Error)", result)
        self.assertIn("console.log(\"Error:\", e);", result)
        self.assertIn("finally {", result)
        self.assertIn("cleanup();", result)

    def test_python_to_javascript_function_conversion(self):
        """测试Python到JavaScript的函数控制流转换"""
        # 测试异步函数转换
        python_code = """
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
"""
        result = convert_function_flow(python_code, "python", "javascript")
        self.assertIn("async function fetchData()", result)
        self.assertIn("const response = await fetch(url);", result)
        self.assertIn("return await response.json();", result)


class TestMemoryConverters(unittest.TestCase):
    """测试内存管理转换功能"""

    def test_python_to_cpp_resource_conversion(self):
        """测试Python到C++的资源管理转换"""
        # 测试文件资源管理
        python_code = """
with open('file.txt', 'r') as f:
    content = f.read()
"""
        result = convert_resource_management(python_code, "python", "cpp")
        self.assertIn("std::ifstream f(\"file.txt\");", result)
        self.assertIn("std::string content;", result)
        self.assertIn("std::getline(f, content, '\\0');", result)

        # 测试锁资源管理
        python_code = """
with lock:
    shared_resource.update()
"""
        result = convert_resource_management(python_code, "python", "cpp")
        self.assertIn("std::lock_guard<std::mutex> guard(lock);", result)
        self.assertIn("sharedResource.update();", result)

    def test_javascript_to_rust_ownership_conversion(self):
        """测试JavaScript到Rust的所有权转换"""
        # 测试数组处理转换
        js_code = """
function processData(data) {
    const result = data.map(x => x * 2);
    return result.filter(x => x > 10);
}
"""
        result = convert_ownership_system(js_code, "javascript", "rust")
        self.assertIn("fn process_data(data: Vec<i32>) -> Vec<i32>", result)
        self.assertIn(".map(|x| x * 2)", result)
        self.assertIn(".filter(|&x| x > 10)", result)
        self.assertIn(".collect()", result)


class TestIntegrationScenarios(unittest.TestCase):
    """测试复杂的集成场景"""

    def test_python_to_typescript_full_conversion(self):
        """测试Python到TypeScript的完整转换"""
        python_code = """
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = []
    
    def load_data(self, filename):
        try:
            with open(filename, 'r') as f:
                self.data = [int(line.strip()) for line in f]
            return True
        except FileNotFoundError:
            print(f"File {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def process(self):
        result = []
        for item in self.data:
            if item > self.config.threshold:
                result.append(item * 2)
            elif item > 0:
                result.append(item)
        return result
    
    async def process_async(self, data_source):
        async with aiohttp.ClientSession() as session:
            async with session.get(data_source) as response:
                data = await response.json()
                return [item for item in data if self.is_valid(item)]
    
    def is_valid(self, item):
        return item.get('active', False) and item.get('value', 0) > self.config.min_value
"""
        # 转换循环结构
        loop_result = convert_loop(python_code, "python", "typescript")
        
        # 转换条件语句
        condition_result = convert_condition(loop_result, "python", "typescript")
        
        # 转换异常处理
        exception_result = convert_exception_handling(condition_result, "python", "typescript")
        
        # 转换函数控制流
        function_result = convert_function_flow(exception_result, "python", "typescript")
        
        # 转换资源管理
        resource_result = convert_resource_management(function_result, "python", "typescript")
        
        # 验证转换结果
        self.assertIn("class DataProcessor", resource_result)
        self.assertIn("constructor(config: Config)", resource_result)
        self.assertIn("async processAsync(dataSource: string)", resource_result)
        self.assertIn("for (const item of this.data)", resource_result)
        self.assertIn("if (item > this.config.threshold)", resource_result)
        self.assertIn("try {", resource_result)
        self.assertIn("catch (e) {", resource_result)
        self.assertIn("const response = await fetch(dataSource);", resource_result)


if __name__ == '__main__':
    unittest.main()