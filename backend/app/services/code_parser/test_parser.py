"""
代码解析服务测试脚本
用于测试代码解析服务的功能，包括单次解析和批量解析
"""
import asyncio
import json

from app.services.code_parser.parser import CodeParserService, Language, FeatureType

# 创建代码解析服务实例
parser_service = CodeParserService()

# 定义Python测试代码
python_code = """
import os

def greet(name):
    print(f"Hello, {name}!")

if __name__ == "__main__":
    greet("World")
"""

# 定义Java测试代码
java_code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""

async def test_single_parse():
    """测试单次解析"""
    print("--- 测试单次解析 ---")
    
    try:
        # 解析Python代码
        result = await parser_service.parse(
            python_code, 
            Language.PYTHON, 
            [FeatureType.AST, FeatureType.METRICS]
        )
        
        # 打印结果
        print("Python解析结果:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"单次解析失败: {e}")

async def test_batch_parse():
    """测试批量解析"""
    print("\n--- 测试批量解析 ---")
    
    try:
        # 创建批量解析任务
        tasks = [
            (python_code, Language.PYTHON, [FeatureType.AST, FeatureType.CFG, FeatureType.METRICS]),
            (java_code, Language.JAVA, [FeatureType.AST, FeatureType.DFG, FeatureType.METRICS])
        ]
        
        # 批量解析
        results = await parser_service.batch_parse(tasks)
        
        # 打印结果
        print("批量解析结果:")
        for i, result in enumerate(results):
            print(f"--- 任务 {i+1} ---")
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"批量解析失败: {e}")

async def main():
    """主函数"""
    # 运行测试
    await test_single_parse()
    await test_batch_parse()

if __name__ == "__main__":
    asyncio.run(main())