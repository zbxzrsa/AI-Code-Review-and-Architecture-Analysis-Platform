import asyncio
import aiohttp
import json
import time

# API基础URL
BASE_URL = "http://localhost:8000/api/v1"

async def test_embed_endpoint():
    """测试代码向量化端点"""
    print("\n测试代码向量化端点...")
    
    url = f"{BASE_URL}/embed"
    payload = {
        "code_structure": {
            "type": "module",
            "children": [
                {
                    "type": "function_definition",
                    "name": "hello_world",
                    "body": {
                        "type": "block",
                        "children": [
                            {
                                "type": "expression_statement",
                                "expression": {
                                    "type": "call",
                                    "function": {"type": "name", "value": "print"},
                                    "arguments": [{"type": "string", "value": "Hello, World!"}]
                                }
                            }
                        ]
                    }
                }
            ]
        },
        "language": "python"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            print(f"状态码: {response.status}")
            print(f"向量维度: {len(result['vector'])}")
            print(f"处理时间: {result['processing_time']:.4f}秒")
            
            # 测试缓存
            print("\n测试缓存...")
            start = time.time()
            async with session.post(url, json=payload) as cached_response:
                cached_result = await cached_response.json()
                print(f"缓存响应时间: {time.time() - start:.4f}秒")
                print(f"是否命中缓存: {cached_response.headers.get('X-Cache-Hit', 'No')}")

async def test_defect_detection_endpoint():
    """测试缺陷检测端点"""
    print("\n测试缺陷检测端点...")
    
    url = f"{BASE_URL}/analyze/defects"
    payload = {
        "code": """
def process_data(data):
    result = data.process()
    if result == None:
        return
    
    file = open('output.txt', 'w')
    file.write(str(result))
    # 文件未关闭
    
    return result
        """,
        "language": "python"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            print(f"状态码: {response.status}")
            print(f"检测到的缺陷数量: {len(result['defects'])}")
            for i, defect in enumerate(result['defects']):
                print(f"缺陷 {i+1}: {defect['defect_type']} (置信度: {defect['confidence']:.2f})")
            print(f"处理时间: {result['processing_time']:.4f}秒")

async def test_architecture_analysis_endpoint():
    """测试架构模式识别端点"""
    print("\n测试架构模式识别端点...")
    
    url = f"{BASE_URL}/analyze/architecture"
    payload = {
        "project_structure": {
            "root": {
                "type": "directory",
                "name": "project",
                "children": [
                    {
                        "type": "directory",
                        "name": "controllers",
                        "children": [
                            {"type": "file", "name": "UserController.java"},
                            {"type": "file", "name": "ProductController.java"}
                        ]
                    },
                    {
                        "type": "directory",
                        "name": "models",
                        "children": [
                            {"type": "file", "name": "User.java"},
                            {"type": "file", "name": "Product.java"}
                        ]
                    },
                    {
                        "type": "directory",
                        "name": "views",
                        "children": [
                            {"type": "file", "name": "user_view.html"},
                            {"type": "file", "name": "product_view.html"}
                        ]
                    },
                    {
                        "type": "directory",
                        "name": "services",
                        "children": [
                            {"type": "file", "name": "UserService.java"},
                            {"type": "file", "name": "AuthService.java"}
                        ]
                    }
                ]
            }
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            print(f"状态码: {response.status}")
            print(f"检测到的架构模式数量: {len(result['patterns'])}")
            for i, pattern in enumerate(result['patterns']):
                print(f"模式 {i+1}: {pattern['pattern_name']} (置信度: {pattern['confidence']:.2f})")
            
            print(f"\n检测到的架构坏味数量: {len(result['smells'])}")
            for i, smell in enumerate(result['smells']):
                print(f"坏味 {i+1}: {smell['smell_type']} (严重程度: {smell['severity']})")
            
            print(f"处理时间: {result['processing_time']:.4f}秒")

async def test_similarity_endpoint():
    """测试代码相似度计算端点"""
    print("\n测试代码相似度计算端点...")
    
    url = f"{BASE_URL}/similarity"
    payload = {
        "code1": """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
        """,
        "code2": """
def sum_array(arr):
    result = 0
    for value in arr:
        result += value
    return result
        """,
        "language": "python",
        "detailed_analysis": True
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            print(f"状态码: {response.status}")
            print(f"相似度分数: {result['similarity_score']:.2f}")
            print(f"相似段数量: {len(result['similar_segments'])}")
            for i, segment in enumerate(result['similar_segments']):
                print(f"相似段 {i+1}: 代码1({segment['code1_start_line']}-{segment['code1_end_line']}) 与 代码2({segment['code2_start_line']}-{segment['code2_end_line']}) 相似度: {segment['similarity_score']:.2f}")
            print(f"处理时间: {result['processing_time']:.4f}秒")

async def test_rate_limiting():
    """测试请求限流功能"""
    print("\n测试请求限流功能...")
    
    url = f"{BASE_URL}/embed"
    payload = {
        "code_structure": {"type": "module"},
        "language": "python"
    }
    
    async with aiohttp.ClientSession() as session:
        for i in range(12):  # 超过限流阈值(10次/分钟)
            start = time.time()
            async with session.post(url, json=payload) as response:
                duration = time.time() - start
                status = response.status
                if status == 429:
                    print(f"请求 {i+1}: 触发限流 (状态码: {status}, 耗时: {duration:.4f}秒)")
                    break
                else:
                    print(f"请求 {i+1}: 成功 (状态码: {status}, 耗时: {duration:.4f}秒)")
                
                # 避免请求过快
                await asyncio.sleep(0.1)

async def main():
    """运行所有测试"""
    print("开始测试AI分析服务API...")
    
    # 测试各个端点
    await test_embed_endpoint()
    await test_defect_detection_endpoint()
    await test_architecture_analysis_endpoint()
    await test_similarity_endpoint()
    
    # 测试限流功能
    await test_rate_limiting()
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    asyncio.run(main())