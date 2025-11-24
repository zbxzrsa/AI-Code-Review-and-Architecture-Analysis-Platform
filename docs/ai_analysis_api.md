# AI分析服务API文档

## 概述

AI分析服务提供了一系列REST API端点，用于代码向量化、缺陷检测、架构模式识别和代码相似度计算。这些API可以帮助开发者和团队分析代码质量、识别潜在问题并优化软件架构。

## 基础信息

- **基础URL**: `http://localhost:8000/api/v1`
- **API文档**: `http://localhost:8000/api/docs`
- **Redoc文档**: `http://localhost:8000/api/redoc`
- **健康检查**: `http://localhost:8000/health`

## 认证

当前版本不需要认证。在生产环境中，建议实现API密钥或OAuth2认证机制。

## 请求限流

为了防止API滥用，所有端点都实现了请求限流机制：

| 端点 | 限流规则 |
|------|----------|
| `/embed` | 10次请求/分钟 |
| `/analyze/defects` | 5次请求/分钟 |
| `/analyze/architecture` | 2次请求/5分钟 |
| `/similarity` | 10次请求/分钟 |

超过限制的请求将收到`429 Too Many Requests`响应。

## 缓存

所有API响应都会被缓存以提高性能：

| 端点 | 缓存时间 |
|------|----------|
| `/embed` | 1小时 |
| `/analyze/defects` | 1小时 |
| `/analyze/architecture` | 24小时 |
| `/similarity` | 1小时 |

## API端点

### 1. 代码向量化

将代码结构转换为向量表示，用于后续的AI分析。

**请求**:
```
POST /api/v1/embed
```

**请求体**:
```json
{
  "code_structure": {
    "type": "module",
    "children": [...]
  },
  "language": "python"
}
```

**参数说明**:
- `code_structure`: 解析后的代码结构（通常由代码解析服务生成）
- `language`: 编程语言（如"python", "java", "javascript", "go"等）

**响应**:
```json
{
  "vector": [0.1, 0.2, ..., 0.7],  // 768维向量
  "processing_time": 0.125
}
```

**响应说明**:
- `vector`: 768维代码向量
- `processing_time`: 处理时间（秒）

### 2. 缺陷检测

分析代码中的潜在缺陷，如空指针引用、资源泄漏等。

**请求**:
```
POST /api/v1/analyze/defects
```

**请求体**:
```json
{
  "code": "def process_data(data):\n    result = data.process()\n    ...",
  "language": "python"
}
```

或

```json
{
  "vector": [0.1, 0.2, ..., 0.7],
  "language": "python"
}
```

**参数说明**:
- `code`: 原始代码（可选）
- `vector`: 代码向量（可选，由`/embed`端点生成）
- `language`: 编程语言

**注意**: 必须提供`code`或`vector`其中之一

**响应**:
```json
{
  "defects": [
    {
      "defect_type": "null_pointer_exception",
      "confidence": 0.92,
      "location": {
        "start_line": 15,
        "end_line": 15,
        "start_column": 10,
        "end_column": 25
      },
      "description": "可能的空指针引用"
    },
    {
      "defect_type": "resource_leak",
      "confidence": 0.85,
      "location": {
        "start_line": 27,
        "end_line": 30
      },
      "description": "资源未正确关闭"
    }
  ],
  "processing_time": 0.235
}
```

**响应说明**:
- `defects`: 检测到的缺陷列表
  - `defect_type`: 缺陷类型
  - `confidence`: 置信度（0-1）
  - `location`: 缺陷位置信息
  - `description`: 缺陷描述
- `processing_time`: 处理时间（秒）

### 3. 架构模式识别

分析项目结构，识别架构模式和潜在的架构问题。

**请求**:
```
POST /api/v1/analyze/architecture
```

**请求体**:
```json
{
  "project_structure": {
    "root": {
      "type": "directory",
      "name": "project",
      "children": [...]
    }
  }
}
```

或

```json
{
  "vectors": {
    "src/main/java/com/example/Controller.java": [0.1, 0.2, ..., 0.7],
    "src/main/java/com/example/Service.java": [0.2, 0.3, ..., 0.8],
    ...
  }
}
```

**参数说明**:
- `project_structure`: 项目文件结构（可选）
- `vectors`: 文件路径到代码向量的映射（可选）

**注意**: 必须提供`project_structure`或`vectors`其中之一

**响应**:
```json
{
  "patterns": [
    {
      "pattern_name": "MVC",
      "confidence": 0.88,
      "components": ["controllers/", "models/", "views/"],
      "description": "Model-View-Controller架构模式"
    },
    {
      "pattern_name": "Repository",
      "confidence": 0.76,
      "components": ["repositories/", "services/"],
      "description": "仓储模式"
    }
  ],
  "smells": [
    {
      "smell_type": "cyclic_dependency",
      "severity": "high",
      "affected_components": ["service/UserService.java", "service/AuthService.java"],
      "description": "服务之间存在循环依赖",
      "recommendation": "考虑引入中介者模式或事件驱动架构"
    }
  ],
  "processing_time": 0.876
}
```

**响应说明**:
- `patterns`: 检测到的架构模式列表
  - `pattern_name`: 模式名称
  - `confidence`: 置信度（0-1）
  - `components`: 相关组件
  - `description`: 模式描述
- `smells`: 检测到的架构坏味列表
  - `smell_type`: 坏味类型
  - `severity`: 严重程度（low/medium/high/critical）
  - `affected_components`: 受影响的组件
  - `description`: 坏味描述
  - `recommendation`: 改进建议
- `processing_time`: 处理时间（秒）

### 4. 代码相似度计算

计算两段代码的相似度，可用于代码克隆检测、抄袭检查等。

**请求**:
```
POST /api/v1/similarity
```

**请求体**:
```json
{
  "code1": "def calculate_sum(numbers):\n    total = 0\n    ...",
  "code2": "def sum_array(arr):\n    result = 0\n    ...",
  "language": "python",
  "detailed_analysis": true
}
```

**参数说明**:
- `code1`: 第一段代码
- `code2`: 第二段代码
- `language`: 编程语言
- `detailed_analysis`: 是否返回详细分析（默认为false）

**响应**:
```json
{
  "similarity_score": 0.75,
  "similar_segments": [
    {
      "code1_start_line": 5,
      "code1_end_line": 10,
      "code2_start_line": 8,
      "code2_end_line": 13,
      "similarity_score": 0.92
    },
    {
      "code1_start_line": 15,
      "code1_end_line": 20,
      "code2_start_line": 25,
      "code2_end_line": 30,
      "similarity_score": 0.85
    }
  ],
  "processing_time": 0.345
}
```

**响应说明**:
- `similarity_score`: 总体相似度分数（0-1）
- `similar_segments`: 相似代码段列表（仅当detailed_analysis=true时返回）
  - `code1_start_line`/`code1_end_line`: 代码1中相似段的行范围
  - `code2_start_line`/`code2_end_line`: 代码2中相似段的行范围
  - `similarity_score`: 该段的相似度分数
- `processing_time`: 处理时间（秒）

## 错误处理

所有API端点在发生错误时都会返回标准的HTTP错误状态码和详细的错误信息：

```json
{
  "detail": "错误描述"
}
```

常见错误状态码：

- `400 Bad Request`: 请求参数无效
- `429 Too Many Requests`: 请求频率超过限制
- `500 Internal Server Error`: 服务器内部错误

## 性能考虑

- 对于大型代码库或复杂分析，处理时间可能较长
- 建议使用异步请求模式，避免阻塞客户端
- 利用缓存机制减少重复分析的开销

## 集成示例

### Python示例

```python
import aiohttp
import asyncio
import json

async def analyze_code_defects(code, language):
    url = "http://localhost:8000/api/v1/analyze/defects"
    payload = {
        "code": code,
        "language": language
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.json()
                print(f"错误: {error['detail']}")
                return None

# 使用示例
code = """
def process_data(data):
    result = data.process()
    if result == None:
        return
    
    file = open('output.txt', 'w')
    file.write(str(result))
    # 文件未关闭
    
    return result
"""

result = asyncio.run(analyze_code_defects(code, "python"))
print(json.dumps(result, indent=2, ensure_ascii=False))
```

### JavaScript示例

```javascript
async function calculateSimilarity(code1, code2, language) {
  const url = "http://localhost:8000/api/v1/similarity";
  const payload = {
    code1,
    code2,
    language,
    detailed_analysis: true
  };
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    
    if (response.ok) {
      return await response.json();
    } else {
      const error = await response.json();
      console.error(`错误: ${error.detail}`);
      return null;
    }
  } catch (error) {
    console.error(`请求失败: ${error.message}`);
    return null;
  }
}

// 使用示例
const code1 = `function sum(a, b) { return a + b; }`;
const code2 = `function add(x, y) { return x + y; }`;

calculateSimilarity(code1, code2, "javascript")
  .then(result => console.log(JSON.stringify(result, null, 2)))
  .catch(error => console.error(error));
```