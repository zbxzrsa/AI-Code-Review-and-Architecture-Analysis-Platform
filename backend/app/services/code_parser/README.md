# 多语言AST解析器和统一中间表示系统

本系统提供了一个强大的多语言代码解析和转换框架，支持Python、JavaScript/TypeScript等语言的解析，并通过统一的中间表示(IR)实现跨语言代码转换。

## 系统架构

系统由以下核心组件构成：

1. **语言解析器（LanguageParser）**：提供统一的解析接口，支持各种编程语言的代码解析
2. **统一中间表示（IR）**：定义语言无关的代码表示格式，作为不同语言间转换的桥梁
3. **AST到IR转换器**：将特定语言的AST转换为统一的IR格式
4. **IR到代码生成器**：将IR转换为目标语言的代码

## 核心组件

### 语言解析器（LanguageParser）

`LanguageParser`是所有语言特定解析器的基类，提供统一的解析接口：

- `parse_file(file_path)`: 解析文件
- `parse_code(code, file_path)`: 解析代码字符串
- `parse_incremental(code, previous_ast, changed_range)`: 增量解析
- `get_node_at_position(ast_root, line, column)`: 获取指定位置的AST节点
- `get_node_range(node)`: 获取节点的源代码范围

目前实现的语言解析器：

- `PythonParser`: Python语言解析器，支持Python 3.8+语法特性
- `JSTypeScriptParser`: JavaScript/TypeScript解析器，支持ES2022+和TypeScript特性

### 统一中间表示（IR）

IR系统定义了一套语言无关的代码表示格式，包括：

- `Module`: 模块/命名空间
- `Function`: 函数/方法定义
- `Class`: 类定义
- `Statement`: 各类语句（赋值、条件、循环等）
- `Expression`: 各类表达式（二元操作、函数调用、字面量等）
- `Type`: 类型系统抽象

IR系统还实现了访问者模式（Visitor Pattern），便于遍历和转换IR节点。

### AST到IR转换器

`ASTToIRConverter`负责将特定语言的AST转换为统一的IR格式：

- `PythonASTToIRConverter`: 将Python AST转换为IR
- `JavaScriptASTToIRConverter`: 将JavaScript/TypeScript AST转换为IR

转换过程保持语义信息，并提供转换质量验证。

### IR到代码生成器

`IRToCodeGenerator`负责将IR转换为目标语言的代码：

- `PythonCodeGenerator`: 生成Python代码
- `JavaScriptCodeGenerator`: 生成JavaScript代码

代码生成过程应用语言惯用法，并保持代码格式和风格。

## 使用示例

### 解析代码

```python
# 解析Python代码
python_parser = PythonParser()
result = python_parser.parse_code("def hello(): print('Hello, world!')")

# 解析JavaScript代码
js_parser = JSTypeScriptParser(use_typescript=False)
result = js_parser.parse_code("function hello() { console.log('Hello, world!'); }")

# 解析TypeScript代码
ts_parser = JSTypeScriptParser(use_typescript=True)
result = ts_parser.parse_code("function hello(): void { console.log('Hello, world!'); }")
```

### 代码转换

```python
# Python到JavaScript的转换
python_parser = PythonParser()
python_converter = PythonASTToIRConverter()
js_generator = JavaScriptCodeGenerator()

# 解析Python代码
parse_result = python_parser.parse_code("def hello(name): return f'Hello, {name}!'")

# 转换到IR
ir_module = python_converter.convert(parse_result.ast)

# 生成JavaScript代码
js_code = js_generator.generate(ir_module)
print(js_code)  # 输出: function hello(name) { return `Hello, ${name}!`; }
```

## 扩展指南

### 添加新的语言支持

1. 创建新的语言解析器，继承`LanguageParser`
2. 实现AST到IR的转换器
3. 实现IR到目标语言的代码生成器
4. 添加单元测试

### 增强IR功能

1. 在`ir_model.py`中添加新的IR节点类型
2. 更新相关的转换器和生成器
3. 添加单元测试验证新功能

## 性能优化

系统实现了以下性能优化：

1. 增量解析：支持只解析变更的代码部分
2. 节点缓存：缓存常用的AST节点查询结果
3. 懒加载：按需加载和处理AST节点

## 错误处理

系统提供了全面的错误处理机制：

1. 语法错误恢复：尝试从语法错误中恢复，继续解析
2. 详细的错误信息：包含错误位置和上下文
3. 降级处理：在转换失败时提供降级处理选项