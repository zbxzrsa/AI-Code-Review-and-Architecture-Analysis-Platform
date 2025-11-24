"""
渐进式转换和错误恢复机制

提供代码转换的渐进式支持，包括：
- 可转换子集识别
- 渐进式代码迁移
- 混合语言项目支持
"""
from typing import Dict, List, Any, Optional, Tuple, Set
import re
import json
import os
import ast


class IncrementalConverter:
    """渐进式转换引擎，支持部分转换和转换进度跟踪"""
    
    def __init__(self):
        """初始化渐进式转换引擎"""
        self.supported_features = self.load_supported_features()
        self.conversion_progress = {}
        
    def load_supported_features(self) -> Dict[str, Dict[str, List[str]]]:
        """
        加载支持的特性列表
        
        Returns:
            按源语言和目标语言分类的支持特性
        """
        # 定义各语言对支持的特性
        return {
            "python": {
                "javascript": [
                    "基本语法", "循环结构", "条件语句", "函数定义", 
                    "类定义", "异常处理", "列表操作", "字典操作",
                    "字符串操作", "模块导入", "Django框架"
                ],
                "typescript": [
                    "基本语法", "循环结构", "条件语句", "函数定义", 
                    "类定义", "异常处理", "列表操作", "字典操作",
                    "字符串操作", "模块导入", "类型注解", "Django框架"
                ],
                "cpp": [
                    "基本语法", "循环结构", "条件语句", "函数定义", 
                    "类定义", "异常处理", "资源管理"
                ],
            },
            "javascript": {
                "python": [
                    "基本语法", "循环结构", "条件语句", "函数定义", 
                    "类定义", "异常处理", "数组操作", "对象操作",
                    "字符串操作", "模块导入", "Express框架"
                ],
                "typescript": [
                    "基本语法", "循环结构", "条件语句", "函数定义", 
                    "类定义", "异常处理", "数组操作", "对象操作",
                    "字符串操作", "模块导入"
                ],
            },
            "java": {
                "csharp": [
                    "基本语法", "循环结构", "条件语句", "函数定义", 
                    "类定义", "异常处理", "集合操作", "字符串操作",
                    "Spring框架", "ORM映射"
                ],
                "kotlin": [
                    "基本语法", "循环结构", "条件语句", "函数定义", 
                    "类定义", "异常处理", "集合操作", "字符串操作"
                ],
            },
        }
    
    def can_convert(self, code_snippet: str, source_lang: str, target_lang: str) -> Tuple[bool, List[str]]:
        """
        检查代码片段是否可转换
        
        Args:
            code_snippet: 代码片段
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            是否可转换及不支持的特性列表
        """
        source_lang = source_lang.lower()
        target_lang = target_lang.lower()
        
        # 检查语言对是否支持
        if source_lang not in self.supported_features or target_lang not in self.supported_features.get(source_lang, {}):
            return False, [f"不支持从 {source_lang} 到 {target_lang} 的转换"]
        
        # 获取支持的特性
        supported = self.supported_features[source_lang][target_lang]
        
        # 分析代码中的特性
        features = self._analyze_code_features(code_snippet, source_lang)
        
        # 检查不支持的特性
        unsupported = [f for f in features if f not in supported]
        
        return len(unsupported) == 0, unsupported
    
    def _analyze_code_features(self, code: str, language: str) -> List[str]:
        """
        分析代码中使用的特性
        
        Args:
            code: 代码
            language: 代码语言
            
        Returns:
            特性列表
        """
        features = ["基本语法"]  # 默认包含基本语法
        
        if language == "python":
            # 检测Python特性
            if re.search(r'\bfor\b.*\bin\b', code):
                features.append("循环结构")
            if re.search(r'\bif\b.*\belse\b', code):
                features.append("条件语句")
            if re.search(r'\bdef\b', code):
                features.append("函数定义")
            if re.search(r'\bclass\b', code):
                features.append("类定义")
            if re.search(r'\btry\b.*\bexcept\b', code):
                features.append("异常处理")
            if re.search(r'\[\s*.*\s*\]', code):
                features.append("列表操作")
            if re.search(r'\{\s*.*\s*\}', code):
                features.append("字典操作")
            if re.search(r'\bimport\b|\bfrom\b.*\bimport\b', code):
                features.append("模块导入")
            if re.search(r'\bJsonResponse\b|\bHttpResponse\b|\brender\b', code):
                features.append("Django框架")
                
        elif language == "javascript":
            # 检测JavaScript特性
            if re.search(r'\bfor\b|\bwhile\b', code):
                features.append("循环结构")
            if re.search(r'\bif\b.*\belse\b', code):
                features.append("条件语句")
            if re.search(r'\bfunction\b|\=\>\s*\{', code):
                features.append("函数定义")
            if re.search(r'\bclass\b', code):
                features.append("类定义")
            if re.search(r'\btry\b.*\bcatch\b', code):
                features.append("异常处理")
            if re.search(r'\[\s*.*\s*\]', code):
                features.append("数组操作")
            if re.search(r'\{\s*.*\s*\}', code):
                features.append("对象操作")
            if re.search(r'\brequire\b|\bimport\b', code):
                features.append("模块导入")
            if re.search(r'\bexpress\b|\bapp\.get\b|\bapp\.post\b|\bres\.json\b', code):
                features.append("Express框架")
                
        elif language == "java":
            # 检测Java特性
            if re.search(r'\bfor\b|\bwhile\b', code):
                features.append("循环结构")
            if re.search(r'\bif\b.*\belse\b', code):
                features.append("条件语句")
            if re.search(r'\bpublic\b.*\b\w+\s*\(', code):
                features.append("函数定义")
            if re.search(r'\bclass\b', code):
                features.append("类定义")
            if re.search(r'\btry\b.*\bcatch\b', code):
                features.append("异常处理")
            if re.search(r'\bList\b|\bArrayList\b|\bSet\b', code):
                features.append("集合操作")
            if re.search(r'\bString\b', code):
                features.append("字符串操作")
            if re.search(r'\bimport\b', code):
                features.append("模块导入")
            if re.search(r'\@RestController\b|\@GetMapping\b|\@PostMapping\b', code):
                features.append("Spring框架")
            if re.search(r'\@Entity\b|\@Table\b|\@Column\b', code):
                features.append("ORM映射")
        
        return features
    
    def partial_convert(self, code: str, source_lang: str, target_lang: str, 
                       strategy: str = 'aggressive') -> Dict[str, Any]:
        """
        尝试部分转换，返回可转换部分和问题列表
        
        Args:
            code: 源代码
            source_lang: 源语言
            target_lang: 目标语言
            strategy: 转换策略，'conservative'(保守)或'aggressive'(激进)
            
        Returns:
            转换结果，包括转换的代码、未转换的代码和问题列表
        """
        # 分割代码为可能的独立块
        blocks = self._split_code_blocks(code, source_lang)
        
        converted_blocks = []
        unconverted_blocks = []
        problems = []
        
        for i, block in enumerate(blocks):
            can_convert, issues = self.can_convert(block, source_lang, target_lang)
            
            if can_convert:
                # 模拟转换过程
                try:
                    # 这里应调用实际的转换函数
                    converted = f"// 已转换: 块 {i+1}\n{block}"
                    converted_blocks.append({
                        "original": block,
                        "converted": converted,
                        "block_index": i
                    })
                except Exception as e:
                    problems.append({
                        "block_index": i,
                        "error": str(e),
                        "code": block
                    })
                    unconverted_blocks.append({
                        "block_index": i,
                        "code": block,
                        "reason": f"转换错误: {str(e)}"
                    })
            else:
                unconverted_blocks.append({
                    "block_index": i,
                    "code": block,
                    "reason": f"不支持的特性: {', '.join(issues)}"
                })
                problems.append({
                    "block_index": i,
                    "unsupported_features": issues,
                    "code": block
                })
        
        # 根据策略决定如何处理未转换的块
        if strategy == 'aggressive':
            # 激进策略：尝试转换尽可能多的代码，保留原始注释
            final_code = []
            for i in range(len(blocks)):
                converted_block = next((b for b in converted_blocks if b["block_index"] == i), None)
                if converted_block:
                    final_code.append(converted_block["converted"])
                else:
                    unconverted = next((b for b in unconverted_blocks if b["block_index"] == i), None)
                    if unconverted:
                        final_code.append(f"// TODO: 未转换的代码块 - {unconverted['reason']}\n{unconverted['code']}")
            
            result_code = "\n\n".join(final_code)
        else:
            # 保守策略：只包含成功转换的部分
            result_code = "\n\n".join([b["converted"] for b in converted_blocks])
        
        # 计算转换完成度
        total_lines = len(code.split('\n'))
        converted_lines = sum(len(b["converted"].split('\n')) for b in converted_blocks)
        completion_percentage = (converted_lines / total_lines) * 100 if total_lines > 0 else 0
        
        return {
            "converted_code": result_code,
            "converted_blocks": converted_blocks,
            "unconverted_blocks": unconverted_blocks,
            "problems": problems,
            "completion_percentage": round(completion_percentage, 2),
            "strategy": strategy
        }
    
    def _split_code_blocks(self, code: str, language: str) -> List[str]:
        """
        将代码分割为可能的独立块
        
        Args:
            code: 源代码
            language: 源代码语言
            
        Returns:
            代码块列表
        """
        if language == "python":
            # 尝试使用AST分割Python代码
            try:
                tree = ast.parse(code)
                blocks = []
                
                # 按函数和类分割
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        start_line = node.lineno
                        end_line = 0
                        
                        # 找到节点的最后一行
                        for child in ast.walk(node):
                            if hasattr(child, 'lineno'):
                                end_line = max(end_line, child.lineno)
                        
                        # 提取代码块
                        lines = code.split('\n')
                        block = '\n'.join(lines[start_line-1:end_line])
                        blocks.append(block)
                
                # 如果没有找到函数或类，返回整个代码
                if not blocks:
                    blocks = [code]
                
                return blocks
            except SyntaxError:
                # 如果AST解析失败，回退到简单分割
                pass
        
        # 简单分割：按空行分割
        blocks = re.split(r'\n\s*\n', code)
        return [b for b in blocks if b.strip()]
    
    def suggest_workarounds(self, unconvertible_code: str, source_lang: str, target_lang: str) -> List[Dict[str, str]]:
        """
        为不可转换代码提供变通方案
        
        Args:
            unconvertible_code: 不可转换的代码
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            变通方案列表
        """
        workarounds = []
        
        # 分析不可转换的原因
        _, issues = self.can_convert(unconvertible_code, source_lang, target_lang)
        
        # 根据不同的问题提供变通方案
        for issue in issues:
            if issue == "Django框架" and target_lang == "javascript":
                workarounds.append({
                    "issue": "Django框架",
                    "suggestion": "使用Express.js的等价功能，例如将Django的视图函数转换为Express路由处理器",
                    "example": """
// Express等价实现
const express = require('express');
const app = express();

app.get('/api/data', (req, res) => {
  // 处理请求
  res.json({ message: 'Hello' });
});
"""
                })
            elif issue == "ORM映射" and source_lang == "java" and target_lang == "csharp":
                workarounds.append({
                    "issue": "ORM映射",
                    "suggestion": "将JPA注解转换为Entity Framework注解",
                    "example": """
// C# Entity Framework等价实现
[Table("users")]
public class User
{
    [Key]
    public int Id { get; set; }
    
    [Required]
    [MaxLength(100)]
    public string Username { get; set; }
}
"""
                })
            # 添加更多特定问题的变通方案...
        
        # 通用变通方案
        workarounds.append({
            "issue": "通用",
            "suggestion": "考虑使用语言互操作性机制，例如通过API或微服务架构分离不同语言的代码",
            "example": "将难以转换的功能封装为独立服务，通过HTTP或消息队列进行通信"
        })
        
        return workarounds
    
    def estimate_conversion_effort(self, codebase: Dict[str, str], source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        估算完整转换所需工作量
        
        Args:
            codebase: 代码库，文件路径到内容的映射
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            工作量估算
        """
        total_files = len(codebase)
        total_lines = sum(len(content.split('\n')) for content in codebase.values())
        
        convertible_files = 0
        convertible_lines = 0
        partial_files = 0
        unconvertible_files = 0
        
        file_details = []
        
        for file_path, content in codebase.items():
            can_convert, issues = self.can_convert(content, source_lang, target_lang)
            
            if can_convert:
                convertible_files += 1
                convertible_lines += len(content.split('\n'))
                file_details.append({
                    "file": file_path,
                    "status": "可完全转换",
                    "issues": []
                })
            elif not issues:  # 部分可转换
                partial_result = self.partial_convert(content, source_lang, target_lang)
                partial_files += 1
                convertible_lines += int(partial_result["completion_percentage"] * len(content.split('\n')) / 100)
                file_details.append({
                    "file": file_path,
                    "status": "部分可转换",
                    "completion": f"{partial_result['completion_percentage']}%",
                    "issues": [p.get("reason", "未知问题") for p in partial_result["unconverted_blocks"]]
                })
            else:
                unconvertible_files += 1
                file_details.append({
                    "file": file_path,
                    "status": "不可转换",
                    "issues": issues
                })
        
        # 估算工作量（人天）
        # 假设：完全可转换=每1000行1天，部分可转换=每500行1天，不可转换=每200行1天
        effort_convertible = convertible_lines / 1000
        effort_partial = (total_lines - convertible_lines) / 500
        effort_manual = (total_lines - convertible_lines) / 200
        
        # 总体完成度
        completion_percentage = (convertible_lines / total_lines) * 100 if total_lines > 0 else 0
        
        return {
            "total_files": total_files,
            "total_lines": total_lines,
            "convertible_files": convertible_files,
            "partial_files": partial_files,
            "unconvertible_files": unconvertible_files,
            "completion_percentage": round(completion_percentage, 2),
            "estimated_effort_days": {
                "optimistic": round(effort_convertible + effort_partial, 1),
                "realistic": round(effort_convertible + effort_manual, 1),
                "pessimistic": round(effort_convertible + effort_manual * 1.5, 1)
            },
            "file_details": file_details
        }


# 单例模式，确保全局只有一个转换器实例
_converter_instance = None

def get_incremental_converter() -> IncrementalConverter:
    """获取渐进式转换器实例"""
    global _converter_instance
    if _converter_instance is None:
        _converter_instance = IncrementalConverter()
    return _converter_instance