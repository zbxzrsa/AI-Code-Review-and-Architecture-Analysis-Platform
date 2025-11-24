"""
代码质量评估系统

提供代码转换质量的评估机制，包括：
- 可读性评分
- 维护性指标
- 最佳实践符合度
- 综合质量评分
"""
from typing import Dict, List, Any, Optional, Tuple
import re
import math
import statistics


class QualityMetrics:
    """代码质量评估器，用于评估转换后代码的质量"""
    
    def __init__(self):
        """初始化质量评估器"""
        # 语言特定的最佳实践规则
        self.best_practices = {
            "python": [
                (r"^\s*import\s+\*\s*$", "避免使用 import *"),
                (r"except\s*:", "避免使用空except子句"),
                (r"except\s+Exception\s*:", "避免捕获所有异常"),
                (r"global\s+", "减少全局变量的使用"),
                (r"exec\s*\(", "避免使用exec"),
            ],
            "javascript": [
                (r"==(?!=)", "使用 === 代替 =="),
                (r"!=(?!=)", "使用 !== 代替 !="),
                (r"var\s+", "使用 let 或 const 代替 var"),
                (r"with\s*\(", "避免使用with语句"),
                (r"eval\s*\(", "避免使用eval"),
            ],
            "typescript": [
                (r"any(?!\w)", "避免使用any类型"),
                (r"==(?!=)", "使用 === 代替 =="),
                (r"!=(?!=)", "使用 !== 代替 !="),
                (r"var\s+", "使用 let 或 const 代替 var"),
                (r"eval\s*\(", "避免使用eval"),
            ],
            "java": [
                (r"catch\s*\(\s*Exception\s+", "避免捕获通用Exception"),
                (r"public\s+static\s+\w+\s+\w+", "减少静态方法的使用"),
                (r"instanceof", "减少instanceof的使用"),
                (r"synchronized", "谨慎使用synchronized"),
            ],
            "csharp": [
                (r"goto\s+", "避免使用goto"),
                (r"catch\s*\(\s*Exception\s+", "避免捕获通用Exception"),
                (r"public\s+static\s+\w+\s+\w+", "减少静态方法的使用"),
                (r"dynamic\s+", "谨慎使用dynamic类型"),
            ],
            "cpp": [
                (r"goto\s+", "避免使用goto"),
                (r"using\s+namespace\s+std", "避免使用using namespace std"),
                (r"#define\s+\w+", "优先使用const和constexpr代替宏"),
                (r"malloc\s*\(", "使用new代替malloc"),
                (r"free\s*\(", "使用delete代替free"),
            ],
            "rust": [
                (r"unsafe\s*\{", "减少unsafe块的使用"),
                (r"unwrap\(\)", "避免使用unwrap()"),
                (r"expect\(\s*\"", "避免使用expect()"),
                (r"panic!\s*\(", "减少panic!的使用"),
            ],
        }
    
    def calculate_conversion_score(self, source: str, target: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        计算转换质量综合评分
        
        Args:
            source: 源代码
            target: 目标代码
            language: 目标语言（可选，未提供时自动推断）
            
        Returns:
            综合评分和详细指标
        """
        # 自动语言推断（简单启发式）
        if not language:
            lowered = target.lower()
            if re.search(r"\bdef\s+\w+\s*\(", lowered):
                language = "python"
            elif re.search(r"\bfunction\s+\w+\s*\(", lowered) or "const " in lowered or "let " in lowered:
                language = "javascript"
            elif re.search(r"class\s+\w+\s*\{", lowered) and "using" in lowered:
                language = "csharp"
            else:
                language = "javascript"

        # 计算各项指标
        readability = self.analyze_readability(target)
        best_practices_score = self.check_best_practices(target, language)
        maintainability = self.assess_maintainability(target)
        
        # 计算综合得分 (加权平均)
        total_score = (
            readability["score"] * 0.3 + 
            best_practices_score["score"] * 0.3 + 
            maintainability["score"] * 0.4
        )
        
        # 评分等级
        grade = "A" if total_score >= 90 else "B" if total_score >= 80 else "C" if total_score >= 70 else "D" if total_score >= 60 else "F"
        
        return {
            "overall_score": round(total_score, 2),
            "grade": grade,
            "readability": readability,
            "best_practices": best_practices_score,
            "maintainability": maintainability,
        }
    
    def analyze_readability(self, code: str) -> Dict[str, Any]:
        """
        分析代码可读性
        
        Args:
            code: 要分析的代码
            
        Returns:
            可读性评分和详细指标
        """
        # 分割代码行
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return {"score": 0, "details": "空代码"}
        
        # 计算行长度
        line_lengths = [len(line) for line in non_empty_lines]
        avg_line_length = statistics.mean(line_lengths) if line_lengths else 0
        max_line_length = max(line_lengths) if line_lengths else 0
        
        # 计算缩进一致性
        indent_sizes = []
        for line in non_empty_lines:
            leading_spaces = len(line) - len(line.lstrip())
            if leading_spaces > 0:
                indent_sizes.append(leading_spaces)
        
        indent_consistency = 1.0
        if indent_sizes:
            # 检查缩进是否一致（是否为同一缩进单位的倍数）
            if len(set(indent_sizes)) > 1:
                # 找出最小的非零缩进
                min_indent = min(i for i in indent_sizes if i > 0)
                # 检查所有缩进是否是最小缩进的倍数
                consistent_indents = all(i % min_indent == 0 for i in indent_sizes)
                indent_consistency = 1.0 if consistent_indents else 0.5
        
        # 计算注释比例
        comment_lines = sum(1 for line in lines if re.match(r'^\s*(//|#|/\*|\*)', line.strip()))
        comment_ratio = comment_lines / len(lines) if lines else 0
        
        # 计算空行比例
        blank_lines = len(lines) - len(non_empty_lines)
        blank_ratio = blank_lines / len(lines) if lines else 0
        
        # 计算标识符命名质量
        identifiers = re.findall(r'\b[a-zA-Z_]\w*\b', code)
        avg_identifier_length = statistics.mean([len(i) for i in identifiers]) if identifiers else 0
        
        # 可读性得分计算
        line_length_score = 100 - max(0, min(100, (avg_line_length - 40) * 2))
        max_line_penalty = max(0, min(20, (max_line_length - 80) / 4))
        comment_score = min(100, comment_ratio * 200)
        blank_score = 100 if 0.1 <= blank_ratio <= 0.3 else max(0, 100 - abs(blank_ratio - 0.2) * 300)
        # 放宽对标识符长度的良好区间，降低惩罚斜率
        identifier_score = 100 if 6 <= avg_identifier_length <= 24 else max(0, 100 - abs(avg_identifier_length - 15) * 7)
        
        # 综合得分
        readability_score = (
            line_length_score * 0.3 +
            (100 - max_line_penalty) * 0.1 +
            comment_score * 0.2 +
            blank_score * 0.1 +
            identifier_score * 0.2 +
            indent_consistency * 100 * 0.1
        )
        
        # 对常见的“结构清晰+有注释+缩进规范”的代码给予适度奖励
        has_function = bool(re.search(r"\bdef\s+\w+\s*\(|\bfunction\s+\w+\s*\(", code))
        has_doc_or_comment = comment_ratio > 0.05 or '"""' in code or "'''" in code
        high_indent_consistency = indent_consistency >= 0.8
        if has_function and has_doc_or_comment and high_indent_consistency:
            readability_score = min(100, readability_score + 20)
            # 对拥有函数+文档字符串+高缩进一致性的代码设定合理下限
            readability_score = max(readability_score, 72)
        
        return {
            "score": round(readability_score, 2),
            "details": {
                "avg_line_length": round(avg_line_length, 2),
                "max_line_length": max_line_length,
                "comment_ratio": round(comment_ratio, 2),
                "blank_line_ratio": round(blank_ratio, 2),
                "avg_identifier_length": round(avg_identifier_length, 2),
                "indent_consistency": round(indent_consistency, 2),
            }
        }
    
    def check_best_practices(self, code: str, language: str) -> Dict[str, Any]:
        """
        检查最佳实践符合度
        
        Args:
            code: 要分析的代码
            language: 代码语言
            
        Returns:
            最佳实践评分和详细问题列表
        """
        language = language.lower()
        if language not in self.best_practices:
            return {"score": 0, "details": f"不支持的语言: {language}"}
        
        # 检查代码中的反模式
        issues = []
        suggestions = []
        suggestion_map = {
            "python": {
                "避免使用 import *": "明确导入需要的符号，例如: from module import name",
                "避免使用空except子句": "捕获特定异常类型，提高错误处理的确定性",
                "避免捕获所有异常": "只捕获可能发生的异常种类，避免吞掉错误",
                "减少全局变量的使用": "改用函数参数或类属性，控制状态作用域",
                "避免使用exec": "使用更安全的替代方案，如 getattr/dispatch 表",
            },
            "javascript": {
                "使用 === 代替 ==": "始终使用严格相等避免隐式类型转换",
                "使用 !== 代替 !=": "使用严格不等避免隐式类型转换",
                "使用 let 或 const 代替 var": "使用块级作用域与不可变变量，提高可维护性",
                "避免使用with语句": "with 会改变作用域链，易产生歧义，避免使用",
                "避免使用eval": "eval 存在安全与性能问题，改用安全的解析方法",
            },
            "typescript": {
                "避免使用any类型": "使用明确的类型或泛型，提升类型安全",
                "使用 === 代替 ==": "始终使用严格相等避免隐式类型转换",
                "使用 !== 代替 !=": "使用严格不等避免隐式类型转换",
                "使用 let 或 const 代替 var": "使用块级作用域与不可变变量",
                "避免使用eval": "eval 存在安全与性能问题，改用安全的解析方法",
            },
            "java": {
                "避免捕获通用Exception": "捕获具体异常类型，或在上层统一处理",
                "减少静态方法的使用": "优先实例方法，利于测试与可扩展",
                "减少instanceof的使用": "使用多态或模式匹配替代",
                "谨慎使用synchronized": "使用并发库 (java.util.concurrent) 更安全",
            },
            "csharp": {
                "避免使用goto": "使用结构化控制流 (if/for/while/foreach) 替代",
                "避免捕获通用Exception": "捕获具体异常类型，或在上层统一处理",
                "减少静态方法的使用": "优先实例方法，利于测试与可扩展",
                "谨慎使用dynamic类型": "仅在必要时使用，优先静态类型",
            },
            "cpp": {
                "避免使用goto": "使用结构化控制流替代，提高可读性",
                "避免使用using namespace std": "限定命名空间或按需 using",
                "优先使用const和constexpr代替宏": "宏缺少类型检查，优先常量/内联函数",
                "使用new代替malloc": "C++ 中 prefer new/delete 或智能指针",
                "使用delete代替free": "与 new/new[] 搭配使用 delete/delete[]",
            },
            "rust": {
                "减少unsafe块的使用": "优先安全抽象，隔离 unsafe 封装",
                "避免使用unwrap()": "使用 ? 操作符或 match 处理错误",
                "避免使用expect()": "以可恢复方式处理错误，或提供合理默认",
                "减少panic!的使用": "在库中避免 panic，返回 Result 更合理",
            },
        }
        for pattern, message in self.best_practices[language]:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "line": line_num,
                    "message": message,
                    "snippet": code.split('\n')[line_num - 1] if line_num <= len(code.split('\n')) else ""
                })
                # 生成对应建议
                suggest = suggestion_map.get(language, {}).get(message)
                if suggest:
                    suggestions.append({
                        "line": line_num,
                        "description": message,
                        "suggestion": suggest,
                        "severity": "medium"
                    })
        
        # 计算得分
        # 基础分100，每个问题扣5分，最低0分
        score = max(0, 100 - len(issues) * 5)
        
        return {
            "score": score,
            "issues": issues,
            "issue_count": len(issues),
            "suggestions": suggestions,
        }
    
    def assess_maintainability(self, code: str) -> Dict[str, Any]:
        """
        评估代码可维护性
        
        Args:
            code: 要分析的代码
            
        Returns:
            可维护性评分和详细指标
        """
        # 分割代码行
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return {"score": 0, "details": "空代码"}
        
        # 计算代码复杂度指标
        
        # 1. 圈复杂度（简化估计）
        decision_points = (
            len(re.findall(r'\bif\b', code)) +
            len(re.findall(r'\belse\b', code)) +
            len(re.findall(r'\bfor\b', code)) +
            len(re.findall(r'\bwhile\b', code)) +
            len(re.findall(r'\bcase\b', code)) +
            len(re.findall(r'\bcatch\b', code)) +
            len(re.findall(r'\b\|\|\b', code)) +
            len(re.findall(r'\b&&\b', code))
        )
        
        # 估算函数/方法数量
        function_count = (
            len(re.findall(r'\bfunction\s+\w+\s*\(', code)) +  # JavaScript/TypeScript
            len(re.findall(r'\bdef\s+\w+\s*\(', code)) +       # Python
            len(re.findall(r'\b\w+\s+\w+\s*\([^)]*\)\s*{', code))  # Java/C#/C++
        )
        
        # 如果没有检测到函数，假设至少有一个
        function_count = max(1, function_count)
        
        # 计算平均圈复杂度
        avg_complexity = decision_points / function_count
        
        # 2. 代码重复（简化估计）
        # 检查重复行（连续3行或更多）
        duplicate_lines = 0
        line_hashes = {}
        
        for i in range(len(non_empty_lines) - 2):
            chunk = '\n'.join(non_empty_lines[i:i+3])
            if chunk in line_hashes:
                duplicate_lines += 3
                line_hashes[chunk] += 1
            else:
                line_hashes[chunk] = 1
        
        duplicate_ratio = duplicate_lines / len(non_empty_lines) if non_empty_lines else 0
        
        # 3. 函数长度
        # 简化：假设每个函数平均长度
        avg_function_length = len(non_empty_lines) / function_count
        
        # 计算维护性指数
        # 基于圈复杂度、重复率和函数长度的加权得分
        complexity_score = 100 - min(100, avg_complexity * 10)
        duplication_score = 100 - min(100, duplicate_ratio * 200)
        length_score = 100 - min(100, max(0, (avg_function_length - 15) * 2))
        
        maintainability_score = (
            complexity_score * 0.4 +
            duplication_score * 0.3 +
            length_score * 0.3
        )
        
        return {
            "score": round(maintainability_score, 2),
            "details": {
                "avg_complexity": round(avg_complexity, 2),
                "duplicate_ratio": round(duplicate_ratio, 2),
                "avg_function_length": round(avg_function_length, 2),
                "function_count": function_count,
                "decision_points": decision_points
            }
        }


# 单例模式，确保全局只有一个评估器实例
_metrics_instance = None

def get_metrics() -> QualityMetrics:
    """获取质量评估器实例"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = QualityMetrics()
    return _metrics_instance