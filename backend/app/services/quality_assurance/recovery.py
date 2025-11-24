"""
错误恢复机制

提供代码转换过程中的错误处理和恢复策略，包括：
- 语法错误自动修复
- 语义损失最小化
- 降级方案提供
- 用户干预接口
"""
from typing import Dict, List, Any, Optional, Tuple, Set
import re
import json
import difflib


class ErrorRecoverySystem:
    """错误恢复系统，处理转换过程中的各类错误"""
    
    def __init__(self):
        """初始化错误恢复系统"""
        self.error_patterns = self._load_error_patterns()
        self.user_corrections = {}  # 存储用户修正，用于学习
        
    def _load_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        加载常见错误模式及修复策略
        
        Returns:
            错误模式字典
        """
        return {
            # Python 到 JavaScript 的常见错误
            "python_to_javascript": {
                "syntax_errors": {
                    # 缩进错误
                    r"IndentationError": {
                        "pattern": r"IndentationError: (.*)",
                        "fix": self._fix_indentation,
                        "description": "缩进错误，JavaScript不依赖缩进表示代码块"
                    },
                    # 字典语法错误
                    r"dict_syntax": {
                        "pattern": r"(\w+)\[['\"](.*?)['\"]\]",
                        "replacement": r"\1.\2",
                        "description": "Python字典访问转换为JavaScript对象属性访问"
                    },
                    # 列表推导式
                    r"list_comprehension": {
                        "pattern": r"\[(.*?) for (.*?) in (.*?)\]",
                        "replacement": r"\3.map((\2) => \1)",
                        "description": "Python列表推导式转换为JavaScript的map方法"
                    }
                },
                "semantic_gaps": {
                    "range": {
                        "pattern": r"range\((\d+)\)",
                        "replacement": r"[...Array(\1).keys()]",
                        "description": "Python的range函数转换为JavaScript数组生成"
                    },
                    "dict_methods": {
                        "pattern": r"(\w+)\.items\(\)",
                        "replacement": r"Object.entries(\1)",
                        "description": "Python字典items方法转换为JavaScript的Object.entries"
                    }
                }
            },
            
            # Java 到 C# 的常见错误
            "java_to_csharp": {
                "syntax_errors": {
                    # 访问修饰符位置
                    r"modifier_order": {
                        "pattern": r"(public|private|protected)\s+static\s+(.*?)\s+(\w+)",
                        "replacement": r"\1 static \2 \3",
                        "description": "Java和C#的修饰符顺序不同"
                    },
                    # 泛型语法
                    r"generic_syntax": {
                        "pattern": r"<(\w+)\s+extends\s+(\w+)>",
                        "replacement": r"<\1> where \1 : \2",
                        "description": "Java泛型约束语法转换为C#的where子句"
                    }
                },
                "semantic_gaps": {
                    "exception_handling": {
                        "pattern": r"throws\s+([\w,\s]+)",
                        "replacement": "",  # C#不需要声明抛出的异常
                        "description": "Java的throws子句在C#中不需要"
                    },
                    "interface_methods": {
                        "pattern": r"(public|private|protected)?\s+interface\s+(\w+)\s*\{",
                        "replacement": r"public interface \2 {",
                        "description": "Java接口成员默认为public，C#需要显式声明"
                    }
                }
            }
        }
    
    def handle_syntax_error(self, error: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理语法转换错误
        
        Args:
            error: 错误信息
            context: 错误上下文，包括源代码、目标语言等
            
        Returns:
            修复建议和修复后的代码
        """
        source_lang = context.get("source_lang", "")
        target_lang = context.get("target_lang", "")
        code = context.get("code", "")
        
        conversion_key = f"{source_lang}_to_{target_lang}"
        
        # 检查是否有匹配的错误模式
        if conversion_key in self.error_patterns:
            syntax_errors = self.error_patterns[conversion_key]["syntax_errors"]
            
            for error_key, error_info in syntax_errors.items():
                if re.search(error_key, error):
                    # 找到匹配的错误模式
                    if callable(error_info.get("fix")):
                        # 使用自定义修复函数
                        fixed_code = error_info["fix"](code, error)
                    elif "pattern" in error_info and "replacement" in error_info:
                        # 使用正则表达式替换
                        fixed_code = re.sub(error_info["pattern"], error_info["replacement"], code)
                    else:
                        fixed_code = code
                    
                    return {
                        "original_error": error,
                        "fixed_code": fixed_code,
                        "description": error_info.get("description", "自动修复"),
                        "confidence": self._calculate_fix_confidence(code, fixed_code)
                    }
        
        # 如果没有匹配的错误模式，尝试通用修复
        return self._generic_syntax_fix(error, code, source_lang, target_lang)
    
    def _fix_indentation(self, code: str, error: str) -> str:
        """
        修复缩进错误
        
        Args:
            code: 源代码
            error: 错误信息
            
        Returns:
            修复后的代码
        """
        # 将Python的缩进块转换为JavaScript的花括号块
        lines = code.split('\n')
        result = []
        indent_stack = [0]
        
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
            
            if not stripped:  # 空行
                result.append(line)
                continue
                
            if current_indent > indent_stack[-1]:
                # 缩进增加，添加左花括号
                result[-1] += " {"
                indent_stack.append(current_indent)
            elif current_indent < indent_stack[-1]:
                # 缩进减少，添加右花括号
                while current_indent < indent_stack[-1]:
                    indent_stack.pop()
                    result.append(" " * indent_stack[-1] + "}")
            
            # 添加当前行
            result.append(line)
        
        # 处理文件末尾的缩进
        while len(indent_stack) > 1:
            indent_stack.pop()
            result.append(" " * indent_stack[-1] + "}")
            
        return '\n'.join(result)
    
    def _generic_syntax_fix(self, error: str, code: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        通用语法修复策略
        
        Args:
            error: 错误信息
            code: 源代码
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            修复建议
        """
        # 提取错误行号（如果有）
        line_match = re.search(r"line\s+(\d+)", error)
        line_num = int(line_match.group(1)) if line_match else -1
        
        lines = code.split('\n')
        suggestions = []
        
        if line_num > 0 and line_num <= len(lines):
            error_line = lines[line_num - 1]
            
            # 常见语法错误检查
            if ";" not in error_line and target_lang in ["javascript", "java", "csharp"]:
                suggestions.append("添加分号")
                lines[line_num - 1] = error_line + ";"
            
            if ":" in error_line and target_lang in ["javascript", "java", "csharp"]:
                suggestions.append("将冒号替换为花括号")
                lines[line_num - 1] = error_line.replace(":", " {")
            
            # 更多通用修复...
        
        return {
            "original_error": error,
            "fixed_code": '\n'.join(lines) if suggestions else code,
            "description": "尝试通用语法修复: " + ", ".join(suggestions) if suggestions else "无法自动修复",
            "confidence": 0.5 if suggestions else 0.1,
            "suggestions": suggestions
        }
    
    def _calculate_fix_confidence(self, original: str, fixed: str) -> float:
        """
        计算修复的置信度
        
        Args:
            original: 原始代码
            fixed: 修复后的代码
            
        Returns:
            置信度（0-1）
        """
        if original == fixed:
            return 0.0
            
        # 使用difflib计算相似度
        matcher = difflib.SequenceMatcher(None, original, fixed)
        similarity = matcher.ratio()
        
        # 变化越小，置信度越高
        return max(0, 1 - (1 - similarity) * 2)
    
    def handle_semantic_gap(self, source_feature: str, target_lang: str, code_context: str) -> Dict[str, Any]:
        """
        处理语义鸿沟
        
        Args:
            source_feature: 源语言特性
            target_lang: 目标语言
            code_context: 代码上下文
            
        Returns:
            处理建议
        """
        # 从源特性名称推断源语言
        source_lang = ""
        if "python" in source_feature.lower():
            source_lang = "python"
        elif "java" in source_feature.lower():
            source_lang = "java"
        elif "javascript" in source_feature.lower() or "js" in source_feature.lower():
            source_lang = "javascript"
        
        conversion_key = f"{source_lang}_to_{target_lang}"
        
        if conversion_key in self.error_patterns and "semantic_gaps" in self.error_patterns[conversion_key]:
            semantic_gaps = self.error_patterns[conversion_key]["semantic_gaps"]
            
            for gap_key, gap_info in semantic_gaps.items():
                if re.search(gap_key, source_feature, re.IGNORECASE):
                    # 找到匹配的语义鸿沟
                    if "pattern" in gap_info and "replacement" in gap_info:
                        # 使用正则表达式替换
                        fixed_code = re.sub(gap_info["pattern"], gap_info["replacement"], code_context)
                        
                        return {
                            "feature": source_feature,
                            "target_lang": target_lang,
                            "description": gap_info.get("description", "语义鸿沟处理"),
                            "original_code": code_context,
                            "suggested_code": fixed_code,
                            "confidence": self._calculate_fix_confidence(code_context, fixed_code)
                        }
        
        # 如果没有匹配的语义鸿沟处理，返回通用建议
        return self._provide_semantic_gap_suggestions(source_feature, source_lang, target_lang, code_context)
    
    def _provide_semantic_gap_suggestions(self, feature: str, source_lang: str, target_lang: str, code: str) -> Dict[str, Any]:
        """
        提供语义鸿沟的通用建议
        
        Args:
            feature: 特性名称
            source_lang: 源语言
            target_lang: 目标语言
            code: 代码上下文
            
        Returns:
            建议信息
        """
        suggestions = []
        
        # 常见语义鸿沟的通用建议
        if source_lang == "python" and target_lang == "javascript":
            if "list comprehension" in feature.lower():
                suggestions.append({
                    "description": "使用Array.map()替代列表推导式",
                    "example": "const newArray = array.map(item => transformFunction(item));"
                })
            elif "generator" in feature.lower():
                suggestions.append({
                    "description": "使用迭代器或生成器函数",
                    "example": "function* generatorFunction() { yield 1; yield 2; }"
                })
            elif "context manager" in feature.lower() or "with" in feature.lower():
                suggestions.append({
                    "description": "使用try/finally或自定义资源管理",
                    "example": "try { const resource = openResource(); /* 使用资源 */ } finally { resource.close(); }"
                })
        
        elif source_lang == "java" and target_lang == "csharp":
            if "checked exception" in feature.lower():
                suggestions.append({
                    "description": "C#不支持检查异常，使用返回值或异常处理",
                    "example": "// 使用返回值表示成功/失败\npublic bool TryOperation(out Result result) { ... }"
                })
            elif "anonymous inner class" in feature.lower():
                suggestions.append({
                    "description": "使用Lambda表达式或委托",
                    "example": "button.Click += (sender, e) => { HandleClick(); };"
                })
        
        return {
            "feature": feature,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "suggestions": suggestions,
            "message": "无法自动处理此语义鸿沟，请考虑以下建议" if suggestions else "无法提供自动建议，需要手动处理"
        }
    
    def provide_fallback(self, original_code: str, error_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        提供降级方案
        
        Args:
            original_code: 原始代码
            error_type: 错误类型
            context: 上下文信息
            
        Returns:
            降级方案
        """
        source_lang = context.get("source_lang", "")
        target_lang = context.get("target_lang", "")
        
        fallback_options = []
        
        # 根据错误类型提供不同的降级方案
        if error_type == "syntax_error":
            # 语法错误的降级方案
            fallback_options.append({
                "type": "comment_out",
                "description": "注释掉有问题的代码，并添加TODO标记",
                "code": self._create_commented_code(original_code, target_lang)
            })
            
            fallback_options.append({
                "type": "simplify",
                "description": "简化代码，移除复杂特性",
                "code": self._simplify_code(original_code, source_lang, target_lang)
            })
            
        elif error_type == "semantic_gap":
            # 语义鸿沟的降级方案
            fallback_options.append({
                "type": "alternative_api",
                "description": "使用目标语言的替代API",
                "code": self._suggest_alternative_api(original_code, source_lang, target_lang)
            })
            
            fallback_options.append({
                "type": "interop",
                "description": "使用语言互操作性",
                "code": self._suggest_interop(original_code, source_lang, target_lang)
            })
        
        # 通用降级方案
        fallback_options.append({
            "type": "manual_conversion",
            "description": "保留原始代码作为注释，提供手动转换指南",
            "code": self._create_manual_conversion_guide(original_code, source_lang, target_lang)
        })
        
        return {
            "original_code": original_code,
            "error_type": error_type,
            "fallback_options": fallback_options
        }
    
    def _create_commented_code(self, code: str, target_lang: str) -> str:
        """
        创建注释版本的代码
        
        Args:
            code: 原始代码
            target_lang: 目标语言
            
        Returns:
            注释后的代码
        """
        comment_prefix = "//" if target_lang in ["javascript", "typescript", "java", "csharp", "cpp"] else "#"
        
        commented_lines = []
        commented_lines.append(f"{comment_prefix} TODO: 以下代码需要手动转换")
        commented_lines.append(f"{comment_prefix} 原始代码:")
        
        for line in code.split('\n'):
            commented_lines.append(f"{comment_prefix} {line}")
            
        commented_lines.append("")  # 空行
        commented_lines.append(f"{comment_prefix} 转换后的代码应放在这里")
        commented_lines.append(f"{comment_prefix} 示例:")
        commented_lines.append(f"{comment_prefix} // 目标代码")
        
        return '\n'.join(commented_lines)
    
    def _simplify_code(self, code: str, source_lang: str, target_lang: str) -> str:
        """
        简化代码，移除复杂特性
        
        Args:
            code: 原始代码
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            简化后的代码
        """
        simplified = code
        
        # 移除复杂的语言特性
        if source_lang == "python":
            # 移除列表推导式
            simplified = re.sub(r"\[(.*?) for (.*?) in (.*?)\]", "// TODO: 替换列表推导式", simplified)
            # 移除装饰器
            simplified = re.sub(r"@\w+(\(.*?\))?\s*", "// TODO: 替换装饰器\n", simplified)
            
        elif source_lang == "java":
            # 简化泛型
            simplified = re.sub(r"<.*?>", "<>", simplified)
            # 移除注解
            simplified = re.sub(r"@\w+(\(.*?\))?\s*", "// TODO: 替换注解\n", simplified)
        
        comment_prefix = "//" if target_lang in ["javascript", "typescript", "java", "csharp", "cpp"] else "#"
        return f"{comment_prefix} 简化版本 - 需要手动完善:\n{simplified}"
    
    def _suggest_alternative_api(self, code: str, source_lang: str, target_lang: str) -> str:
        """
        建议替代API
        
        Args:
            code: 原始代码
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            替代API建议
        """
        comment_prefix = "//" if target_lang in ["javascript", "typescript", "java", "csharp", "cpp"] else "#"
        
        suggestions = []
        suggestions.append(f"{comment_prefix} 建议使用以下替代API:")
        
        if source_lang == "python" and target_lang == "javascript":
            if "dict.items()" in code:
                suggestions.append(f"{comment_prefix} Python dict.items() -> JavaScript Object.entries()")
                suggestions.append(f"{comment_prefix} 例如: for (const [key, value] of Object.entries(obj)) {{ ... }}")
                
            if "enumerate(" in code:
                suggestions.append(f"{comment_prefix} Python enumerate() -> JavaScript array.entries()")
                suggestions.append(f"{comment_prefix} 例如: for (const [index, value] of array.entries()) {{ ... }}")
        
        elif source_lang == "java" and target_lang == "csharp":
            if "ArrayList" in code:
                suggestions.append(f"{comment_prefix} Java ArrayList -> C# List<T>")
                suggestions.append(f"{comment_prefix} 例如: List<string> items = new List<string>();")
                
            if "HashMap" in code:
                suggestions.append(f"{comment_prefix} Java HashMap -> C# Dictionary<K,V>")
                suggestions.append(f"{comment_prefix} 例如: Dictionary<string, int> map = new Dictionary<string, int>();")
        
        suggestions.append("")
        suggestions.append(code)
        
        return '\n'.join(suggestions)
    
    def _suggest_interop(self, code: str, source_lang: str, target_lang: str) -> str:
        """
        建议使用语言互操作性
        
        Args:
            code: 原始代码
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            互操作性建议
        """
        comment_prefix = "//" if target_lang in ["javascript", "typescript", "java", "csharp", "cpp"] else "#"
        
        suggestions = []
        suggestions.append(f"{comment_prefix} 建议使用语言互操作性:")
        
        if source_lang == "python" and target_lang == "javascript":
            suggestions.append(f"{comment_prefix} 方案1: 将此功能封装为Python微服务，通过HTTP API调用")
            suggestions.append(f"{comment_prefix} 方案2: 使用WebAssembly将Python代码编译为浏览器可执行格式")
            suggestions.append(f"{comment_prefix} 方案3: 使用Node.js的child_process模块执行Python脚本")
            
        elif source_lang == "java" and target_lang == "csharp":
            suggestions.append(f"{comment_prefix} 方案1: 使用C#的P/Invoke调用Java本地方法")
            suggestions.append(f"{comment_prefix} 方案2: 将Java代码封装为Web服务，通过HTTP调用")
            suggestions.append(f"{comment_prefix} 方案3: 使用IKVM.NET在.NET中运行Java代码")
        
        suggestions.append("")
        suggestions.append(f"{comment_prefix} 原始代码:")
        for line in code.split('\n'):
            suggestions.append(f"{comment_prefix} {line}")
        
        return '\n'.join(suggestions)
    
    def _create_manual_conversion_guide(self, code: str, source_lang: str, target_lang: str) -> str:
        """
        创建手动转换指南
        
        Args:
            code: 原始代码
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            手动转换指南
        """
        comment_prefix = "//" if target_lang in ["javascript", "typescript", "java", "csharp", "cpp"] else "#"
        
        guide = []
        guide.append(f"{comment_prefix} 手动转换指南")
        guide.append(f"{comment_prefix} 从 {source_lang} 转换到 {target_lang}")
        guide.append(f"{comment_prefix} ===================================")
        guide.append(f"{comment_prefix} 原始代码:")
        
        for line in code.split('\n'):
            guide.append(f"{comment_prefix} {line}")
            
        guide.append(f"{comment_prefix} ===================================")
        guide.append(f"{comment_prefix} 转换步骤:")
        
        # 根据语言对提供具体的转换步骤
        if source_lang == "python" and target_lang == "javascript":
            guide.append(f"{comment_prefix} 1. 将Python的缩进块替换为JavaScript的花括号")
            guide.append(f"{comment_prefix} 2. 将Python的函数定义 'def' 替换为 'function'")
            guide.append(f"{comment_prefix} 3. 添加分号到每行语句结尾")
            guide.append(f"{comment_prefix} 4. 将Python的列表推导式转换为Array.map()或Array.filter()")
            guide.append(f"{comment_prefix} 5. 将Python的字典访问方式 dict['key'] 转换为 dict.key 或 dict['key']")
            
        elif source_lang == "java" and target_lang == "csharp":
            guide.append(f"{comment_prefix} 1. 将Java包声明转换为C#命名空间")
            guide.append(f"{comment_prefix} 2. 将Java的泛型语法调整为C#格式")
            guide.append(f"{comment_prefix} 3. 移除Java的checked异常声明")
            guide.append(f"{comment_prefix} 4. 将Java的getter/setter转换为C#属性")
            guide.append(f"{comment_prefix} 5. 将Java的集合类型转换为C#对应类型")
        
        guide.append(f"{comment_prefix} ===================================")
        guide.append(f"{comment_prefix} 转换后的代码应放在下方:")
        guide.append("")
        
        return '\n'.join(guide)
    
    def learn_from_corrections(self, user_fixes: Dict[str, Any]) -> Dict[str, Any]:
        """
        从用户修正中学习
        
        Args:
            user_fixes: 用户修正信息
            
        Returns:
            学习结果
        """
        original_code = user_fixes.get("original_code", "")
        corrected_code = user_fixes.get("corrected_code", "")
        error_type = user_fixes.get("error_type", "")
        source_lang = user_fixes.get("source_lang", "")
        target_lang = user_fixes.get("target_lang", "")
        
        if not original_code or not corrected_code:
            return {"status": "error", "message": "缺少原始代码或修正后的代码"}
        
        # 计算差异
        diff = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            corrected_code.splitlines(keepends=True),
            n=3
        )
        
        # 存储用户修正
        correction_id = f"{source_lang}_{target_lang}_{len(self.user_corrections)}"
        self.user_corrections[correction_id] = {
            "original": original_code,
            "corrected": corrected_code,
            "diff": ''.join(diff),
            "error_type": error_type,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "timestamp": "current_time"  # 实际应用中应使用真实时间戳
        }
        
        # 尝试从修正中提取模式
        patterns = self._extract_patterns_from_correction(original_code, corrected_code, source_lang, target_lang)
        
        return {
            "status": "success",
            "message": "成功学习用户修正",
            "correction_id": correction_id,
            "patterns_extracted": len(patterns),
            "patterns": patterns
        }
    
    def _extract_patterns_from_correction(self, original: str, corrected: str, source_lang: str, target_lang: str) -> List[Dict[str, Any]]:
        """
        从修正中提取模式
        
        Args:
            original: 原始代码
            corrected: 修正后的代码
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            提取的模式列表
        """
        patterns = []
        
        # 分割为行
        original_lines = original.split('\n')
        corrected_lines = corrected.split('\n')
        
        # 如果行数不同，使用difflib查找差异
        if len(original_lines) != len(corrected_lines):
            matcher = difflib.SequenceMatcher(None, original_lines, corrected_lines)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag in ['replace', 'insert', 'delete']:
                    # 找到差异
                    orig_block = '\n'.join(original_lines[i1:i2])
                    corr_block = '\n'.join(corrected_lines[j1:j2])
                    
                    if orig_block and corr_block:
                        # 尝试提取模式
                        pattern = self._create_pattern(orig_block, corr_block)
                        if pattern:
                            patterns.append(pattern)
        else:
            # 行数相同，逐行比较
            for i, (orig_line, corr_line) in enumerate(zip(original_lines, corrected_lines)):
                if orig_line != corr_line:
                    pattern = self._create_pattern(orig_line, corr_line)
                    if pattern:
                        patterns.append(pattern)
        
        return patterns
    
    def _create_pattern(self, original: str, corrected: str) -> Optional[Dict[str, Any]]:
        """
        从原始代码和修正后的代码创建模式
        
        Args:
            original: 原始代码
            corrected: 修正后的代码
            
        Returns:
            创建的模式，如果无法创建则返回None
        """
        # 如果代码太长或太短，不创建模式
        if len(original) < 3 or len(corrected) < 3 or len(original) > 200 or len(corrected) > 200:
            return None
            
        # 尝试创建正则表达式模式
        try:
            # 转义正则表达式特殊字符
            escaped_original = re.escape(original)
            
            # 将具体的标识符替换为通配符
            generalized = escaped_original
            for word in re.findall(r'\b\w+\b', original):
                if len(word) > 2 and word not in ['for', 'if', 'else', 'while', 'def', 'class', 'function', 'return', 'var', 'let', 'const']:
                    generalized = generalized.replace(re.escape(word), r'(\w+)')
            
            # 创建替换模板
            replacement_template = corrected
            for word in re.findall(r'\b\w+\b', original):
                if len(word) > 2 and word not in ['for', 'if', 'else', 'while', 'def', 'class', 'function', 'return', 'var', 'let', 'const']:
                    if word in replacement_template:
                        replacement_template = replacement_template.replace(word, r'\1')
            
            # 测试模式是否有效
            if generalized != escaped_original and replacement_template != corrected:
                return {
                    "pattern": generalized,
                    "replacement": replacement_template,
                    "description": "从用户修正中学习的模式",
                    "original_example": original,
                    "corrected_example": corrected
                }
        except Exception:
            pass
            
        return None


# 单例模式，确保全局只有一个错误恢复系统实例
_recovery_system_instance = None

def get_error_recovery_system() -> ErrorRecoverySystem:
    """获取错误恢复系统实例"""
    global _recovery_system_instance
    if _recovery_system_instance is None:
        _recovery_system_instance = ErrorRecoverySystem()
    return _recovery_system_instance