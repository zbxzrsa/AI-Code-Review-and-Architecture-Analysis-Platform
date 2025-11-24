"""
转换质量验证和测试系统

提供代码转换质量的验证机制，包括：
- 语法正确性验证
- 语义等价性验证
- 测试用例生成和执行
- 边界条件覆盖
"""
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import tempfile
import os
import re
import ast
import json
import importlib
import sys
import difflib
from enum import Enum


class SyntaxValidationResult(Enum):
    """语法验证结果枚举"""
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class ConversionValidator:
    """代码转换验证器，用于验证转换后代码的质量"""
    
    def __init__(self):
        """初始化验证器"""
        self.language_parsers = {
            "python": self._validate_python_syntax,
            "javascript": self._validate_javascript_syntax,
            "typescript": self._validate_typescript_syntax,
            "java": self._validate_java_syntax,
            "csharp": self._validate_csharp_syntax,
            "cpp": self._validate_cpp_syntax,
            "rust": self._validate_rust_syntax,
        }
        
    def validate_syntax(self, source_code: str, target_lang: str) -> Dict[str, Any]:
        """
        验证目标代码语法正确性
        
        Args:
            source_code: 要验证的代码
            target_lang: 目标语言
            
        Returns:
            结构化验证结果：{"valid": bool, "status": str, "errors": List[str]}
        """
        if target_lang.lower() not in self.language_parsers:
            return {
                "valid": False,
                "status": SyntaxValidationResult.UNKNOWN.value,
                "errors": [f"不支持的语言: {target_lang}"]
            }
        
        status, error = self.language_parsers[target_lang.lower()](source_code)
        return {
            "valid": status == SyntaxValidationResult.VALID,
            "status": status.value,
            "errors": [] if error is None else [error]
        }
    
    def _validate_python_syntax(self, code: str) -> Tuple[SyntaxValidationResult, Optional[str]]:
        """验证Python代码语法"""
        try:
            ast.parse(code)
            return SyntaxValidationResult.VALID, None
        except SyntaxError as e:
            return SyntaxValidationResult.INVALID, f"语法错误: {str(e)}"
        except Exception as e:
            return SyntaxValidationResult.UNKNOWN, f"未知错误: {str(e)}"
    
    def _validate_javascript_syntax(self, code: str) -> Tuple[SyntaxValidationResult, Optional[str]]:
        """验证JavaScript代码语法"""
        with tempfile.NamedTemporaryFile(suffix='.js', delete=False) as temp:
            temp_name = temp.name
            temp.write(code.encode('utf-8'))
        
        try:
            # 使用Node.js检查语法
            result = subprocess.run(
                ['node', '--check', temp_name],
                capture_output=True,
                text=True
            )
            os.unlink(temp_name)
            
            if result.returncode == 0:
                return SyntaxValidationResult.VALID, None
            else:
                return SyntaxValidationResult.INVALID, result.stderr.strip()
        except Exception as e:
            os.unlink(temp_name)
            return SyntaxValidationResult.UNKNOWN, f"验证过程出错: {str(e)}"
    
    def _validate_typescript_syntax(self, code: str) -> Tuple[SyntaxValidationResult, Optional[str]]:
        """验证TypeScript代码语法"""
        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as temp:
            temp_name = temp.name
            temp.write(code.encode('utf-8'))
        
        try:
            # 使用tsc检查语法
            result = subprocess.run(
                ['tsc', '--noEmit', temp_name],
                capture_output=True,
                text=True
            )
            os.unlink(temp_name)
            
            if result.returncode == 0:
                return SyntaxValidationResult.VALID, None
            else:
                return SyntaxValidationResult.INVALID, result.stderr.strip()
        except Exception as e:
            os.unlink(temp_name)
            return SyntaxValidationResult.UNKNOWN, f"验证过程出错: {str(e)}"
    
    def _validate_java_syntax(self, code: str) -> Tuple[SyntaxValidationResult, Optional[str]]:
        """验证Java代码语法"""
        # 提取类名
        class_match = re.search(r'class\s+(\w+)', code)
        if not class_match:
            return SyntaxValidationResult.INVALID, "无法找到Java类定义"
        
        class_name = class_match.group(1)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, f"{class_name}.java")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            try:
                # 使用javac编译检查语法
                result = subprocess.run(
                    ['javac', file_path],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    return SyntaxValidationResult.VALID, None
                else:
                    return SyntaxValidationResult.INVALID, result.stderr.strip()
            except Exception as e:
                return SyntaxValidationResult.UNKNOWN, f"验证过程出错: {str(e)}"
    
    def _validate_csharp_syntax(self, code: str) -> Tuple[SyntaxValidationResult, Optional[str]]:
        """验证C#代码语法"""
        with tempfile.NamedTemporaryFile(suffix='.cs', delete=False) as temp:
            temp_name = temp.name
            temp.write(code.encode('utf-8'))
        
        try:
            # 使用csc编译器检查语法
            result = subprocess.run(
                ['csc', '/nologo', '/t:library', temp_name],
                capture_output=True,
                text=True
            )
            os.unlink(temp_name)
            
            if result.returncode == 0:
                return SyntaxValidationResult.VALID, None
            else:
                return SyntaxValidationResult.INVALID, result.stderr.strip()
        except Exception as e:
            os.unlink(temp_name)
            return SyntaxValidationResult.UNKNOWN, f"验证过程出错: {str(e)}"
    
    def _validate_cpp_syntax(self, code: str) -> Tuple[SyntaxValidationResult, Optional[str]]:
        """验证C++代码语法"""
        with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False) as temp:
            temp_name = temp.name
            temp.write(code.encode('utf-8'))
        
        try:
            # 使用g++编译器检查语法
            result = subprocess.run(
                ['g++', '-fsyntax-only', temp_name],
                capture_output=True,
                text=True
            )
            os.unlink(temp_name)
            
            if result.returncode == 0:
                return SyntaxValidationResult.VALID, None
            else:
                return SyntaxValidationResult.INVALID, result.stderr.strip()
        except Exception as e:
            os.unlink(temp_name)
            return SyntaxValidationResult.UNKNOWN, f"验证过程出错: {str(e)}"
    
    def _validate_rust_syntax(self, code: str) -> Tuple[SyntaxValidationResult, Optional[str]]:
        """验证Rust代码语法"""
        with tempfile.NamedTemporaryFile(suffix='.rs', delete=False) as temp:
            temp_name = temp.name
            temp.write(code.encode('utf-8'))
        
        try:
            # 使用rustc编译器检查语法
            result = subprocess.run(
                ['rustc', '--emit=mir', temp_name],
                capture_output=True,
                text=True
            )
            os.unlink(temp_name)
            
            if result.returncode == 0:
                return SyntaxValidationResult.VALID, None
            else:
                return SyntaxValidationResult.INVALID, result.stderr.strip()
        except Exception as e:
            os.unlink(temp_name)
            return SyntaxValidationResult.UNKNOWN, f"验证过程出错: {str(e)}"
    
    def validate_semantics(self, source_code: str, converted_code: str, 
                          source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        验证语义等价性
        
        Args:
            source_code: 源代码
            converted_code: 转换后的代码
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            验证结果详情
        """
        # 生成测试用例
        test_cases = self.generate_test_cases(source_code, source_lang)
        
        # 运行对比测试
        results = {
            "equivalent": True,
            "test_cases": len(test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for test_case in test_cases:
            try:
                source_result = self._execute_test(source_code, test_case, source_lang)
                target_result = self._execute_test(converted_code, test_case, target_lang)
                
                is_equivalent = self._compare_results(source_result, target_result)
                
                if is_equivalent:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["equivalent"] = False
                
                results["details"].append({
                    "test_case": test_case,
                    "source_result": source_result,
                    "target_result": target_result,
                    "equivalent": is_equivalent
                })
            except Exception as e:
                results["failed"] += 1
                results["equivalent"] = False
                results["details"].append({
                    "test_case": test_case,
                    "error": str(e)
                })
        
        return results
    
    def generate_test_cases(self, source_code: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        基于源代码生成测试用例
        
        Args:
            source_code: 源代码
            language: 源代码语言（可选，未提供时自动推断）
            
        Returns:
            测试用例列表
        """
        # 自动语言推断（简化）
        if not language:
            lowered = source_code.lower()
            if re.search(r"\bdef\s+\w+\s*\(", lowered):
                language = "python"
            elif re.search(r"\bfunction\s+\w+\s*\(", lowered):
                language = "javascript"
            else:
                language = "python"

        # 简单实现：提取函数参数并生成基本测试用例
        test_cases = []
        
        if language.lower() == "python":
            try:
                tree = ast.parse(source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # 提取函数参数
                        args = [arg.arg for arg in node.args.args if arg.arg != 'self']
                        
                        # 生成标准化测试用例格式：{"inputs": [...], "expected_output": ...}
                        arg_count = len(args)
                        if arg_count > 0:
                            # 三组代表性输入
                            test_cases.append({
                                "inputs": [0] * arg_count,
                                "expected_output": None
                            })
                            test_cases.append({
                                "inputs": ["test"] * arg_count,
                                "expected_output": None
                            })
                            test_cases.append({
                                "inputs": [[1, 2, 3]] * arg_count,
                                "expected_output": None
                            })
            except Exception:
                # 解析失败时使用默认测试用例
                pass
        
        # 如果没有生成测试用例，提供默认测试用例
        if not test_cases:
            test_cases = [
                {"inputs": [0, "test", []], "expected_output": None},
                {"inputs": [[1, 2, 3], {"key": "value"}], "expected_output": None}
            ]
        
        return test_cases
    
    def _execute_test(self, code: str, test_case: Dict[str, Any], language: str) -> Any:
        """
        执行测试用例
        
        Args:
            code: 要测试的代码
            test_case: 测试用例
            language: 代码语言
            
        Returns:
            执行结果
        """
        # 这里是一个简化的实现，实际应用中需要更复杂的执行环境
        # 模拟执行结果
        return {
            "output": f"模拟执行 {language} 代码的结果",
            "test_case": test_case
        }
    
    def _compare_results(self, source_result: Any, target_result: Any) -> bool:
        """
        比较源代码和目标代码的执行结果
        
        Args:
            source_result: 源代码执行结果
            target_result: 目标代码执行结果
            
        Returns:
            是否等价
        """
        # 简化实现，实际应用中需要更复杂的比较逻辑
        return source_result == target_result
    
    def run_comparison_tests(self, source_func: str, target_func: str, 
                            source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        运行对比测试验证行为一致性
        
        Args:
            source_func: 源函数代码
            target_func: 目标函数代码
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            测试结果
        """
        # 生成测试用例
        test_cases = self.generate_test_cases(source_func, source_lang)
        
        # 运行测试
        results = {
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for i, test_case in enumerate(test_cases):
            try:
                source_result = self._execute_test(source_func, test_case, source_lang)
                target_result = self._execute_test(target_func, test_case, target_lang)
                
                is_equivalent = self._compare_results(source_result, target_result)
                
                if is_equivalent:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                results["details"].append({
                    "test_id": i + 1,
                    "test_case": test_case,
                    "source_result": source_result,
                    "target_result": target_result,
                    "passed": is_equivalent
                })
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "test_id": i + 1,
                    "test_case": test_case,
                    "error": str(e),
                    "passed": False
                })
        
        return results


# 单例模式，确保全局只有一个验证器实例
_validator_instance = None

def get_validator() -> ConversionValidator:
    """获取验证器实例"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ConversionValidator()
    return _validator_instance