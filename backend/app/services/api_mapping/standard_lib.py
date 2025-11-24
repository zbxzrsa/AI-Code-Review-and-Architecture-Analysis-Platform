"""
跨语言标准库 API 映射系统

核心能力：
- API 功能等价性分析（输入/输出、错误处理、性能特征）
- API 映射规则库（函数签名转换、参数类型映射、返回值处理）
- 替代方案推荐（第三方库、兼容层、自定义实现与功能降级）
- 迁移指南生成

支持示例：
- Python → JavaScript：os.path.join、json.loads、datetime.now 等
- Java → C#：集合框架（List、Map、Set）与方法（add/put）等
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import re


@dataclass
class ApiRule:
    """单条 API 映射规则"""
    src_lang: str
    dst_lang: str
    src_pattern: str  # 用于识别源代码调用的正则/提示
    replace: Any      # 可为模板字符串或回调函数(code)->str
    note: str = ""


class ApiMappingDB:
    """API 映射规则库：维护跨语言等价 API 映射"""

    def __init__(self) -> None:
        self.rules: List[ApiRule] = []
        self._bootstrap_rules()

    def _bootstrap_rules(self) -> None:
        """初始化内置规则，用于常见标准库映射"""
        # Python → JavaScript
        self.rules += [
            ApiRule(
                src_lang="python",
                dst_lang="javascript",
                src_pattern=r"\bos\.path\.join\(([^)]+)\)",
                replace=lambda m: f"path.join({m.group(1)})",
                note="os.path.join → path.join"
            ),
            ApiRule(
                src_lang="python",
                dst_lang="javascript",
                src_pattern=r"\bjson\.loads\(([^)]+)\)",
                replace=lambda m: f"JSON.parse({m.group(1)})",
                note="json.loads → JSON.parse"
            ),
            ApiRule(
                src_lang="python",
                dst_lang="javascript",
                src_pattern=r"\bdatetime\.datetime\.now\(\)",
                replace=lambda m: "new Date()",
                note="datetime.now → new Date()"
            ),
        ]

        # Java → C# 集合框架
        self.rules += [
            ApiRule(
                src_lang="java",
                dst_lang="csharp",
                src_pattern=r"\bList<([A-Za-z0-9_]+)>\s+(\w+)\s*=\s*new\s+ArrayList<>\(\)\s*;",
                replace=lambda m: (lambda t, v: f"var {v} = new List<{t}>();")({
                    "String": "string",
                    "Integer": "int",
                    "Long": "long",
                    "Double": "double",
                    "Float": "float",
                    "Boolean": "bool",
                }.get(m.group(1), m.group(1)), m.group(2)),
                note="Java List → C# List"
            ),
            ApiRule(
                src_lang="java",
                dst_lang="csharp",
                src_pattern=r"\bMap<([A-Za-z0-9_]+),\s*([A-Za-z0-9_]+)>\s+(\w+)\s*=\s*new\s+HashMap<>\(\)\s*;",
                replace=lambda m: (lambda kt, vt, v: f"var {v} = new Dictionary<{kt}, {vt}>();")({
                    "String": "string",
                    "Integer": "int",
                    "Long": "long",
                    "Double": "double",
                    "Float": "float",
                    "Boolean": "bool",
                }.get(m.group(1), m.group(1)), {
                    "String": "string",
                    "Integer": "int",
                    "Long": "long",
                    "Double": "double",
                    "Float": "float",
                    "Boolean": "bool",
                }.get(m.group(2), m.group(2)), m.group(3)),
                note="Java Map → C# Dictionary"
            ),
            ApiRule(
                src_lang="java",
                dst_lang="csharp",
                src_pattern=r"\b(\w+)\.add\(([^)]+)\)\s*;",
                replace=lambda m: f"{m.group(1)}.Add({m.group(2)});",
                note="list.add → List.Add"
            ),
            ApiRule(
                src_lang="java",
                dst_lang="csharp",
                src_pattern=r"\b(\w+)\.put\(([^,]+),\s*([^)]+)\)\s*;",
                replace=lambda m: f"{m.group(1)}[{m.group(2)}] = {m.group(3)};",
                note="map.put → dict[key] = value"
            ),
        ]

    def find_rules(self, src_lang: str, dst_lang: str) -> List[ApiRule]:
        return [r for r in self.rules if r.src_lang == src_lang and r.dst_lang == dst_lang]


class ApiMappingEngine:
    """API 映射引擎：组合规则库并提供分析、替换与指南生成"""

    def __init__(self) -> None:
        self.db = ApiMappingDB()

    def analyze_equivalence(self, src_lang: str, dst_lang: str, code: str) -> Dict[str, Any]:
        """功能等价分析：返回输入/输出、错误处理与性能特征提示"""
        hints: List[Dict[str, str]] = []
        for rule in self.db.find_rules(src_lang, dst_lang):
            if re.search(rule.src_pattern, code):
                hints.append({
                    "rule": rule.note,
                    "io_behavior": "保持行为一致",
                    "error_handling": "遵循目标语言标准错误模式",
                    "performance": "目标实现等效或更快"
                })
        return {"matches": hints}

    def recommend_alternatives(self, src_lang: str, dst_lang: str, code: str) -> List[str]:
        """替代方案推荐：当无直接规则命中时给出替代建议"""
        suggestions: List[str] = []
        if src_lang == "python" and dst_lang in ("javascript", "typescript"):
            if "pathlib" in code:
                suggestions.append("使用 Node.js 内置 'path' 处理路径")
            if "ujson" in code:
                suggestions.append("使用 JSON.parse 或引入 'fast-json-stringify' 进行优化")
        if src_lang == "java" and dst_lang == "csharp":
            suggestions.append("Java Stream → C# LINQ；Optional → Nullable/Option 模式")
        return suggestions

    def convert(self, code: str, src_lang: str, dst_lang: str) -> Tuple[str, Dict[str, Any]]:
        """执行映射转换并返回转换后的代码与详情"""
        converted = code
        details: Dict[str, Any] = {"applied_rules": []}

        # 可能需要目标语言的导入提示
        header_inserts: List[str] = []
        if src_lang == "python" and dst_lang == "javascript":
            if re.search(r"\bos\.path\.join\(", code):
                header_inserts.append("const path = require('path');")
            if re.search(r"\bjson\.loads\(", code):
                header_inserts.append("// JSON.parse 为全局函数，无需额外导入")
            if re.search(r"\bopen\(", code):
                header_inserts.append("const fs = require('fs');")

        for rule in self.db.find_rules(src_lang, dst_lang):
            regex = re.compile(rule.src_pattern)
            def _do_replace(m: re.Match) -> str:
                return rule.replace(m) if callable(rule.replace) else str(rule.replace)
            new_converted, n = regex.subn(_do_replace, converted)
            if n > 0:
                converted = new_converted
                details["applied_rules"].append(rule.note)

        # 统一：Python→JS 的 datetime 导入移除提示
        if src_lang == "python" and dst_lang == "javascript":
            if re.search(r"\bdatetime\.datetime\.now\(\)", code):
                details.setdefault("notes", []).append("将 datetime.now 替换为 new Date()，无需额外库")

        # 插入目标语言依赖声明（简单拼接在顶部）
        if header_inserts:
            header_block = "\n".join(header_inserts) + "\n\n"
            converted = header_block + converted

        # 生成替代建议与分析
        details["equivalence"] = self.analyze_equivalence(src_lang, dst_lang, code)
        alt = self.recommend_alternatives(src_lang, dst_lang, code)
        if alt:
            details["alternatives"] = alt
        return converted, details


def convert_standard_api(source_code: str, source_language: str, target_language: str) -> str:
    """面向 API 的单入口转换函数，返回转换代码字符串"""
    engine = ApiMappingEngine()
    converted, _ = engine.convert(source_code, source_language.lower(), target_language.lower())
    return converted


def generate_api_migration_guide(source_code: str, source_language: str, target_language: str) -> str:
    """生成迁移指南：说明替换规则、依赖与注意事项"""
    engine = ApiMappingEngine()
    _, details = engine.convert(source_code, source_language.lower(), target_language.lower())
    lines: List[str] = [
        f"从 {source_language} 迁移到 {target_language} 的标准库API指南",
        "",
        "已应用规则："
    ]
    for r in details.get("applied_rules", []):
        lines.append(f"- {r}")
    if "alternatives" in details:
        lines.append("")
        lines.append("替代方案建议：")
        for s in details["alternatives"]:
            lines.append(f"- {s}")
    if "equivalence" in details:
        lines.append("")
        lines.append("等价性分析匹配：")
        for h in details["equivalence"].get("matches", []):
            lines.append(f"- {h['rule']} | I/O: {h['io_behavior']} | 错误: {h['error_handling']} | 性能: {h['performance']}")
    return "\n".join(lines)