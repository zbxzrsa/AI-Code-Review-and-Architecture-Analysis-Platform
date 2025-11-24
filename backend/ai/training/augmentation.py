"""
数据增强：
- 语法保持的代码变换（变量重命名、函数重排、注释插入）
- 多语言样本生成（跨语言简单等价片段）
"""
from typing import Dict, Any, List
import random
import re


def rename_identifiers(code: str, language: str) -> str:
    """粗粒度标识符重命名（保留语法结构，简化实现）。"""
    # 仅示例：替换常见短变量名
    mapping = {"i": "idx", "j": "jdx", "tmp": "temp"}
    def repl(m):
        name = m.group(0)
        return mapping.get(name, name)
    return re.sub(r"\b(i|j|tmp)\b", repl, code)


def reorder_functions(code: str, language: str) -> str:
    """将独立函数块重排（简单基于正则）。"""
    blocks = re.findall(r"(def\s+\w+\(.*?\):[\s\S]*?(?=\n\w|$))", code)
    if len(blocks) <= 1:
        return code
    random.shuffle(blocks)
    return "\n\n".join(blocks)


def insert_comments(code: str, language: str) -> str:
    """插入无害注释。"""
    lines = code.splitlines()
    for idx in range(0, len(lines), max(1, len(lines)//5)):
        lines.insert(idx, f"# auto-generated comment")
    return "\n".join(lines)


def augment_code(code: str, language: str, strategies: List[str]) -> List[str]:
    out = [code]
    for s in strategies:
        if s == "rename":
            out.append(rename_identifiers(code, language))
        elif s == "reorder":
            out.append(reorder_functions(code, language))
        elif s == "comment":
            out.append(insert_comments(code, language))
    return out