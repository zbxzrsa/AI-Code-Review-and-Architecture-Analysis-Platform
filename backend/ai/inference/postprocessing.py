"""
后处理优化：规则引擎、上下文过滤、置信度校准
此处提供简化实现，便于替换为更复杂的业务规则。
"""
from typing import List, Dict, Any


class RuleEngine:
    """简单规则引擎，用于修正或过滤缺陷结果。"""

    def apply(self, defects: List[Dict[str, Any]], language: str, code: str = "") -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d in defects:
            t = d.get("defect_type")
            conf = float(d.get("confidence", 0.0))
            # 示例规则：低置信度且非关键缺陷类型的结果过滤
            if conf < 0.2 and t in {"style_issue", "minor_smell"}:
                continue
            # 示例规则：空指针仅在存在可能为None的对象访问时提升置信度
            if t == "null_pointer_exception" and language == "python":
                if _heuristic_maybe_none_access(code):
                    d["confidence"] = min(1.0, conf + 0.05)
            out.append(d)
        return out


def _heuristic_maybe_none_access(code: str) -> bool:
    # 非严格：根据常见模式判断可能的None访问
    needles = ["if x is None", "if obj is None", "return None", "NoneType", ".foo("]
    lc = code.lower()
    return any(n.lower() in lc for n in needles)


class ContextFilter:
    """上下文感知过滤，用于去除显著不合逻辑的结果。"""

    def filter(self, defects: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d in defects:
            t = d.get("defect_type")
            # 示例：Go语言不应该出现Python特定缺陷类型
            if language == "go" and t in {"python_specific_issue"}:
                continue
            out.append(d)
        return out


class ConfidenceCalibrator:
    """置信度校准（简化版）：映射到更稳健的区间。"""

    def calibrate(self, defects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for d in defects:
            c = float(d.get("confidence", 0.0))
            # 简化的Platt缩放替代：拉近到[0.1, 0.9]区间以减少过度自信
            d["confidence"] = 0.1 + 0.8 * c
        return defects