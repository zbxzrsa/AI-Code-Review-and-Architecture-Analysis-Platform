"""
模型集成与不确定性估计
提供多个专家模型的投票/加权融合、基于分歧的不确定性估计。
当前示例与AIModelService对接，可用于真实模型替换。
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class ExpertOutput:
    defects: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class EnsembleAggregator:
    """集合多个专家输出，进行投票与加权融合。"""

    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights

    def fuse_defects(self, experts: List[ExpertOutput]) -> Tuple[List[Dict[str, Any]], float]:
        """
        简化融合策略：
        - 按 (defect_type, location) 聚类
        - 置信度加权平均
        - 返回总体不确定性（专家间置信度方差的均值）
        """
        if not experts:
            return [], float("nan")

        # 收集所有候选
        buckets: Dict[Tuple[str, str], List[Tuple[float, Dict[str, Any]]]] = {}
        for idx, e in enumerate(experts):
            w = self.weights[idx] if self.weights and idx < len(self.weights) else 1.0
            for d in e.defects:
                key = (d.get("defect_type", "unknown"), _loc_key(d.get("location")))
                buckets.setdefault(key, []).append((w * float(d.get("confidence", 0.0)), d))

        fused: List[Dict[str, Any]] = []
        variances: List[float] = []
        for key, items in buckets.items():
            # 投票：超过半数专家出现则保留
            count = len(items)
            if count < max(2, math.ceil(len(experts) / 2)):
                continue
            # 加权平均置信度
            confs = [c for c, _ in items]
            avg_conf = sum(confs) / max(1, len(confs))
            var = _variance(confs)
            variances.append(var)
            # 使用第一个的结构，更新置信度
            d_sample = items[0][1].copy()
            d_sample["confidence"] = max(0.0, min(1.0, avg_conf))
            fused.append(d_sample)

        mean_var = sum(variances) / len(variances) if variances else 0.0
        return fused, mean_var


def _loc_key(loc: Dict[str, Any]) -> str:
    if not loc:
        return ""
    s = loc.get("start_line"), loc.get("end_line"), loc.get("start_column"), loc.get("end_column")
    return ":".join([str(x) for x in s])


def _variance(vals: List[float]) -> float:
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)