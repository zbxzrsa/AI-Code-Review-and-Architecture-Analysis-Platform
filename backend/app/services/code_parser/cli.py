"""
命令行入口：项目代码解析
示例：
  python -m app.services.code_parser.cli --root C:\repo --features ast cfg dfg metrics --resume --concurrency 8
"""
import argparse
import asyncio
from pathlib import Path
from typing import List

from .parser import FeatureType
from .project_parser import ProjectParser


def parse_args():
    p = argparse.ArgumentParser(description="Large project code parsing with caching & resume")
    p.add_argument("--root", required=True, help="项目根目录")
    p.add_argument("--features", nargs="*", default=["ast", "metrics"], help="需要提取的特征列表，默认 ast metrics")
    p.add_argument("--concurrency", type=int, default=8, help="并发解析任务数")
    p.add_argument("--resume", action="store_true", help="开启断点续解析，跳过未变更文件")
    p.add_argument("--memory_limit_mb", type=int, default=2048, help="内存限制（MB），用于并发调节")
    return p.parse_args()


def to_features(names: List[str]) -> List[FeatureType]:
    out = []
    for n in names:
        try:
            out.append(FeatureType(n))
        except Exception:
            # 忽略未知特征
            pass
    if not out:
        out = [FeatureType.AST, FeatureType.METRICS]
    return out


async def main():
    args = parse_args()
    root = Path(args.root).resolve()
    features = to_features(args.features)
    memory_limit_bytes = args.memory_limit_mb * 1024 * 1024

    parser = ProjectParser(
        project_root=root,
        max_concurrency=args.concurrency,
        memory_limit_bytes=memory_limit_bytes,
    )

    summary = await parser.parse_project(features=features, resume=args.resume)
    # 输出摘要
    print("=== Parse Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())