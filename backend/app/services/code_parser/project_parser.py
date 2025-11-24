"""
项目级代码解析器
实现大规模项目的增量解析、缓存、并行与断点续解析能力，并对内存进行约束。
"""
import os
import time
import json
import sqlite3
import hashlib
import asyncio
import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .parser import parser_service, Language, FeatureType


# 支持的文件扩展名到语言映射
EXT_LANGUAGE_MAP = {
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".js": Language.JAVASCRIPT,
    ".ts": Language.JAVASCRIPT,
    ".go": Language.GO,
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件SHA256，避免一次性读取大文件至内存。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines_stream(path: Path, chunk_size: int = 1024 * 1024) -> int:
    """流式统计文件行数，用于粗略估计大小与性能指标。"""
    count = 0
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            count += data.count(b"\n")
    return count


@dataclass
class FileTask:
    file_path: Path
    language: Language
    features: List[FeatureType]
    file_hash: str
    line_count: int


class ParseCache:
    """基于SQLite的持久化解析缓存，支持断点续解析。"""

    def __init__(self, db_path: Path, results_dir: Path):
        self.db_path = db_path
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                language TEXT NOT NULL,
                features TEXT NOT NULL,
                result_path TEXT,
                parsed_at REAL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY,
                project_root TEXT,
                processed_count INTEGER,
                total_count INTEGER,
                updated_at REAL
            )
            """
        )
        conn.commit()
        conn.close()

    def is_up_to_date(self, file_path: Path, file_hash: str, features: List[FeatureType]) -> bool:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT file_hash, features FROM files WHERE file_path=?", (str(file_path),))
        row = c.fetchone()
        conn.close()
        if not row:
            return False
        cached_hash, cached_features = row
        # 特征列表按名称排序后对比，避免顺序影响
        feat_str = ",".join(sorted([f.value for f in features]))
        return cached_hash == file_hash and cached_features == feat_str

    def write_result(
        self,
        file_path: Path,
        file_hash: str,
        language: Language,
        features: List[FeatureType],
        result_rel_path: Path,
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO files (file_path, file_hash, language, features, result_path, parsed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                file_hash=excluded.file_hash,
                language=excluded.language,
                features=excluded.features,
                result_path=excluded.result_path,
                parsed_at=excluded.parsed_at
            """,
            (
                str(file_path),
                file_hash,
                language.value,
                ",".join(sorted([f.value for f in features])),
                str(result_rel_path),
                time.time(),
            ),
        )
        conn.commit()
        conn.close()

    def update_checkpoint(self, project_root: Path, processed_count: int, total_count: int) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO checkpoints (id, project_root, processed_count, total_count, updated_at)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                project_root=excluded.project_root,
                processed_count=excluded.processed_count,
                total_count=excluded.total_count,
                updated_at=excluded.updated_at
            """,
            (str(project_root), processed_count, total_count, time.time()),
        )
        conn.commit()
        conn.close()


class ProjectParser:
    """
    项目解析器：
    - 增量：按文件hash与特征列表跳过未变更文件
    - 缓存：结果JSON持久化 + SQLite元数据
    - 并行：基于异步+信号量并发
    - 流式：哈希与行数统计流式处理，大文件解析受并发控制
    - 断点续解析：缓存+检查点记录，重跑自动跳过已完成文件
    """

    def __init__(
        self,
        project_root: Path,
        cache_dir: Optional[Path] = None,
        max_concurrency: int = 8,
        memory_limit_bytes: int = 2 * 1024 * 1024 * 1024,
    ):
        self.project_root = project_root
        self.cache_dir = cache_dir or (project_root / ".parse_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.cache_dir / "results"
        self.db_path = self.cache_dir / "parse_cache.sqlite"
        self.cache = ParseCache(self.db_path, self.results_dir)
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.memory_limit_bytes = memory_limit_bytes

    def _discover_files(self) -> List[Path]:
        files = []
        for ext in EXT_LANGUAGE_MAP.keys():
            files.extend(self.project_root.rglob(f"*{ext}"))
        return files

    def _language_for(self, file: Path) -> Optional[Language]:
        return EXT_LANGUAGE_MAP.get(file.suffix.lower())

    def _estimate_safe_concurrency(self, file_sizes: List[int], base_concurrency: int) -> int:
        # 保守估计每个并发解析占用约 ~64MB（源码+AST结构体），大文件更高；
        # 根据总内存限制动态下调并发，以避免超过2GB。
        est_per_task = 64 * 1024 * 1024
        max_tasks_by_mem = max(1, int(self.memory_limit_bytes / est_per_task))
        return max(1, min(base_concurrency, max_tasks_by_mem))

    async def _parse_one(self, task: FileTask) -> Tuple[Path, Dict[str, Any]]:
        async with self.semaphore:
            # 以只读内存映射方式读取，尽量避免复制
            try:
                with task.file_path.open("rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    code = mm.read().decode("utf-8", errors="ignore")
                    mm.close()
            except Exception:
                # 退回普通读取
                code = task.file_path.read_text(encoding="utf-8", errors="ignore")

            result = await parser_service.parse_code(code, task.language, task.features)
            return task.file_path, result

    async def parse_project(self, features: List[FeatureType], resume: bool = True) -> Dict[str, Any]:
        start = time.time()
        files = self._discover_files()
        total_files = len(files)

        # 准备任务（增量跳过）
        tasks: List[FileTask] = []
        to_process_sizes: List[int] = []

        for f in files:
            lang = self._language_for(f)
            if not lang:
                continue
            file_hash = sha256_file(f)
            line_count = count_lines_stream(f)
            up_to_date = self.cache.is_up_to_date(f, file_hash, features)
            if resume and up_to_date:
                continue
            tasks.append(FileTask(file_path=f, language=lang, features=features, file_hash=file_hash, line_count=line_count))
            try:
                to_process_sizes.append(f.stat().st_size)
            except Exception:
                pass

        # 动态并发调节（尽量不超过内存限制）
        base_concurrency = self.semaphore._value if hasattr(self.semaphore, "_value") else 8
        safe_concurrency = self._estimate_safe_concurrency(to_process_sizes, base_concurrency)
        # 更新信号量值
        self.semaphore = asyncio.Semaphore(safe_concurrency)

        processed = 0
        skipped = total_files - len(tasks)
        errors: List[Tuple[Path, str]] = []

        async def _worker(task: FileTask):
            nonlocal processed
            try:
                file_path, result = await self._parse_one(task)
                # 写入结果到JSON
                result_name = f"{task.file_hash}.json"
                result_path = self.results_dir / result_name
                with result_path.open("w", encoding="utf-8") as rf:
                    json.dump({
                        "file": str(file_path),
                        "language": task.language.value,
                        "features": [f.value for f in task.features],
                        "lines": task.line_count,
                        "result": result,
                    }, rf, ensure_ascii=False)

                # 更新缓存元数据
                self.cache.write_result(file_path, task.file_hash, task.language, task.features, result_path.relative_to(self.cache_dir))
                processed += 1
                self.cache.update_checkpoint(self.project_root, processed_count=processed, total_count=total_files)
            except Exception as e:
                errors.append((task.file_path, str(e)))

        # 并发执行
        await asyncio.gather(*[_worker(t) for t in tasks])

        elapsed = time.time() - start
        return {
            "project_root": str(self.project_root),
            "total_files": total_files,
            "processed": processed,
            "skipped": skipped,
            "errors": [(str(p), msg) for p, msg in errors],
            "elapsed_sec": elapsed,
            "concurrency": safe_concurrency,
        }