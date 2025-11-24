import os
import re
import json
from typing import List, Dict


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def list_files(base: str, exts: List[str]) -> List[str]:
    matches = []
    for root, _, files in os.walk(base):
        for f in files:
            if any(f.lower().endswith(e.lower()) for e in exts):
                matches.append(os.path.join(root, f))
    return matches


def grep_references(search_roots: List[str], needle: str) -> List[str]:
    hits = []
    pattern = re.compile(re.escape(needle), re.IGNORECASE)
    for root in search_roots:
        for dirpath, _, files in os.walk(root):
            for name in files:
                # Only scan reasonable text-based files
                if not any(name.endswith(ext) for ext in (
                    ".ts", ".tsx", ".js", ".jsx", ".md", ".json", ".html", ".css", ".yml", ".yaml", ".py"
                )):
                    continue
                file_path = os.path.join(dirpath, name)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                        if pattern.search(content):
                            hits.append(file_path)
                except Exception:
                    # Ignore unreadable files
                    pass
    return sorted(set(hits))


def relative(path: str) -> str:
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except Exception:
        return path


def main() -> None:
    frontend_public = os.path.join(PROJECT_ROOT, "frontend", "public")
    frontend_src = os.path.join(PROJECT_ROOT, "frontend", "src")
    assets_dir = os.path.join(PROJECT_ROOT, "assets")

    # Collect candidate assets
    images = list_files(frontend_public, [".png", ".jpg", ".jpeg", ".svg", ".ico"])
    css = list_files(os.path.join(frontend_src, "styles"), [".css"])
    fonts = list_files(frontend_public, [".woff", ".woff2", ".ttf", ".otf"])
    misc_assets = list_files(assets_dir, [".svg", ".png", ".ico"])

    search_roots = [
        frontend_src,
        frontend_public,
        os.path.join(PROJECT_ROOT, "backend"),
        os.path.join(PROJECT_ROOT, "docs"),
        PROJECT_ROOT,
    ]

    report: Dict[str, List[Dict[str, List[str]]]] = {
        "images": [],
        "css": [],
        "fonts": [],
        "misc": [],
    }

    for bucket_name, files in (
        ("images", images),
        ("css", css),
        ("fonts", fonts),
        ("misc", misc_assets),
    ):
        for f in files:
            name = os.path.basename(f)
            refs = grep_references(search_roots, name)
            report[bucket_name].append({
                "file": relative(f),
                "referenced_in": [relative(r) for r in refs],
            })

    # Identify unused candidates
    unused: Dict[str, List[str]] = {"images": [], "css": [], "fonts": [], "misc": []}
    for bucket_name, entries in report.items():
        for entry in entries:
            if not entry["referenced_in"]:
                unused[bucket_name].append(entry["file"])

    output = {
        "root": PROJECT_ROOT,
        "unused_candidates": unused,
        "full_report": report,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()