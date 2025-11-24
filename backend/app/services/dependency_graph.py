"""
依赖图解析与反向闭包计算

功能：
1. 解析 Python/JavaScript 代码中的 import 语句
2. 构建模块级依赖图
3. 计算反向依赖闭包（传播性影响）
4. 支持增量分析中的变更传播
"""

import re
import hashlib
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImportInfo:
    """导入信息"""
    source_file: str           # 导入所在文件
    imported_module: str       # 被导入模块名
    import_type: str          # 'from' or 'import'
    line_number: int
    is_relative: bool = False
    imported_items: List[str] = field(default_factory=list)  # from X import Y, Z


@dataclass
class DependencyNode:
    """依赖图节点"""
    file_path: str
    import_hash: str          # 导入集合的哈希值
    imports: Set[str]         # 直接导入的模块集合
    imported_by: Set[str] = field(default_factory=set)  # 被谁导入

    def __hash__(self):
        return hash(self.file_path)

    def __eq__(self, other):
        return self.file_path == other.file_path


class DependencyParser:
    """依赖解析器"""

    # Python import 正则表达式
    PYTHON_IMPORT_RE = re.compile(
        r'^\s*(?:from\s+([\w.]+)\s+)?import\s+([\w\s,*]+)',
        re.MULTILINE
    )

    # JavaScript import 正则表达式
    JS_IMPORT_RE = re.compile(
        r'^\s*(?:import\s+(?:{[\w\s,]*}|[\w]+|[\w]+\s+as\s+[\w]+)?\s+from\s+[\'"]([^\'"]+)[\'"]|'
        r'require\([\'"]([^\'"]+)[\'"]\))',
        re.MULTILINE
    )

    @staticmethod
    def parse_python_imports(file_path: str, content: str) -> List[ImportInfo]:
        """解析 Python 文件中的导入"""
        imports = []

        for match in DependencyParser.PYTHON_IMPORT_RE.finditer(content):
            from_module = match.group(1)
            imported_items = match.group(2).strip()

            # 处理 from X import Y, Z 形式
            if from_module:
                items = [item.strip() for item in imported_items.split(',')]
                imports.append(ImportInfo(
                    source_file=file_path,
                    imported_module=from_module,
                    import_type='from',
                    line_number=content[:match.start()].count('\n') + 1,
                    is_relative=from_module.startswith('.'),
                    imported_items=items
                ))
            else:
                # 处理 import X, Y, Z 形式
                items = [item.strip() for item in imported_items.split(',')]
                for item in items:
                    # 去掉 "as alias" 部分
                    module = item.split()[0] if ' ' in item else item
                    imports.append(ImportInfo(
                        source_file=file_path,
                        imported_module=module,
                        import_type='import',
                        line_number=content[:match.start()].count('\n') + 1,
                    ))

        return imports

    @staticmethod
    def parse_js_imports(file_path: str, content: str) -> List[ImportInfo]:
        """解析 JavaScript 文件中的导入"""
        imports = []

        for match in DependencyParser.JS_IMPORT_RE.finditer(content):
            # 获取导入路径（可能在 group(1) 或 group(2)）
            module_path = match.group(1) or match.group(2)
            if not module_path:
                continue

            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=module_path,
                import_type='import',
                line_number=content[:match.start()].count('\n') + 1,
                is_relative=module_path.startswith('.')
            ))

        return imports

    @staticmethod
    def normalize_module_path(module_name: str, source_file: str,
                             language: str = 'python') -> str:
        """规范化模块路径

        Args:
            module_name: 原始模块名
            source_file: 源文件路径
            language: 编程语言

        Returns:
            规范化后的文件路径
        """
        if language == 'python':
            # Python: 'a.b.c' -> 'a/b/c.py' 或 'a/b/c/__init__.py'
            path = module_name.replace('.', '/')
            return path
        else:  # javascript
            # JS: './utils' -> 'utils.js' 或 'utils/index.js'
            if module_name.startswith('.'):
                # 相对路径
                source_dir = '/'.join(source_file.split('/')[:-1])
                path = f"{source_dir}/{module_name}"
            else:
                path = module_name

            # 标准化 .js 后缀
            if not path.endswith('.js') and not path.endswith('/index.js'):
                path += '.js'

            return path


class DependencyGraph:
    """依赖图管理"""

    def __init__(self):
        self.nodes: Dict[str, DependencyNode] = {}  # file_path -> node
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # module -> importing files
        self.language_map: Dict[str, str] = {}  # file_path -> language

    def add_file(self, file_path: str, content: str, language: str = 'python'):
        """添加文件到依赖图"""
        self.language_map[file_path] = language

        # 解析导入
        if language == 'python':
            imports = DependencyParser.parse_python_imports(file_path, content)
        else:  # javascript
            imports = DependencyParser.parse_js_imports(file_path, content)

        # 规范化模块名
        imported_modules = set()
        for imp in imports:
            normalized = DependencyParser.normalize_module_path(
                imp.imported_module, file_path, language
            )
            imported_modules.add(normalized)
            self.reverse_graph[normalized].add(file_path)

        # 创建或更新节点
        import_hash = hashlib.sha256(
            '|'.join(sorted(imported_modules)).encode()
        ).hexdigest()[:16]

        self.nodes[file_path] = DependencyNode(
            file_path=file_path,
            import_hash=import_hash,
            imports=imported_modules
        )

    def get_reverse_closure(self, changed_files: Set[str],
                           depth: int = 10) -> Set[str]:
        """计算反向闭包（受影响的文件集）

        Args:
            changed_files: 变更文件集合
            depth: 最大遍历深度

        Returns:
            所有受影响的文件集合
        """
        affected = set(changed_files)
        queue = deque(changed_files)
        visited = set(changed_files)
        current_depth = 0

        while queue and current_depth < depth:
            next_level = set()
            level_size = len(queue)

            for _ in range(level_size):
                file_path = queue.popleft()

                # 找出导入该文件的所有文件
                # 考虑：file_path 中的模块可能被其他文件导入
                for module, importing_files in self.reverse_graph.items():
                    if file_path in module or module in file_path:
                        for importing_file in importing_files:
                            if importing_file not in visited:
                                affected.add(importing_file)
                                next_level.add(importing_file)
                                visited.add(importing_file)

            queue.extend(next_level)
            current_depth += 1

        return affected

    def get_impact_chain(self, changed_file: str, max_depth: int = 5) -> List[List[str]]:
        """获取影响链（分层显示）

        Args:
            changed_file: 变更文件
            max_depth: 最大显示深度

        Returns:
            按层级组织的影响文件列表
        """
        chains = [[changed_file]]
        visited = {changed_file}

        for depth in range(max_depth - 1):
            current_level = chains[-1]
            next_level = set()

            for file_path in current_level:
                for module, importing_files in self.reverse_graph.items():
                    if file_path in module or module in file_path:
                        for importing_file in importing_files:
                            if importing_file not in visited:
                                next_level.add(importing_file)
                                visited.add(importing_file)

            if next_level:
                chains.append(list(next_level))
            else:
                break

        return chains


class IncrementalAnalyzer:
    """增量分析器 - 基于 git diff 确定需要重新分析的文件"""

    def __init__(self, dependency_graph: DependencyGraph,
                 content_cache: Dict[str, str]):
        """
        Args:
            dependency_graph: 依赖图实例
            content_cache: {file_path: file_content}
        """
        self.graph = dependency_graph
        self.cache = content_cache

    def get_changed_files_from_diff(self, diff_output: str) -> Tuple[Set[str], Set[str]]:
        """从 git diff 输出解析变更文件

        Returns:
            (added_files, modified_files)
        """
        added = set()
        modified = set()

        for line in diff_output.split('\n'):
            if line.startswith('diff --git'):
                # 解析: diff --git a/path/file b/path/file
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[3][2:]  # 去掉 'b/' 前缀

                    # 检查后续行确定是新增还是修改
                    # new file 表示新增，否则为修改
                    pass
            elif line.startswith('new file'):
                # 提取最后一行的文件路径
                pass

        return added, modified

    def determine_analysis_scope(self, changed_files: Set[str]) -> Dict[str, str]:
        """确定分析范围与类型

        Args:
            changed_files: 变更文件集合

        Returns:
            {file_path: 'full'|'incremental'}
        """
        affected = self.graph.get_reverse_closure(changed_files)
        scope = {}

        for file_path in affected:
            if file_path in changed_files:
                scope[file_path] = 'full'  # 变更文件需要全量分析
            else:
                scope[file_path] = 'incremental'  # 受影响文件可增量分析

        return scope


def compute_dependency_hash(imports: Set[str]) -> str:
    """计算依赖集合的哈希值"""
    sorted_imports = sorted(list(imports))
    combined = '|'.join(sorted_imports)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
