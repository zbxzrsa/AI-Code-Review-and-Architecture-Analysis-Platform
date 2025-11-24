"""
Tree-sitter集成模块
提供与Tree-sitter库的集成，用于快速语法解析
"""
import os
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import asyncio
import json

# 导入Tree-sitter库
try:
    from tree_sitter import Language as TSLanguage, Parser
except ImportError:
    raise ImportError("请安装tree-sitter库: pip install tree-sitter")

# 配置日志
logger = logging.getLogger(__name__)

# 语言映射
LANGUAGE_MAP = {
    "python": "python",
    "java": "java",
    "javascript": "javascript",
    "go": "go"
}

class TreeSitterManager:
    """Tree-sitter管理器，负责加载语言库和创建解析器"""
    
    def __init__(self, languages_dir: Optional[str] = None):
        """
        初始化Tree-sitter管理器
        
        Args:
            languages_dir: Tree-sitter语言库目录，默认为当前目录下的languages
        """
        self.languages_dir = languages_dir or os.path.join(os.path.dirname(__file__), "languages")
        self.languages = {}
        self.parsers = {}
        
        # 确保语言库目录存在
        os.makedirs(self.languages_dir, exist_ok=True)
    
    async def load_language(self, language_name: str) -> TSLanguage:
        """
        加载指定语言的Tree-sitter语言库
        
        Args:
            language_name: 语言名称
            
        Returns:
            Tree-sitter语言对象
        """
        if language_name in self.languages:
            return self.languages[language_name]
        
        # 检查语言库是否存在
        lib_path = os.path.join(self.languages_dir, f"{language_name}.so")
        if not os.path.exists(lib_path):
            # 如果语言库不存在，则尝试构建
            await self._build_language(language_name)
        
        # 加载语言库
        try:
            ts_language = TSLanguage.build_library(
                lib_path,
                [os.path.join(self.languages_dir, language_name)]
            )
            self.languages[language_name] = ts_language
            return ts_language
        except Exception as e:
            logger.error(f"加载语言库失败: {str(e)}")
            raise ValueError(f"加载语言库失败: {str(e)}")
    
    async def _build_language(self, language_name: str) -> None:
        """
        构建语言库
        
        Args:
            language_name: 语言名称
        """
        import subprocess
        
        # 克隆语言仓库
        repo_url = f"https://github.com/tree-sitter/tree-sitter-{language_name}.git"
        repo_path = os.path.join(self.languages_dir, language_name)
        
        if not os.path.exists(repo_path):
            logger.info(f"克隆语言仓库: {repo_url}")
            try:
                # 使用异步子进程执行git命令
                proc = await asyncio.create_subprocess_exec(
                    "git", "clone", repo_url, repo_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                
                if proc.returncode != 0:
                    logger.error(f"克隆仓库失败: {stderr.decode()}")
                    raise ValueError(f"克隆仓库失败: {stderr.decode()}")
            except Exception as e:
                logger.error(f"克隆仓库时出错: {str(e)}")
                raise ValueError(f"克隆仓库时出错: {str(e)}")
    
    def get_parser(self, language_name: str) -> Parser:
        """
        获取指定语言的解析器
        
        Args:
            language_name: 语言名称
            
        Returns:
            Tree-sitter解析器
        """
        if language_name in self.parsers:
            return self.parsers[language_name]
        
        # 创建新的解析器
        parser = Parser()
        ts_language_name = LANGUAGE_MAP.get(language_name, language_name)
        
        # 设置解析器语言
        try:
            ts_language = self.languages.get(ts_language_name)
            if not ts_language:
                # 同步调用异步方法
                loop = asyncio.get_event_loop()
                ts_language = loop.run_until_complete(self.load_language(ts_language_name))
            
            parser.set_language(ts_language)
            self.parsers[language_name] = parser
            return parser
        except Exception as e:
            logger.error(f"设置解析器语言失败: {str(e)}")
            raise ValueError(f"设置解析器语言失败: {str(e)}")

class TreeSitterParser:
    """Tree-sitter解析器，用于解析代码并生成AST"""
    
    def __init__(self, manager: TreeSitterManager = None):
        """
        初始化Tree-sitter解析器
        
        Args:
            manager: Tree-sitter管理器，如果为None则创建新的管理器
        """
        self.manager = manager or TreeSitterManager()
    
    async def parse_code(self, code: str, language: str) -> Dict[str, Any]:
        """
        解析代码并生成AST
        
        Args:
            code: 源代码字符串
            language: 编程语言
            
        Returns:
            AST字典
        """
        try:
            # 获取解析器
            parser = self.manager.get_parser(language)
            
            # 解析代码
            tree = parser.parse(bytes(code, "utf8"))
            
            # 转换为字典
            return self._tree_to_dict(tree.root_node)
        except Exception as e:
            logger.error(f"解析代码失败: {str(e)}")
            raise ValueError(f"解析代码失败: {str(e)}")
    
    def _tree_to_dict(self, node) -> Dict[str, Any]:
        """
        将Tree-sitter节点转换为字典
        
        Args:
            node: Tree-sitter节点
            
        Returns:
            节点字典
        """
        result = {
            "type": node.type,
            "start_point": {
                "row": node.start_point[0],
                "column": node.start_point[1]
            },
            "end_point": {
                "row": node.end_point[0],
                "column": node.end_point[1]
            },
            "start_byte": node.start_byte,
            "end_byte": node.end_byte
        }
        
        # 添加子节点
        if len(node.children) > 0:
            result["children"] = [self._tree_to_dict(child) for child in node.children]
        
        return result

# 创建全局Tree-sitter解析器实例
ts_parser = TreeSitterParser()