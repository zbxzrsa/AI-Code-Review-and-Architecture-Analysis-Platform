import os
import json
import logging
import requests
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from tqdm import tqdm
import subprocess
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitHubDataCollector:
    """
    从GitHub收集代码缺陷数据
    """
    def __init__(self, token: str, output_dir: str):
        """
        初始化GitHub数据收集器
        
        Args:
            token: GitHub API令牌
            output_dir: 输出目录
        """
        self.token = token
        self.output_dir = output_dir
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = "https://api.github.com"
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
    def search_bug_fix_commits(self, query: str, max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        搜索修复bug的提交
        
        Args:
            query: 搜索查询，例如 "fix bug language:python"
            max_results: 最大结果数
            
        Returns:
            commits: 提交列表
        """
        logger.info(f"搜索修复bug的提交: {query}")
        
        # 构建搜索URL
        search_url = f"{self.base_url}/search/commits"
        params = {
            'q': query,
            'sort': 'committer-date',
            'order': 'desc',
            'per_page': 100
        }
        
        # 收集提交
        commits = []
        page = 1
        
        with tqdm(total=max_results) as pbar:
            while len(commits) < max_results:
                params['page'] = page
                response = requests.get(search_url, headers=self.headers, params=params)
                
                if response.status_code != 200:
                    logger.error(f"API请求失败: {response.status_code} - {response.text}")
                    break
                
                data = response.json()
                items = data.get('items', [])
                
                if not items:
                    break
                
                commits.extend(items)
                pbar.update(len(items))
                
                # 检查是否达到最大结果数
                if len(commits) >= max_results:
                    commits = commits[:max_results]
                    break
                
                # 检查是否有下一页
                if 'next' not in response.links:
                    break
                
                page += 1
                
                # 避免触发GitHub API速率限制
                if page % 10 == 0:
                    logger.info(f"已收集 {len(commits)} 个提交，暂停5秒...")
                    time.sleep(5)
        
        logger.info(f"共收集 {len(commits)} 个提交")
        return commits
    
    def clone_repository(self, repo_url: str, repo_dir: str) -> bool:
        """
        克隆代码库
        
        Args:
            repo_url: 代码库URL
            repo_dir: 代码库目录
            
        Returns:
            success: 是否成功
        """
        logger.info(f"克隆代码库: {repo_url}")
        
        try:
            # 如果目录已存在，先删除
            if os.path.exists(repo_dir):
                import shutil
                shutil.rmtree(repo_dir)
            
            # 克隆代码库
            subprocess.run(['git', 'clone', repo_url, repo_dir], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"克隆代码库失败: {e}")
            return False
    
    def get_commit_diff(self, repo_dir: str, commit_hash: str) -> Dict[str, Any]:
        """
        获取提交差异
        
        Args:
            repo_dir: 代码库目录
            commit_hash: 提交哈希
            
        Returns:
            diff_data: 差异数据
        """
        logger.info(f"获取提交差异: {commit_hash}")
        
        try:
            # 切换到提交
            subprocess.run(['git', '-C', repo_dir, 'checkout', commit_hash + '~1'], check=True)
            
            # 获取修改前的文件
            before_files = {}
            modified_files = subprocess.check_output(
                ['git', '-C', repo_dir, 'diff', '--name-only', commit_hash],
                text=True
            ).strip().split('\n')
            
            for file_path in modified_files:
                if file_path.endswith('.py') or file_path.endswith('.java') or file_path.endswith('.js'):
                    try:
                        content = subprocess.check_output(
                            ['git', '-C', repo_dir, 'show', f'{commit_hash}~1:{file_path}'],
                            text=True
                        )
                        before_files[file_path] = content
                    except subprocess.CalledProcessError:
                        # 文件可能是新增的
                        pass
            
            # 切换到提交后的版本
            subprocess.run(['git', '-C', repo_dir, 'checkout', commit_hash], check=True)
            
            # 获取修改后的文件
            after_files = {}
            for file_path in modified_files:
                if file_path.endswith('.py') or file_path.endswith('.java') or file_path.endswith('.js'):
                    try:
                        content = subprocess.check_output(
                            ['git', '-C', repo_dir, 'show', f'{commit_hash}:{file_path}'],
                            text=True
                        )
                        after_files[file_path] = content
                    except subprocess.CalledProcessError:
                        # 文件可能被删除
                        pass
            
            # 获取提交信息
            commit_message = subprocess.check_output(
                ['git', '-C', repo_dir, 'log', '-1', '--pretty=%B', commit_hash],
                text=True
            ).strip()
            
            return {
                'commit_hash': commit_hash,
                'commit_message': commit_message,
                'before_files': before_files,
                'after_files': after_files
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"获取提交差异失败: {e}")
            return {}
    
    def extract_defect_type(self, commit_message: str) -> List[str]:
        """
        从提交信息中提取缺陷类型
        
        Args:
            commit_message: 提交信息
            
        Returns:
            defect_types: 缺陷类型列表
        """
        # 定义缺陷类型关键词
        defect_patterns = {
            'security_vulnerability': [
                r'security', r'vulnerability', r'exploit', r'cve', r'xss', 
                r'injection', r'overflow', r'leak', r'auth', r'permission'
            ],
            'performance_issue': [
                r'performance', r'slow', r'speed', r'memory', r'cpu', 
                r'optimize', r'efficient', r'leak', r'timeout'
            ],
            'logic_error': [
                r'logic', r'incorrect', r'wrong', r'error', r'bug', 
                r'fix', r'issue', r'mistake', r'fault'
            ],
            'exception_handling': [
                r'exception', r'error', r'crash', r'handle', r'catch', 
                r'try', r'except', r'throw', r'null'
            ],
            'concurrency_issue': [
                r'concurrency', r'race', r'deadlock', r'thread', r'sync', 
                r'atomic', r'lock', r'mutex'
            ]
        }
        
        # 检查提交信息中的关键词
        defect_types = []
        for defect_type, patterns in defect_patterns.items():
            for pattern in patterns:
                if re.search(pattern, commit_message, re.IGNORECASE):
                    defect_types.append(defect_type)
                    break
        
        return list(set(defect_types))
    
    def extract_severity(self, commit_message: str) -> str:
        """
        从提交信息中提取严重程度
        
        Args:
            commit_message: 提交信息
            
        Returns:
            severity: 严重程度 (low, medium, high)
        """
        # 定义严重程度关键词
        severity_patterns = {
            'high': [
                r'critical', r'severe', r'major', r'high', r'serious', 
                r'important', r'urgent', r'emergency', r'crash', r'security'
            ],
            'medium': [
                r'medium', r'moderate', r'normal', r'average', r'common', 
                r'standard', r'regular', r'general'
            ],
            'low': [
                r'low', r'minor', r'trivial', r'small', r'simple', 
                r'easy', r'cosmetic', r'typo', r'documentation'
            ]
        }
        
        # 检查提交信息中的关键词
        for severity, patterns in severity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, commit_message, re.IGNORECASE):
                    return severity
        
        # 默认为中等严重程度
        return 'medium'
    
    def process_commits(self, commits: List[Dict[str, Any]], temp_dir: str) -> List[Dict[str, Any]]:
        """
        处理提交列表，提取缺陷数据
        
        Args:
            commits: 提交列表
            temp_dir: 临时目录
            
        Returns:
            defect_data: 缺陷数据列表
        """
        logger.info(f"处理 {len(commits)} 个提交")
        
        # 创建临时目录
        os.makedirs(temp_dir, exist_ok=True)
        
        defect_data = []
        
        for commit in tqdm(commits):
            # 提取提交信息
            repo_url = commit['repository']['html_url']
            repo_name = commit['repository']['full_name']
            commit_hash = commit['sha']
            commit_message = commit['commit']['message']
            
            # 创建代码库目录
            repo_dir = os.path.join(temp_dir, repo_name.replace('/', '_'))
            
            # 克隆代码库
            if not self.clone_repository(repo_url, repo_dir):
                continue
            
            # 获取提交差异
            diff_data = self.get_commit_diff(repo_dir, commit_hash)
            
            if not diff_data:
                continue
            
            # 提取缺陷类型和严重程度
            defect_types = self.extract_defect_type(commit_message)
            severity = self.extract_severity(commit_message)
            
            # 构建缺陷数据
            for file_path in diff_data['before_files']:
                if file_path in diff_data['after_files']:
                    defect_data.append({
                        'repo_name': repo_name,
                        'commit_hash': commit_hash,
                        'commit_message': commit_message,
                        'file_path': file_path,
                        'before_code': diff_data['before_files'][file_path],
                        'after_code': diff_data['after_files'][file_path],
                        'defect_types': defect_types,
                        'severity': severity,
                        'has_defect': True
                    })
        
        logger.info(f"共处理 {len(defect_data)} 个缺陷数据")
        return defect_data
    
    def save_data(self, defect_data: List[Dict[str, Any]], filename: str):
        """
        保存缺陷数据
        
        Args:
            defect_data: 缺陷数据列表
            filename: 文件名
        """
        logger.info(f"保存 {len(defect_data)} 个缺陷数据到 {filename}")
        
        # 保存为JSON文件
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(defect_data, f, ensure_ascii=False, indent=2)
    
    def collect_data(self, queries: List[str], max_results_per_query: int = 100, temp_dir: str = './temp'):
        """
        收集缺陷数据
        
        Args:
            queries: 搜索查询列表
            max_results_per_query: 每个查询的最大结果数
            temp_dir: 临时目录
        """
        all_defect_data = []
        
        for query in queries:
            # 搜索修复bug的提交
            commits = self.search_bug_fix_commits(query, max_results_per_query)
            
            # 处理提交
            defect_data = self.process_commits(commits, temp_dir)
            
            all_defect_data.extend(defect_data)
        
        # 保存数据
        self.save_data(all_defect_data, 'defect_data.json')
        
        # 创建数据集划分
        self.create_dataset_splits(all_defect_data)
    
    def create_dataset_splits(self, defect_data: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1):
        """
        创建数据集划分
        
        Args:
            defect_data: 缺陷数据列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        logger.info(f"创建数据集划分: 训练集 {train_ratio}, 验证集 {val_ratio}, 测试集 {1-train_ratio-val_ratio}")
        
        # 随机打乱数据
        import random
        random.shuffle(defect_data)
        
        # 计算划分索引
        n = len(defect_data)
        train_idx = int(n * train_ratio)
        val_idx = train_idx + int(n * val_ratio)
        
        # 划分数据集
        train_data = defect_data[:train_idx]
        val_data = defect_data[train_idx:val_idx]
        test_data = defect_data[val_idx:]
        
        # 保存数据集
        self.save_data(train_data, 'train_data.json')
        self.save_data(val_data, 'val_data.json')
        self.save_data(test_data, 'test_data.json')
        
        logger.info(f"数据集划分完成: 训练集 {len(train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")

class DefectDataAnnotator:
    """
    代码缺陷数据标注工具
    """
    def __init__(self, data_dir: str, output_dir: str):
        """
        初始化代码缺陷数据标注工具
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self, filename: str) -> List[Dict[str, Any]]:
        """
        加载数据
        
        Args:
            filename: 文件名
            
        Returns:
            data: 数据列表
        """
        logger.info(f"加载数据: {filename}")
        
        # 加载JSON文件
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"共加载 {len(data)} 条数据")
        return data
    
    def save_data(self, data: List[Dict[str, Any]], filename: str):
        """
        保存数据
        
        Args:
            data: 数据列表
            filename: 文件名
        """
        logger.info(f"保存 {len(data)} 条数据到 {filename}")
        
        # 保存为JSON文件
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def annotate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        标注数据
        
        Args:
            data: 数据列表
            
        Returns:
            annotated_data: 标注后的数据列表
        """
        logger.info(f"标注 {len(data)} 条数据")
        
        annotated_data = []
        
        for item in tqdm(data):
            # 检查是否已标注
            if 'defect_types' in item and 'severity' in item:
                annotated_data.append(item)
                continue
            
            # 提取提交信息
            commit_message = item['commit_message']
            
            # 提取缺陷类型和严重程度
            defect_types = self.extract_defect_type(commit_message)
            severity = self.extract_severity(commit_message)
            
            # 更新数据
            item['defect_types'] = defect_types
            item['severity'] = severity
            
            annotated_data.append(item)
        
        logger.info(f"共标注 {len(annotated_data)} 条数据")
        return annotated_data
    
    def extract_defect_type(self, commit_message: str) -> List[str]:
        """
        从提交信息中提取缺陷类型
        
        Args:
            commit_message: 提交信息
            
        Returns:
            defect_types: 缺陷类型列表
        """
        # 定义缺陷类型关键词
        defect_patterns = {
            'security_vulnerability': [
                r'security', r'vulnerability', r'exploit', r'cve', r'xss', 
                r'injection', r'overflow', r'leak', r'auth', r'permission'
            ],
            'performance_issue': [
                r'performance', r'slow', r'speed', r'memory', r'cpu', 
                r'optimize', r'efficient', r'leak', r'timeout'
            ],
            'logic_error': [
                r'logic', r'incorrect', r'wrong', r'error', r'bug', 
                r'fix', r'issue', r'mistake', r'fault'
            ],
            'exception_handling': [
                r'exception', r'error', r'crash', r'handle', r'catch', 
                r'try', r'except', r'throw', r'null'
            ],
            'concurrency_issue': [
                r'concurrency', r'race', r'deadlock', r'thread', r'sync', 
                r'atomic', r'lock', r'mutex'
            ]
        }
        
        # 检查提交信息中的关键词
        defect_types = []
        for defect_type, patterns in defect_patterns.items():
            for pattern in patterns:
                if re.search(pattern, commit_message, re.IGNORECASE):
                    defect_types.append(defect_type)
                    break
        
        return list(set(defect_types))
    
    def extract_severity(self, commit_message: str) -> str:
        """
        从提交信息中提取严重程度
        
        Args:
            commit_message: 提交信息
            
        Returns:
            severity: 严重程度 (low, medium, high)
        """
        # 定义严重程度关键词
        severity_patterns = {
            'high': [
                r'critical', r'severe', r'major', r'high', r'serious', 
                r'important', r'urgent', r'emergency', r'crash', r'security'
            ],
            'medium': [
                r'medium', r'moderate', r'normal', r'average', r'common', 
                r'standard', r'regular', r'general'
            ],
            'low': [
                r'low', r'minor', r'trivial', r'small', r'simple', 
                r'easy', r'cosmetic', r'typo', r'documentation'
            ]
        }
        
        # 检查提交信息中的关键词
        for severity, patterns in severity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, commit_message, re.IGNORECASE):
                    return severity
        
        # 默认为中等严重程度
        return 'medium'
    
    def process_data(self):
        """
        处理数据
        """
        # 加载数据
        train_data = self.load_data('train_data.json')
        val_data = self.load_data('val_data.json')
        test_data = self.load_data('test_data.json')
        
        # 标注数据
        train_data = self.annotate_data(train_data)
        val_data = self.annotate_data(val_data)
        test_data = self.annotate_data(test_data)
        
        # 保存数据
        self.save_data(train_data, 'train_data_annotated.json')
        self.save_data(val_data, 'val_data_annotated.json')
        self.save_data(test_data, 'test_data_annotated.json')

# 使用示例
if __name__ == "__main__":
    import time
    
    # GitHub数据收集
    collector = GitHubDataCollector(
        token="YOUR_GITHUB_TOKEN",
        output_dir="./data/defect_detection"
    )
    
    # 定义搜索查询
    queries = [
        "fix bug language:python",
        "fix security vulnerability language:python",
        "fix performance issue language:python",
        "fix exception language:python",
        "fix concurrency issue language:python"
    ]
    
    # 收集数据
    collector.collect_data(
        queries=queries,
        max_results_per_query=100,
        temp_dir="./temp"
    )
    
    # 数据标注
    annotator = DefectDataAnnotator(
        data_dir="./data/defect_detection",
        output_dir="./data/defect_detection/annotated"
    )
    
    # 处理数据
    annotator.process_data()