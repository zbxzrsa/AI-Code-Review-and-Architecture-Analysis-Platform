"""
GitHub API 集成 - PR 信息、Checks 注解、提交元数据

功能：
1. 认证与连接管理
2. 获取 PR 变更文件列表
3. 发布 Checks API 结果
4. 获取提交元数据
5. 发布 PR 评论
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

try:
    import requests
    from requests.auth import HTTPBasicAuth
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


@dataclass
class GitHubCheckResult:
    """GitHub Check 结果"""
    conclusion: str  # 'success', 'failure', 'neutral', 'cancelled', 'skipped', 'timed_out'
    title: str
    summary: str
    annotations: List[Dict]  # [{path, start_line, end_line, annotation_level, message, title}]
    completed_at: str = None

    def to_dict(self):
        if not self.completed_at:
            self.completed_at = datetime.utcnow().isoformat() + 'Z'

        return {
            'conclusion': self.conclusion,
            'completed_at': self.completed_at,
            'output': {
                'title': self.title,
                'summary': self.summary,
                'annotations': self.annotations[:50]  # API 限制 50 个注解/请求
            }
        }


@dataclass
class PullRequestInfo:
    """PR 信息"""
    number: int
    head_sha: str                    # PR 最新 commit SHA
    base_sha: str                    # Base 分支 commit SHA
    title: str
    description: str
    author: str
    changed_files: List[str]         # 变更文件列表
    additions: int
    deletions: int
    changed_lines_count: int = 0


class GitHubAPI:
    """GitHub API 客户端"""

    API_BASE = "https://api.github.com"

    def __init__(self, owner: str, repo: str, token: str = None, base_url: str = None):
        """
        Args:
            owner: 仓库所有者
            repo: 仓库名
            token: Personal Access Token（可从 GITHUB_TOKEN 环境变量读取）
            base_url: GitHub API 基础 URL（支持 GitHub Enterprise）
        """
        self.owner = owner
        self.repo = repo
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.base_url = base_url or self.API_BASE
        self.session = None

        if not self.token:
            logger.warning("GITHUB_TOKEN not set; API calls may be rate-limited")

    def _get_session(self):
        """获取 requests session"""
        if not self.session:
            self.session = requests.Session() if requests else None
            if self.session and self.token:
                self.session.headers.update({
                    'Authorization': f'token {self.token}',
                    'Accept': 'application/vnd.github.v3+json',
                    'X-GitHub-Api-Version': '2022-11-28'
                })
        return self.session

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """发送 API 请求"""
        if not requests:
            logger.error("requests library not available")
            return {}

        session = self._get_session()
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}{endpoint}"

        try:
            resp = session.request(method, url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp.json() if resp.text else {}
        except requests.exceptions.RequestException as e:
            logger.error(f"GitHub API request failed: {e}")
            return {}

    def get_pr_info(self, pr_number: int) -> Optional[PullRequestInfo]:
        """获取 PR 信息"""
        pr_data = self._request('GET', f'/pulls/{pr_number}')
        if not pr_data:
            return None

        # 获取变更文件列表
        changed_files = self.get_pr_changed_files(pr_number)

        return PullRequestInfo(
            number=pr_number,
            head_sha=pr_data['head']['sha'],
            base_sha=pr_data['base']['sha'],
            title=pr_data['title'],
            description=pr_data['body'] or '',
            author=pr_data['user']['login'],
            changed_files=changed_files,
            additions=pr_data['additions'],
            deletions=pr_data['deletions']
        )

    def get_pr_changed_files(self, pr_number: int) -> List[str]:
        """获取 PR 变更文件列表"""
        files = []
        page = 1

        while True:
            data = self._request('GET', f'/pulls/{pr_number}/files',
                                params={'page': page, 'per_page': 100})

            if not data or not isinstance(data, list):
                break

            files.extend([f['filename'] for f in data])

            if len(data) < 100:
                break
            page += 1

        return files

    def get_commit_info(self, commit_sha: str) -> Dict:
        """获取提交元数据"""
        return self._request('GET', f'/commits/{commit_sha}')

    def create_check_run(self, head_sha: str, check_name: str) -> Optional[str]:
        """创建 Check Run（返回 check_run_id）"""
        data = {
            'name': check_name,
            'head_sha': head_sha,
            'status': 'in_progress'
        }

        response = self._request('POST', '/check-runs', json=data)
        return response.get('id') if response else None

    def update_check_run(self, check_run_id: int, result: GitHubCheckResult):
        """更新 Check Run 结果"""
        data = result.to_dict()
        data['status'] = 'completed'

        self._request('PATCH', f'/check-runs/{check_run_id}', json=data)

    def create_pr_comment(self, pr_number: int, comment_body: str):
        """在 PR 上创建评论"""
        data = {'body': comment_body}
        self._request('POST', f'/issues/{pr_number}/comments', json=data)

    def create_review_comment(self, pr_number: int, commit_sha: str,
                             file_path: str, line: int, comment: str):
        """在 PR 具体行上创建 Review 评论"""
        data = {
            'commit_id': commit_sha,
            'path': file_path,
            'line': line,
            'body': comment
        }
        self._request('POST', f'/pulls/{pr_number}/comments', json=data)


class ChecksAnnotationBuilder:
    """Check 注解生成器"""

    ANNOTATION_LEVELS = {
        'error': 'failure',
        'warning': 'warning',
        'note': 'notice',
        'info': 'notice'
    }

    @staticmethod
    def build_from_analysis_result(
        result: Dict,
        file_path: str,
        base_sha: str = None
    ) -> List[Dict]:
        """从分析结果生成 Check 注解

        Args:
            result: 分析结果 {issues: [{line, col, rule, message, severity}], ...}
            file_path: 文件路径
            base_sha: Base commit SHA（用于增量模式）

        Returns:
            注解列表
        """
        annotations = []

        if 'issues' not in result:
            return annotations

        for issue in result['issues'][:50]:  # 每个文件最多 50 个注解
            annotation = {
                'path': file_path,
                'start_line': issue.get('line', 1),
                'end_line': issue.get('end_line', issue.get('line', 1)),
                'annotation_level': ChecksAnnotationBuilder.ANNOTATION_LEVELS.get(
                    issue.get('severity', 'note'), 'notice'
                ),
                'message': issue.get('message', ''),
                'title': issue.get('rule', 'Analysis Issue'),
            }

            # 添加代码片段
            if 'code_snippet' in issue:
                annotation['raw_details'] = issue['code_snippet']

            annotations.append(annotation)

        return annotations


class GitHubPRAnalysisIntegration:
    """GitHub PR 分析集成 - 完整工作流"""

    def __init__(self, owner: str, repo: str, token: str = None):
        self.github = GitHubAPI(owner, repo, token)
        self.check_runs = {}  # pr_number -> check_run_id

    def start_analysis(self, pr_number: int, head_sha: str) -> bool:
        """启动分析 - 创建 Check Run"""
        check_id = self.github.create_check_run(head_sha, 'Code Analysis')
        if check_id:
            self.check_runs[pr_number] = check_id
            return True
        return False

    def publish_results(self, pr_number: int, head_sha: str,
                       analysis_results: Dict[str, Dict]) -> bool:
        """发布分析结果到 GitHub

        Args:
            pr_number: PR 号
            head_sha: 提交 SHA
            analysis_results: {file_path: {issues: [...], ...}}

        Returns:
            是否成功
        """
        check_id = self.check_runs.get(pr_number)
        if not check_id:
            logger.warning(f"No check run found for PR {pr_number}")
            return False

        # 汇总注解
        all_annotations = []
        total_issues = 0
        critical_count = 0

        for file_path, result in analysis_results.items():
            annotations = ChecksAnnotationBuilder.build_from_analysis_result(result, file_path)
            all_annotations.extend(annotations)

            issues = result.get('issues', [])
            total_issues += len(issues)
            critical_count += sum(1 for i in issues if i.get('severity') == 'error')

        # 生成摘要
        if total_issues == 0:
            conclusion = 'success'
            summary = "✅ All checks passed!"
        else:
            conclusion = 'failure' if critical_count > 0 else 'neutral'
            summary = f"Found {total_issues} issues ({critical_count} critical)"

        # 更新 Check Run
        check_result = GitHubCheckResult(
            conclusion=conclusion,
            title="Code Analysis Report",
            summary=summary,
            annotations=all_annotations
        )

        self.github.update_check_run(check_id, check_result)
        return True

    def publish_pr_summary_comment(self, pr_number: int,
                                  analysis_results: Dict[str, Dict],
                                  performance_metrics: Dict = None):
        """发布 PR 摘要评论"""
        total_files = len(analysis_results)
        total_issues = sum(len(r.get('issues', [])) for r in analysis_results.values())

        comment = f"""## 🔍 Code Analysis Summary

**Files analyzed**: {total_files}
**Total issues found**: {total_issues}

"""

        # 按文件列出问题
        for file_path, result in analysis_results.items():
            issues = result.get('issues', [])
            if issues:
                comment += f"### {file_path}\n"
                for issue in issues[:5]:  # 每个文件最多显示 5 个
                    severity = issue.get('severity', 'info').upper()
                    comment += f"- **[{severity}]** Line {issue['line']}: {issue['message']}\n"
                if len(issues) > 5:
                    comment += f"- ... and {len(issues) - 5} more issues\n"
                comment += "\n"

        # 添加性能指标
        if performance_metrics:
            comment += f"""
### ⚡ Performance

- **Total time**: {performance_metrics.get('total_time', '?')}s
- **Cache hit ratio**: {performance_metrics.get('cache_hit_ratio', '?'):.1%}
- **Files cached**: {performance_metrics.get('cached_files', '?')}

[View full report](#)
"""

        self.github.create_pr_comment(pr_number, comment)


def create_github_integration(owner: str, repo: str) -> Optional[GitHubPRAnalysisIntegration]:
    """工厂函数 - 创建 GitHub 集成实例"""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        logger.warning("GITHUB_TOKEN not set; GitHub integration will be disabled")
        return None

    return GitHubPRAnalysisIntegration(owner, repo, token)
