"""
P2 核心 API 端点 - 集成依赖图、GitHub、规则引擎

路由：
- POST /api/v1/pr/{pr_number}/analyze - 触发 PR 分析
- GET /api/v1/pr/{pr_number}/analysis - 获取分析进度与结果
- POST /api/v1/rules/filter - 应用规则过滤
- GET /api/v1/dependency-graph/{file_path} - 获取依赖图
"""

import os
import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Header
from pydantic import BaseModel

from app.services.dependency_graph import (
    DependencyGraph, DependencyParser, IncrementalAnalyzer
)
from app.services.github_integration import (
    GitHubPRAnalysisIntegration, ChecksAnnotationBuilder
)
from app.services.rule_engine import RuleEngine, RuleRegistry, NoiseDetector
from app.db.session import get_db
from app.tasks.analysis_tasks import run_analysis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["P2"])

# 全局实例
dependency_graph = DependencyGraph()
rule_engine = RuleRegistry.create_default_engine()
github_integration: Optional[GitHubPRAnalysisIntegration] = None

# 初始化 GitHub 集成
def init_github_integration():
    global github_integration
    owner = os.getenv('GITHUB_OWNER')
    repo = os.getenv('GITHUB_REPO')
    if owner and repo:
        github_integration = GitHubPRAnalysisIntegration(owner, repo)
        logger.info(f"GitHub integration initialized for {owner}/{repo}")


class AnalyzePRRequest(BaseModel):
    """PR 分析请求"""
    rulepack_version: str = "default"
    include_categories: List[str] = None  # 要分析的分类
    exclude_rules: List[str] = None


class FilterIssuesRequest(BaseModel):
    """Issue 过滤请求"""
    issues: List[Dict]
    severity_threshold: str = "warning"  # 最低严重级别


class DependencyNodeResponse(BaseModel):
    """依赖图节点响应"""
    file_path: str
    imports: List[str]
    imported_by: List[str]
    import_hash: str


# ============================================================================
# PR 分析端点
# ============================================================================

@router.post("/pr/{pr_number}/analyze")
async def analyze_pr(
    pr_number: int,
    request: AnalyzePRRequest,
    background_tasks: BackgroundTasks,
    x_github_token: Optional[str] = Header(None),
):
    """触发 PR 分析

    流程：
    1. 获取 PR 信息 & 变更文件列表
    2. 确定分析范围（基于依赖图）
    3. 触发增量分析任务
    4. 创建 GitHub Check Run
    """

    if not github_integration:
        init_github_integration()

    if not github_integration:
        raise HTTPException(
            status_code=503,
            detail="GitHub integration not configured"
        )

    try:
        # 1. 获取 PR 信息
        pr_info = github_integration.github.get_pr_info(pr_number)
        if not pr_info:
            raise HTTPException(status_code=404, detail=f"PR {pr_number} not found")

        # 2. 确定分析范围（基于依赖图的影响分析）
        analyzer = IncrementalAnalyzer(dependency_graph, {})
        analysis_scope = analyzer.determine_analysis_scope(set(pr_info.changed_files))

        logger.info(f"PR {pr_number}: {len(pr_info.changed_files)} changed files, "
                   f"{len(analysis_scope)} files in scope")

        # 3. 创建 GitHub Check Run
        if not github_integration.start_analysis(pr_number, pr_info.head_sha):
            raise HTTPException(status_code=500, detail="Failed to create GitHub Check Run")

        # 4. 后台触发分析任务
        async def run_pr_analysis():
            results = {}
            for file_path, scope_type in analysis_scope.items():
                # 触发分析任务
                task_result = await run_analysis(
                    file_path=file_path,
                    commit_sha=pr_info.head_sha,
                    rulepack_version=request.rulepack_version,
                    analysis_type=scope_type  # 'full' or 'incremental'
                )
                results[file_path] = task_result

            # 应用规则过滤
            all_issues = []
            for file_result in results.values():
                if 'issues' in file_result:
                    all_issues.extend(file_result['issues'])

            filtered_issues = rule_engine.filter_issues(all_issues)

            # 汇总结果
            file_results = {}
            for file_path, result in results.items():
                file_issues = [i for i in filtered_issues if i.get('file_path') == file_path]
                file_results[file_path] = {'issues': file_issues}

            # 发布到 GitHub
            github_integration.publish_results(pr_number, pr_info.head_sha, file_results)

            # 发布 PR 评论摘要
            summary = rule_engine.get_issues_summary(filtered_issues)
            github_integration.publish_pr_summary_comment(
                pr_number,
                file_results,
                performance_metrics={
                    'total_time': '?',  # TODO: 从分析结果获取
                    'cache_hit_ratio': 0.6,
                    'cached_files': 5
                }
            )

        background_tasks.add_task(run_pr_analysis)

        return {
            "status": "queued",
            "pr_number": pr_number,
            "head_sha": pr_info.head_sha,
            "files_to_analyze": len(analysis_scope),
            "message": "Analysis queued successfully"
        }

    except Exception as e:
        logger.error(f"Failed to analyze PR {pr_number}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pr/{pr_number}/analysis")
async def get_pr_analysis(
    pr_number: int,
    sha: str = Query(...),
):
    """获取 PR 分析进度与结果

    返回格式：
    {
      "status": "running|completed|failed",
      "progress": 45,
      "total_files": 10,
      "analyzed_files": 4,
      "cached_files": 2,
      "issues": [...],
      "summary": {...},
      "performance": {...},
      "github_check_url": "..."
    }
    """

    # TODO: 从数据库查询分析结果
    # 这里返回示例数据
    return {
        "status": "completed",
        "progress": 100,
        "total_files": 10,
        "analyzed_files": 10,
        "cached_files": 4,
        "issues": [],
        "summary": {
            "total_issues": 0,
            "by_severity": {},
            "by_category": {},
            "top_rules": []
        },
        "performance": {
            "total_time": 45.2,
            "cache_hit_ratio": 0.4
        },
        "github_check_url": f"https://github.com/repo/commit/{sha}/checks"
    }


# ============================================================================
# 规则引擎端点
# ============================================================================

@router.post("/rules/filter")
async def filter_issues(request: FilterIssuesRequest):
    """应用规则引擎过滤

    功能：
    1. 过滤禁用的规则
    2. 检测噪音（重复/虚假正例）
    3. 应用豁免条件
    4. 按优先级排序
    """

    filtered = rule_engine.filter_issues(request.issues)
    sorted_issues = rule_engine.sort_issues_by_priority(filtered)
    summary = rule_engine.get_issues_summary(sorted_issues)

    return {
        "total_input": len(request.issues),
        "total_output": len(sorted_issues),
        "filtered_out": len(request.issues) - len(sorted_issues),
        "issues": sorted_issues,
        "summary": summary
    }


@router.get("/rules")
async def list_rules():
    """列出所有规则及其配置"""
    rules = {}
    for rule_id, config in rule_engine.rules.items():
        rules[rule_id] = {
            "name": config.name,
            "category": config.category.value,
            "severity": config.default_severity.value,
            "enabled": config.enabled,
            "priority": config.priority,
            "description": config.description
        }
    return rules


@router.patch("/rules/{rule_id}")
async def update_rule(rule_id: str, updates: Dict):
    """更新规则配置

    支持的更新：
    - enabled: bool
    - priority: int (0-100)
    - exemptions: List[str]
    - thresholds: Dict
    """

    if rule_id not in rule_engine.rules:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")

    rule = rule_engine.rules[rule_id]

    if 'enabled' in updates:
        rule.enabled = updates['enabled']

    if 'priority' in updates:
        rule_engine.set_rule_priority(rule_id, updates['priority'])

    if 'exemptions' in updates:
        rule.exemptions = updates['exemptions']

    if 'thresholds' in updates:
        rule.thresholds = updates['thresholds']

    return {
        "rule_id": rule_id,
        "updated": True,
        "rule": {
            "name": rule.name,
            "enabled": rule.enabled,
            "priority": rule.priority
        }
    }


# ============================================================================
# 依赖图端点
# ============================================================================

@router.get("/dependency-graph/{file_path:path}")
async def get_dependency_info(file_path: str):
    """获取文件的依赖信息

    返回：
    - 直接导入
    - 被谁导入（反向依赖）
    - 影响链（分层显示）
    """

    if file_path not in dependency_graph.nodes:
        raise HTTPException(status_code=404, detail=f"File {file_path} not found in dependency graph")

    node = dependency_graph.nodes[file_path]
    impact_chain = dependency_graph.get_impact_chain(file_path, max_depth=3)

    return {
        "file_path": file_path,
        "import_hash": node.import_hash,
        "imports": list(node.imports),
        "imported_by": list(node.imported_by),
        "impact_chain": impact_chain,
        "impact_count": sum(len(level) for level in impact_chain[1:])  # 不包括文件本身
    }


@router.post("/dependency-graph/analyze-change")
async def analyze_change_impact(
    changed_files: List[str],
    max_depth: int = Query(10)
):
    """分析变更对代码库的影响

    功能：
    1. 从变更文件集合出发
    2. 计算反向依赖闭包
    3. 返回所有受影响的文件
    """

    affected = dependency_graph.get_reverse_closure(set(changed_files), depth=max_depth)

    return {
        "changed_files": changed_files,
        "affected_files": list(affected),
        "affected_count": len(affected),
        "impact_ratio": len(affected) / max(1, len(dependency_graph.nodes)) if dependency_graph.nodes else 0
    }


# ============================================================================
# 初始化
# ============================================================================

@router.on_event("startup")
async def startup_event():
    """启动时初始化集成"""
    init_github_integration()
    logger.info("P2 module initialized")
