"""
规则引擎与噪音过滤

功能：
1. 规则生命周期管理（启用/禁用/优先级）
2. Issue 分类（错误/警告/信息）
3. 噪音检测（重复/虚假正例）
4. 规则配置（阈值、豁免）
"""

import re
import json
import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class IssueSeverity(str, Enum):
    """Issue 严重级别"""
    CRITICAL = "critical"   # 必须立即修复
    ERROR = "error"         # 需要修复
    WARNING = "warning"     # 建议修复
    INFO = "info"          # 信息性


class IssueCategory(str, Enum):
    """Issue 分类"""
    SECURITY = "security"          # 安全问题
    PERFORMANCE = "performance"    # 性能问题
    MAINTAINABILITY = "maintainability"  # 可维护性
    TESTING = "testing"            # 测试覆盖
    STYLE = "style"                # 代码风格
    DOCUMENTATION = "documentation" # 文档
    BUG = "bug"                     # 潜在 Bug


@dataclass
class RuleConfig:
    """规则配置"""
    rule_id: str
    name: str
    category: IssueCategory
    default_severity: IssueSeverity
    enabled: bool = True
    priority: int = 100  # 0-100, 越高越重要
    description: str = ""
    exemptions: List[str] = field(default_factory=list)  # 豁免条件（正则表达式）
    thresholds: Dict[str, any] = field(default_factory=dict)  # 阈值参数

    def to_dict(self):
        return asdict(self)


@dataclass
class IssueFingerprint:
    """Issue 指纹 - 用于去重"""
    file_path: str
    line: int
    rule_id: str
    message_hash: str  # 消息内容的哈希

    def __hash__(self):
        return hash((self.file_path, self.line, self.rule_id))

    def __eq__(self, other):
        return (self.file_path == other.file_path and
                self.line == other.line and
                self.rule_id == other.rule_id)


class NoiseDetector:
    """噪音检测器"""

    def __init__(self):
        self.issue_history: Dict[IssueFingerprint, List[datetime]] = defaultdict(list)
        self.false_positive_patterns: List[str] = []  # 已知的虚假正例模式
        self.repeated_issue_threshold = 3  # 同一问题出现 3 次以上视为噪音

    def is_duplicate(self, fingerprint: IssueFingerprint,
                    within_hours: int = 24) -> bool:
        """检测是否为重复 Issue"""
        if fingerprint not in self.issue_history:
            return False

        recent = [ts for ts in self.issue_history[fingerprint]
                 if datetime.now() - ts < timedelta(hours=within_hours)]

        return len(recent) >= self.repeated_issue_threshold

    def is_likely_false_positive(self, issue: Dict) -> bool:
        """检测是否为虚假正例"""
        message = issue.get('message', '').lower()

        for pattern in self.false_positive_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return True

        # 简单启发式：非常短的消息通常是虚假正例
        if len(message) < 10:
            return True

        return False

    def should_suppress(self, fingerprint: IssueFingerprint, issue: Dict) -> bool:
        """是否应该抑制此 Issue"""
        return self.is_duplicate(fingerprint) or self.is_likely_false_positive(issue)

    def record_issue(self, fingerprint: IssueFingerprint):
        """记录 Issue（用于后续的重复检测）"""
        self.issue_history[fingerprint].append(datetime.now())

        # 保留最近 100 个记录
        if len(self.issue_history[fingerprint]) > 100:
            self.issue_history[fingerprint] = self.issue_history[fingerprint][-100:]

    def add_false_positive_pattern(self, pattern: str):
        """添加虚假正例模式"""
        self.false_positive_patterns.append(pattern)


class RuleEngine:
    """规则引擎"""

    def __init__(self):
        self.rules: Dict[str, RuleConfig] = {}
        self.rule_categories: Dict[IssueCategory, Set[str]] = defaultdict(set)
        self.noise_detector = NoiseDetector()

    def register_rule(self, config: RuleConfig):
        """注册规则"""
        self.rules[config.rule_id] = config
        self.rule_categories[config.category].add(config.rule_id)

    def enable_rule(self, rule_id: str):
        """启用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True

    def disable_rule(self, rule_id: str):
        """禁用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False

    def set_rule_priority(self, rule_id: str, priority: int):
        """设置规则优先级"""
        if rule_id in self.rules:
            self.rules[rule_id].priority = max(0, min(100, priority))

    def filter_issues(self, issues: List[Dict]) -> List[Dict]:
        """过滤 Issue 列表

        Args:
            issues: 原始 Issue 列表

        Returns:
            过滤后的 Issue 列表
        """
        filtered = []

        for issue in issues:
            rule_id = issue.get('rule_id')

            # 1. 检查规则是否启用
            if rule_id not in self.rules or not self.rules[rule_id].enabled:
                continue

            # 2. 检查豁免条件
            rule = self.rules[rule_id]
            if self._is_exempted(issue, rule):
                continue

            # 3. 检查是否噪音
            fingerprint = IssueFingerprint(
                file_path=issue.get('file_path', ''),
                line=issue.get('line', 0),
                rule_id=rule_id,
                message_hash=self._hash_message(issue.get('message', ''))
            )

            if self.noise_detector.should_suppress(fingerprint, issue):
                continue

            # 4. 应用 thresholds（如果需要）
            if not self._check_thresholds(issue, rule):
                continue

            filtered.append(issue)
            self.noise_detector.record_issue(fingerprint)

        return filtered

    def sort_issues_by_priority(self, issues: List[Dict]) -> List[Dict]:
        """按优先级排序 Issue

        排序优先级：
        1. 严重级别（CRITICAL > ERROR > WARNING > INFO）
        2. 规则优先级
        3. 行号
        """
        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.ERROR: 1,
            IssueSeverity.WARNING: 2,
            IssueSeverity.INFO: 3,
        }

        def sort_key(issue):
            rule_id = issue.get('rule_id')
            rule = self.rules.get(rule_id)

            severity = issue.get('severity', 'info')
            severity_rank = severity_order.get(severity, 4)

            priority = rule.priority if rule else 50
            line = issue.get('line', float('inf'))

            # 返回元组进行多级排序
            return (severity_rank, -priority, line)

        return sorted(issues, key=sort_key)

    def group_issues_by_category(self, issues: List[Dict]) -> Dict[IssueCategory, List[Dict]]:
        """按分类分组 Issue"""
        grouped = defaultdict(list)

        for issue in issues:
            rule_id = issue.get('rule_id')
            rule = self.rules.get(rule_id)
            if rule:
                grouped[rule.category].append(issue)

        return dict(grouped)

    def get_issues_summary(self, issues: List[Dict]) -> Dict:
        """获取 Issue 摘要"""
        severity_counts = Counter(i.get('severity', 'info') for i in issues)
        category_counts = Counter()

        for issue in issues:
            rule_id = issue.get('rule_id')
            rule = self.rules.get(rule_id)
            if rule:
                category_counts[rule.category] += 1

        return {
            'total_issues': len(issues),
            'by_severity': dict(severity_counts),
            'by_category': dict(category_counts),
            'top_rules': self._get_top_rules(issues, limit=5),
        }

    @staticmethod
    def _is_exempted(issue: Dict, rule: RuleConfig) -> bool:
        """检查是否符合豁免条件"""
        message = issue.get('message', '')

        for exemption_pattern in rule.exemptions:
            if re.search(exemption_pattern, message, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def _check_thresholds(issue: Dict, rule: RuleConfig) -> bool:
        """检查阈值"""
        # 这里可以根据具体规则实现阈值检查
        # 例如：影响行数、文件数等
        return True

    @staticmethod
    def _hash_message(message: str) -> str:
        """计算消息哈希"""
        import hashlib
        return hashlib.sha256(message.encode()).hexdigest()[:16]

    @staticmethod
    def _get_top_rules(issues: List[Dict], limit: int = 5) -> List[Tuple[str, int]]:
        """获取最常出现的规则"""
        rule_counts = Counter(i.get('rule_id') for i in issues)
        return rule_counts.most_common(limit)


class RuleRegistry:
    """规则注册表 - 预定义规则集"""

    # Python 安全规则
    RULE_SQL_INJECTION = RuleConfig(
        rule_id="py-sql-injection",
        name="SQL Injection Risk",
        category=IssueCategory.SECURITY,
        default_severity=IssueSeverity.CRITICAL,
        priority=99,
        description="Detects potential SQL injection vulnerabilities"
    )

    RULE_HARDCODED_PASSWORD = RuleConfig(
        rule_id="hardcoded-password",
        name="Hardcoded Password/Secret",
        category=IssueCategory.SECURITY,
        default_severity=IssueSeverity.CRITICAL,
        priority=98,
        exemptions=[r"password.*=.*\*+", r"(?:test|example|demo).*password"]
    )

    # 性能规则
    RULE_N_PLUS_ONE = RuleConfig(
        rule_id="n-plus-one-query",
        name="N+1 Query Pattern",
        category=IssueCategory.PERFORMANCE,
        default_severity=IssueSeverity.WARNING,
        priority=85,
        description="Database query inside a loop"
    )

    RULE_UNOPTIMIZED_LOOP = RuleConfig(
        rule_id="unoptimized-loop",
        name="Unoptimized Loop",
        category=IssueCategory.PERFORMANCE,
        default_severity=IssueSeverity.WARNING,
        priority=70,
        description="Loop that could be replaced with list comprehension or built-in"
    )

    # 可维护性规则
    RULE_HIGH_COMPLEXITY = RuleConfig(
        rule_id="high-cyclomatic-complexity",
        name="High Cyclomatic Complexity",
        category=IssueCategory.MAINTAINABILITY,
        default_severity=IssueSeverity.WARNING,
        priority=75,
        thresholds={'max_complexity': 10}
    )

    RULE_MISSING_DOCSTRING = RuleConfig(
        rule_id="missing-docstring",
        name="Missing Docstring",
        category=IssueCategory.DOCUMENTATION,
        default_severity=IssueSeverity.INFO,
        priority=50,
        exemptions=[r"^_.*"]  # Private methods exempted
    )

    @classmethod
    def create_default_engine(cls) -> RuleEngine:
        """创建默认规则引擎"""
        engine = RuleEngine()

        # 注册所有默认规则
        for attr_name in dir(cls):
            if attr_name.startswith('RULE_'):
                rule_config = getattr(cls, attr_name)
                if isinstance(rule_config, RuleConfig):
                    engine.register_rule(rule_config)

        return engine


class RuleConfigLoader:
    """规则配置加载器"""

    @staticmethod
    def load_from_json(json_path: str) -> Dict[str, RuleConfig]:
        """从 JSON 文件加载规则配置"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            rules = {}
            for rule_id, rule_data in data.items():
                rules[rule_id] = RuleConfig(**rule_data)

            return rules
        except Exception as e:
            logger.error(f"Failed to load rules from {json_path}: {e}")
            return {}

    @staticmethod
    def save_to_json(json_path: str, rules: Dict[str, RuleConfig]):
        """保存规则配置到 JSON 文件"""
        try:
            data = {rule_id: rule.to_dict()
                   for rule_id, rule in rules.items()}

            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save rules to {json_path}: {e}")
