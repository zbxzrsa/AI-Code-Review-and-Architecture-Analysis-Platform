"""
增强的语义Diff分析服务
支持函数级变更检测、调用影响分析和风险热区识别
"""

import ast
import difflib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """变更类型"""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    MOVED = "moved"


class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FunctionSignature:
    """函数签名"""
    name: str
    parameters: List[str]
    return_type: Optional[str]
    decorators: List[str]
    is_async: bool
    line_start: int
    line_end: int
    file_path: str


@dataclass
class ClassSignature:
    """类签名"""
    name: str
    bases: List[str]
    methods: List[FunctionSignature]
    line_start: int
    line_end: int
    file_path: str


@dataclass
class SemanticChange:
    """语义变更"""
    change_type: ChangeType
    entity_type: str  # function, class, variable, import
    entity_name: str
    old_signature: Optional[FunctionSignature]
    new_signature: Optional[FunctionSignature]
    risk_level: RiskLevel
    description: str
    affected_lines: List[int]
    impact_scope: List[str]  # 受影响的文件/函数
    suggested_actions: List[str]


@dataclass
class CallGraphChange:
    """调用图变更"""
    caller: str
    callee: str
    change_type: ChangeType
    risk_level: RiskLevel
    description: str


@dataclass
class RiskHotspot:
    """风险热区"""
    file_path: str
    line_range: Tuple[int, int]
    risk_level: RiskLevel
    risk_factors: List[str]
    affected_functions: List[str]
    recommendations: List[str]


class SemanticDiffAnalyzer:
    """语义Diff分析器"""
    
    def __init__(self, dependency_service=None):
        self.dependency_service = dependency_service
        self.risk_patterns = self._load_risk_patterns()
        
    def analyze_semantic_diff(
        self,
        old_content: str,
        new_content: str,
        file_path: str,
        old_file_path: Optional[str] = None
    ) -> List[SemanticChange]:
        """分析语义差异"""
        try:
            # 解析AST
            old_ast = ast.parse(old_content, type_comments=True) if old_content.strip() else None
            new_ast = ast.parse(new_content, type_comments=True) if new_content.strip() else None
            
            # 提取函数和类签名
            old_functions = self._extract_functions(old_ast, file_path) if old_ast else {}
            new_functions = self._extract_functions(new_ast, file_path) if new_ast else {}
            
            old_classes = self._extract_classes(old_ast, file_path) if old_ast else {}
            new_classes = self._extract_classes(new_ast, file_path) if new_ast else {}
            
            # 分析变更
            changes = []
            
            # 分析函数变更
            function_changes = self._analyze_function_changes(
                old_functions, new_functions, file_path
            )
            changes.extend(function_changes)
            
            # 分析类变更
            class_changes = self._analyze_class_changes(
                old_classes, new_classes, file_path
            )
            changes.extend(class_changes)
            
            # 分析导入变更
            import_changes = self._analyze_import_changes(
                old_ast, new_ast, file_path
            )
            changes.extend(import_changes)
            
            # 分析变量变更
            variable_changes = self._analyze_variable_changes(
                old_ast, new_ast, file_path
            )
            changes.extend(variable_changes)
            
            # 计算风险级别和影响范围
            for change in changes:
                change.risk_level = self._calculate_risk_level(change)
                change.impact_scope = self._calculate_impact_scope(change)
                change.suggested_actions = self._generate_suggested_actions(change)
            
            return changes
            
        except Exception as e:
            logger.error(f"Failed to analyze semantic diff for {file_path}: {e}")
            return []
    
    def analyze_call_graph_changes(
        self,
        old_content: str,
        new_content: str,
        file_path: str
    ) -> List[CallGraphChange]:
        """分析调用图变更"""
        try:
            old_calls = self._extract_function_calls(old_content, file_path)
            new_calls = self._extract_function_calls(new_content, file_path)
            
            changes = []
            
            # 检测新增的调用
            for call in new_calls - old_calls:
                changes.append(CallGraphChange(
                    caller=self._get_caller_function(call, new_content),
                    callee=call,
                    change_type=ChangeType.ADDED,
                    risk_level=self._assess_call_risk(call, ChangeType.ADDED),
                    description=f"New call to {call}"
                ))
            
            # 检测移除的调用
            for call in old_calls - new_calls:
                changes.append(CallGraphChange(
                    caller=self._get_caller_function(call, old_content),
                    callee=call,
                    change_type=ChangeType.REMOVED,
                    risk_level=self._assess_call_risk(call, ChangeType.REMOVED),
                    description=f"Removed call to {call}"
                ))
            
            return changes
            
        except Exception as e:
            logger.error(f"Failed to analyze call graph changes for {file_path}: {e}")
            return []
    
    def identify_risk_hotspots(
        self,
        changes: List[SemanticChange],
        file_path: str
    ) -> List[RiskHotspot]:
        """识别风险热区"""
        hotspots = []
        
        # 按行范围分组变更
        line_groups = self._group_changes_by_lines(changes)
        
        for line_range, group_changes in line_groups.items():
            # 计算风险因素
            risk_factors = []
            max_risk = RiskLevel.LOW
            
            for change in group_changes:
                if change.risk_level.value > max_risk.value:
                    max_risk = change.risk_level
                
                # 收集风险因素
                if change.entity_type == "function" and change.change_type == ChangeType.MODIFIED:
                    risk_factors.append("Function signature modification")
                
                if "security" in change.description.lower():
                    risk_factors.append("Security-related change")
                
                if "database" in change.description.lower() or "sql" in change.description.lower():
                    risk_factors.append("Database operation change")
                
                if "api" in change.description.lower() or "endpoint" in change.description.lower():
                    risk_factors.append("API interface change")
            
            # 生成建议
            recommendations = self._generate_hotspot_recommendations(
                max_risk, risk_factors, group_changes
            )
            
            if max_risk != RiskLevel.LOW:  # 只保留有风险的区域
                hotspots.append(RiskHotspot(
                    file_path=file_path,
                    line_range=line_range,
                    risk_level=max_risk,
                    risk_factors=risk_factors,
                    affected_functions=[c.entity_name for c in group_changes if c.entity_type == "function"],
                    recommendations=recommendations
                ))
        
        return hotspots
    
    def generate_changelog(
        self,
        changes: List[SemanticChange],
        file_path: str
    ) -> Dict[str, Any]:
        """生成变更日志"""
        changelog = {
            "file_path": file_path,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_changes": len(changes),
                "by_type": {},
                "by_risk": {},
                "by_entity": {}
            },
            "changes": [],
            "impact_assessment": {
                "high_risk_count": 0,
                "affected_functions": [],
                "breaking_changes": []
            }
        }
        
        # 统计变更
        for change in changes:
            # 按类型统计
            change_type = change.change_type.value
            changelog["summary"]["by_type"][change_type] = \
                changelog["summary"]["by_type"].get(change_type, 0) + 1
            
            # 按风险统计
            risk_level = change.risk_level.value
            changelog["summary"]["by_risk"][risk_level] = \
                changelog["summary"]["by_risk"].get(risk_level, 0) + 1
            
            # 按实体类型统计
            entity_type = change.entity_type
            changelog["summary"]["by_entity"][entity_type] = \
                changelog["summary"]["by_entity"].get(entity_type, 0) + 1
            
            # 影响评估
            if change.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                changelog["impact_assessment"]["high_risk_count"] += 1
            
            if change.entity_type == "function":
                changelog["impact_assessment"]["affected_functions"].append(change.entity_name)
            
            if self._is_breaking_change(change):
                changelog["impact_assessment"]["breaking_changes"].append({
                    "entity": change.entity_name,
                    "description": change.description
                })
            
            # 添加变更详情
            changelog["changes"].append({
                "type": change.change_type.value,
                "entity_type": change.entity_type,
                "entity_name": change.entity_name,
                "risk_level": change.risk_level.value,
                "description": change.description,
                "affected_lines": change.affected_lines,
                "suggested_actions": change.suggested_actions
            })
        
        return changelog
    
    def _extract_functions(
        self, 
        tree: ast.AST, 
        file_path: str
    ) -> Dict[str, FunctionSignature]:
        """提取函数签名"""
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 提取参数
                parameters = []
                for arg in node.args.args:
                    parameters.append(arg.arg)
                
                # 提取返回类型注解
                return_type = None
                if node.returns:
                    return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
                
                # 提取装饰器
                decorators = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Attribute):
                        decorators.append(ast.unparse(decorator) if hasattr(ast, 'unparse') else str(decorator))
                
                signature = FunctionSignature(
                    name=node.name,
                    parameters=parameters,
                    return_type=return_type,
                    decorators=decorators,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    file_path=file_path
                )
                
                functions[node.name] = signature
        
        return functions
    
    def _extract_classes(
        self, 
        tree: ast.AST, 
        file_path: str
    ) -> Dict[str, ClassSignature]:
        """提取类签名"""
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 提取基类
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif hasattr(ast, 'unparse'):
                        bases.append(ast.unparse(base))
                
                # 提取方法
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_sig = self._extract_functions(
                            ast.Module(body=[item]), file_path
                        ).get(item.name)
                        if method_sig:
                            methods.append(method_sig)
                
                signature = ClassSignature(
                    name=node.name,
                    bases=bases,
                    methods=methods,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    file_path=file_path
                )
                
                classes[node.name] = signature
        
        return classes
    
    def _analyze_function_changes(
        self,
        old_functions: Dict[str, FunctionSignature],
        new_functions: Dict[str, FunctionSignature],
        file_path: str
    ) -> List[SemanticChange]:
        """分析函数变更"""
        changes = []
        
        # 检测新增函数
        for name, new_sig in new_functions.items():
            if name not in old_functions:
                changes.append(SemanticChange(
                    change_type=ChangeType.ADDED,
                    entity_type="function",
                    entity_name=name,
                    old_signature=None,
                    new_signature=new_sig,
                    risk_level=RiskLevel.LOW,  # 将在后面重新计算
                    description=f"Added function '{name}'",
                    affected_lines=list(range(new_sig.line_start, new_sig.line_end + 1)),
                    impact_scope=[],
                    suggested_actions=[]
                ))
        
        # 检测删除函数
        for name, old_sig in old_functions.items():
            if name not in new_functions:
                changes.append(SemanticChange(
                    change_type=ChangeType.REMOVED,
                    entity_type="function",
                    entity_name=name,
                    old_signature=old_sig,
                    new_signature=None,
                    risk_level=RiskLevel.LOW,
                    description=f"Removed function '{name}'",
                    affected_lines=list(range(old_sig.line_start, old_sig.line_end + 1)),
                    impact_scope=[],
                    suggested_actions=[]
                ))
        
        # 检测修改函数
        for name in old_functions:
            if name in new_functions:
                old_sig = old_functions[name]
                new_sig = new_functions[name]
                
                if not self._signatures_equal(old_sig, new_sig):
                    changes.append(SemanticChange(
                        change_type=ChangeType.MODIFIED,
                        entity_type="function",
                        entity_name=name,
                        old_signature=old_sig,
                        new_signature=new_sig,
                        risk_level=RiskLevel.LOW,
                        description=self._describe_function_change(old_sig, new_sig),
                        affected_lines=list(range(
                            min(old_sig.line_start, new_sig.line_start),
                            max(old_sig.line_end, new_sig.line_end) + 1
                        )),
                        impact_scope=[],
                        suggested_actions=[]
                    ))
        
        return changes
    
    def _analyze_class_changes(
        self,
        old_classes: Dict[str, ClassSignature],
        new_classes: Dict[str, ClassSignature],
        file_path: str
    ) -> List[SemanticChange]:
        """分析类变更"""
        changes = []
        
        # 类似于函数变更分析
        for name, new_class in new_classes.items():
            if name not in old_classes:
                changes.append(SemanticChange(
                    change_type=ChangeType.ADDED,
                    entity_type="class",
                    entity_name=name,
                    old_signature=None,
                    new_signature=None,
                    risk_level=RiskLevel.LOW,
                    description=f"Added class '{name}'",
                    affected_lines=list(range(new_class.line_start, new_class.line_end + 1)),
                    impact_scope=[],
                    suggested_actions=[]
                ))
        
        for name, old_class in old_classes.items():
            if name not in new_classes:
                changes.append(SemanticChange(
                    change_type=ChangeType.REMOVED,
                    entity_type="class",
                    entity_name=name,
                    old_signature=None,
                    new_signature=None,
                    risk_level=RiskLevel.LOW,
                    description=f"Removed class '{name}'",
                    affected_lines=list(range(old_class.line_start, old_class.line_end + 1)),
                    impact_scope=[],
                    suggested_actions=[]
                ))
        
        return changes
    
    def _analyze_import_changes(
        self,
        old_ast: Optional[ast.AST],
        new_ast: Optional[ast.AST],
        file_path: str
    ) -> List[SemanticChange]:
        """分析导入变更"""
        changes = []
        
        old_imports = self._extract_imports(old_ast) if old_ast else set()
        new_imports = self._extract_imports(new_ast) if new_ast else set()
        
        # 新增导入
        for imp in new_imports - old_imports:
            changes.append(SemanticChange(
                change_type=ChangeType.ADDED,
                entity_type="import",
                entity_name=imp,
                old_signature=None,
                new_signature=None,
                risk_level=RiskLevel.LOW,
                description=f"Added import '{imp}'",
                affected_lines=[1],  # 导入通常在文件顶部
                impact_scope=[],
                suggested_actions=[]
            ))
        
        # 删除导入
        for imp in old_imports - new_imports:
            changes.append(SemanticChange(
                change_type=ChangeType.REMOVED,
                entity_type="import",
                entity_name=imp,
                old_signature=None,
                new_signature=None,
                risk_level=RiskLevel.MEDIUM,  # 删除导入可能有风险
                description=f"Removed import '{imp}'",
                affected_lines=[1],
                impact_scope=[],
                suggested_actions=["Verify that the import is no longer used"]
            ))
        
        return changes
    
    def _analyze_variable_changes(
        self,
        old_ast: Optional[ast.AST],
        new_ast: Optional[ast.AST],
        file_path: str
    ) -> List[SemanticChange]:
        """分析变量变更"""
        # 简化实现，实际可以更复杂
        return []
    
    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """提取导入语句"""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.add(f"{module}.{alias.name}" if module else alias.name)
        
        return imports
    
    def _signatures_equal(
        self, 
        sig1: FunctionSignature, 
        sig2: FunctionSignature
    ) -> bool:
        """比较函数签名是否相等"""
        return (
            sig1.name == sig2.name and
            sig1.parameters == sig2.parameters and
            sig1.return_type == sig2.return_type and
            sig1.is_async == sig2.is_async
        )
    
    def _describe_function_change(
        self, 
        old_sig: FunctionSignature, 
        new_sig: FunctionSignature
    ) -> str:
        """描述函数变更"""
        changes = []
        
        if old_sig.parameters != new_sig.parameters:
            changes.append("parameters changed")
        
        if old_sig.return_type != new_sig.return_type:
            changes.append("return type changed")
        
        if old_sig.is_async != new_sig.is_async:
            changes.append("async/await changed")
        
        if not changes:
            changes.append("implementation changed")
        
        return f"Function '{old_sig.name}': {', '.join(changes)}"
    
    def _calculate_risk_level(self, change: SemanticChange) -> RiskLevel:
        """计算风险级别"""
        risk_score = 0
        
        # 基于实体类型
        if change.entity_type == "function":
            risk_score += 1
        elif change.entity_type == "class":
            risk_score += 2
        
        # 基于变更类型
        if change.change_type == ChangeType.REMOVED:
            risk_score += 3
        elif change.change_type == ChangeType.MODIFIED:
            risk_score += 2
        
        # 基于名称模式（公共API风险更高）
        if change.entity_name.startswith('_'):
            risk_score -= 1  # 私有成员风险较低
        elif change.entity_name in ['__init__', '__str__', '__repr__']:
            risk_score += 2  # 特殊方法风险较高
        
        # 基于风险模式
        for pattern in self.risk_patterns:
            if re.search(pattern["regex"], change.entity_name):
                risk_score += pattern["risk_weight"]
        
        # 转换为风险级别
        if risk_score >= 5:
            return RiskLevel.CRITICAL
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _calculate_impact_scope(self, change: SemanticChange) -> List[str]:
        """计算影响范围"""
        impact_scope = [change.entity_name]
        
        # 如果有依赖服务，可以计算更精确的影响范围
        if self.dependency_service:
            try:
                # 这里需要根据实际的依赖服务API调整
                # dependents = self.dependency_service.get_dependents(change.entity_name)
                # impact_scope.extend(dependents)
                pass
            except Exception as e:
                logger.error(f"Failed to calculate impact scope: {e}")
        
        return impact_scope
    
    def _generate_suggested_actions(self, change: SemanticChange) -> List[str]:
        """生成建议操作"""
        actions = []
        
        if change.change_type == ChangeType.REMOVED:
            actions.append("Check for references to this entity")
            actions.append("Update documentation")
        
        if change.change_type == ChangeType.MODIFIED:
            actions.append("Review function signature changes")
            actions.append("Update calling code if needed")
        
        if change.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            actions.append("Consider adding unit tests")
            actions.append("Review with team before merging")
        
        if "security" in change.description.lower():
            actions.append("Security review required")
        
        return actions
    
    def _extract_function_calls(self, content: str, file_path: str) -> Set[str]:
        """提取函数调用"""
        calls = set()
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        # 处理 method calls
                        if hasattr(ast, 'unparse'):
                            calls.add(ast.unparse(node.func))
                        else:
                            calls.add(str(node.func))
        
        except Exception as e:
            logger.error(f"Failed to extract function calls from {file_path}: {e}")
        
        return calls
    
    def _get_caller_function(self, call: str, content: str) -> str:
        """获取调用者函数名"""
        # 简化实现，实际需要更复杂的AST分析
        return "unknown"
    
    def _assess_call_risk(self, call: str, change_type: ChangeType) -> RiskLevel:
        """评估调用风险"""
        # 基于调用名称评估风险
        high_risk_patterns = ['execute', 'eval', 'exec', 'open', 'file', 'db', 'sql']
        
        for pattern in high_risk_patterns:
            if pattern in call.lower():
                return RiskLevel.HIGH
        
        return RiskLevel.MEDIUM
    
    def _group_changes_by_lines(
        self, 
        changes: List[SemanticChange]
    ) -> Dict[Tuple[int, int], List[SemanticChange]]:
        """按行范围分组变更"""
        line_groups = {}
        
        for change in changes:
            if not change.affected_lines:
                continue
            
            line_range = (min(change.affected_lines), max(change.affected_lines))
            
            # 尝试合并相邻的变更
            merged = False
            for existing_range in line_groups:
                existing_start, existing_end = existing_range
                if line_range[0] <= existing_end + 5:  # 5行内认为是相邻
                    # 合并范围
                    new_range = (
                        min(existing_start, line_range[0]),
                        max(existing_end, line_range[1])
                    )
                    line_groups[new_range] = line_groups.pop(existing_range)
                    line_groups[new_range].append(change)
                    merged = True
                    break
            
            if not merged:
                line_groups[line_range] = [change]
        
        return line_groups
    
    def _generate_hotspot_recommendations(
        self,
        risk_level: RiskLevel,
        risk_factors: List[str],
        changes: List[SemanticChange]
    ) -> List[str]:
        """生成热区建议"""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("Immediate code review required")
            recommendations.append("Consider rollback if already deployed")
        
        if "Security-related change" in risk_factors:
            recommendations.append("Security team review required")
            recommendations.append("Add security tests")
        
        if "API interface change" in risk_factors:
            recommendations.append("Update API documentation")
            recommendations.append("Notify downstream consumers")
        
        if len(changes) > 5:
            recommendations.append("Consider splitting into smaller changes")
        
        return recommendations
    
    def _is_breaking_change(self, change: SemanticChange) -> bool:
        """判断是否为破坏性变更"""
        if change.change_type == ChangeType.REMOVED:
            return True
        
        if change.entity_type == "function" and change.change_type == ChangeType.MODIFIED:
            # 检查参数变更
            if change.old_signature and change.new_signature:
                return (
                    change.old_signature.parameters != change.new_signature.parameters or
                    change.old_signature.return_type != change.new_signature.return_type
                )
        
        return False
    
    def _load_risk_patterns(self) -> List[Dict[str, Any]]:
        """加载风险模式"""
        return [
            {"regex": r".*execute.*", "risk_weight": 3},
            {"regex": r".*eval.*", "risk_weight": 3},
            {"regex": r".*sql.*", "risk_weight": 2},
            {"regex": r".*auth.*", "risk_weight": 2},
            {"regex": r".*password.*", "risk_weight": 3},
            {"regex": r".*token.*", "risk_weight": 2},
            {"regex": r".*admin.*", "risk_weight": 2},
        ]