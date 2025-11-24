"""
DSL规则管理API端点
提供规则的CRUD操作和执行功能
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body, Depends
from pydantic import BaseModel, Field

from app.services.dsl_rules import (
    rule_manager, 
    dsl_engine,
    DSLRule,
    RuleType,
    Severity,
    RuleContext,
    RuleViolation
)
from app.core.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rules", tags=["dsl-rules"])


class RuleCreateRequest(BaseModel):
    """创建规则请求"""
    rule_id: str = Field(..., description="Rule ID")
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    rule_type: RuleType = Field(..., description="Rule type")
    severity: Severity = Field(..., description="Rule severity")
    enabled: bool = Field(True, description="Whether rule is enabled")
    tags: List[str] = Field(default_factory=list, description="Rule tags")
    dsl_code: str = Field(..., description="DSL code")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RuleUpdateRequest(BaseModel):
    """更新规则请求"""
    name: Optional[str] = Field(None, description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    rule_type: Optional[RuleType] = Field(None, description="Rule type")
    severity: Optional[Severity] = Field(None, description="Rule severity")
    enabled: Optional[bool] = Field(None, description="Whether rule is enabled")
    tags: Optional[List[str]] = Field(None, description="Rule tags")
    dsl_code: Optional[str] = Field(None, description="DSL code")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RuleExecuteRequest(BaseModel):
    """执行规则请求"""
    file_path: str = Field(..., description="File path to analyze")
    file_content: str = Field(..., description="File content")
    language: str = Field("python", description="Programming language")
    rule_ids: Optional[List[str]] = Field(None, description="Specific rule IDs to execute")
    rule_types: Optional[List[RuleType]] = Field(None, description="Rule types to execute")


class DSLValidationRequest(BaseModel):
    """DSL验证请求"""
    dsl_code: str = Field(..., description="DSL code to validate")


@router.get("/", response_model=List[Dict[str, Any]])
async def list_rules(
    rule_type: Optional[RuleType] = Query(None, description="Filter by rule type"),
    enabled_only: bool = Query(False, description="Only return enabled rules"),
    current_user: Dict = Depends(get_current_user)
):
    """获取规则列表"""
    if rule_type:
        rules = rule_manager.get_rules_by_type(rule_type)
    else:
        rules = list(rule_manager.rules.values())
    
    if enabled_only:
        rules = [rule for rule in rules if rule.enabled]
    
    return [rule.to_dict() for rule in rules]


@router.get("/{rule_id}", response_model=Dict[str, Any])
async def get_rule(
    rule_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """获取单个规则"""
    rule = rule_manager.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    return rule.to_dict()


@router.post("/", response_model=Dict[str, Any])
async def create_rule(
    request: RuleCreateRequest,
    current_user: Dict = Depends(get_current_user)
):
    """创建新规则"""
    # 检查规则ID是否已存在
    if rule_manager.get_rule(request.rule_id):
        raise HTTPException(status_code=409, detail="Rule ID already exists")
    
    # 验证DSL代码
    validation_errors = dsl_engine.validate_dsl_code(request.dsl_code)
    if validation_errors:
        raise HTTPException(
            status_code=400, 
            detail=f"DSL validation failed: {'; '.join(validation_errors)}"
        )
    
    # 创建规则
    rule = DSLRule(
        rule_id=request.rule_id,
        name=request.name,
        description=request.description,
        rule_type=request.rule_type,
        severity=request.severity,
        enabled=request.enabled,
        tags=request.tags,
        dsl_code=request.dsl_code,
        metadata=request.metadata
    )
    
    rule_manager.add_rule(rule)
    
    logger.info(f"User {current_user.get('id')} created rule {request.rule_id}")
    
    return {
        "message": "Rule created successfully",
        "rule": rule.to_dict()
    }


@router.put("/{rule_id}", response_model=Dict[str, Any])
async def update_rule(
    rule_id: str,
    request: RuleUpdateRequest,
    current_user: Dict = Depends(get_current_user)
):
    """更新规则"""
    rule = rule_manager.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    # 更新字段
    if request.name is not None:
        rule.name = request.name
    if request.description is not None:
        rule.description = request.description
    if request.rule_type is not None:
        rule.rule_type = request.rule_type
    if request.severity is not None:
        rule.severity = request.severity
    if request.enabled is not None:
        rule.enabled = request.enabled
    if request.tags is not None:
        rule.tags = request.tags
    if request.dsl_code is not None:
        # 验证新的DSL代码
        validation_errors = dsl_engine.validate_dsl_code(request.dsl_code)
        if validation_errors:
            raise HTTPException(
                status_code=400, 
                detail=f"DSL validation failed: {'; '.join(validation_errors)}"
            )
        rule.dsl_code = request.dsl_code
    if request.metadata is not None:
        rule.metadata = request.metadata
    
    logger.info(f"User {current_user.get('id')} updated rule {rule_id}")
    
    return {
        "message": "Rule updated successfully",
        "rule": rule.to_dict()
    }


@router.delete("/{rule_id}", response_model=Dict[str, Any])
async def delete_rule(
    rule_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """删除规则"""
    if not rule_manager.remove_rule(rule_id):
        raise HTTPException(status_code=404, detail="Rule not found")
    
    logger.info(f"User {current_user.get('id')} deleted rule {rule_id}")
    
    return {"message": "Rule deleted successfully"}


@router.post("/{rule_id}/enable", response_model=Dict[str, Any])
async def enable_rule(
    rule_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """启用规则"""
    if not rule_manager.enable_rule(rule_id):
        raise HTTPException(status_code=404, detail="Rule not found")
    
    logger.info(f"User {current_user.get('id')} enabled rule {rule_id}")
    
    return {"message": "Rule enabled successfully"}


@router.post("/{rule_id}/disable", response_model=Dict[str, Any])
async def disable_rule(
    rule_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """禁用规则"""
    if not rule_manager.disable_rule(rule_id):
        raise HTTPException(status_code=404, detail="Rule not found")
    
    logger.info(f"User {current_user.get('id')} disabled rule {rule_id}")
    
    return {"message": "Rule disabled successfully"}


@router.post("/execute", response_model=Dict[str, Any])
async def execute_rules(
    request: RuleExecuteRequest,
    current_user: Dict = Depends(get_current_user)
):
    """执行规则"""
    try:
        import ast
        
        # 解析AST
        ast_tree = None
        try:
            ast_tree = ast.parse(request.file_content, type_comments=True)
        except SyntaxError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Syntax error in file content: {e}"
            )
        
        # 创建规则上下文
        context = RuleContext(
            file_path=request.file_path,
            file_content=request.file_content,
            ast_tree=ast_tree,
            language=request.language
        )
        
        # 执行规则
        violations = rule_manager.execute_rules(
            context=context,
            rule_ids=request.rule_ids,
            rule_types=request.rule_types
        )
        
        # 转换违规为字典
        violation_dicts = []
        for violation in violations:
            violation_dict = {
                "rule_id": violation.rule_id,
                "rule_name": violation.rule_name,
                "severity": violation.severity.value,
                "message": violation.message,
                "line_number": violation.line_number,
                "column_number": violation.column_number,
                "end_line_number": violation.end_line_number,
                "end_column_number": violation.end_column_number,
                "suggestion": violation.suggestion,
                "metadata": violation.metadata
            }
            violation_dicts.append(violation_dict)
        
        logger.info(f"User {current_user.get('id')} executed rules on {request.file_path}")
        
        return {
            "file_path": request.file_path,
            "violations": violation_dicts,
            "total_violations": len(violation_dicts),
            "violations_by_severity": {
                severity.value: len([v for v in violation_dicts if v["severity"] == severity.value])
                for severity in Severity
            }
        }
        
    except Exception as e:
        logger.error(f"Error executing rules: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing rules: {str(e)}")


@router.post("/validate", response_model=Dict[str, Any])
async def validate_dsl(
    request: DSLValidationRequest,
    current_user: Dict = Depends(get_current_user)
):
    """验证DSL代码"""
    validation_errors = dsl_engine.validate_dsl_code(request.dsl_code)
    
    return {
        "is_valid": len(validation_errors) == 0,
        "errors": validation_errors
    }


@router.get("/functions/list", response_model=List[Dict[str, str]])
async def list_dsl_functions(
    current_user: Dict = Depends(get_current_user)
):
    """获取可用的DSL函数列表"""
    return dsl_engine.get_function_list()


@router.get("/statistics", response_model=Dict[str, Any])
async def get_rule_statistics(
    current_user: Dict = Depends(get_current_user)
):
    """获取规则统计信息"""
    return rule_manager.get_rule_statistics()


@router.post("/import", response_model=Dict[str, Any])
async def import_rules(
    file_content: str = Body(..., description="Rule file content (JSON or YAML)"),
    current_user: Dict = Depends(get_current_user)
):
    """导入规则"""
    try:
        import json
        import yaml
        from io import StringIO
        
        # 尝试解析为JSON
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError:
            # 尝试解析为YAML
            try:
                data = yaml.safe_load(StringIO(file_content))
            except yaml.YAMLError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file format. Must be valid JSON or YAML."
                )
        
        # 提取规则数据
        if isinstance(data, list):
            rules_data = data
        elif isinstance(data, dict) and 'rules' in data:
            rules_data = data['rules']
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid rule file format. Expected list of rules or object with 'rules' key."
            )
        
        # 导入规则
        imported_count = 0
        errors = []
        
        for rule_data in rules_data:
            try:
                rule = DSLRule.from_dict(rule_data)
                
                # 检查是否已存在
                if rule_manager.get_rule(rule.rule_id):
                    errors.append(f"Rule {rule.rule_id} already exists, skipping")
                    continue
                
                # 验证DSL代码
                validation_errors = dsl_engine.validate_dsl_code(rule.dsl_code)
                if validation_errors:
                    errors.append(f"Rule {rule.rule_id} has invalid DSL: {'; '.join(validation_errors)}")
                    continue
                
                rule_manager.add_rule(rule)
                imported_count += 1
                
            except Exception as e:
                errors.append(f"Error importing rule: {str(e)}")
        
        logger.info(f"User {current_user.get('id')} imported {imported_count} rules")
        
        return {
            "imported_count": imported_count,
            "total_rules": len(rules_data),
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Error importing rules: {e}")
        raise HTTPException(status_code=500, detail=f"Error importing rules: {str(e)}")


@router.get("/export", response_model=Dict[str, Any])
async def export_rules(
    rule_ids: Optional[str] = Query(None, description="Comma-separated rule IDs to export"),
    format: str = Query("json", description="Export format (json or yaml)"),
    current_user: Dict = Depends(get_current_user)
):
    """导出规则"""
    try:
        import json
        import yaml
        
        # 确定要导出的规则
        target_rule_ids = None
        if rule_ids:
            target_rule_ids = [rule_id.strip() for rule_id in rule_ids.split(",")]
        
        # 获取规则数据
        if target_rule_ids:
            rules = [
                rule_manager.rules[rule_id].to_dict()
                for rule_id in target_rule_ids
                if rule_id in rule_manager.rules
            ]
        else:
            rules = [rule.to_dict() for rule in rule_manager.rules.values()]
        
        # 格式化输出
        export_data = {"rules": rules}
        
        if format.lower() == "yaml":
            content = yaml.dump(export_data, default_flow_style=False, indent=2)
            media_type = "application/x-yaml"
        else:
            content = json.dumps(export_data, indent=2)
            media_type = "application/json"
        
        logger.info(f"User {current_user.get('id')} exported {len(rules)} rules")
        
        from fastapi.responses import Response
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=rules.{format.lower()}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting rules: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting rules: {str(e)}")


@router.get("/examples", response_model=List[Dict[str, Any]])
async def get_example_rules(
    current_user: Dict = Depends(get_current_user)
):
    """获取示例规则"""
    from app.services.dsl_rules import EXAMPLE_RULES
    return EXAMPLE_RULES