"""
代码度量计算模块
计算代码的各种度量指标，如圈复杂度、代码行数等
"""
from typing import Dict, Any, List


def calculate_cyclomatic_complexity(cfg: Dict[str, Any]) -> int:
    """
    计算代码的圈复杂度
    
    圈复杂度 = 边数 - 节点数 + 2 * 连通分量数
    对于单一连通图，公式简化为：圈复杂度 = 边数 - 节点数 + 2
    
    Args:
        cfg: 控制流图
        
    Returns:
        圈复杂度
    """
    # 获取边数和节点数
    edges = cfg.get("edges", [])
    nodes = cfg.get("nodes", [])
    
    # 计算圈复杂度
    # 假设图是连通的，连通分量数为1
    return len(edges) - len(nodes) + 2


def count_lines_of_code(ast: Dict[str, Any]) -> Dict[str, int]:
    """
    计算代码行数
    
    Args:
        ast: 抽象语法树
        
    Returns:
        包含不同类型代码行数的字典
    """
    # 获取AST的起始和结束位置
    start_point = ast.get("start_point", {"row": 0, "column": 0})
    end_point = ast.get("end_point", {"row": 0, "column": 0})
    
    # 计算总行数
    total_lines = end_point.get("row", 0) - start_point.get("row", 0) + 1
    
    # 计算注释行数（简化实现，实际应该分析AST中的注释节点）
    comment_lines = _count_comment_lines(ast)
    
    # 计算空行数（简化实现，实际应该分析源代码）
    blank_lines = _count_blank_lines(ast)
    
    # 计算代码行数
    code_lines = total_lines - comment_lines - blank_lines
    
    return {
        "total": total_lines,
        "code": code_lines,
        "comment": comment_lines,
        "blank": blank_lines
    }


def _count_comment_lines(ast: Dict[str, Any]) -> int:
    """
    计算注释行数
    
    Args:
        ast: 抽象语法树
        
    Returns:
        注释行数
    """
    comment_count = 0
    
    # 检查当前节点是否为注释
    if ast.get("type") in ["comment", "block_comment", "line_comment"]:
        # 获取注释的起始和结束位置
        start_point = ast.get("start_point", {"row": 0, "column": 0})
        end_point = ast.get("end_point", {"row": 0, "column": 0})
        
        # 计算注释行数
        comment_count += end_point.get("row", 0) - start_point.get("row", 0) + 1
    
    # 递归处理子节点
    if "children" in ast:
        for child in ast["children"]:
            comment_count += _count_comment_lines(child)
    
    return comment_count


def _count_blank_lines(ast: Dict[str, Any]) -> int:
    """
    计算空行数（简化实现）
    
    Args:
        ast: 抽象语法树
        
    Returns:
        空行数
    """
    # 简化实现，实际应该分析源代码
    # 这里假设空行数为总行数的10%
    start_point = ast.get("start_point", {"row": 0, "column": 0})
    end_point = ast.get("end_point", {"row": 0, "column": 0})
    total_lines = end_point.get("row", 0) - start_point.get("row", 0) + 1
    
    return int(total_lines * 0.1)


def calculate_halstead_metrics(ast: Dict[str, Any]) -> Dict[str, float]:
    """
    计算Halstead复杂度度量
    
    Args:
        ast: 抽象语法树
        
    Returns:
        Halstead度量指标字典
    """
    # 计算操作符和操作数
    operators = set()
    operands = set()
    
    # 提取操作符和操作数
    _extract_operators_and_operands(ast, operators, operands)
    
    # 计算Halstead度量
    n1 = len(operators)  # 不同操作符数量
    n2 = len(operands)   # 不同操作数数量
    N1 = sum(1 for _ in _get_all_operators(ast))  # 操作符总数
    N2 = sum(1 for _ in _get_all_operands(ast))   # 操作数总数
    
    # 计算程序长度
    program_length = N1 + N2
    
    # 计算程序词汇量
    vocabulary = n1 + n2
    
    # 计算程序体积
    volume = program_length * (vocabulary.bit_length() if vocabulary > 0 else 0)
    
    # 计算难度
    difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
    
    # 计算工作量
    effort = difficulty * volume
    
    return {
        "n1": n1,
        "n2": n2,
        "N1": N1,
        "N2": N2,
        "program_length": program_length,
        "vocabulary": vocabulary,
        "volume": volume,
        "difficulty": difficulty,
        "effort": effort
    }


def _extract_operators_and_operands(ast: Dict[str, Any], operators: set, operands: set) -> None:
    """
    从AST中提取操作符和操作数
    
    Args:
        ast: 抽象语法树
        operators: 操作符集合
        operands: 操作数集合
    """
    node_type = ast.get("type", "")
    
    # 检查节点类型
    if node_type in ["binary_operator", "unary_operator", "assignment_operator"]:
        # 添加操作符
        if "value" in ast:
            operators.add(ast["value"])
    elif node_type in ["identifier", "number", "string", "boolean"]:
        # 添加操作数
        if "value" in ast:
            operands.add(ast["value"])
    
    # 递归处理子节点
    if "children" in ast:
        for child in ast["children"]:
            _extract_operators_and_operands(child, operators, operands)


def _get_all_operators(ast: Dict[str, Any]) -> List[str]:
    """
    获取AST中的所有操作符
    
    Args:
        ast: 抽象语法树
        
    Returns:
        操作符列表
    """
    operators = []
    
    # 检查节点类型
    if ast.get("type") in ["binary_operator", "unary_operator", "assignment_operator"]:
        # 添加操作符
        if "value" in ast:
            operators.append(ast["value"])
    
    # 递归处理子节点
    if "children" in ast:
        for child in ast["children"]:
            operators.extend(_get_all_operators(child))
    
    return operators


def _get_all_operands(ast: Dict[str, Any]) -> List[str]:
    """
    获取AST中的所有操作数
    
    Args:
        ast: 抽象语法树
        
    Returns:
        操作数列表
    """
    operands = []
    
    # 检查节点类型
    if ast.get("type") in ["identifier", "number", "string", "boolean"]:
        # 添加操作数
        if "value" in ast:
            operands.append(ast["value"])
    
    # 递归处理子节点
    if "children" in ast:
        for child in ast["children"]:
            operands.extend(_get_all_operands(child))
    
    return operands


def calculate_maintainability_index(ast: Dict[str, Any], cyclomatic_complexity: int, halstead_volume: float) -> float:
    """
    计算可维护性指数
    
    可维护性指数 = 171 - 5.2 * ln(Halstead Volume) - 0.23 * (Cyclomatic Complexity) - 16.2 * ln(Lines of Code)
    
    Args:
        ast: 抽象语法树
        cyclomatic_complexity: 圈复杂度
        halstead_volume: Halstead体积
        
    Returns:
        可维护性指数
    """
    import math
    
    # 计算代码行数
    loc = count_lines_of_code(ast)["code"]
    
    # 计算可维护性指数
    maintainability_index = 171 - 5.2 * math.log(halstead_volume) - 0.23 * cyclomatic_complexity - 16.2 * math.log(loc)
    
    # 归一化到0-100范围
    normalized_mi = max(0, min(100, maintainability_index * 100 / 171))
    
    return normalized_mi