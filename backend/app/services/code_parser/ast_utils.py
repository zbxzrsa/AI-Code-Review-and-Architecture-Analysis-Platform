"""
AST标准化工具
将Tree-sitter生成的AST转换为标准化格式
"""
from typing import Dict, Any, List, Optional


def standardize_ast(raw_ast: Dict[str, Any], language: str) -> Dict[str, Any]:
    """
    将原始Tree-sitter AST转换为标准化格式
    
    Args:
        raw_ast: Tree-sitter生成的原始AST
        language: 编程语言
        
    Returns:
        标准化的AST
    """
    # 创建标准化AST的根节点
    standardized_ast = {
        "type": raw_ast.get("type", "program"),
        "language": language,
        "start_point": raw_ast.get("start_point", {"row": 0, "column": 0}),
        "end_point": raw_ast.get("end_point", {"row": 0, "column": 0}),
        "children": []
    }
    
    # 递归处理子节点
    if "children" in raw_ast:
        for child in raw_ast["children"]:
            standardized_child = _standardize_node(child, language)
            if standardized_child:
                standardized_ast["children"].append(standardized_child)
    
    return standardized_ast


def _standardize_node(node: Dict[str, Any], language: str) -> Optional[Dict[str, Any]]:
    """
    标准化单个AST节点
    
    Args:
        node: Tree-sitter AST节点
        language: 编程语言
        
    Returns:
        标准化的节点
    """
    if not node:
        return None
    
    # 创建标准化节点
    standardized_node = {
        "type": node.get("type", "unknown"),
        "start_point": node.get("start_point", {"row": 0, "column": 0}),
        "end_point": node.get("end_point", {"row": 0, "column": 0}),
    }
    
    # 添加节点值（如果存在）
    if "value" in node:
        standardized_node["value"] = node["value"]
    
    # 递归处理子节点
    if "children" in node:
        standardized_node["children"] = []
        for child in node["children"]:
            standardized_child = _standardize_node(child, language)
            if standardized_child:
                standardized_node["children"].append(standardized_child)
    
    return standardized_node


def get_node_by_type(ast: Dict[str, Any], node_type: str) -> List[Dict[str, Any]]:
    """
    从AST中获取指定类型的所有节点
    
    Args:
        ast: 抽象语法树
        node_type: 节点类型
        
    Returns:
        指定类型的节点列表
    """
    result = []
    
    # 检查当前节点
    if ast.get("type") == node_type:
        result.append(ast)
    
    # 递归检查子节点
    if "children" in ast:
        for child in ast["children"]:
            result.extend(get_node_by_type(child, node_type))
    
    return result


def get_node_by_location(ast: Dict[str, Any], row: int, column: int) -> Optional[Dict[str, Any]]:
    """
    从AST中获取指定位置的节点
    
    Args:
        ast: 抽象语法树
        row: 行号
        column: 列号
        
    Returns:
        指定位置的节点，如果不存在则返回None
    """
    # 检查当前节点是否包含指定位置
    start_point = ast.get("start_point", {"row": 0, "column": 0})
    end_point = ast.get("end_point", {"row": 0, "column": 0})
    
    if not _is_point_in_range(row, column, start_point, end_point):
        return None
    
    # 检查子节点
    if "children" in ast:
        for child in ast["children"]:
            node = get_node_by_location(child, row, column)
            if node:
                return node
    
    # 如果没有子节点包含指定位置，则返回当前节点
    return ast


def _is_point_in_range(row: int, column: int, start_point: Dict[str, int], end_point: Dict[str, int]) -> bool:
    """
    检查指定点是否在范围内
    
    Args:
        row: 行号
        column: 列号
        start_point: 起始点
        end_point: 结束点
        
    Returns:
        如果点在范围内则返回True，否则返回False
    """
    start_row = start_point.get("row", 0)
    start_col = start_point.get("column", 0)
    end_row = end_point.get("row", 0)
    end_col = end_point.get("column", 0)
    
    # 检查点是否在范围内
    if start_row < row < end_row:
        return True
    elif start_row == row and row == end_row:
        return start_col <= column <= end_col
    elif start_row == row:
        return start_col <= column
    elif row == end_row:
        return column <= end_col
    
    return False


def get_ast_depth(ast: Dict[str, Any]) -> int:
    """
    计算AST的深度
    
    Args:
        ast: 抽象语法树
        
    Returns:
        AST的深度
    """
    if not ast:
        return 0
    
    # 如果没有子节点，深度为1
    if "children" not in ast or not ast["children"]:
        return 1
    
    # 递归计算子节点的最大深度
    max_child_depth = 0
    for child in ast["children"]:
        child_depth = get_ast_depth(child)
        max_child_depth = max(max_child_depth, child_depth)
    
    # 当前节点的深度为最大子节点深度加1
    return max_child_depth + 1


def get_ast_size(ast: Dict[str, Any]) -> int:
    """
    计算AST的大小（节点数量）
    
    Args:
        ast: 抽象语法树
        
    Returns:
        AST的节点数量
    """
    if not ast:
        return 0
    
    # 当前节点计数为1
    count = 1
    
    # 递归计算子节点数量
    if "children" in ast:
        for child in ast["children"]:
            count += get_ast_size(child)
    
    return count


def ast_to_string(ast: Dict[str, Any], indent: int = 0) -> str:
    """
    将AST转换为字符串表示
    
    Args:
        ast: 抽象语法树
        indent: 缩进级别
        
    Returns:
        AST的字符串表示
    """
    if not ast:
        return ""
    
    # 创建缩进
    indent_str = "  " * indent
    
    # 构建节点字符串
    node_str = f"{indent_str}{ast.get('type', 'unknown')}"
    
    # 添加值（如果存在）
    if "value" in ast:
        node_str += f": {ast['value']}"
    
    # 添加位置信息
    start_point = ast.get("start_point", {"row": 0, "column": 0})
    end_point = ast.get("end_point", {"row": 0, "column": 0})
    node_str += f" ({start_point.get('row', 0)}:{start_point.get('column', 0)} - {end_point.get('row', 0)}:{end_point.get('column', 0)})"
    
    # 递归处理子节点
    if "children" in ast and ast["children"]:
        node_str += "\n"
        for child in ast["children"]:
            node_str += ast_to_string(child, indent + 1)
    else:
        node_str += "\n"
    
    return node_str