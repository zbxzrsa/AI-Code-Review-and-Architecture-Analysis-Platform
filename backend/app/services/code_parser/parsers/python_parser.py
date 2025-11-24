"""
Python代码解析器
实现Python代码的解析、AST生成、CFG/DFG提取和度量计算
"""
import asyncio
from typing import Dict, List, Any, Optional, Set
import logging

from ..parser import BaseParser, FeatureType
from ..tree_sitter_integration import ts_parser
from ..ast_utils import standardize_ast
from ..metrics import calculate_cyclomatic_complexity, count_lines_of_code

# 配置日志
logger = logging.getLogger(__name__)

class PythonParser(BaseParser):
    """Python代码解析器"""
    
    async def parse(self, code: str, features: List[FeatureType]) -> Dict[str, Any]:
        """
        解析Python代码并提取请求的特征
        
        Args:
            code: Python源代码
            features: 需要提取的特征列表
            
        Returns:
            包含请求特征的字典
        """
        result = {}
        
        # 生成AST
        if FeatureType.AST in features or FeatureType.CFG in features or FeatureType.DFG in features or FeatureType.METRICS in features:
            ast = await self.generate_ast(code)
            if FeatureType.AST in features:
                result[FeatureType.AST] = ast
        
        # 生成CFG
        if FeatureType.CFG in features or FeatureType.DFG in features:
            cfg = await self.generate_cfg(ast)
            if FeatureType.CFG in features:
                result[FeatureType.CFG] = cfg
        
        # 生成DFG
        if FeatureType.DFG in features:
            dfg = await self.generate_dfg(ast, cfg if FeatureType.CFG in features or FeatureType.DFG in features else None)
            result[FeatureType.DFG] = dfg
        
        # 计算度量指标
        if FeatureType.METRICS in features:
            metrics = await self.calculate_metrics(ast, cfg if FeatureType.CFG in features or FeatureType.METRICS in features else None)
            result[FeatureType.METRICS] = metrics
        
        return result
    
    async def generate_ast(self, code: str) -> Dict[str, Any]:
        """
        生成Python代码的抽象语法树
        
        Args:
            code: Python源代码
            
        Returns:
            标准化的AST字典
        """
        try:
            # 使用Tree-sitter解析代码
            raw_ast = await ts_parser.parse_code(code, "python")
            
            # 标准化AST
            standardized_ast = standardize_ast(raw_ast, "python")
            
            return standardized_ast
        except Exception as e:
            logger.error(f"生成AST失败: {str(e)}")
            raise ValueError(f"生成AST失败: {str(e)}")
    
    async def generate_cfg(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成Python代码的控制流图
        
        Args:
            ast: 抽象语法树
            
        Returns:
            控制流图字典
        """
        try:
            # 初始化CFG
            cfg = {
                "nodes": [],
                "edges": [],
                "entry": None,
                "exit": None
            }
            
            # 提取控制流节点和边
            await self._extract_control_flow(ast, cfg)
            
            return cfg
        except Exception as e:
            logger.error(f"生成CFG失败: {str(e)}")
            raise ValueError(f"生成CFG失败: {str(e)}")
    
    async def _extract_control_flow(self, ast: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        """
        从AST提取控制流信息
        
        Args:
            ast: 抽象语法树
            cfg: 控制流图
        """
        # 创建入口节点
        entry_id = len(cfg["nodes"])
        cfg["nodes"].append({
            "id": entry_id,
            "type": "entry",
            "location": ast.get("start_point", {"row": 0, "column": 0})
        })
        cfg["entry"] = entry_id
        
        # 创建出口节点
        exit_id = len(cfg["nodes"])
        cfg["nodes"].append({
            "id": exit_id,
            "type": "exit",
            "location": ast.get("end_point", {"row": 0, "column": 0})
        })
        cfg["exit"] = exit_id
        
        # 处理AST节点
        current_id = entry_id
        
        # 递归处理AST节点
        if "children" in ast:
            for child in ast["children"]:
                node_type = child.get("type", "")
                
                # 处理控制流结构
                if node_type in ["if_statement", "for_statement", "while_statement", "try_statement"]:
                    # 创建条件/循环节点
                    cond_id = len(cfg["nodes"])
                    cfg["nodes"].append({
                        "id": cond_id,
                        "type": node_type,
                        "location": child.get("start_point")
                    })
                    
                    # 连接当前节点到条件/循环节点
                    cfg["edges"].append({
                        "source": current_id,
                        "target": cond_id,
                        "type": "flow"
                    })
                    
                    # 更新当前节点
                    current_id = cond_id
                
                # 处理函数定义
                elif node_type == "function_definition":
                    # 创建函数节点
                    func_id = len(cfg["nodes"])
                    cfg["nodes"].append({
                        "id": func_id,
                        "type": "function",
                        "name": self._get_function_name(child),
                        "location": child.get("start_point")
                    })
                    
                    # 连接当前节点到函数节点
                    cfg["edges"].append({
                        "source": current_id,
                        "target": func_id,
                        "type": "flow"
                    })
                    
                    # 递归处理函数体
                    if "children" in child:
                        for func_child in child["children"]:
                            if func_child.get("type") == "block":
                                await self._extract_control_flow(func_child, cfg)
                    
                    # 连接函数节点到出口节点
                    cfg["edges"].append({
                        "source": func_id,
                        "target": exit_id,
                        "type": "flow"
                    })
        
        # 连接最后一个节点到出口节点
        if current_id != exit_id:
            cfg["edges"].append({
                "source": current_id,
                "target": exit_id,
                "type": "flow"
            })
    
    def _get_function_name(self, node: Dict[str, Any]) -> str:
        """
        从函数定义节点获取函数名
        
        Args:
            node: 函数定义节点
            
        Returns:
            函数名
        """
        if "children" in node:
            for child in node["children"]:
                if child.get("type") == "identifier":
                    return child.get("value", "anonymous")
        
        return "anonymous"
    
    async def generate_dfg(self, ast: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成Python代码的数据流图
        
        Args:
            ast: 抽象语法树
            cfg: 控制流图（可选）
            
        Returns:
            数据流图字典
        """
        try:
            # 初始化DFG
            dfg = {
                "nodes": [],
                "edges": []
            }
            
            # 提取变量定义和使用
            variables = {}
            await self._extract_variables(ast, variables)
            
            # 创建变量节点
            for var_name, var_info in variables.items():
                var_id = len(dfg["nodes"])
                dfg["nodes"].append({
                    "id": var_id,
                    "type": "variable",
                    "name": var_name,
                    "locations": var_info["locations"]
                })
                
                # 添加定义-使用边
                for def_loc in var_info["definitions"]:
                    for use_loc in var_info["uses"]:
                        dfg["edges"].append({
                            "source": var_id,
                            "target": var_id,
                            "type": "def-use",
                            "def_location": def_loc,
                            "use_location": use_loc
                        })
            
            return dfg
        except Exception as e:
            logger.error(f"生成DFG失败: {str(e)}")
            raise ValueError(f"生成DFG失败: {str(e)}")
    
    async def _extract_variables(self, node: Dict[str, Any], variables: Dict[str, Dict[str, Any]]) -> None:
        """
        从AST提取变量定义和使用
        
        Args:
            node: AST节点
            variables: 变量信息字典
        """
        node_type = node.get("type", "")
        
        # 处理变量定义
        if node_type in ["assignment", "augmented_assignment"]:
            if "children" in node:
                for child in node["children"]:
                    if child.get("type") == "identifier":
                        var_name = child.get("value", "")
                        if var_name:
                            if var_name not in variables:
                                variables[var_name] = {
                                    "definitions": [],
                                    "uses": [],
                                    "locations": []
                                }
                            
                            variables[var_name]["definitions"].append(child.get("start_point"))
                            variables[var_name]["locations"].append(child.get("start_point"))
        
        # 处理变量使用
        elif node_type == "identifier":
            var_name = node.get("value", "")
            if var_name:
                if var_name not in variables:
                    variables[var_name] = {
                        "definitions": [],
                        "uses": [],
                        "locations": []
                    }
                
                variables[var_name]["uses"].append(node.get("start_point"))
                variables[var_name]["locations"].append(node.get("start_point"))
        
        # 递归处理子节点
        if "children" in node:
            for child in node["children"]:
                await self._extract_variables(child, variables)
    
    async def calculate_metrics(self, ast: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        计算Python代码的度量指标
        
        Args:
            ast: 抽象语法树
            cfg: 控制流图（可选）
            
        Returns:
            度量指标字典
        """
        try:
            metrics = {}
            
            # 计算圈复杂度
            if cfg:
                metrics["cyclomatic_complexity"] = calculate_cyclomatic_complexity(cfg)
            else:
                metrics["cyclomatic_complexity"] = await self._calculate_cyclomatic_complexity_from_ast(ast)
            
            # 计算代码行数
            metrics["loc"] = count_lines_of_code(ast)
            
            # 计算函数数量
            metrics["function_count"] = await self._count_functions(ast)
            
            # 计算类数量
            metrics["class_count"] = await self._count_classes(ast)
            
            return metrics
        except Exception as e:
            logger.error(f"计算度量指标失败: {str(e)}")
            raise ValueError(f"计算度量指标失败: {str(e)}")
    
    async def _calculate_cyclomatic_complexity_from_ast(self, ast: Dict[str, Any]) -> int:
        """
        从AST计算圈复杂度
        
        Args:
            ast: 抽象语法树
            
        Returns:
            圈复杂度
        """
        complexity = 1  # 基础复杂度
        
        # 递归计算复杂度
        await self._calculate_complexity_recursive(ast, complexity)
        
        return complexity
    
    async def _calculate_complexity_recursive(self, node: Dict[str, Any], complexity: int) -> int:
        """
        递归计算复杂度
        
        Args:
            node: AST节点
            complexity: 当前复杂度
            
        Returns:
            更新后的复杂度
        """
        node_type = node.get("type", "")
        
        # 增加复杂度的节点类型
        if node_type in ["if_statement", "elif_clause", "for_statement", "while_statement", "except_clause"]:
            complexity += 1
        
        # 递归处理子节点
        if "children" in node:
            for child in node["children"]:
                complexity = await self._calculate_complexity_recursive(child, complexity)
        
        return complexity
    
    async def _count_functions(self, ast: Dict[str, Any]) -> int:
        """
        计算函数数量
        
        Args:
            ast: 抽象语法树
            
        Returns:
            函数数量
        """
        count = 0
        
        # 检查当前节点
        if ast.get("type") == "function_definition":
            count += 1
        
        # 递归处理子节点
        if "children" in ast:
            for child in ast["children"]:
                count += await self._count_functions(child)
        
        return count
    
    async def _count_classes(self, ast: Dict[str, Any]) -> int:
        """
        计算类数量
        
        Args:
            ast: 抽象语法树
            
        Returns:
            类数量
        """
        count = 0
        
        # 检查当前节点
        if ast.get("type") == "class_definition":
            count += 1
        
        # 递归处理子节点
        if "children" in ast:
            for child in ast["children"]:
                count += await self._count_classes(child)
        
        return count