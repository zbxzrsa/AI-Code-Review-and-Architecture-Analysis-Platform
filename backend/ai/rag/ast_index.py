"""
AST-based RAG indexing for intelligent code retrieval.
Indexes code structure, functions, classes, and relationships for better context retrieval.
"""

import ast
import os
import json
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import pickle

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of AST nodes for indexing."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    DECORATOR = "decorator"
    CALL = "call"
    ASSIGNMENT = "assignment"
    LOOP = "loop"
    CONDITION = "condition"
    TRY_EXCEPT = "try_except"
    WITH = "with"


@dataclass
class CodeNode:
    """Represents a code element extracted from AST."""
    node_type: NodeType
    name: str
    line_number: int
    end_line_number: int
    file_path: str
    content: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    hash_id: Optional[str] = None
    
    def __post_init__(self):
        if self.hash_id is None:
            self.hash_id = self._generate_hash()
    
    def _generate_hash(self) -> str:
        """Generate unique hash for this node."""
        content = f"{self.node_type.value}:{self.name}:{self.file_path}:{self.line_number}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Relationship:
    """Represents relationships between code nodes."""
    source_id: str
    target_id: str
    relationship_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ASTIndexer:
    """Indexes Python code using AST analysis."""
    
    def __init__(self):
        self.nodes: Dict[str, CodeNode] = {}
        self.relationships: List[Relationship] = []
        self.file_index: Dict[str, List[str]] = {}
        self.type_index: Dict[NodeType, List[str]] = {}
        self.name_index: Dict[str, str] = {}  # name -> node_id
    
    def index_file(self, file_path: str) -> List[CodeNode]:
        """Index a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract nodes
            nodes = self._extract_nodes(tree, file_path, content)
            
            # Index nodes
            for node in nodes:
                self.nodes[node.hash_id] = node
                
                # Update file index
                if file_path not in self.file_index:
                    self.file_index[file_path] = []
                self.file_index[file_path].append(node.hash_id)
                
                # Update type index
                if node.node_type not in self.type_index:
                    self.type_index[node.node_type] = []
                self.type_index[node.node_type].append(node.hash_id)
                
                # Update name index
                key = f"{node.node_type.value}:{node.name}"
                self.name_index[key] = node.hash_id
            
            # Extract relationships
            relationships = self._extract_relationships(tree, nodes)
            self.relationships.extend(relationships)
            
            logger.info(f"Indexed {len(nodes)} nodes from {file_path}")
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            return []
    
    def index_directory(self, directory_path: str, pattern: str = "*.py") -> Dict[str, Any]:
        """Index all Python files in a directory."""
        indexed_files = 0
        total_nodes = 0
        
        for root, dirs, files in os.walk(directory_path):
            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    nodes = self.index_file(file_path)
                    indexed_files += 1
                    total_nodes += len(nodes)
        
        stats = {
            "indexed_files": indexed_files,
            "total_nodes": total_nodes,
            "directory": directory_path,
            "node_types": {node_type.value: len(nodes) for node_type, nodes in self.type_index.items()},
            "relationships": len(self.relationships)
        }
        
        logger.info(f"Indexed {indexed_files} files with {total_nodes} total nodes")
        return stats
    
    def _extract_nodes(self, tree: ast.AST, file_path: str, content: str) -> List[CodeNode]:
        """Extract code nodes from AST."""
        nodes = []
        lines = content.split('\n')
        
        class NodeVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Extract function information
                start_line = node.lineno
                end_line = getattr(node, 'end_lineno', start_line)
                func_content = '\n'.join(lines[start_line-1:end_line])
                
                func_node = CodeNode(
                    node_type=NodeType.FUNCTION,
                    name=node.name,
                    line_number=start_line,
                    end_line_number=end_line,
                    file_path=file_path,
                    content=func_content,
                    metadata={
                        "args": [arg.arg for arg in node.args.args],
                        "returns": ast.unparse(node.returns) if node.returns else None,
                        "decorators": [ast.unparse(d) for d in node.decorator_list],
                        "docstring": ast.get_docstring(node)
                    }
                )
                nodes.append(func_node)
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Extract class information
                start_line = node.lineno
                end_line = getattr(node, 'end_lineno', start_line)
                class_content = '\n'.join(lines[start_line-1:end_line])
                
                class_node = CodeNode(
                    node_type=NodeType.CLASS,
                    name=node.name,
                    line_number=start_line,
                    end_line_number=end_line,
                    file_path=file_path,
                    content=class_content,
                    metadata={
                        "bases": [ast.unparse(base) for base in node.bases],
                        "decorators": [ast.unparse(d) for d in node.decorator_list],
                        "docstring": ast.get_docstring(node)
                    }
                )
                nodes.append(class_node)
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Extract import information
                for alias in node.names:
                    import_node = CodeNode(
                        node_type=NodeType.IMPORT,
                        name=alias.name,
                        line_number=node.lineno,
                        end_line_number=getattr(node, 'end_lineno', node.lineno),
                        file_path=file_path,
                        content=ast.unparse(node),
                        metadata={
                            "module": alias.name,
                            "alias": alias.asname,
                            "level": "module"
                        }
                    )
                    nodes.append(import_node)
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                # Extract from-import information
                module = node.module or ""
                for alias in node.names:
                    import_node = CodeNode(
                        node_type=NodeType.IMPORT,
                        name=alias.name,
                        line_number=node.lineno,
                        end_line_number=getattr(node, 'end_lineno', node.lineno),
                        file_path=file_path,
                        content=ast.unparse(node),
                        metadata={
                            "module": module,
                            "alias": alias.asname,
                            "level": "from_import"
                        }
                    )
                    nodes.append(import_node)
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Extract function call information
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_name = ast.unparse(node.func)
                else:
                    call_name = "unknown"
                
                call_node = CodeNode(
                    node_type=NodeType.CALL,
                    name=call_name,
                    line_number=node.lineno,
                    end_line_number=getattr(node, 'end_lineno', node.lineno),
                    file_path=file_path,
                    content=ast.unparse(node),
                    metadata={
                        "args_count": len(node.args),
                        "keywords": [kw.arg for kw in node.keywords]
                    }
                )
                nodes.append(call_node)
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                # Extract assignment information
                targets = []
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        targets.append(target.id)
                    elif isinstance(target, ast.Attribute):
                        targets.append(ast.unparse(target))
                
                assign_node = CodeNode(
                    node_type=NodeType.ASSIGNMENT,
                    name=",".join(targets),
                    line_number=node.lineno,
                    end_line_number=getattr(node, 'end_lineno', node.lineno),
                    file_path=file_path,
                    content=ast.unparse(node),
                    metadata={
                        "targets": targets,
                        "type": "simple"
                    }
                )
                nodes.append(assign_node)
                self.generic_visit(node)
        
        # Visit all nodes
        visitor = NodeVisitor()
        visitor.visit(tree)
        
        return nodes
    
    def _extract_relationships(self, tree: ast.AST, nodes: List[CodeNode]) -> List[Relationship]:
        """Extract relationships between code nodes."""
        relationships = []
        node_map = {node.name: node.hash_id for node in nodes}
        
        class RelationshipVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Function call relationships
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in node_map:
                        # Find calling function context
                        current_scope = self._find_current_scope(node, nodes)
                        if current_scope:
                            relationship = Relationship(
                                source_id=current_scope.hash_id,
                                target_id=node_map[func_name],
                                relationship_type="calls",
                                metadata={"line": node.lineno}
                            )
                            relationships.append(relationship)
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Inheritance relationships
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_name = base.id
                        if base_name in node_map:
                            class_node_id = node_map.get(node.name)
                            if class_node_id:
                                relationship = Relationship(
                                    source_id=class_node_id,
                                    target_id=node_map[base_name],
                                    relationship_type="inherits",
                                    metadata={"line": node.lineno}
                                )
                                relationships.append(relationship)
                self.generic_visit(node)
            
            def _find_current_scope(self, node: ast.AST, nodes: List[CodeNode]) -> Optional[CodeNode]:
                """Find the function/class containing this node."""
                for code_node in nodes:
                    if (code_node.node_type in [NodeType.FUNCTION, NodeType.CLASS] and
                        code_node.line_number <= node.lineno <= code_node.end_line_number):
                        return code_node
                return None
        
        # Visit to extract relationships
        visitor = RelationshipVisitor()
        visitor.visit(tree)
        
        return relationships
    
    def search_nodes(
        self, 
        query: str, 
        node_types: Optional[List[NodeType]] = None,
        file_path: Optional[str] = None,
        limit: int = 10
    ) -> List[CodeNode]:
        """Search for nodes matching criteria."""
        results = []
        
        for node in self.nodes.values():
            # Filter by node type
            if node_types and node.node_type not in node_types:
                continue
            
            # Filter by file path
            if file_path and node.file_path != file_path:
                continue
            
            # Simple text matching (can be enhanced with embeddings)
            if (query.lower() in node.name.lower() or 
                query.lower() in node.content.lower()):
                results.append(node)
        
        # Sort by relevance (simple heuristic for now)
        results.sort(key=lambda n: (
            query.lower() in n.name.lower(),
            n.line_number
        ), reverse=True)
        
        return results[:limit]
    
    def get_node_by_id(self, node_id: str) -> Optional[CodeNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[CodeNode]:
        """Get all nodes of a specific type."""
        node_ids = self.type_index.get(node_type, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_nodes_in_file(self, file_path: str) -> List[CodeNode]:
        """Get all nodes in a specific file."""
        node_ids = self.file_index.get(file_path, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_relationships(self, source_id: Optional[str] = None, target_id: Optional[str] = None) -> List[Relationship]:
        """Get relationships matching criteria."""
        relationships = self.relationships
        
        if source_id:
            relationships = [r for r in relationships if r.source_id == source_id]
        
        if target_id:
            relationships = [r for r in relationships if r.target_id == target_id]
        
        return relationships
    
    def save_index(self, output_path: str):
        """Save index to disk."""
        index_data = {
            "nodes": {nid: self._node_to_dict(node) for nid, node in self.nodes.items()},
            "relationships": [self._relationship_to_dict(rel) for rel in self.relationships],
            "metadata": {
                "total_nodes": len(self.nodes),
                "total_relationships": len(self.relationships),
                "file_count": len(self.file_index),
                "node_types": {nt.value: len(nodes) for nt, nodes in self.type_index.items()}
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"Saved index with {len(self.nodes)} nodes to {output_path}")
    
    def load_index(self, index_path: str):
        """Load index from disk."""
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        # Reconstruct nodes
        self.nodes = {}
        for nid, node_data in index_data["nodes"].items():
            self.nodes[nid] = self._dict_to_node(node_data)
        
        # Reconstruct relationships
        self.relationships = [
            self._dict_to_relationship(rel_data) 
            for rel_data in index_data["relationships"]
        ]
        
        # Rebuild indexes
        self._rebuild_indexes()
        
        logger.info(f"Loaded index with {len(self.nodes)} nodes from {index_path}")
    
    def _node_to_dict(self, node: CodeNode) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "node_type": node.node_type.value,
            "name": node.name,
            "line_number": node.line_number,
            "end_line_number": node.end_line_number,
            "file_path": node.file_path,
            "content": node.content,
            "parent": node.parent,
            "children": node.children,
            "metadata": node.metadata,
            "embeddings": node.embeddings,
            "hash_id": node.hash_id
        }
    
    def _dict_to_node(self, node_data: Dict[str, Any]) -> CodeNode:
        """Convert dictionary to node."""
        return CodeNode(
            node_type=NodeType(node_data["node_type"]),
            name=node_data["name"],
            line_number=node_data["line_number"],
            end_line_number=node_data["end_line_number"],
            file_path=node_data["file_path"],
            content=node_data["content"],
            parent=node_data.get("parent"),
            children=node_data.get("children", []),
            metadata=node_data.get("metadata", {}),
            embeddings=node_data.get("embeddings"),
            hash_id=node_data.get("hash_id")
        )
    
    def _relationship_to_dict(self, relationship: Relationship) -> Dict[str, Any]:
        """Convert relationship to dictionary."""
        return {
            "source_id": relationship.source_id,
            "target_id": relationship.target_id,
            "relationship_type": relationship.relationship_type,
            "metadata": relationship.metadata
        }
    
    def _dict_to_relationship(self, rel_data: Dict[str, Any]) -> Relationship:
        """Convert dictionary to relationship."""
        return Relationship(
            source_id=rel_data["source_id"],
            target_id=rel_data["target_id"],
            relationship_type=rel_data["relationship_type"],
            metadata=rel_data.get("metadata", {})
        )
    
    def _rebuild_indexes(self):
        """Rebuild file and type indexes."""
        self.file_index.clear()
        self.type_index.clear()
        self.name_index.clear()
        
        for node in self.nodes.values():
            # File index
            if node.file_path not in self.file_index:
                self.file_index[node.file_path] = []
            self.file_index[node.file_path].append(node.hash_id)
            
            # Type index
            if node.node_type not in self.type_index:
                self.type_index[node.node_type] = []
            self.type_index[node.node_type].append(node.hash_id)
            
            # Name index
            key = f"{node.node_type.value}:{node.name}"
            self.name_index[key] = node.hash_id


class ASTRAGIndex:
    """AST-based RAG index for enhanced code retrieval."""
    
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path
        self.indexer = ASTIndexer()
        self.embedding_cache = {}
        
        if index_path and os.path.exists(index_path):
            self.load_index()
    
    def build_index(self, directory_path: str, rebuild: bool = False):
        """Build AST index for directory."""
        if rebuild:
            self.indexer = ASTIndexer()
        
        stats = self.indexer.index_directory(directory_path)
        
        # Save index
        if self.index_path:
            self.indexer.save_index(self.index_path)
        
        return stats
    
    def search(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for relevant code nodes."""
        context = context or {}
        
        # Determine search parameters from context
        node_types = None
        if "node_types" in context:
            node_types = [NodeType(nt) for nt in context["node_types"]]
        
        file_path = context.get("file_path")
        
        # Search nodes
        nodes = self.indexer.search_nodes(
            query=query,
            node_types=node_types,
            file_path=file_path,
            limit=limit
        )
        
        # Format results for RAG
        results = []
        for node in nodes:
            # Get surrounding context
            context_lines = self._get_context_lines(node, context_lines=5)
            
            result = {
                "file": node.file_path,
                "start": node.line_number,
                "end": node.end_line_number,
                "text": node.content,
                "context": context_lines,
                "node_type": node.node_type.value,
                "name": node.name,
                "metadata": node.metadata,
                "relevance_score": self._calculate_relevance(query, node),
                "hash_id": node.hash_id
            }
            results.append(result)
        
        return results
    
    def get_related_nodes(self, node_id: str, relationship_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get nodes related to the given node."""
        relationships = self.indexer.get_relationships(source_id=node_id)
        
        if relationship_types:
            relationships = [r for r in relationships if r.relationship_type in relationship_types]
        
        related_nodes = []
        for rel in relationships:
            target_node = self.indexer.get_node_by_id(rel.target_id)
            if target_node:
                related_nodes.append({
                    "node": self._node_to_dict(target_node),
                    "relationship": rel.relationship_type,
                    "metadata": rel.metadata
                })
        
        return related_nodes
    
    def _get_context_lines(self, node: CodeNode, context_lines: int = 5) -> str:
        """Get context lines around a node."""
        try:
            with open(node.file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            
            start = max(0, node.line_number - context_lines - 1)
            end = min(len(all_lines), node.end_line_number + context_lines)
            
            context = ''.join(all_lines[start:end])
            return context.strip()
            
        except Exception:
            return node.content
    
    def _calculate_relevance(self, query: str, node: CodeNode) -> float:
        """Calculate relevance score for query-node match."""
        query_lower = query.lower()
        
        # Name match
        name_score = 1.0 if query_lower in node.name.lower() else 0.0
        
        # Content match
        content_score = 0.0
        if query_lower in node.content.lower():
            # Higher score for exact name matches
            content_score = 0.8
        
        # Type-specific scoring
        type_bonus = 0.0
        if node.node_type == NodeType.FUNCTION and "function" in query_lower:
            type_bonus = 0.3
        elif node.node_type == NodeType.CLASS and "class" in query_lower:
            type_bonus = 0.3
        
        return name_score + content_score + type_bonus
    
    def _node_to_dict(self, node: CodeNode) -> Dict[str, Any]:
        """Convert node to dictionary format."""
        return {
            "hash_id": node.hash_id,
            "node_type": node.node_type.value,
            "name": node.name,
            "line_number": node.line_number,
            "end_line_number": node.end_line_number,
            "file_path": node.file_path,
            "content": node.content,
            "metadata": node.metadata
        }
    
    def save_index(self, path: Optional[str] = None):
        """Save index to disk."""
        save_path = path or self.index_path
        if save_path:
            self.indexer.save_index(save_path)
    
    def load_index(self, path: Optional[str] = None):
        """Load index from disk."""
        load_path = path or self.index_path
        if load_path and os.path.exists(load_path):
            self.indexer.load_index(load_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_nodes": len(self.indexer.nodes),
            "total_relationships": len(self.indexer.relationships),
            "file_count": len(self.indexer.file_index),
            "node_types": {
                nt.value: len(nodes) 
                for nt, nodes in self.indexer.type_index.items()
            }
        }


# Global AST RAG index instance
_ast_rag_index = None


def get_ast_rag_index(index_path: Optional[str] = None) -> ASTRAGIndex:
    """Get global AST RAG index instance."""
    global _ast_rag_index
    if _ast_rag_index is None:
        _ast_rag_index = ASTRAGIndex(index_path)
    return _ast_rag_index


def build_ast_index(directory_path: str, index_path: Optional[str] = None) -> Dict[str, Any]:
    """Build AST index for directory."""
    index = get_ast_rag_index(index_path)
    return index.build_index(directory_path)


def search_ast_index(query: str, context: Optional[Dict[str, Any]] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """Search AST index for relevant code."""
    index = get_ast_rag_index()
    return index.search(query, context, limit)