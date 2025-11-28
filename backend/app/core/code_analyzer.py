#!/usr/bin/env python3
"""
Enhanced Code Analyzer with Multi-Language Support
Uses tree-sitter for universal parsing and supports incremental analysis
"""

import ast
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import json
import time

# Tree-sitter imports
TREE_SITTER_AVAILABLE = False
Language = None
Parser = None

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    logging.warning("Tree-sitter not available, falling back to AST-only parsing")

# Language-specific parsers
language_parsers_available = {}
try:
    import tree_sitter_python as tspython
    language_parsers_available['python'] = tspython
except ImportError:
    pass

try:
    import tree_sitter_javascript as tsjavascript
    language_parsers_available['javascript'] = tsjavascript
except ImportError:
    pass

try:
    import tree_sitter_java as tsjava
    language_parsers_available['java'] = tsjava
except ImportError:
    pass

try:
    import tree_sitter_cpp as tscpp
    language_parsers_available['cpp'] = tscpp
    language_parsers_available['c'] = tscpp
except ImportError:
    pass

try:
    import tree_sitter_typescript as tstypescript
    language_parsers_available['typescript'] = tstypescript
except ImportError:
    pass

try:
    import tree_sitter_go as tsgo
    language_parsers_available['go'] = tsgo
except ImportError:
    pass

try:
    import tree_sitter_rust as tsrust
    language_parsers_available['rust'] = tsrust
except ImportError:
    pass

if not language_parsers_available:
    logging.warning("No tree-sitter language parsers available")

logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """Enhanced code analyzer with multi-language support and incremental parsing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.incremental_enabled = self.config.get('incremental_enabled', True)
        self.max_file_size = self.config.get('max_file_size', 1024 * 1024)  # 1MB
        
        # Initialize parsers
        self.parsers = {}
        self.language_grammars = {}
        self._init_parsers()
        
        # Analysis cache for incremental updates
        self.analysis_cache = {}
        
    def _init_parsers(self):
        """Initialize tree-sitter parsers for supported languages"""
        if not TREE_SITTER_AVAILABLE or not Language or not Parser:
            logger.warning("Tree-sitter not available, using AST-only mode")
            return
            
        try:
            # Initialize language grammars
            self.language_grammars = {}
            
            # Create grammars for available languages
            if 'python' in language_parsers_available:
                self.language_grammars['python'] = Language(language_parsers_available['python'].language(), 'python')
            if 'javascript' in language_parsers_available:
                self.language_grammars['javascript'] = Language(language_parsers_available['javascript'].language(), 'javascript')
            if 'typescript' in language_parsers_available:
                self.language_grammars['typescript'] = Language(language_parsers_available['typescript'].language(), 'typescript')
            if 'java' in language_parsers_available:
                self.language_grammars['java'] = Language(language_parsers_available['java'].language(), 'java')
            if 'cpp' in language_parsers_available:
                self.language_grammars['cpp'] = Language(language_parsers_available['cpp'].language(), 'cpp')
                self.language_grammars['c'] = Language(language_parsers_available['cpp'].language(), 'c')
            if 'go' in language_parsers_available:
                self.language_grammars['go'] = Language(language_parsers_available['go'].language(), 'go')
            if 'rust' in language_parsers_available:
                self.language_grammars['rust'] = Language(language_parsers_available['rust'].language(), 'rust')
            
            # Create parsers for each language
            for lang, grammar in self.language_grammars.items():
                if grammar:
                    parser = Parser()
                    parser.set_language(grammar)
                    self.parsers[lang] = parser
                    
            logger.info(f"Initialized parsers for languages: {list(self.parsers.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter parsers: {e}")
            
    def detect_language(self, file_path: str, content: Optional[str] = None) -> str:
        """Detect programming language from file extension and content"""
        file_ext = Path(file_path).suffix.lower()
        
        # Extension-based detection
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
        }
        
        detected_lang = ext_map.get(file_ext, 'unknown')
        
        # Content-based detection for ambiguous cases
        if detected_lang == 'unknown' and content:
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in ['def ', 'import ', 'from ', 'class ']):
                detected_lang = 'python'
            elif any(keyword in content_lower for keyword in ['function ', 'const ', 'let ', 'var ']):
                detected_lang = 'javascript'
            elif any(keyword in content_lower for keyword in ['public class ', 'import java.', 'package ']):
                detected_lang = 'java'
                
        return detected_lang
    
    def analyze_code(self, code: str, language: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze code and extract comprehensive metrics
        
        Args:
            code: Source code to analyze
            language: Programming language
            file_path: Optional file path for context
            
        Returns:
            Dictionary containing analysis results and metrics
        """
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(code, language, file_path)
        
        # Check cache
        if self.cache_enabled and cache_key in self.analysis_cache:
            logger.debug(f"Using cached analysis for {file_path}")
            cached_result = self.analysis_cache[cache_key]
            cached_result['cached'] = True
            return cached_result
        
        try:
            # Initialize result
            result = {
                'file_path': file_path,
                'language': language,
                'timestamp': time.time(),
                'cached': False,
                'metrics': {},
                'structure': {},
                'issues': [],
                'complexity': {},
                'dependencies': [],
                'functions': [],
                'classes': [],
                'imports': [],
                'exports': []
            }
            
            # Choose analysis method based on language and availability
            if language in self.parsers and TREE_SITTER_AVAILABLE:
                result = self._analyze_with_tree_sitter(code, language, result)
            elif language == 'python':
                result = self._analyze_python_ast(code, result)
            else:
                result = self._analyze_fallback(code, language, result)
            
            # Add processing time
            result['processing_time'] = time.time() - start_time
            
            # Cache result
            if self.cache_enabled:
                self.analysis_cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Code analysis failed for {file_path}: {e}")
            return {
                'file_path': file_path,
                'language': language,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'cached': False
            }
    
    def _analyze_with_tree_sitter(self, code: str, language: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code using tree-sitter for accurate parsing"""
        parser = self.parsers[language]
        
        # Parse code into AST
        tree = parser.parse(bytes(code, "utf-8"))
        root_node = tree.root_node
        
        # Extract basic metrics
        result['metrics'] = {
            'lines_of_code': len(code.splitlines()),
            'characters': len(code),
            'functions': self._count_node_type(root_node, self._get_function_nodes(language)),
            'classes': self._count_node_type(root_node, self._get_class_nodes(language)),
            'imports': self._count_node_type(root_node, self._get_import_nodes(language)),
            'exports': self._count_node_type(root_node, self._get_export_nodes(language)),
            'comments': self._count_comments(root_node, code),
            'complexity': self._calculate_cyclomatic_complexity(root_node, language)
        }
        
        # Extract structure
        result['structure'] = {
            'root_type': root_node.type,
            'children_count': len(root_node.children),
            'tree_depth': self._calculate_tree_depth(root_node)
        }
        
        # Extract functions and classes
        result['functions'] = self._extract_functions(root_node, language, code)
        result['classes'] = self._extract_classes(root_node, language, code)
        result['imports'] = self._extract_imports(root_node, language, code)
        result['exports'] = self._extract_exports(root_node, language, code)
        
        # Extract dependencies
        result['dependencies'] = self._extract_dependencies(root_node, language)
        
        # Detect issues
        result['issues'] = self._detect_issues_tree_sitter(root_node, language, code)
        
        return result
    
    def _analyze_python_ast(self, code: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Python code using built-in AST"""
        try:
            tree = ast.parse(code)
            
            # Basic metrics
            result['metrics'] = {
                'lines_of_code': len(code.splitlines()),
                'characters': len(code),
                'functions': len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                'classes': len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                'imports': len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]),
                'exports': len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.startswith('_') is False]),
                'comments': code.count('#'),
                'complexity': self._calculate_python_complexity(tree)
            }
            
            # Extract functions
            result['functions'] = self._extract_python_functions(tree, code)
            result['classes'] = self._extract_python_classes(tree, code)
            result['imports'] = self._extract_python_imports(tree)
            
            # Detect issues
            result['issues'] = self._detect_python_issues(tree, code)
            
        except SyntaxError as e:
            result['error'] = f"Python syntax error: {e}"
            
        return result
    
    def _analyze_fallback(self, code: str, language: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis for unsupported languages"""
        lines = code.splitlines()
        
        # Basic text-based metrics
        result['metrics'] = {
            'lines_of_code': len(lines),
            'characters': len(code),
            'functions': sum(1 for line in lines if self._contains_function_definition(line, language)),
            'classes': sum(1 for line in lines if self._contains_class_definition(line, language)),
            'imports': sum(1 for line in lines if self._contains_import_definition(line, language)),
            'exports': 0,  # Hard to detect without proper parsing
            'comments': sum(1 for line in lines if line.strip().startswith(('//', '#', '/*', '*'))),
            'complexity': 1  # Default complexity
        }
        
        # Basic issue detection
        result['issues'] = self._detect_basic_issues(code, language)
        
        return result
    
    def _generate_cache_key(self, code: str, language: str, file_path: Optional[str]) -> str:
        """Generate cache key for analysis results"""
        content = f"{language}:{file_path or 'unknown'}:{code}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _count_node_type(self, node: Any, node_types: List[str]) -> int:
        """Count specific node types in tree"""
        count = 0
        if node.type in node_types:
            count += 1
        for child in node.children:
            count += self._count_node_type(child, node_types)
        return count
    
    def _get_function_nodes(self, language: str) -> List[str]:
        """Get function node types for language"""
        function_nodes = {
            'python': ['function_definition', 'async_function_definition'],
            'javascript': ['function_declaration', 'function_expression', 'arrow_function'],
            'typescript': ['function_declaration', 'function_expression', 'arrow_function'],
            'java': ['method_declaration', 'constructor_declaration'],
            'cpp': ['function_definition', 'method_definition'],
            'c': ['function_definition'],
            'go': ['function_declaration', 'function_literal'],
            'rust': ['function_item', 'method_item']
        }
        return function_nodes.get(language, [])
    
    def _get_class_nodes(self, language: str) -> List[str]:
        """Get class node types for language"""
        class_nodes = {
            'python': ['class_definition'],
            'javascript': ['class_declaration', 'class_expression'],
            'typescript': ['class_declaration', 'class_expression'],
            'java': ['class_declaration', 'interface_declaration', 'enum_declaration'],
            'cpp': ['class_specifier', 'struct_specifier'],
            'go': ['type_declaration'],  # Go uses structs
            'rust': ['struct_item', 'enum_item', 'impl_item']
        }
        return class_nodes.get(language, [])
    
    def _get_import_nodes(self, language: str) -> List[str]:
        """Get import node types for language"""
        import_nodes = {
            'python': ['import_statement', 'import_from_statement'],
            'javascript': ['import_statement', 'require_expression'],
            'typescript': ['import_statement', 'require_expression'],
            'java': ['import_declaration'],
            'cpp': ['include_directive'],
            'go': ['import_declaration'],
            'rust': ['use_declaration', 'mod_item']
        }
        return import_nodes.get(language, [])
    
    def _get_export_nodes(self, language: str) -> List[str]:
        """Get export node types for language"""
        export_nodes = {
            'python': [],  # Python doesn't have explicit exports
            'javascript': ['export_statement', 'export_default_declaration'],
            'typescript': ['export_statement', 'export_default_declaration'],
            'java': [],  # Java uses public modifiers
            'cpp': [],  # C++ uses header files
            'go': [],  # Go uses capitalization
            'rust': ['pub_item']
        }
        return export_nodes.get(language, [])
    
    def _calculate_tree_depth(self, node: Any, current_depth: int = 0) -> int:
        """Calculate maximum depth of AST"""
        if not node.children:
            return current_depth
        return max(self._calculate_tree_depth(child, current_depth + 1) for child in node.children)
    
    def _count_comments(self, root_node: Any, code: str) -> int:
        """Count comments in code (simplified)"""
        comment_indicators = ['#', '//', '/*', '*']
        return sum(1 for line in code.splitlines() if line.strip().startswith(tuple(comment_indicators)))
    
    def _calculate_cyclomatic_complexity(self, node: Any, language: str) -> int:
        """Calculate cyclomatic complexity using tree-sitter"""
        complexity = 1  # Base complexity
        
        # Decision points that increase complexity
        decision_nodes = {
            'python': ['if_statement', 'while_statement', 'for_statement', 'except_clause', 'with_statement'],
            'javascript': ['if_statement', 'while_statement', 'for_statement', 'switch_statement', 'catch_clause'],
            'typescript': ['if_statement', 'while_statement', 'for_statement', 'switch_statement', 'catch_clause'],
            'java': ['if_statement', 'while_statement', 'for_statement', 'switch_statement', 'catch_clause'],
            'cpp': ['if_statement', 'while_statement', 'for_statement', 'switch_statement', 'catch_clause'],
            'c': ['if_statement', 'while_statement', 'for_statement', 'switch_statement'],
            'go': ['if_statement', 'for_statement', 'select_statement'],
            'rust': ['if_expression', 'while_expression', 'for_expression', 'match_expression']
        }
        
        decision_types = decision_nodes.get(language, [])
        
        def count_decisions(node):
            nonlocal complexity
            if node.type in decision_types:
                complexity += 1
            for child in node.children:
                count_decisions(child)
        
        count_decisions(node)
        return complexity
    
    def _extract_functions(self, root_node: Any, language: str, code: str) -> List[Dict[str, Any]]:
        """Extract function information from AST"""
        functions = []
        function_types = self._get_function_nodes(language)
        
        def traverse(node):
            if node.type in function_types:
                func_info = self._extract_function_info(node, language, code)
                if func_info:
                    functions.append(func_info)
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return functions
    
    def _extract_function_info(self, node: Any, language: str, code: str) -> Optional[Dict[str, Any]]:
        """Extract detailed function information"""
        try:
            lines = code.splitlines()
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            
            func_info = {
                'name': self._get_function_name(node, language),
                'start_line': start_line + 1,
                'end_line': end_line + 1,
                'line_count': end_line - start_line + 1,
                'parameters': self._extract_function_parameters(node, language),
                'return_type': self._extract_return_type(node, language),
                'visibility': self._extract_visibility(node, language),
                'is_async': self._is_async_function(node, language),
                'complexity': self._calculate_node_complexity(node, language)
            }
            
            # Extract function body (first few lines)
            if start_line < len(lines):
                body_start = start_line + 1
                body_end = min(body_start + 5, end_line + 1)
                func_info['body_preview'] = '\n'.join(lines[body_start:body_end])
            
            return func_info
            
        except Exception as e:
            logger.warning(f"Failed to extract function info: {e}")
            return None
    
    def _get_function_name(self, node: Any, language: str) -> str:
        """Extract function name from node"""
        # This is language-specific and would need proper implementation
        return f"function_{node.start_point[0]}_{node.start_point[1]}"
    
    def _extract_function_parameters(self, node: Any, language: str) -> List[str]:
        """Extract function parameters"""
        # Language-specific parameter extraction
        return []
    
    def _extract_return_type(self, node: Any, language: str) -> Optional[str]:
        """Extract return type"""
        return None
    
    def _extract_visibility(self, node: Any, language: str) -> str:
        """Extract function visibility"""
        return 'public'
    
    def _is_async_function(self, node: Any, language: str) -> bool:
        """Check if function is async"""
        return node.type in ['async_function_definition', 'async_function_declaration']
    
    def _calculate_node_complexity(self, node: Any, language: str) -> int:
        """Calculate complexity for a specific node"""
        return 1  # Simplified
    
    def _extract_classes(self, root_node: Any, language: str, code: str) -> List[Dict[str, Any]]:
        """Extract class information from AST"""
        classes = []
        class_types = self._get_class_nodes(language)
        
        def traverse(node):
            if node.type in class_types:
                class_info = self._extract_class_info(node, language, code)
                if class_info:
                    classes.append(class_info)
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return classes
    
    def _extract_class_info(self, node: Any, language: str, code: str) -> Optional[Dict[str, Any]]:
        """Extract detailed class information"""
        try:
            lines = code.splitlines()
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            
            class_info = {
                'name': self._get_class_name(node, language),
                'start_line': start_line + 1,
                'end_line': end_line + 1,
                'line_count': end_line - start_line + 1,
                'methods': self._extract_class_methods(node, language),
                'properties': self._extract_class_properties(node, language),
                'inheritance': self._extract_inheritance(node, language),
                'visibility': self._extract_class_visibility(node, language)
            }
            
            return class_info
            
        except Exception as e:
            logger.warning(f"Failed to extract class info: {e}")
            return None
    
    def _get_class_name(self, node: Any, language: str) -> str:
        """Extract class name from node"""
        return f"class_{node.start_point[0]}_{node.start_point[1]}"
    
    def _extract_class_methods(self, node: Any, language: str) -> List[str]:
        """Extract class methods"""
        return []
    
    def _extract_class_properties(self, node: Any, language: str) -> List[str]:
        """Extract class properties"""
        return []
    
    def _extract_inheritance(self, node: Any, language: str) -> List[str]:
        """Extract inheritance information"""
        return []
    
    def _extract_class_visibility(self, node: Any, language: str) -> str:
        """Extract class visibility"""
        return 'public'
    
    def _extract_imports(self, root_node: Any, language: str, code: str) -> List[Dict[str, Any]]:
        """Extract import information"""
        imports = []
        import_types = self._get_import_nodes(language)
        
        def traverse(node):
            if node.type in import_types:
                import_info = self._extract_import_info(node, language, code)
                if import_info:
                    imports.append(import_info)
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return imports
    
    def _extract_import_info(self, node: Any, language: str, code: str) -> Optional[Dict[str, Any]]:
        """Extract import information"""
        return {
            'module': 'unknown',
            'line': node.start_point[0] + 1,
            'type': node.type
        }
    
    def _extract_exports(self, root_node: Any, language: str, code: str) -> List[Dict[str, Any]]:
        """Extract export information"""
        exports = []
        export_types = self._get_export_nodes(language)
        
        def traverse(node):
            if node.type in export_types:
                export_info = self._extract_export_info(node, language, code)
                if export_info:
                    exports.append(export_info)
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return exports
    
    def _extract_export_info(self, node: Any, language: str, code: str) -> Optional[Dict[str, Any]]:
        """Extract export information"""
        return {
            'name': 'unknown',
            'line': node.start_point[0] + 1,
            'type': node.type
        }
    
    def _extract_dependencies(self, root_node: Any, language: str) -> List[str]:
        """Extract dependency information"""
        # This would analyze function calls, class instantiations, etc.
        return []
    
    def _detect_issues_tree_sitter(self, root_node: Any, language: str, code: str) -> List[Dict[str, Any]]:
        """Detect code issues using tree-sitter"""
        issues = []
        
        # Common issue patterns
        issues.extend(self._detect_long_functions(root_node, language, code))
        issues.extend(self._detect_long_classes(root_node, language, code))
        issues.extend(self._detect_deep_nesting(root_node, language, code))
        issues.extend(self._detect_large_parameter_lists(root_node, language, code))
        
        return issues
    
    def _detect_long_functions(self, root_node: Any, language: str, code: str, max_lines: int = 50) -> List[Dict[str, Any]]:
        """Detect functions that are too long"""
        issues = []
        function_types = self._get_function_nodes(language)
        
        def traverse(node):
            if node.type in function_types:
                line_count = node.end_point[0] - node.start_point[0] + 1
                if line_count > max_lines:
                    issues.append({
                        'type': 'quality',
                        'severity': 'medium',
                        'rule_id': 'long_function',
                        'title': f'Long function detected ({line_count} lines)',
                        'description': f'Function is {line_count} lines long, consider refactoring',
                        'line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'recommendation': 'Break down into smaller functions'
                    })
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return issues
    
    def _detect_long_classes(self, root_node: Any, language: str, code: str, max_lines: int = 200) -> List[Dict[str, Any]]:
        """Detect classes that are too long"""
        issues = []
        class_types = self._get_class_nodes(language)
        
        def traverse(node):
            if node.type in class_types:
                line_count = node.end_point[0] - node.start_point[0] + 1
                if line_count > max_lines:
                    issues.append({
                        'type': 'quality',
                        'severity': 'medium',
                        'rule_id': 'long_class',
                        'title': f'Long class detected ({line_count} lines)',
                        'description': f'Class is {line_count} lines long, consider refactoring',
                        'line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'recommendation': 'Split into multiple classes or extract functionality'
                    })
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return issues
    
    def _detect_deep_nesting(self, root_node: Any, language: str, code: str, max_depth: int = 4) -> List[Dict[str, Any]]:
        """Detect deeply nested code"""
        issues = []
        
        def traverse(node, depth=0):
            if depth > max_depth:
                issues.append({
                    'type': 'quality',
                    'severity': 'medium',
                    'rule_id': 'deep_nesting',
                    'title': f'Deep nesting detected (depth {depth})',
                    'description': f'Code is nested {depth} levels deep, consider refactoring',
                    'line': node.start_point[0] + 1,
                    'recommendation': 'Extract nested logic into separate functions'
                })
            
            for child in node.children:
                traverse(child, depth + 1)
        
        traverse(root_node)
        return issues
    
    def _detect_large_parameter_lists(self, root_node: Any, language: str, code: str, max_params: int = 5) -> List[Dict[str, Any]]:
        """Detect functions with too many parameters"""
        issues = []
        function_types = self._get_function_nodes(language)
        
        def traverse(node):
            if node.type in function_types:
                param_count = self._count_function_parameters(node, language)
                if param_count > max_params:
                    issues.append({
                        'type': 'quality',
                        'severity': 'low',
                        'rule_id': 'too_many_parameters',
                        'title': f'Too many parameters ({param_count})',
                        'description': f'Function has {param_count} parameters, consider using a parameter object',
                        'line': node.start_point[0] + 1,
                        'recommendation': 'Group related parameters into objects or configuration structs'
                    })
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return issues
    
    def _count_function_parameters(self, node: Any, language: str) -> int:
        """Count number of function parameters"""
        # This would be language-specific
        return 0
    
    def _extract_python_functions(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Extract Python functions using AST"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno or node.lineno,
                    'line_count': (node.end_lineno or node.lineno) - node.lineno + 1,
                    'parameters': [arg.arg for arg in node.args.args],
                    'is_async': False,
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                    'docstring': ast.get_docstring(node) or '',
                    'complexity': self._calculate_python_function_complexity(node)
                }
                functions.append(func_info)
            elif isinstance(node, ast.AsyncFunctionDef):
                func_info = {
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno or node.lineno,
                    'line_count': (node.end_lineno or node.lineno) - node.lineno + 1,
                    'parameters': [arg.arg for arg in node.args.args],
                    'is_async': True,
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                    'docstring': ast.get_docstring(node) or '',
                    'complexity': self._calculate_python_function_complexity(node)
                }
                functions.append(func_info)
        
        return functions
    
    def _extract_python_classes(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Extract Python classes using AST"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                
                class_info = {
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno or node.lineno,
                    'line_count': (node.end_lineno or node.lineno) - node.lineno + 1,
                    'methods': methods,
                    'base_classes': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                    'docstring': ast.get_docstring(node) or ''
                }
                classes.append(class_info)
        
        return classes
    
    def _extract_python_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract Python imports using AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno,
                        'type': 'import'
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno,
                        'type': 'from_import'
                    })
        
        return imports
    
    def _calculate_python_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity for Python code"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_python_function_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """Calculate complexity for a specific Python function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _detect_python_issues(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Detect Python-specific issues"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Long function
                line_count = (node.end_lineno or node.lineno) - node.lineno + 1
                if line_count > 50:
                    issues.append({
                        'type': 'quality',
                        'severity': 'medium',
                        'rule_id': 'long_function',
                        'title': f'Long function ({line_count} lines)',
                        'description': f'Function {node.name} is {line_count} lines long',
                        'line': node.lineno,
                        'recommendation': 'Consider breaking down into smaller functions'
                    })
                
                # Too many parameters
                if len(node.args.args) > 5:
                    issues.append({
                        'type': 'quality',
                        'severity': 'low',
                        'rule_id': 'too_many_parameters',
                        'title': f'Too many parameters ({len(node.args.args)})',
                        'description': f'Function {node.name} has {len(node.args.args)} parameters',
                        'line': node.lineno,
                        'recommendation': 'Consider using a parameter object'
                    })
        
        return issues
    
    def _contains_function_definition(self, line: str, language: str) -> bool:
        """Check if line contains function definition (fallback)"""
        function_keywords = {
            'python': ['def '],
            'javascript': ['function ', '=>', 'const '],
            'java': ['public ', 'private ', 'protected '],
            'cpp': ['auto ', 'void ', 'int ', 'string '],
            'go': ['func '],
            'rust': ['fn ']
        }
        
        keywords = function_keywords.get(language, [])
        return any(keyword in line for keyword in keywords)
    
    def _contains_class_definition(self, line: str, language: str) -> bool:
        """Check if line contains class definition (fallback)"""
        class_keywords = {
            'python': ['class '],
            'javascript': ['class '],
            'java': ['class ', 'interface ', 'enum '],
            'cpp': ['class ', 'struct '],
            'go': ['type '],
            'rust': ['struct ', 'enum ', 'impl ']
        }
        
        keywords = class_keywords.get(language, [])
        return any(keyword in line for keyword in keywords)
    
    def _contains_import_definition(self, line: str, language: str) -> bool:
        """Check if line contains import definition (fallback)"""
        import_keywords = {
            'python': ['import ', 'from '],
            'javascript': ['import ', 'require('],
            'java': ['import '],
            'cpp': ['#include '],
            'go': ['import '],
            'rust': ['use ', 'mod ']
        }
        
        keywords = import_keywords.get(language, [])
        return any(keyword in line for keyword in keywords)
    
    def _detect_basic_issues(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Detect basic issues without proper parsing"""
        issues = []
        lines = code.splitlines()
        
        for i, line in enumerate(lines, 1):
            # Long lines
            if len(line) > 120:
                issues.append({
                    'type': 'quality',
                    'severity': 'low',
                    'rule_id': 'long_line',
                    'title': f'Long line ({len(line)} characters)',
                    'description': f'Line {i} is {len(line)} characters long',
                    'line': i,
                    'recommendation': 'Break line into multiple lines'
                })
        
        return issues
    
    def clear_cache(self):
        """Clear analysis cache"""
        self.analysis_cache.clear()
        logger.info("Analysis cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.analysis_cache),
            'cache_enabled': self.cache_enabled,
            'incremental_enabled': self.incremental_enabled
        }