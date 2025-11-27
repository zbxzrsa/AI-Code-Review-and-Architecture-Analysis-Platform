"""
RAG context formatting and citation utilities.
"""

from typing import List, Dict, Tuple, Any, Optional
import os
from pathlib import Path


class Citation:
    """Represents a citation with source and reasoning."""
    
    def __init__(self, source: str, reason: str):
        self.source = source
        self.reason = reason
    
    def to_dict(self) -> Dict[str, str]:
        return {"source": self.source, "reason": self.reason}


def format_context(snippets: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
    """
    Format RAG snippets for inclusion in prompts and generate citations.
    
    Args:
        snippets: List of snippet dictionaries with keys:
                 - file: file path
                 - start: line number (int)
                 - end: line number (int) 
                 - text: snippet content (string)
                 
    Returns:
        Tuple of (formatted_context_text, citations_list)
    """
    if not snippets:
        return "", []
    
    context_lines = ["Context:"]
    citations = []
    
    for snippet in snippets:
        file_path = snippet.get("file", "unknown")
        start_line = snippet.get("start", 0)
        end_line = snippet.get("end", start_line)
        text = snippet.get("text", "").strip()
        
        # Format source identifier
        source_id = f"{file_path}:{start_line}-{end_line}"
        
        # Add to context
        context_lines.append(f"{source_id} text: |")
        context_lines.append(f"  {text}")
        
        # Add citation
        citations.append({
            "source": source_id,
            "reason": "retrieved via RAG"
        })
    
    formatted_context = "\n".join(context_lines)
    return formatted_context, citations


def extract_citations_from_text(text: str, base_path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Extract file:line references from text and convert to citations.
    
    Args:
        text: Text that may contain file references
        base_path: Base path for resolving relative file paths
        
    Returns:
        List of citation dictionaries
    """
    import re
    
    # Pattern to match file:line or file:start-end references
    pattern = r'(\S+(?:\.\w+)?):(\d+)(?:-(\d+))?'
    
    citations = []
    seen = set()
    
    for match in re.finditer(pattern, text):
        file_path = match.group(1)
        start_line = match.group(2)
        end_line = match.group(3) or start_line
        
        # Skip if it doesn't look like a real file path
        if not _looks_like_file_path(file_path):
            continue
            
        source_id = f"{file_path}:{start_line}-{end_line}"
        
        if source_id not in seen:
            citations.append({
                "source": source_id,
                "reason": "referenced in analysis"
            })
            seen.add(source_id)
    
    return citations


def _looks_like_file_path(path: str) -> bool:
    """
    Check if a string looks like a file path.
    
    Args:
        path: String to check
        
    Returns:
        True if it looks like a file path
    """
    # Common indicators of file paths
    indicators = [
        '/' in path,  # Unix path separator
        '\\' in path,  # Windows path separator
        '.' in path and path.split('.')[-1] in ['py', 'js', 'ts', 'java', 'cpp', 'c', 'h', 'go', 'rs', 'md', 'yaml', 'yml', 'json'],
        path.startswith(('src/', 'app/', 'lib/', 'components/', 'utils/', 'services/')),
    ]
    
    return any(indicators)


def merge_citations(*citation_lists: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Merge multiple citation lists, removing duplicates.
    
    Args:
        *citation_lists: Variable number of citation lists
        
    Returns:
        Merged citation list with unique entries
    """
    merged = []
    seen_sources = set()
    
    for citations in citation_lists:
        for citation in citations:
            source = citation.get("source", "")
            if source and source not in seen_sources:
                merged.append(citation)
                seen_sources.add(source)
    
    return merged


def validate_snippet_format(snippet: Dict[str, Any]) -> bool:
    """
    Validate that a snippet has the required format.
    
    Args:
        snippet: Snippet dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["file", "text"]
    numeric_fields = ["start", "end"]
    
    # Check required string fields
    for field in required_fields:
        if field not in snippet or not isinstance(snippet[field], str):
            return False
    
    # Check numeric fields if present
    for field in numeric_fields:
        if field in snippet and not isinstance(snippet[field], (int, float)):
            return False
    
    return True


def create_mock_snippets(file_path: str, content: str, max_snippets: int = 3) -> List[Dict[str, Any]]:
    """
    Create mock snippets from file content for testing.
    
    Args:
        file_path: Path to the file
        content: File content
        max_snippets: Maximum number of snippets to create
        
    Returns:
        List of snippet dictionaries
    """
    lines = content.split('\n')
    snippets = []
    
    # Create snippets of reasonable size (around 10 lines each)
    snippet_size = 10
    total_lines = len(lines)
    
    for i in range(0, total_lines, snippet_size * 2):  # Overlap slightly
        if len(snippets) >= max_snippets:
            break
            
        start_line = i + 1  # 1-based line numbers
        end_line = min(i + snippet_size, total_lines)
        snippet_text = '\n'.join(lines[i:end_line])
        
        if snippet_text.strip():  # Skip empty snippets
            snippets.append({
                "file": file_path,
                "start": start_line,
                "end": end_line,
                "text": snippet_text
            })
    
    return snippets