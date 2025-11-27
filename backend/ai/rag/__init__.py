"""
RAG module for context retrieval and citations.
"""

from .citations import (
    format_context,
    extract_citations_from_text, 
    merge_citations,
    validate_snippet_format,
    create_mock_snippets,
    Citation
)

__all__ = [
    "format_context",
    "extract_citations_from_text",
    "merge_citations", 
    "validate_snippet_format",
    "create_mock_snippets",
    "Citation"
]