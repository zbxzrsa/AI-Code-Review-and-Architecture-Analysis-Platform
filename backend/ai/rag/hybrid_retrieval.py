"""
Hybrid retrieval system combining semantic search, AST indexing, and traditional search.
Provides intelligent code context retrieval for AI review systems.
"""

import os
import json
import math
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from .ast_index import get_ast_rag_index, ASTRAGIndex
from .citations import format_context, extract_citations_from_text

logger = logging.getLogger(__name__)


class RetrievalMethod(Enum):
    """Types of retrieval methods."""
    SEMANTIC = "semantic"
    AST_STRUCTURED = "ast_structured"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    CITATION = "citation"


@dataclass
class RetrievalResult:
    """Result from retrieval system."""
    content: str
    source: str
    method: RetrievalMethod
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Optional[str] = None
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class RetrievalQuery:
    """Query for retrieval system."""
    query: str
    methods: List[RetrievalMethod]
    context: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    weights: Dict[RetrievalMethod, float] = field(default_factory=dict)


class SemanticRetriever:
    """Semantic search using embeddings."""
    
    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings_cache = {}
        self.document_store = {}
        
        # Try to import sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.embedding_model)
            self.available = True
        except ImportError:
            logger.warning("sentence-transformers not available, semantic search disabled")
            self.available = False
            self.model = None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to semantic store."""
        if not self.available:
            return
        
        for doc in documents:
            doc_id = doc.get("id", str(len(self.document_store)))
            text = doc.get("content", "")
            
            # Generate embedding
            if text not in self.embeddings_cache:
                embedding = self.model.encode(text)
                self.embeddings_cache[text] = embedding
            
            self.document_store[doc_id] = {
                "content": text,
                "embedding": self.embeddings_cache[text],
                "metadata": doc.get("metadata", {}),
                "file_path": doc.get("file_path"),
                "line_start": doc.get("line_start"),
                "line_end": doc.get("line_end")
            }
    
    def search(self, query: str, limit: int = 10) -> List[RetrievalResult]:
        """Search documents semantically."""
        if not self.available or not self.document_store:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode(query)
        
        # Calculate similarities
        results = []
        for doc_id, doc in self.document_store.items():
            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc["embedding"])
            
            result = RetrievalResult(
                content=doc["content"],
                source=doc_id,
                method=RetrievalMethod.SEMANTIC,
                relevance_score=similarity,
                metadata=doc["metadata"],
                file_path=doc.get("file_path"),
                line_start=doc.get("line_start"),
                line_end=doc.get("line_end")
            )
            results.append(result)
        
        # Sort by relevance and limit
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)


class KeywordRetriever:
    """Traditional keyword-based search."""
    
    def __init__(self):
        self.document_store = {}
        self.inverted_index = {}
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to keyword store."""
        for doc in documents:
            doc_id = doc.get("id", str(len(self.document_store)))
            text = doc.get("content", "")
            
            self.document_store[doc_id] = {
                "content": text,
                "metadata": doc.get("metadata", {}),
                "file_path": doc.get("file_path"),
                "line_start": doc.get("line_start"),
                "line_end": doc.get("line_end")
            }
            
            # Build inverted index
            terms = self._extract_terms(text)
            for term in terms:
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(doc_id)
    
    def search(self, query: str, limit: int = 10) -> List[RetrievalResult]:
        """Search documents by keywords."""
        query_terms = self._extract_terms(query)
        
        # Score documents by term frequency
        doc_scores = {}
        for term in query_terms:
            if term in self.inverted_index:
                for doc_id in self.inverted_index[term]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1
        
        # Create results
        results = []
        for doc_id, score in doc_scores.items():
            if doc_id in self.document_store:
                doc = self.document_store[doc_id]
                
                result = RetrievalResult(
                    content=doc["content"],
                    source=doc_id,
                    method=RetrievalMethod.KEYWORD,
                    relevance_score=score / len(query_terms),  # Normalize by query terms
                    metadata=doc["metadata"],
                    file_path=doc.get("file_path"),
                    line_start=doc.get("line_start"),
                    line_end=doc.get("line_end")
                )
                results.append(result)
        
        # Sort by score and limit
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract searchable terms from text."""
        import re
        
        # Simple term extraction - can be enhanced
        terms = []
        
        # Split on non-alphanumeric
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter common stop words and short terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for word in words:
            if len(word) > 2 and word not in stop_words:
                terms.append(word)
        
        return terms


class HybridRetrievalSystem:
    """Hybrid retrieval system combining multiple search methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize retrievers
        self.semantic_retriever = SemanticRetriever(
            self.config.get("semantic_model")
        )
        self.keyword_retriever = KeywordRetriever()
        self.ast_index = get_ast_rag_index(
            self.config.get("ast_index_path")
        )
        
        # Default weights
        self.default_weights = {
            RetrievalMethod.SEMANTIC: 0.4,
            RetrievalMethod.AST_STRUCTURED: 0.3,
            RetrievalMethod.KEYWORD: 0.2,
            RetrievalMethod.CITATION: 0.1
        }
        
        self.document_count = 0
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to all retrievers."""
        # Add to semantic and keyword retrievers
        self.semantic_retriever.add_documents(documents)
        self.keyword_retriever.add_documents(documents)
        
        # Add to AST index (convert format)
        ast_docs = []
        for doc in documents:
            if doc.get("file_path") and doc.get("content"):
                ast_doc = {
                    "file_path": doc["file_path"],
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {})
                }
                ast_docs.append(ast_doc)
        
        if ast_docs:
            # Build AST index for each file
            file_paths = set(doc["file_path"] for doc in ast_docs)
            for file_path in file_paths:
                try:
                    self.ast_index.build_index(os.path.dirname(file_path), rebuild=False)
                except Exception as e:
                    logger.warning(f"Failed to build AST index for {file_path}: {e}")
        
        self.document_count += len(documents)
        logger.info(f"Added {len(documents)} documents to hybrid retrieval system")
    
    def search(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Perform hybrid search using multiple methods.
        
        Args:
            query: RetrievalQuery with search parameters
            
        Returns:
            Combined and ranked results
        """
        # Use default methods if not specified
        methods = query.methods or [
            RetrievalMethod.SEMANTIC,
            RetrievalMethod.AST_STRUCTURED,
            RetrievalMethod.KEYWORD
        ]
        
        # Use provided weights or defaults
        weights = query.weights or self.default_weights
        
        # Collect results from different methods
        all_results = []
        method_results = {}
        
        # Semantic search
        if RetrievalMethod.SEMANTIC in methods:
            semantic_results = self.semantic_retriever.search(query.query, query.limit)
            method_results[RetrievalMethod.SEMANTIC] = semantic_results
            all_results.extend(semantic_results)
        
        # AST-based search
        if RetrievalMethod.AST_STRUCTURED in methods:
            ast_results = self._search_ast(query)
            method_results[RetrievalMethod.AST_STRUCTURED] = ast_results
            all_results.extend(ast_results)
        
        # Keyword search
        if RetrievalMethod.KEYWORD in methods:
            keyword_results = self.keyword_retriever.search(query.query, query.limit)
            method_results[RetrievalMethod.KEYWORD] = keyword_results
            all_results.extend(keyword_results)
        
        # Citation-based search (if context provided)
        if RetrievalMethod.CITATION in methods and query.context.get("citations"):
            citation_results = self._search_citations(query)
            method_results[RetrievalMethod.CITATION] = citation_results
            all_results.extend(citation_results)
        
        # Combine and re-rank results
        combined_results = self._combine_results(
            all_results, method_results, weights, query
        )
        
        return combined_results[:query.limit]
    
    def _search_ast(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Search using AST index."""
        try:
            # Build search context
            ast_context = {}
            if query.filters.get("node_types"):
                ast_context["node_types"] = query.filters["node_types"]
            if query.filters.get("file_path"):
                ast_context["file_path"] = query.filters["file_path"]
            
            # Search AST index
            ast_results = self.ast_index.search(query.query, ast_context, query.limit)
            
            # Convert to RetrievalResult format
            results = []
            for ast_result in ast_results:
                result = RetrievalResult(
                    content=ast_result["text"],
                    source=ast_result["hash_id"],
                    method=RetrievalMethod.AST_STRUCTURED,
                    relevance_score=ast_result.get("relevance_score", 0.5),
                    metadata=ast_result.get("metadata", {}),
                    file_path=ast_result.get("file"),
                    line_start=ast_result.get("start"),
                    line_end=ast_result.get("end"),
                    context=ast_result.get("context")
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"AST search failed: {e}")
            return []
    
    def _search_citations(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Search using citation information."""
        citations = query.context.get("citations", [])
        query_lower = query.query.lower()
        
        results = []
        for citation in citations:
            # Check if citation matches query
            citation_text = f"{citation.get('source', '')} {citation.get('reason', '')}"
            if query_lower in citation_text.lower():
                result = RetrievalResult(
                    content=citation.get("reason", ""),
                    source=citation.get("source", ""),
                    method=RetrievalMethod.CITATION,
                    relevance_score=0.8,  # High relevance for citation matches
                    metadata={"citation": citation}
                )
                results.append(result)
        
        return results
    
    def _combine_results(
        self, 
        all_results: List[RetrievalResult],
        method_results: Dict[RetrievalMethod, List[RetrievalResult]],
        weights: Dict[RetrievalMethod, float],
        query: RetrievalQuery
    ) -> List[RetrievalResult]:
        """Combine and re-rank results from multiple methods."""
        
        # Group results by content to avoid duplicates
        content_groups = {}
        
        for result in all_results:
            content_key = result.content[:100]  # First 100 chars as key
            
            if content_key not in content_groups:
                content_groups[content_key] = {
                    "results": [],
                    "methods": set(),
                    "max_score": 0.0
                }
            
            group = content_groups[content_key]
            group["results"].append(result)
            group["methods"].add(result.method)
            group["max_score"] = max(group["max_score"], result.relevance_score)
        
        # Calculate combined scores
        combined_results = []
        
        for content_key, group in content_groups.items():
            # Weighted score based on methods that found this content
            combined_score = 0.0
            method_weight_sum = 0.0
            
            for method in group["methods"]:
                weight = weights.get(method, 0.0)
                method_weight_sum += weight
                
                # Use the best score from this method
                method_results_list = method_results.get(method, [])
                method_scores = [r.relevance_score for r in method_results_list if r.content[:100] == content_key]
                
                if method_scores:
                    best_score = max(method_scores)
                    combined_score += weight * best_score
            
            # Normalize by total weight
            if method_weight_sum > 0:
                combined_score /= method_weight_sum
            
            # Boost score for multi-method matches
            method_boost = min(len(group["methods"]) / 3.0, 1.0)  # Max 33% boost
            combined_score *= (1.0 + method_boost)
            
            # Create combined result
            best_result = max(group["results"], key=lambda r: r.relevance_score)
            
            combined_result = RetrievalResult(
                content=best_result.content,
                source=best_result.source,
                method=RetrievalMethod.HYBRID,
                relevance_score=combined_score,
                metadata={
                    **best_result.metadata,
                    "methods_found": list(group["methods"]),
                    "original_scores": {method.value: r.relevance_score for r in group["results"]},
                    "method_boost": method_boost
                },
                file_path=best_result.file_path,
                line_start=best_result.line_start,
                line_end=best_result.line_end,
                context=best_result.context
            )
            
            combined_results.append(combined_result)
        
        # Sort by combined score
        combined_results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        return combined_results
    
    def get_context_for_query(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get formatted context for AI query."""
        retrieval_query = RetrievalQuery(
            query=query,
            methods=[RetrievalMethod.HYBRID],
            limit=limit
        )
        
        results = self.search(retrieval_query)
        
        # Format as RAG context
        context_snippets = []
        for result in results:
            snippet = {
                "file": result.file_path or "unknown",
                "start": result.line_start or 1,
                "end": result.line_end or result.line_start or 1,
                "text": result.content
            }
            context_snippets.append(snippet)
        
        # Format context and citations
        context_text, citations = format_context(context_snippets)
        
        return {
            "context": context_text,
            "citations": citations,
            "results": results,
            "metadata": {
                "query": query,
                "total_results": len(results),
                "methods_used": list(set(r.method for r in results))
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        return {
            "document_count": self.document_count,
            "semantic_available": self.semantic_retriever.available,
            "ast_index_stats": self.ast_index.get_statistics() if self.ast_index else {},
            "keyword_index_size": len(self.keyword_retriever.document_store),
            "inverted_index_size": len(self.keyword_retriever.inverted_index),
            "config": self.config
        }


# Global hybrid retrieval system instance
_hybrid_retrieval_system = None


def get_hybrid_retrieval_system(config: Optional[Dict[str, Any]] = None) -> HybridRetrievalSystem:
    """Get global hybrid retrieval system instance."""
    global _hybrid_retrieval_system
    if _hybrid_retrieval_system is None:
        _hybrid_retrieval_system = HybridRetrievalSystem(config)
    return _hybrid_retrieval_system


def search_code_context(
    query: str, 
    methods: Optional[List[str]] = None,
    limit: int = 5,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function to search code context."""
    system = get_hybrid_retrieval_system()
    
    # Convert string methods to enum
    method_enums = []
    if methods:
        method_map = {
            "semantic": RetrievalMethod.SEMANTIC,
            "ast": RetrievalMethod.AST_STRUCTURED,
            "keyword": RetrievalMethod.KEYWORD,
            "hybrid": RetrievalMethod.HYBRID,
            "citation": RetrievalMethod.CITATION
        }
        for method in methods:
            if method in method_map:
                method_enums.append(method_map[method])
    
    retrieval_query = RetrievalQuery(
        query=query,
        methods=method_enums or [RetrievalMethod.HYBRID],
        limit=limit,
        context=context or {}
    )
    
    return system.get_context_for_query(query, limit)