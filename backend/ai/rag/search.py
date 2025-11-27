#!/usr/bin/env python3
"""
RAG Search - Retrieve relevant context for AI reviews
Usage: python -m backend.ai.rag.search --query "function name" --index ./ai_models/rag_index
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

class RAGSearcher:
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.metadata_path = self.index_path / 'metadata.json'
        self.keyword_index_path = self.index_path / 'keyword_index.json'
        self.faiss_index_path = self.index_path / 'faiss.index'
        
        # Load metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.chunks = json.load(f)
        else:
            self.chunks = []
        
        # Load index based on type
        if self.faiss_index_path.exists() and RAG_AVAILABLE:
            self.index_type = 'vector'
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = faiss.read_index(str(self.faiss_index_path))
        elif self.keyword_index_path.exists():
            self.index_type = 'keyword'
            with open(self.keyword_index_path, 'r') as f:
                self.keyword_index = json.load(f)
        else:
            self.index_type = 'none'
            print("Warning: No RAG index found")
    
    def search_vector(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using vector similarity"""
        if self.index_type != 'vector':
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(1 / (1 + dist))  # Convert distance to similarity
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results
    
    def search_keyword(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using keyword matching"""
        if self.index_type != 'keyword':
            return []
        
        query_words = set(query.lower().split())
        scored_chunks = {}
        
        # Find chunks matching keywords
        for word in query_words:
            if word in self.keyword_index:
                for chunk_idx in self.keyword_index[word]:
                    if chunk_idx not in scored_chunks:
                        scored_chunks[chunk_idx] = {
                            'chunk': self.chunks[chunk_idx],
                            'matched_words': [],
                            'score': 0
                        }
                    scored_chunks[chunk_idx]['matched_words'].append(word)
                    scored_chunks[chunk_idx]['score'] += 1
        
        # Sort by score and limit to k results
        sorted_chunks = sorted(
            scored_chunks.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:k]
        
        results = []
        for i, item in enumerate(sorted_chunks):
            chunk = item['chunk'].copy()
            chunk['similarity_score'] = item['score'] / len(query_words)  # Normalize by query length
            chunk['matched_words'] = item['matched_words']
            chunk['rank'] = i + 1
            results.append(chunk)
        
        return results
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Universal search method"""
        if self.index_type == 'vector':
            return self.search_vector(query, k)
        elif self.index_type == 'keyword':
            return self.search_keyword(query, k)
        else:
            return []
    
    def format_context(self, results: List[Dict[str, Any]], max_chars: int = 2000) -> str:
        """Format search results as context for AI prompt"""
        if not results:
            return "No relevant context found."
        
        context_parts = []
        current_chars = 0
        
        for result in results:
            if current_chars >= max_chars:
                break
            
            chunk_text = result['text']
            file_path = result.get('relative_path', result['file'])
            line_info = f"{result.get('line_start', '?')}-{result.get('line_end', '?')}"
            
            context_part = f"File: {file_path} (lines {line_info})\n{chunk_text}\n"
            
            if current_chars + len(context_part) <= max_chars:
                context_parts.append(context_part)
                current_chars += len(context_part)
            else:
                # Add partial chunk
                remaining_chars = max_chars - current_chars
                partial_text = chunk_text[:remaining_chars - 50] + "..."
                context_part = f"File: {file_path} (lines {line_info})\n{partial_text}\n"
                context_parts.append(context_part)
                break
        
        return "Context:\n" + "\n".join(context_parts)

def main():
    parser = argparse.ArgumentParser(description="RAG Search")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--index", default="./ai_models/rag_index", help="RAG index directory")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--format", choices=["json", "context"], default="json", help="Output format")
    args = parser.parse_args()
    
    searcher = RAGSearcher(args.index)
    results = searcher.search(args.query, args.k)
    
    if args.format == "context":
        output = searcher.format_context(results)
    else:
        output = {
            'query': args.query,
            'index_type': searcher.index_type,
            'num_results': len(results),
            'results': results
        }
        output = json.dumps(output, indent=2)
    
    print(output)
    return 0

if __name__ == "__main__":
    exit(main())