#!/usr/bin/env python3
"""
RAG Ingestion - Index repository code and documentation for context-aware reviews
Usage: python -m backend.ai.rag.ingest --repo ./ --output ./ai_models/rag_index
"""
import argparse
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

class RAGIngestor:
    def __init__(self, repo_path: str, output_path: str):
        self.repo_path = Path(repo_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model if available
        if RAG_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.chunk_size = 512  # tokens
            self.chunk_overlap = 64
        else:
            self.model = None
            print("Warning: RAG dependencies not available. Using keyword-only indexing.")
    
    def should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored"""
        ignore_patterns = [
            'node_modules', '.git', '.vscode', '.idea', 
            'ai_models', '__pycache__', '.pytest_cache',
            'dist', 'build', 'coverage', 'htmlcov',
            '*.min.js', '*.min.css', '*.map',
            '.env', '.DS_Store'
        ]
        
        path_str = str(file_path)
        for pattern in ignore_patterns:
            if pattern in path_str or path_str.endswith(pattern.replace('*', '')):
                return True
        return False
    
    def read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content with encoding handling"""
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception:
                return None
        except Exception:
            return None
    
    def chunk_text(self, text: str, file_path: Path) -> List[Dict[str, Any]]:
        """Chunk text into overlapping segments"""
        if not self.model:
            # Simple line-based chunking for keyword search
            lines = text.split('\n')
            chunks = []
            for i, line in enumerate(lines):
                if line.strip():  # Skip empty lines
                    chunks.append({
                        'text': line.strip(),
                        'file': str(file_path),
                        'line_start': i + 1,
                        'line_end': i + 1,
                        'chunk_id': hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()[:8]
                    })
            return chunks
        
        # Token-based chunking with overlap
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Estimate line numbers (rough approximation)
            line_start = text[:text.find(chunk_text)].count('\n') + 1
            line_end = line_start + chunk_text.count('\n')
            
            chunks.append({
                'text': chunk_text,
                'file': str(file_path),
                'line_start': line_start,
                'line_end': line_end,
                'chunk_id': hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()[:8]
            })
        
        return chunks
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file path"""
        return {
            'file_path': str(file_path),
            'relative_path': str(file_path.relative_to(self.repo_path)),
            'extension': file_path.suffix,
            'size': file_path.stat().st_size if file_path.exists() else 0,
            'language': self.detect_language(file_path)
        }
    
    def detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.sh': 'bash',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.less': 'less',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.xml': 'xml',
            '.md': 'markdown',
            '.txt': 'text'
        }
        return ext_map.get(file_path.suffix.lower(), 'unknown')
    
    def ingest_repository(self) -> Dict[str, Any]:
        """Main ingestion process"""
        print(f"Starting RAG ingestion for {self.repo_path}")
        
        all_chunks = []
        file_count = 0
        skipped_count = 0
        
        # Walk through repository
        for file_path in self.repo_path.rglob('*'):
            if file_path.is_file() and not self.should_ignore_file(file_path):
                content = self.read_file_content(file_path)
                if content:
                    chunks = self.chunk_text(content, file_path)
                    for chunk in chunks:
                        chunk.update(self.extract_metadata(file_path))
                        all_chunks.append(chunk)
                    file_count += 1
                else:
                    skipped_count += 1
        
        print(f"Processed {file_count} files, skipped {skipped_count}")
        print(f"Generated {len(all_chunks)} chunks")
        
        # Create embeddings and index
        if RAG_AVAILABLE and self.model:
            return self.create_vector_index(all_chunks)
        else:
            return self.create_keyword_index(all_chunks)
    
    def create_vector_index(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create FAISS vector index"""
        print("Creating vector embeddings...")
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save index and metadata
        index_path = self.output_path / 'faiss.index'
        faiss.write_index(index, str(index_path))
        
        # Save metadata
        metadata_path = self.output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(chunks, f, indent=2)
        
        print(f"Vector index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return {
            'type': 'vector',
            'index_path': str(index_path),
            'metadata_path': str(metadata_path),
            'num_chunks': len(chunks),
            'dimension': dimension
        }
    
    def create_keyword_index(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create keyword-based index (fallback)"""
        print("Creating keyword index...")
        
        # Build inverted index
        keyword_index = {}
        
        for i, chunk in enumerate(chunks):
            words = set(chunk['text'].lower().split())
            for word in words:
                if word not in keyword_index:
                    keyword_index[word] = []
                keyword_index[word].append(i)
        
        # Save index and metadata
        index_path = self.output_path / 'keyword_index.json'
        with open(index_path, 'w') as f:
            json.dump(keyword_index, f, indent=2)
        
        metadata_path = self.output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(chunks, f, indent=2)
        
        print(f"Keyword index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return {
            'type': 'keyword',
            'index_path': str(index_path),
            'metadata_path': str(metadata_path),
            'num_chunks': len(chunks),
            'num_keywords': len(keyword_index)
        }

def main():
    parser = argparse.ArgumentParser(description="RAG Ingestion")
    parser.add_argument("--repo", default=".", help="Repository path to index")
    parser.add_argument("--output", default="./ai_models/rag_index", help="Output directory for index")
    args = parser.parse_args()
    
    ingestor = RAGIngestor(args.repo, args.output)
    result = ingestor.ingest_repository()
    
    # Save ingestion summary
    summary_path = Path(args.output) / 'ingestion_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'repo_path': args.repo,
            'result': result
        }, f, indent=2)
    
    print(f"Ingestion completed: {summary_path}")
    return 0

if __name__ == "__main__":
    exit(main())