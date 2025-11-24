# pgvector Vector Search Integration
# Feature flag: USE_PGVECTOR=true

import os
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from app.core.feature_flags import FeatureFlags

class VectorSearchService:
    """Vector search service with pgvector integration."""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.use_pgvector = FeatureFlags.USE_PGVECTOR
    
    async def init_pgvector(self):
        """Initialize pgvector extension if enabled."""
        if not self.use_pgvector:
            return
        
        try:
            await self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await self.db.commit()
            print("pgvector extension initialized")
        except Exception as e:
            print(f"Failed to initialize pgvector: {e}")
    
    async def create_embedding_index(self, table_name: str, column_name: str):
        """Create vector index for faster similarity search."""
        if not self.use_pgvector:
            return
        
        try:
            index_sql = f"""
                CREATE INDEX IF NOT EXISTS {table_name}_{column_name}_idx 
                ON {table_name} 
                USING ivfflat ({column_name} vector_cosine_ops);
            """
            await self.db.execute(text(index_sql))
            await self.db.commit()
            print(f"Created vector index for {table_name}.{column_name}")
        except Exception as e:
            print(f"Failed to create index: {e}")
    
    async def store_code_embedding(
        self,
        code_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store code embedding in PostgreSQL with pgvector."""
        if self.use_pgvector:
            await self._store_pgvector_embedding(code_id, content, embedding, metadata)
        else:
            await self._store_neo4j_embedding(code_id, content, embedding, metadata)
    
    async def _store_pgvector_embedding(
        self,
        code_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store embedding using pgvector."""
        try:
            embedding_str = f"[{','.join(map(str, embedding))}]"
            
            sql = text("""
                INSERT INTO code_embeddings (id, content, embedding, metadata)
                VALUES (:id, :content, :embedding::vector, :metadata)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """)
            
            await self.db.execute(sql, {
                'id': code_id,
                'content': content,
                'embedding': embedding_str,
                'metadata': metadata or {}
            })
            await self.db.commit()
            
        except Exception as e:
            print(f"Failed to store pgvector embedding: {e}")
            await self.db.rollback()
    
    async def _store_neo4j_embedding(
        self,
        code_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store embedding using Neo4j (fallback)."""
        # Fallback to existing Neo4j implementation
        from app.services.neo4j_service import Neo4jService
        
        neo4j_service = Neo4jService()
        await neo4j_service.store_code_embedding(code_id, content, embedding, metadata)
    
    async def search_similar_code(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar code using vector similarity."""
        if self.use_pgvector:
            return await self._search_pgvector_similar(query_embedding, limit, threshold)
        else:
            return await self._search_neo4j_similar(query_embedding, limit, threshold)
    
    async def _search_pgvector_similar(
        self,
        query_embedding: List[float],
        limit: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Search using pgvector similarity."""
        try:
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            sql = text("""
                SELECT id, content, metadata,
                       1 - (embedding <=> :query_embedding::vector) as similarity
                FROM code_embeddings
                WHERE 1 - (embedding <=> :query_embedding::vector) > :threshold
                ORDER BY embedding <=> :query_embedding::vector
                LIMIT :limit
            """)
            
            result = await self.db.execute(sql, {
                'query_embedding': embedding_str,
                'threshold': threshold,
                'limit': limit
            })
            
            rows = result.fetchall()
            return [
                {
                    'id': row.id,
                    'content': row.content,
                    'metadata': row.metadata,
                    'similarity': float(row.similarity)
                }
                for row in rows
            ]
            
        except Exception as e:
            print(f"Failed to search pgvector: {e}")
            return []
    
    async def _search_neo4j_similar(
        self,
        query_embedding: List[float],
        limit: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Fallback to Neo4j similarity search."""
        # Implement Neo4j similarity search as fallback
        return []
    
    async def create_tables(self):
        """Create necessary tables for vector search."""
        if not self.use_pgvector:
            return
        
        try:
            sql = text("""
                CREATE TABLE IF NOT EXISTS code_embeddings (
                    id VARCHAR(255) PRIMARY KEY,
                    content TEXT,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            await self.db.execute(sql)
            await self.db.commit()
            print("Created code_embeddings table")
            
        except Exception as e:
            print(f"Failed to create tables: {e}")
            await self.db.rollback()