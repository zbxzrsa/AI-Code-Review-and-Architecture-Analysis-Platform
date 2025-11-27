# RAG Index Rebuild Guide

## Overview

This guide covers rebuilding the RAG (Retrieval-Augmented Generation) index for context-aware AI reviews.

## When to Rebuild

- **Codebase changes**: Major refactoring or new code added
- **Index corruption**: Search results are poor or missing
- **Model updates**: When changing embedding models
- **Scheduled**: Weekly or bi-weekly updates

## Rebuild Process

### 1. Clean Existing Index

```bash
# Remove existing RAG index
rm -rf ./ai_models/rag_index/

# Or backup first
mv ./ai_models/rag_index/ ./ai_models/rag_index_backup_$(date +%Y%m%d_%H%M%S)/
```

### 2. Run Ingestion

```bash
# Full repository ingestion
python -m backend.ai.rag.ingest \
    --repo . \
    --output ./ai_models/rag_index \
    --verbose

# Specific directory ingestion
python -m backend.ai.rag.ingest \
    --repo ./backend \
    --output ./ai_models/rag_index_backend \
    --verbose
```

### 3. Verify Index

```bash
# Test search functionality
python -m backend.ai.rag.search \
    --query "function name" \
    --index ./ai_models/rag_index \
    --k 5

# Check index statistics
ls -la ./ai_models/rag_index/
cat ./ai_models/rag_index/ingestion_summary.json
```

### 4. Update Configuration

```bash
# Update RAG index path in environment
echo "RAG_INDEX_PATH=./ai_models/rag_index" >> .env

# Restart services to pick up new index
docker compose restart api
```

## Advanced Options

### Custom Chunking

```bash
# Custom chunk size and overlap
python -m backend.ai.rag.ingest \
    --repo . \
    --output ./ai_models/rag_index \
    --chunk-size 1024 \
    --chunk-overlap 128
```

### Selective Ingestion

```bash
# Only specific file types
python -m backend.ai.rag.ingest \
    --repo . \
    --output ./ai_models/rag_index \
    --include "*.py,*.js,*.ts,*.tsx"

# Exclude directories
python -m backend.ai.rag.ingest \
    --repo . \
    --output ./ai_models/rag_index \
    --exclude "tests/*,node_modules/*,*.min.js"
```

### Model Selection

```bash
# Use different embedding model
export EMBEDDING_MODEL="all-mpnet-base-v2"
python -m backend.ai.rag.ingest \
    --repo . \
    --output ./ai_models/rag_index
```

## Troubleshooting

### Memory Issues

```bash
# Reduce chunk size for large repositories
python -m backend.ai.rag.ingest \
    --repo . \
    --output ./ai_models/rag_index \
    --chunk-size 256 \
    --chunk-overlap 32

# Process in batches
python -m backend.ai.rag.ingest \
    --repo . \
    --output ./ai_models/rag_index \
    --batch-size 1000
```

### Slow Ingestion

```bash
# Use keyword-only indexing (faster)
python -m backend.ai.rag.ingest \
    --repo . \
    --output ./ai_models/rag_index \
    --keyword-only

# Skip large files
python -m backend.ai.rag.ingest \
    --repo . \
    --output ./ai_models/rag_index \
    --max-file-size 1048576  # 1MB
```

### Poor Search Results

```bash
# Check index quality
python -m backend.ai.rag.search \
    --query "test query" \
    --index ./ai_models/rag_index \
    --k 10 \
    --format context

# Rebuild with different parameters
python -m backend.ai.rag.ingest \
    --repo . \
    --output ./ai_models/rag_index \
    --chunk-size 512 \
    --chunk-overlap 64
```

## Monitoring

### Index Statistics

```bash
# View ingestion summary
cat ./ai_models/rag_index/ingestion_summary.json

# Expected output:
{
  "timestamp": 1640995200.0,
  "repo_path": ".",
  "result": {
    "type": "vector",
    "num_chunks": 1234,
    "dimension": 384
  }
}
```

### Search Performance

```bash
# Benchmark search performance
time python -m backend.ai.rag.search \
    --query "function name" \
    --index ./ai_models/rag_index \
    --k 5

# Expected: < 100ms for typical queries
```

## Automation

### Scheduled Rebuild

```bash
# Add to crontab for weekly rebuild
0 2 * * 0 /path/to/project/scripts/rebuild_rag.sh

# rebuild_rag.sh
#!/bin/bash
cd /path/to/project
python -m backend.ai.rag.ingest --repo . --output ./ai_models/rag_index
docker compose restart api
```

### Git Hook Integration

```bash
# .git/hooks/post-merge
#!/bin/bash
# Auto-rebuild RAG index after merges to main
if [ "$(git rev-parse --abbrev-ref HEAD)" = "main" ]; then
    echo "Rebuilding RAG index after main merge..."
    python -m backend.ai.rag.ingest --repo . --output ./ai_models/rag_index
fi
```

## Best Practices

1. **Regular backups**: Always backup before rebuilding
2. **Test after rebuild**: Verify search quality
3. **Monitor performance**: Track ingestion and search times
4. **Version control**: Keep track of index configurations
5. **Resource monitoring**: Watch memory usage during ingestion

## Recovery

### From Backup

```bash
# Restore from backup
rm -rf ./ai_models/rag_index/
mv ./ai_models/rag_index_backup_20231201_120000/ ./ai_models/rag_index/
docker compose restart api
```

### Partial Rebuild

```bash
# Rebuild only specific directories
python -m backend.ai.rag.ingest \
    --repo ./backend/app \
    --output ./ai_models/rag_index_backend \
    --append  # Append to existing index
```
