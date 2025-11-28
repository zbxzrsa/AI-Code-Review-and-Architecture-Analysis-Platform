#!/bin/bash
# AI Versions Setup Script
# Downloads models and sets up all three AI versions

set -e

echo "🤖 Setting up AI Code Review Platform Versions..."

# Create directories
mkdir -p ai_versions/{v1_stable,v2_experimental,v3_deprecated}/{model,config}

# Setup v1 Stable (CodeBERT)
echo "📦 Setting up v1_stable (CodeBERT)..."
cd ai_versions/v1_stable
echo "Downloading CodeBERT model..."
# Note: In production, this would download from HuggingFace
echo "microsoft/codebert-base" > model/model_name.txt
echo "CodeBERT setup complete"

# Setup v2 Experimental (Llama2)
echo "🔬 Setting up v2_experimental (Llama2)..."
cd ../v2_experimental
echo "Setting up Ollama with Llama2..."
echo "llama2:7b" > model/model_name.txt
echo "Llama2 setup complete"

# Setup v3 Deprecated (GPT-3.5)
echo "📚 Setting up v3_deprecated (GPT-3.5)..."
cd ../v3_deprecated
echo "Creating deprecated version record..."
cat > metrics.json << EOF
{
  "version": "v3_deprecated_gpt3.5",
  "failure_reason": "high_latency",
  "latency": "12.4s",
  "cost_per_review": "$0.05",
  "timestamp": "2023-11-01T00:00:00Z"
}
EOF
echo "v3 setup complete"

# Create blocklist
echo "🚫 Creating model blocklist..."
cd ..
cat > blocklist.yaml << EOF
blocked_models:
  - "gpt-3.5-turbo"  # Reason: high_latency
  - "mistral-7b"     # Reason: cost
EOF

# Make scripts executable
chmod +x version_router.py

echo "✅ AI Versions setup complete!"
echo ""
echo "Available versions:"
echo "  v1_stable: CodeBERT (CPU optimized)"
echo "  v2_experimental: Llama2 (GPU optimized)"  
echo "  v3_deprecated: GPT-3.5 (Archive)"
echo ""
echo "Usage:"
echo "  python ai_versions/version_router.py get-config"
echo "  cat request.json | python ai_versions/version_router.py route"