# PyTorch Geometric Integration
# Feature flag: USE_PYG=true

import os
from typing import Optional, Dict, Any
import torch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from app.core.feature_flags import FeatureFlags

class GraphLearningService:
    """Graph learning service with PyTorch Geometric support."""
    
    def __init__(self):
        self.use_pyg = FeatureFlags.USE_PYG
        self._init_models()
    
    def _init_models(self):
        """Initialize graph neural network models."""
        if self.use_pyg:
            self.gcn_model = GCNConv(in_channels=64, out_channels=32)
            self.gat_model = GATConv(in_channels=64, out_channels=32, heads=4)
        else:
            # Fallback to DGL
            import dgl
            self.gcn_model = dgl.nn.GraphConv(64, 32)
            self.gat_model = None  # DGL equivalent
    
    def create_graph_data(self, nodes: list, edges: list, features: list) -> Any:
        """Create graph data structure."""
        if self.use_pyg:
            edge_index = torch.tensor([list(edge) for edge in edges], dtype=torch.long)
            x = torch.tensor(features, dtype=torch.float)
            return Data(x=x, edge_index=edge_index)
        else:
            import dgl
            import numpy as np
            src, dst = zip(*edges)
            g = dgl.graph((src, dst))
            g.ndata['feat'] = np.array(features)
            return g
    
    def analyze_code_graph(self, code_ast: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code AST as graph for pattern detection."""
        if self.use_pyg:
            return self._analyze_with_pyg(code_ast)
        else:
            return self._analyze_with_dgl(code_ast)
    
    def _analyze_with_pyg(self, code_ast: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze using PyTorch Geometric."""
        # Extract nodes and edges from AST
        nodes = self._extract_nodes(code_ast)
        edges = self._extract_edges(code_ast)
        features = self._extract_features(code_ast)
        
        # Create PyG data
        data = self.create_graph_data(nodes, edges, features)
        
        # Run graph neural network
        with torch.no_grad():
            # GCN analysis
            gcn_output = self.gcn_model(data.x, data.edge_index)
            
            # Global pooling
            pooled = global_mean_pool(gcn_output, data.batch)
            
            # Pattern detection
            patterns = self._detect_patterns(pooled)
            
            return {
                'patterns': patterns,
                'embeddings': pooled.tolist(),
                'graph_stats': {
                    'nodes': len(nodes),
                    'edges': len(edges),
                    'features': len(features[0]) if features else 0
                }
            }
    
    def _analyze_with_dgl(self, code_ast: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze using DGL (fallback)."""
        import dgl
        import numpy as np
        
        # Create DGL graph
        nodes = self._extract_nodes(code_ast)
        edges = self._extract_edges(code_ast)
        features = self._extract_features(code_ast)
        
        src, dst = zip(*edges)
        g = dgl.graph((src, dst))
        g.ndata['feat'] = np.array(features)
        
        # Run DGL GCN
        output = self.gcn_model(g, g.ndata['feat'])
        
        return {
            'patterns': [],
            'embeddings': output.numpy().tolist(),
            'graph_stats': {
                'nodes': len(nodes),
                'edges': len(edges),
                'features': len(features[0]) if features else 0
            }
        }
    
    def _extract_nodes(self, code_ast: Dict[str, Any]) -> list:
        """Extract nodes from AST."""
        nodes = []
        for item in code_ast.get('functions', []):
            nodes.append({
                'id': item.get('name', ''),
                'type': 'function',
                'line': item.get('line', 0)
            })
        return nodes
    
    def _extract_edges(self, code_ast: Dict[str, Any]) -> list:
        """Extract edges from AST."""
        edges = []
        for item in code_ast.get('functions', []):
            # Add edges based on function calls
            for call in item.get('calls', []):
                edges.append([item.get('name', ''), call])
        return edges
    
    def _extract_features(self, code_ast: Dict[str, Any]) -> list:
        """Extract node features."""
        features = []
        for item in code_ast.get('functions', []):
            # Create feature vector
            feature_vector = [
                len(item.get('name', '')),  # name length
                len(item.get('params', [])),  # parameter count
                item.get('complexity', 0),  # complexity
                1 if 'async' in item.get('name', '') else 0,  # is async
            ]
            features.append(feature_vector)
        return features
    
    def _detect_patterns(self, embeddings: torch.Tensor) -> list:
        """Detect patterns using graph embeddings."""
        patterns = []
        
        # Simple pattern detection based on embeddings
        for i, embedding in enumerate(embeddings):
            if embedding.max() > 0.8:  # threshold for pattern detection
                patterns.append({
                    'type': 'complex_pattern',
                    'confidence': float(embedding.max()),
                    'node_id': i
                })
        
        return patterns