import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    """
    基于图注意力网络(GAT)的图结构编码器
    用于编码AST、CFG和DFG
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=8, dropout=0.1):
        """
        初始化GAT编码器
        
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出特征维度
            num_layers: GAT层数
            heads: 注意力头数
            dropout: Dropout比率
        """
        super(GATEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 第一层GAT
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        
        # 中间层GAT
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
        
        # 最后一层GAT (多头注意力结果合并为单一输出)
        if num_layers > 1:
            self.conv_layers.append(
                GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
            )
        
        # 节点级别到图级别的池化层
        self.pool = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            batch: 批处理索引 [num_nodes]
            
        Returns:
            node_embeddings: 节点嵌入 [num_nodes, out_channels]
            graph_embedding: 图嵌入 [batch_size, out_channels]
        """
        # 应用GAT层
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 获取节点嵌入
        node_embeddings = x
        
        # 图级别池化 (如果提供了batch信息)
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            graph_embedding = global_mean_pool(node_embeddings, batch)
            graph_embedding = self.pool(graph_embedding)
        else:
            # 如果没有batch信息，使用平均池化
            graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)
            graph_embedding = self.pool(graph_embedding)
        
        return node_embeddings, graph_embedding
    
    def predict_edge(self, node_embeddings, edge_index):
        """
        预测边的存在概率，用于图结构预测任务
        
        Args:
            node_embeddings: 节点嵌入 [num_nodes, out_channels]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            edge_scores: 边存在概率 [num_edges]
        """
        # 获取源节点和目标节点的嵌入
        src_embeddings = node_embeddings[edge_index[0]]  # [num_edges, out_channels]
        dst_embeddings = node_embeddings[edge_index[1]]  # [num_edges, out_channels]
        
        # 计算边的存在概率
        edge_scores = torch.sum(src_embeddings * dst_embeddings, dim=1)  # [num_edges]
        edge_scores = torch.sigmoid(edge_scores)
        
        return edge_scores