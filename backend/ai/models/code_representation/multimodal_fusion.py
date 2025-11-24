import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制，用于融合不同模态的特征
    """
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(CrossModalAttention, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 查询、键、值投影
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, query, key_value):
        """
        前向传播
        
        Args:
            query: 查询特征 [batch_size, hidden_size]
            key_value: 键值特征 [batch_size, hidden_size]
            
        Returns:
            output: 注意力输出 [batch_size, hidden_size]
        """
        # 投影查询、键、值
        q = self.query_proj(query).unsqueeze(1)  # [batch_size, 1, hidden_size]
        k = self.key_proj(key_value).unsqueeze(1)  # [batch_size, 1, hidden_size]
        v = self.value_proj(key_value).unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, 1, 1]
        attn_scores = attn_scores / (self.hidden_size ** 0.5)
        
        # 应用softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 加权求和
        context = torch.matmul(attn_probs, v)  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]
        
        # 输出投影
        output = self.output_proj(context)
        
        # 残差连接和层归一化
        output = self.layer_norm(query + output)
        
        return output

class MultimodalFusion(nn.Module):
    """
    多模态融合模块，融合文本和图特征
    """
    def __init__(self, hidden_size, num_modalities=3, dropout_rate=0.1):
        """
        初始化多模态融合模块
        
        Args:
            hidden_size: 隐藏层维度
            num_modalities: 模态数量 (文本、AST、CFG、DFG)
            dropout_rate: Dropout比率
        """
        super(MultimodalFusion, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        
        # 跨模态注意力层
        self.cross_attentions = nn.ModuleList([
            CrossModalAttention(hidden_size, dropout_rate)
            for _ in range(num_modalities)
        ])
        
        # 模态融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * num_modalities, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, modality_features):
        """
        前向传播
        
        Args:
            modality_features: 模态特征列表 [batch_size, hidden_size] * num_modalities
                索引0: 文本特征
                索引1: AST特征
                索引2: CFG特征
                索引3: DFG特征 (如果有)
            
        Returns:
            fused_embedding: 融合后的嵌入 [batch_size, hidden_size]
        """
        batch_size = modality_features[0].size(0)
        
        # 跨模态注意力
        attended_features = []
        for i, feature in enumerate(modality_features):
            # 对每个模态，使用其他模态的平均特征作为查询
            other_features = [f for j, f in enumerate(modality_features) if j != i]
            query = torch.stack(other_features, dim=0).mean(dim=0)  # [batch_size, hidden_size]
            
            # 应用跨模态注意力
            attended_feature = self.cross_attentions[i](query, feature)  # [batch_size, hidden_size]
            attended_features.append(attended_feature)
        
        # 拼接所有特征
        concat_features = torch.cat(attended_features, dim=1)  # [batch_size, hidden_size * num_modalities]
        
        # 融合层
        fused_embedding = self.fusion_layer(concat_features)  # [batch_size, hidden_size]
        
        # 残差连接和层归一化 (使用第一个模态特征作为残差)
        fused_embedding = self.layer_norm(modality_features[0] + fused_embedding)
        
        return fused_embedding