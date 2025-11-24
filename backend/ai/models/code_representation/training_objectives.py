import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLearningLoss(nn.Module):
    """
    对比学习损失，使相似代码片段的表示相近，不相似的表示相远
    """
    def __init__(self, temperature=0.07):
        """
        初始化对比学习损失
        
        Args:
            temperature: 温度参数，控制分布的平滑程度
        """
        super(ContrastiveLearningLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels=None):
        """
        计算对比学习损失
        
        Args:
            embeddings: 代码嵌入 [batch_size, hidden_size]
            labels: 相似性标签 [batch_size]，相同标签表示相似样本
                   如果为None，则假设每个样本只与自身相似
        
        Returns:
            loss: 对比学习损失
        """
        batch_size = embeddings.size(0)
        
        # 计算余弦相似度矩阵
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.transpose(0, 1)) / self.temperature
        
        # 创建标签矩阵
        if labels is None:
            # 如果没有提供标签，则假设对角线为正样本
            labels = torch.arange(batch_size, device=embeddings.device)
        
        # 创建掩码，标识正样本对
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
        mask.fill_diagonal_(False)  # 排除自身
        
        # 对每个样本，计算与其他正样本的相似度
        pos_similarity = torch.log_softmax(similarity_matrix, dim=1)
        
        # 计算正样本对的损失
        pos_mask = mask.float()
        pos_count = pos_mask.sum(dim=1)
        pos_loss = torch.zeros(batch_size, device=embeddings.device)
        
        for i in range(batch_size):
            if pos_count[i] > 0:
                pos_loss[i] = -torch.sum(pos_similarity[i] * pos_mask[i]) / pos_count[i]
        
        # 平均损失
        loss = pos_loss.mean()
        
        return loss

class MaskedLanguageModelingLoss(nn.Module):
    """
    掩码语言建模损失，预测被掩盖的代码token
    """
    def __init__(self, vocab_size):
        """
        初始化掩码语言建模损失
        
        Args:
            vocab_size: 词汇表大小
        """
        super(MaskedLanguageModelingLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, prediction_logits, masked_lm_labels):
        """
        计算掩码语言建模损失
        
        Args:
            prediction_logits: 预测logits [batch_size, seq_len, vocab_size]
            masked_lm_labels: 掩码标签 [batch_size, seq_len]，被掩盖的位置为实际token id，其他位置为-100
        
        Returns:
            loss: 掩码语言建模损失
        """
        batch_size, seq_len, vocab_size = prediction_logits.size()
        
        # 重塑logits和标签
        prediction_logits = prediction_logits.view(-1, vocab_size)
        masked_lm_labels = masked_lm_labels.view(-1)
        
        # 计算损失
        loss = self.loss_fn(prediction_logits, masked_lm_labels)
        
        return loss

class GraphStructurePredictionLoss(nn.Module):
    """
    图结构预测损失，预测AST中的父子关系
    """
    def __init__(self):
        """
        初始化图结构预测损失
        """
        super(GraphStructurePredictionLoss, self).__init__()
        self.loss_fn = nn.BCELoss()
        
    def forward(self, edge_scores, edge_labels):
        """
        计算图结构预测损失
        
        Args:
            edge_scores: 边存在概率 [num_edges]
            edge_labels: 边标签 [num_edges]，1表示存在边，0表示不存在
        
        Returns:
            loss: 图结构预测损失
        """
        # 计算二元交叉熵损失
        loss = self.loss_fn(edge_scores, edge_labels)
        
        return loss