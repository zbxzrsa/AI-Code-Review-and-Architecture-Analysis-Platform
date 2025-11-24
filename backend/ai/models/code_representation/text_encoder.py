import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class CodeBERTEncoder(nn.Module):
    """
    基于CodeBERT的代码文本编码器
    """
    def __init__(self, hidden_size=768, dropout_rate=0.1, freeze_base=False):
        """
        初始化CodeBERT编码器
        
        Args:
            hidden_size: 隐藏层维度
            dropout_rate: Dropout比率
            freeze_base: 是否冻结预训练模型参数
        """
        super(CodeBERTEncoder, self).__init__()
        
        # 加载预训练的CodeBERT模型
        self.codebert = RobertaModel.from_pretrained('microsoft/codebert-base')
        
        # 冻结基础模型参数（可选）
        if freeze_base:
            for param in self.codebert.parameters():
                param.requires_grad = False
        
        # 输出层
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(self.codebert.config.hidden_size, hidden_size)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: token类型ID [batch_size, seq_len]
            
        Returns:
            sequence_output: 序列输出 [batch_size, seq_len, hidden_size]
            pooled_output: 池化输出 [batch_size, hidden_size]
        """
        # 获取CodeBERT输出
        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # 获取序列输出和[CLS]表示
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # 应用dropout和输出层
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.output_layer(pooled_output)
        
        return sequence_output, pooled_output
    
    def get_masked_prediction_logits(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        获取用于掩码语言建模的预测logits
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: token类型ID [batch_size, seq_len]
            
        Returns:
            prediction_logits: 预测logits [batch_size, seq_len, vocab_size]
        """
        # 获取CodeBERT输出
        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # 获取序列输出
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 获取预测logits
        prediction_logits = self.codebert.lm_head(sequence_output)  # [batch_size, seq_len, vocab_size]
        
        return prediction_logits