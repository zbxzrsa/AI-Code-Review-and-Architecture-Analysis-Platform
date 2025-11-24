import torch
import torch.nn as nn

from .text_encoder import CodeBERTEncoder
from .graph_encoder import GATEncoder
from .multimodal_fusion import MultimodalFusion
from .training_objectives import ContrastiveLearningLoss, MaskedLanguageModelingLoss, GraphStructurePredictionLoss

class MultimodalCodeRepresentation(nn.Module):
    """
    多模态代码表示学习模型
    """
    def __init__(
        self,
        hidden_size=768,
        graph_hidden_size=256,
        graph_in_channels=300,
        num_graph_layers=2,
        num_attention_heads=8,
        dropout_rate=0.1,
        vocab_size=50265,  # CodeBERT词汇表大小
        freeze_text_encoder=False
    ):
        """
        初始化多模态代码表示学习模型
        
        Args:
            hidden_size: 隐藏层维度
            graph_hidden_size: 图编码器隐藏层维度
            graph_in_channels: 图节点初始特征维度
            num_graph_layers: 图编码器层数
            num_attention_heads: 注意力头数
            dropout_rate: Dropout比率
            vocab_size: 词汇表大小
            freeze_text_encoder: 是否冻结文本编码器
        """
        super(MultimodalCodeRepresentation, self).__init__()
        
        # 文本编码器 (CodeBERT)
        self.text_encoder = CodeBERTEncoder(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            freeze_base=freeze_text_encoder
        )
        
        # AST图编码器
        self.ast_encoder = GATEncoder(
            in_channels=graph_in_channels,
            hidden_channels=graph_hidden_size,
            out_channels=hidden_size,
            num_layers=num_graph_layers,
            heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # CFG图编码器
        self.cfg_encoder = GATEncoder(
            in_channels=graph_in_channels,
            hidden_channels=graph_hidden_size,
            out_channels=hidden_size,
            num_layers=num_graph_layers,
            heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # DFG图编码器
        self.dfg_encoder = GATEncoder(
            in_channels=graph_in_channels,
            hidden_channels=graph_hidden_size,
            out_channels=hidden_size,
            num_layers=num_graph_layers,
            heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # 多模态融合
        self.multimodal_fusion = MultimodalFusion(
            hidden_size=hidden_size,
            num_modalities=4,  # 文本、AST、CFG、DFG
            dropout_rate=dropout_rate
        )
        
        # 训练目标
        self.contrastive_loss_fn = ContrastiveLearningLoss(temperature=0.07)
        self.mlm_loss_fn = MaskedLanguageModelingLoss(vocab_size=vocab_size)
        self.graph_loss_fn = GraphStructurePredictionLoss()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        ast_node_features=None,
        ast_edge_index=None,
        ast_batch=None,
        cfg_node_features=None,
        cfg_edge_index=None,
        cfg_batch=None,
        dfg_node_features=None,
        dfg_edge_index=None,
        dfg_batch=None,
        masked_lm_labels=None,
        edge_labels=None,
        similar_code_ids=None,
        return_embeddings=False
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: token类型ID [batch_size, seq_len]
            ast_node_features: AST节点特征 [num_ast_nodes, in_channels]
            ast_edge_index: AST边索引 [2, num_ast_edges]
            ast_batch: AST批处理索引 [num_ast_nodes]
            cfg_node_features: CFG节点特征 [num_cfg_nodes, in_channels]
            cfg_edge_index: CFG边索引 [2, num_cfg_edges]
            cfg_batch: CFG批处理索引 [num_cfg_nodes]
            dfg_node_features: DFG节点特征 [num_dfg_nodes, in_channels]
            dfg_edge_index: DFG边索引 [2, num_dfg_edges]
            dfg_batch: DFG批处理索引 [num_dfg_nodes]
            masked_lm_labels: 掩码语言建模标签 [batch_size, seq_len]
            edge_labels: 边标签 [num_edges]
            similar_code_ids: 相似代码ID [batch_size]
            return_embeddings: 是否返回嵌入
            
        Returns:
            outputs: 模型输出
        """
        outputs = {}
        
        # 1. 文本编码
        if input_ids is not None:
            text_sequence_output, text_pooled_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            # 如果没有文本输入，创建零张量
            batch_size = ast_batch[-1].item() + 1 if ast_batch is not None else 1
            text_pooled_output = torch.zeros(batch_size, self.text_encoder.output_layer.out_features, device=self.device)
        
        # 2. AST编码
        if ast_node_features is not None and ast_edge_index is not None:
            ast_node_embeddings, ast_graph_embedding = self.ast_encoder(
                x=ast_node_features,
                edge_index=ast_edge_index,
                batch=ast_batch
            )
        else:
            # 如果没有AST输入，创建零张量
            batch_size = input_ids.size(0) if input_ids is not None else 1
            ast_graph_embedding = torch.zeros(batch_size, self.ast_encoder.pool[0].out_features, device=self.device)
            ast_node_embeddings = None
        
        # 3. CFG编码
        if cfg_node_features is not None and cfg_edge_index is not None:
            cfg_node_embeddings, cfg_graph_embedding = self.cfg_encoder(
                x=cfg_node_features,
                edge_index=cfg_edge_index,
                batch=cfg_batch
            )
        else:
            # 如果没有CFG输入，创建零张量
            batch_size = input_ids.size(0) if input_ids is not None else 1
            cfg_graph_embedding = torch.zeros(batch_size, self.cfg_encoder.pool[0].out_features, device=self.device)
            cfg_node_embeddings = None
        
        # 4. DFG编码
        if dfg_node_features is not None and dfg_edge_index is not None:
            dfg_node_embeddings, dfg_graph_embedding = self.dfg_encoder(
                x=dfg_node_features,
                edge_index=dfg_edge_index,
                batch=dfg_batch
            )
        else:
            # 如果没有DFG输入，创建零张量
            batch_size = input_ids.size(0) if input_ids is not None else 1
            dfg_graph_embedding = torch.zeros(batch_size, self.dfg_encoder.pool[0].out_features, device=self.device)
            dfg_node_embeddings = None
        
        # 5. 多模态融合
        modality_features = [
            text_pooled_output,
            ast_graph_embedding,
            cfg_graph_embedding,
            dfg_graph_embedding
        ]
        
        code_embeddings = self.multimodal_fusion(modality_features)
        outputs['code_embeddings'] = code_embeddings
        
        # 如果只需要返回嵌入，提前返回
        if return_embeddings:
            return outputs
        
        # 6. 计算训练目标损失
        
        # 6.1 对比学习损失
        if similar_code_ids is not None:
            contrastive_loss = self.contrastive_loss_fn(code_embeddings, similar_code_ids)
            outputs['contrastive_loss'] = contrastive_loss
        
        # 6.2 掩码语言建模损失
        if masked_lm_labels is not None and input_ids is not None:
            prediction_logits = self.text_encoder.get_masked_prediction_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            mlm_loss = self.mlm_loss_fn(prediction_logits, masked_lm_labels)
            outputs['mlm_loss'] = mlm_loss
            outputs['prediction_logits'] = prediction_logits
        
        # 6.3 图结构预测损失
        if edge_labels is not None and ast_node_embeddings is not None and ast_edge_index is not None:
            edge_scores = self.ast_encoder.predict_edge(ast_node_embeddings, ast_edge_index)
            graph_loss = self.graph_loss_fn(edge_scores, edge_labels)
            outputs['graph_loss'] = graph_loss
            outputs['edge_scores'] = edge_scores
        
        return outputs
    
    @property
    def device(self):
        """获取模型设备"""
        return next(self.parameters()).device