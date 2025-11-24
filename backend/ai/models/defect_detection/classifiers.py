import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional

class BinaryDefectClassifier(nn.Module):
    """
    二进制代码缺陷分类器（有缺陷/无缺陷）
    """
    def __init__(self, input_dim: int = 768, hidden_dims: List[int] = [512, 256], dropout: float = 0.3):
        """
        初始化二进制代码缺陷分类器
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            dropout: Dropout比率
        """
        super(BinaryDefectClassifier, self).__init__()
        
        # 构建多层感知机
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            output: 输出张量 [batch_size, 1]
        """
        return self.mlp(x)

class MultiLabelDefectClassifier(nn.Module):
    """
    多标签代码缺陷分类器（缺陷类型）
    """
    def __init__(self, input_dim: int = 768, hidden_dims: List[int] = [512, 256], num_classes: int = 5, dropout: float = 0.3):
        """
        初始化多标签代码缺陷分类器
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            num_classes: 类别数量
            dropout: Dropout比率
        """
        super(MultiLabelDefectClassifier, self).__init__()
        
        # 构建多层感知机
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            output: 输出张量 [batch_size, num_classes]
        """
        features = self.mlp(x)
        logits = self.classifier(features)
        return torch.sigmoid(logits)

class SeverityClassifier(nn.Module):
    """
    严重程度评估分类器
    """
    def __init__(self, input_dim: int = 768, hidden_dims: List[int] = [512, 256], num_classes: int = 3, dropout: float = 0.3):
        """
        初始化严重程度评估分类器
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            num_classes: 类别数量（低、中、高）
            dropout: Dropout比率
        """
        super(SeverityClassifier, self).__init__()
        
        # 构建多层感知机
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            output: 输出张量 [batch_size, num_classes]
        """
        features = self.mlp(x)
        logits = self.classifier(features)
        return logits

class DefectDetectionModel(nn.Module):
    """
    代码缺陷检测模型，集成三个分类器
    """
    def __init__(
        self, 
        input_dim: int = 768, 
        hidden_dims: List[int] = [512, 256], 
        num_defect_types: int = 5,
        num_severity_levels: int = 3,
        dropout: float = 0.3,
        freeze_code_model: bool = True
    ):
        """
        初始化代码缺陷检测模型
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            num_defect_types: 缺陷类型数量
            num_severity_levels: 严重程度级别数量
            dropout: Dropout比率
            freeze_code_model: 是否冻结代码表示模型
        """
        super(DefectDetectionModel, self).__init__()
        
        # 二进制分类器
        self.binary_classifier = BinaryDefectClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        # 多标签分类器
        self.multilabel_classifier = MultiLabelDefectClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_defect_types,
            dropout=dropout
        )
        
        # 严重程度分类器
        self.severity_classifier = SeverityClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_severity_levels,
            dropout=dropout
        )
    
    def forward(self, code_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            code_embedding: 代码嵌入 [batch_size, input_dim]
            
        Returns:
            outputs: 输出字典，包含三个分类器的输出
        """
        binary_output = self.binary_classifier(code_embedding)
        multilabel_output = self.multilabel_classifier(code_embedding)
        severity_output = self.severity_classifier(code_embedding)
        
        return {
            'binary': binary_output,
            'multilabel': multilabel_output,
            'severity': severity_output
        }
    
    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float] = {'binary': 1.0, 'multilabel': 1.0, 'severity': 1.0}
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            outputs: 模型输出
            targets: 目标标签
            loss_weights: 损失权重
            
        Returns:
            losses: 损失字典
        """
        # 二进制分类损失
        binary_loss = F.binary_cross_entropy(
            outputs['binary'].view(-1),
            targets['binary_label']
        )
        
        # 多标签分类损失
        multilabel_loss = F.binary_cross_entropy(
            outputs['multilabel'],
            targets['multi_label']
        )
        
        # 严重程度分类损失
        severity_loss = F.cross_entropy(
            outputs['severity'],
            targets['severity_label']
        )
        
        # 总损失
        total_loss = (
            loss_weights['binary'] * binary_loss +
            loss_weights['multilabel'] * multilabel_loss +
            loss_weights['severity'] * severity_loss
        )
        
        return {
            'binary_loss': binary_loss,
            'multilabel_loss': multilabel_loss,
            'severity_loss': severity_loss,
            'total_loss': total_loss
        }