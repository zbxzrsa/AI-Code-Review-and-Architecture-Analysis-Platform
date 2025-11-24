import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional
import sys
import os

# 添加父目录到路径，以便导入多模态代码表示模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_representation.data_utils import CodeDataset as BaseCodeDataset
from code_representation.model import MultimodalCodeRepresentation

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DefectDataset(Dataset):
    """
    代码缺陷检测数据集
    """
    def __init__(
        self, 
        data_file: str,
        code_model: MultimodalCodeRepresentation,
        tokenizer,
        max_length: int = 512,
        cache_dir: Optional[str] = None
    ):
        """
        初始化代码缺陷检测数据集
        
        Args:
            data_file: 数据文件路径
            code_model: 预训练的多模态代码表示模型
            tokenizer: 分词器
            max_length: 最大序列长度
            cache_dir: 缓存目录
        """
        self.data_file = data_file
        self.code_model = code_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # 加载数据
        self.data = self._load_data()
        
        # 缺陷类型映射
        self.defect_types = [
            'security_vulnerability',
            'performance_issue',
            'logic_error',
            'exception_handling',
            'concurrency_issue'
        ]
        self.defect_type_to_idx = {t: i for i, t in enumerate(self.defect_types)}
        
        # 严重程度映射
        self.severity_levels = ['low', 'medium', 'high']
        self.severity_to_idx = {s: i for i, s in enumerate(self.severity_levels)}
        
        # 创建缓存目录
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        加载数据
        
        Returns:
            data: 数据列表
        """
        logger.info(f"加载数据: {self.data_file}")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"共加载 {len(data)} 条数据")
        return data
    
    def _get_cache_path(self, idx: int) -> str:
        """
        获取缓存路径
        
        Args:
            idx: 数据索引
            
        Returns:
            cache_path: 缓存路径
        """
        if not self.cache_dir:
            return None
        
        # 使用数据文件名和索引作为缓存文件名
        data_file_name = os.path.basename(self.data_file).split('.')[0]
        cache_path = os.path.join(self.cache_dir, f"{data_file_name}_{idx}.pt")
        
        return cache_path
    
    def _get_code_embedding(self, code: str, idx: int) -> torch.Tensor:
        """
        获取代码嵌入
        
        Args:
            code: 代码文本
            idx: 数据索引
            
        Returns:
            embedding: 代码嵌入
        """
        # 检查缓存
        cache_path = self._get_cache_path(idx)
        if cache_path and os.path.exists(cache_path):
            return torch.load(cache_path)
        
        # 使用多模态代码表示模型获取代码嵌入
        inputs = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 将输入移动到与模型相同的设备
        device = next(self.code_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 获取代码嵌入
        with torch.no_grad():
            embedding = self.code_model.get_code_embedding(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                # 注意：这里简化了，实际应该提供AST、CFG和DFG
                ast_data=None,
                cfg_data=None,
                dfg_data=None
            )
        
        # 缓存嵌入
        if cache_path:
            torch.save(embedding.cpu(), cache_path)
        
        return embedding.cpu()
    
    def __len__(self) -> int:
        """
        获取数据集长度
        
        Returns:
            length: 数据集长度
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            item: 数据项
        """
        item = self.data[idx]
        
        # 获取代码嵌入
        code_embedding = self._get_code_embedding(item['before_code'], idx)
        
        # 二进制标签：有缺陷/无缺陷
        binary_label = torch.tensor(1 if item.get('has_defect', True) else 0, dtype=torch.float)
        
        # 多标签：缺陷类型
        multi_label = torch.zeros(len(self.defect_types), dtype=torch.float)
        for defect_type in item.get('defect_types', []):
            if defect_type in self.defect_type_to_idx:
                multi_label[self.defect_type_to_idx[defect_type]] = 1.0
        
        # 严重程度
        severity = item.get('severity', 'medium')
        severity_label = torch.tensor(self.severity_to_idx.get(severity, 1), dtype=torch.long)
        
        return {
            'code_embedding': code_embedding,
            'binary_label': binary_label,
            'multi_label': multi_label,
            'severity_label': severity_label
        }

def get_dataloaders(
    train_file: str,
    val_file: str,
    test_file: str,
    code_model: MultimodalCodeRepresentation,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    cache_dir: Optional[str] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取数据加载器
    
    Args:
        train_file: 训练数据文件
        val_file: 验证数据文件
        test_file: 测试数据文件
        code_model: 预训练的多模态代码表示模型
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        cache_dir: 缓存目录
        num_workers: 工作进程数
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    logger.info("创建数据加载器")
    
    # 创建数据集
    train_dataset = DefectDataset(
        data_file=train_file,
        code_model=code_model,
        tokenizer=tokenizer,
        max_length=max_length,
        cache_dir=cache_dir
    )
    
    val_dataset = DefectDataset(
        data_file=val_file,
        code_model=code_model,
        tokenizer=tokenizer,
        max_length=max_length,
        cache_dir=cache_dir
    )
    
    test_dataset = DefectDataset(
        data_file=test_file,
        code_model=code_model,
        tokenizer=tokenizer,
        max_length=max_length,
        cache_dir=cache_dir
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"数据加载器创建完成: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader