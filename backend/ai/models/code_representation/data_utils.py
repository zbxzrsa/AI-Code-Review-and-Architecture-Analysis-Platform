import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import networkx as nx
import dgl

class CodeDataset(Dataset):
    """
    多模态代码表示学习数据集
    处理源代码文本、AST、CFG和DFG
    """
    def __init__(self, data_dir, split='train', max_length=512, transform=None):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            split: 数据集划分 (train, valid, test)
            max_length: 代码文本最大长度
            transform: 数据转换函数
        """
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        self.transform = transform
        
        # 加载数据索引
        self.data_path = os.path.join(data_dir, f"{split}.jsonl")
        self.data_samples = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data_samples.append(json.loads(line.strip()))
                
        # 初始化tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        """获取一个数据样本的所有模态"""
        sample = self.data_samples[idx]
        
        # 1. 处理源代码文本
        code_text = sample['code']
        code_tokens = self.tokenizer(
            code_text, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 2. 处理AST (抽象语法树)
        ast_data = sample['ast']
        ast_graph = self._build_graph_from_edges(
            nodes=ast_data['nodes'],
            edges=ast_data['edges'],
            node_features=ast_data['node_features']
        )
        
        # 3. 处理CFG (控制流图)
        cfg_data = sample['cfg']
        cfg_graph = self._build_graph_from_edges(
            nodes=cfg_data['nodes'],
            edges=cfg_data['edges'],
            node_features=cfg_data['node_features']
        )
        
        # 4. 处理DFG (数据流图)
        dfg_data = sample['dfg']
        dfg_graph = self._build_graph_from_edges(
            nodes=dfg_data['nodes'],
            edges=dfg_data['edges'],
            node_features=dfg_data['node_features']
        )
        
        # 5. 获取标签信息（如果有）
        label = sample.get('label', -1)
        
        # 构建返回字典
        result = {
            'code_ids': code_tokens['input_ids'].squeeze(0),
            'code_mask': code_tokens['attention_mask'].squeeze(0),
            'ast_graph': ast_graph,
            'cfg_graph': cfg_graph,
            'dfg_graph': dfg_graph,
            'label': torch.tensor(label, dtype=torch.long),
            'code_text': code_text,  # 原始文本，用于调试
        }
        
        # 应用转换（如果有）
        if self.transform:
            result = self.transform(result)
            
        return result
    
    def _build_graph_from_edges(self, nodes, edges, node_features):
        """
        从节点和边构建DGL图
        
        Args:
            nodes: 节点ID列表
            edges: 边列表，每个边是(src, dst)对
            node_features: 节点特征字典
            
        Returns:
            DGL图对象
        """
        g = dgl.graph(([], []), num_nodes=len(nodes))
        
        # 添加边
        src_nodes, dst_nodes = zip(*edges) if edges else ([], [])
        g.add_edges(src_nodes, dst_nodes)
        
        # 添加节点特征
        for feat_name, feat_values in node_features.items():
            g.ndata[feat_name] = torch.tensor(feat_values, dtype=torch.float)
            
        return g
    
    def get_mask_prediction_batch(self, batch_size=32):
        """
        获取用于掩码语言建模的批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            包含掩码token的批次
        """
        indices = np.random.choice(len(self), batch_size)
        batch = [self[i] for i in indices]
        
        # 处理每个样本，添加掩码
        for i, sample in enumerate(batch):
            code_ids = sample['code_ids'].clone()
            attention_mask = sample['code_mask'].clone()
            
            # 找出所有非特殊token的位置
            special_tokens = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]
            mask_candidates = [(i, id_) for i, id_ in enumerate(code_ids) 
                              if id_ not in special_tokens and attention_mask[i] == 1]
            
            if mask_candidates:
                # 随机选择15%的token进行掩码
                num_to_mask = max(1, int(len(mask_candidates) * 0.15))
                mask_indices = np.random.choice(len(mask_candidates), num_to_mask, replace=False)
                
                # 保存原始token用于计算损失
                original_ids = code_ids.clone()
                
                # 应用掩码
                for idx in mask_indices:
                    token_pos = mask_candidates[idx][0]
                    code_ids[token_pos] = self.tokenizer.mask_token_id
                
                # 更新样本
                batch[i]['masked_code_ids'] = code_ids
                batch[i]['original_code_ids'] = original_ids
                batch[i]['mask_positions'] = [mask_candidates[idx][0] for idx in mask_indices]
            else:
                # 如果没有可掩码的token，则复制原样本
                batch[i]['masked_code_ids'] = code_ids
                batch[i]['original_code_ids'] = code_ids
                batch[i]['mask_positions'] = []
        
        return self.collate_fn(batch)
    
    def collate_fn(self, batch):
        """
        将批次数据整合为tensor
        
        Args:
            batch: 样本列表
            
        Returns:
            整合后的批次数据
        """
        # 文本模态
        code_ids = torch.stack([sample['code_ids'] for sample in batch])
        code_mask = torch.stack([sample['code_mask'] for sample in batch])
        
        # 图模态
        ast_graphs = [sample['ast_graph'] for sample in batch]
        cfg_graphs = [sample['cfg_graph'] for sample in batch]
        dfg_graphs = [sample['dfg_graph'] for sample in batch]
        
        # 批量化图
        batched_ast_graph = dgl.batch(ast_graphs)
        batched_cfg_graph = dgl.batch(cfg_graphs)
        batched_dfg_graph = dgl.batch(dfg_graphs)
        
        # 标签
        labels = torch.stack([sample['label'] for sample in batch])
        
        # 掩码预测相关（如果有）
        result = {
            'code_ids': code_ids,
            'code_mask': code_mask,
            'ast_graph': batched_ast_graph,
            'cfg_graph': batched_cfg_graph,
            'dfg_graph': batched_dfg_graph,
            'labels': labels,
        }
        
        # 添加掩码预测相关字段（如果有）
        if 'masked_code_ids' in batch[0]:
            result['masked_code_ids'] = torch.stack([sample['masked_code_ids'] for sample in batch])
            result['original_code_ids'] = torch.stack([sample['original_code_ids'] for sample in batch])
            # mask_positions需要特殊处理，因为每个样本的掩码位置数量可能不同
            result['mask_positions'] = [sample['mask_positions'] for sample in batch]
            
        return result

def get_dataloader(data_dir, batch_size=32, num_workers=4):
    """
    获取数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        
    Returns:
        训练、验证和测试数据加载器
    """
    train_dataset = CodeDataset(data_dir, split='train')
    valid_dataset = CodeDataset(data_dir, split='valid')
    test_dataset = CodeDataset(data_dir, split='test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=valid_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_dataset.collate_fn
    )
    
    return train_loader, valid_loader, test_loader