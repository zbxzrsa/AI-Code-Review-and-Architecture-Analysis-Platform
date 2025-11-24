import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from tqdm import tqdm

class CodeRepresentationTrainer:
    """
    多模态代码表示学习模型的训练器
    """
    def __init__(
        self, 
        model, 
        train_dataloader, 
        val_dataloader=None,
        test_dataloader=None,
        learning_rate=1e-4,
        weight_decay=1e-5,
        num_epochs=10,
        warmup_steps=1000,
        checkpoint_dir='./checkpoints',
        device=None
    ):
        """
        初始化训练器
        
        Args:
            model: 多模态代码表示模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            test_dataloader: 测试数据加载器
            learning_rate: 学习率
            weight_decay: 权重衰减
            num_epochs: 训练轮数
            warmup_steps: 预热步数
            checkpoint_dir: 检查点保存目录
            device: 训练设备
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.checkpoint_dir = checkpoint_dir
        
        # 设置设备
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        # 最佳验证性能
        self.best_val_loss = float('inf')
        
    def train(self, contrastive_weight=1.0, mlm_weight=1.0, graph_weight=1.0):
        """
        训练模型
        
        Args:
            contrastive_weight: 对比学习损失权重
            mlm_weight: 掩码语言建模损失权重
            graph_weight: 图结构预测损失权重
            
        Returns:
            训练历史
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # 训练循环
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # 训练一个轮次
            train_loss, train_metrics = self._train_epoch(
                contrastive_weight, mlm_weight, graph_weight
            )
            history['train_loss'].append(train_loss)
            history['train_metrics'].append(train_metrics)
            
            # 验证
            if self.val_dataloader is not None:
                val_loss, val_metrics = self._evaluate(
                    self.val_dataloader, contrastive_weight, mlm_weight, graph_weight
                )
                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_metrics)
                
                print(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Metrics: {val_metrics}")
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(f"best_model_epoch_{epoch+1}.pt")
            else:
                print(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Train Metrics: {train_metrics}")
            
            # 保存检查点
            if (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1:
                self._save_checkpoint(f"model_epoch_{epoch+1}.pt")
        
        return history
    
    def _train_epoch(self, contrastive_weight, mlm_weight, graph_weight):
        """
        训练一个轮次
        
        Args:
            contrastive_weight: 对比学习损失权重
            mlm_weight: 掩码语言建模损失权重
            graph_weight: 图结构预测损失权重
            
        Returns:
            avg_loss: 平均损失
            metrics: 训练指标
        """
        self.model.train()
        total_loss = 0
        
        # 指标收集
        all_contrastive_losses = []
        all_mlm_losses = []
        all_graph_losses = []
        
        # 进度条
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # 将数据移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(**batch)
            
            # 计算损失
            contrastive_loss = outputs['contrastive_loss'] if 'contrastive_loss' in outputs else 0
            mlm_loss = outputs['mlm_loss'] if 'mlm_loss' in outputs else 0
            graph_loss = outputs['graph_loss'] if 'graph_loss' in outputs else 0
            
            # 总损失
            loss = (
                contrastive_weight * contrastive_loss +
                mlm_weight * mlm_loss +
                graph_weight * graph_loss
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # 更新统计
            total_loss += loss.item()
            
            # 收集指标
            if isinstance(contrastive_loss, torch.Tensor):
                all_contrastive_losses.append(contrastive_loss.item())
            if isinstance(mlm_loss, torch.Tensor):
                all_mlm_losses.append(mlm_loss.item())
            if isinstance(graph_loss, torch.Tensor):
                all_graph_losses.append(graph_loss.item())
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'c_loss': contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0,
                'm_loss': mlm_loss.item() if isinstance(mlm_loss, torch.Tensor) else 0,
                'g_loss': graph_loss.item() if isinstance(graph_loss, torch.Tensor) else 0
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_dataloader)
        
        # 计算指标
        metrics = {
            'contrastive_loss': np.mean(all_contrastive_losses) if all_contrastive_losses else 0,
            'mlm_loss': np.mean(all_mlm_losses) if all_mlm_losses else 0,
            'graph_loss': np.mean(all_graph_losses) if all_graph_losses else 0
        }
        
        return avg_loss, metrics
    
    def _evaluate(self, dataloader, contrastive_weight, mlm_weight, graph_weight):
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            contrastive_weight: 对比学习损失权重
            mlm_weight: 掩码语言建模损失权重
            graph_weight: 图结构预测损失权重
            
        Returns:
            avg_loss: 平均损失
            metrics: 评估指标
        """
        self.model.eval()
        total_loss = 0
        
        # 指标收集
        all_contrastive_losses = []
        all_mlm_losses = []
        all_graph_losses = []
        
        # 收集预测和标签
        all_edge_preds = []
        all_edge_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 将数据移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                
                # 计算损失
                contrastive_loss = outputs['contrastive_loss'] if 'contrastive_loss' in outputs else 0
                mlm_loss = outputs['mlm_loss'] if 'mlm_loss' in outputs else 0
                graph_loss = outputs['graph_loss'] if 'graph_loss' in outputs else 0
                
                # 总损失
                loss = (
                    contrastive_weight * contrastive_loss +
                    mlm_weight * mlm_loss +
                    graph_weight * graph_loss
                )
                
                # 更新统计
                total_loss += loss.item()
                
                # 收集指标
                if isinstance(contrastive_loss, torch.Tensor):
                    all_contrastive_losses.append(contrastive_loss.item())
                if isinstance(mlm_loss, torch.Tensor):
                    all_mlm_losses.append(mlm_loss.item())
                if isinstance(graph_loss, torch.Tensor):
                    all_graph_losses.append(graph_loss.item())
                
                # 收集图结构预测结果
                if 'edge_scores' in outputs and 'edge_labels' in batch:
                    edge_scores = outputs['edge_scores'].cpu().numpy()
                    edge_labels = batch['edge_labels'].cpu().numpy()
                    
                    # 二值化预测
                    edge_preds = (edge_scores > 0.5).astype(np.int32)
                    
                    all_edge_preds.extend(edge_preds)
                    all_edge_labels.extend(edge_labels)
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        
        # 计算指标
        metrics = {
            'contrastive_loss': np.mean(all_contrastive_losses) if all_contrastive_losses else 0,
            'mlm_loss': np.mean(all_mlm_losses) if all_mlm_losses else 0,
            'graph_loss': np.mean(all_graph_losses) if all_graph_losses else 0
        }
        
        # 计算图结构预测指标
        if all_edge_preds and all_edge_labels:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_edge_labels, all_edge_preds, average='binary'
            )
            metrics.update({
                'edge_precision': precision,
                'edge_recall': recall,
                'edge_f1': f1
            })
        
        return avg_loss, metrics
    
    def test(self, contrastive_weight=1.0, mlm_weight=1.0, graph_weight=1.0):
        """
        测试模型
        
        Args:
            contrastive_weight: 对比学习损失权重
            mlm_weight: 掩码语言建模损失权重
            graph_weight: 图结构预测损失权重
            
        Returns:
            test_loss: 测试损失
            test_metrics: 测试指标
        """
        if self.test_dataloader is None:
            raise ValueError("Test dataloader is not provided")
        
        test_loss, test_metrics = self._evaluate(
            self.test_dataloader, contrastive_weight, mlm_weight, graph_weight
        )
        
        print(f"Test Loss: {test_loss:.4f}, Test Metrics: {test_metrics}")
        
        return test_loss, test_metrics
    
    def _save_checkpoint(self, filename):
        """
        保存检查点
        
        Args:
            filename: 文件名
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded from {checkpoint_path}")
        
    def get_code_embeddings(self, dataloader):
        """
        获取代码嵌入
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            embeddings: 代码嵌入 [num_samples, hidden_size]
            labels: 标签 [num_samples]
        """
        self.model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 将数据移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 获取嵌入
                outputs = self.model(**batch, return_embeddings=True)
                embeddings = outputs['code_embeddings']
                
                # 收集嵌入和标签
                all_embeddings.append(embeddings.cpu().numpy())
                if 'labels' in batch:
                    all_labels.append(batch['labels'].cpu().numpy())
        
        # 拼接结果
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0) if all_labels else None
        
        return embeddings, labels