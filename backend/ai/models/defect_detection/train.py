import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 添加父目录到路径，以便导入多模态代码表示模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_representation.model import MultimodalCodeRepresentation
from code_representation.trainer import CodeRepresentationTrainer

# 导入缺陷检测模块
from data_preprocessing import DefectDataset, get_dataloaders
from classifiers import DefectDetectionModel
from trainer import DefectDetectionTrainer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='代码缺陷检测模型训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作线程数')
    
    # 模型参数
    parser.add_argument('--code_repr_model_path', type=str, required=True, help='预训练的多模态代码表示模型路径')
    parser.add_argument('--hidden_size', type=int, default=512, help='分类器隐藏层大小')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout概率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--early_stopping', type=int, default=5, help='早停耐心值')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='检查点目录')
    parser.add_argument('--results_dir', type=str, default='./results', help='结果目录')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    # 评估参数
    parser.add_argument('--eval_only', action='store_true', help='仅评估模型')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='评估时加载的检查点路径')
    
    return parser.parse_args()

def plot_training_history(history, output_dir):
    """
    绘制训练历史
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制损失
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    plt.close()
    
    # 绘制二进制分类F1分数
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_binary_f1'], label='Train Binary F1')
    plt.plot(history['val_binary_f1'], label='Validation Binary F1')
    plt.title('Binary Classification F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'binary_f1_history.png'))
    plt.close()
    
    # 绘制多标签分类F1分数
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_multilabel_f1'], label='Train Multilabel F1')
    plt.plot(history['val_multilabel_f1'], label='Validation Multilabel F1')
    plt.title('Multilabel Classification F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'multilabel_f1_history.png'))
    plt.close()
    
    # 绘制严重程度分类F1分数
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_severity_f1'], label='Train Severity F1')
    plt.plot(history['val_severity_f1'], label='Validation Severity F1')
    plt.title('Severity Classification F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'severity_f1_history.png'))
    plt.close()

def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 加载数据集
    logger.info("加载数据集...")
    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        code_repr_model_path=args.code_repr_model_path
    )
    
    # 获取缺陷类型和严重程度级别
    defect_types = metadata['defect_types']
    severity_levels = metadata['severity_levels']
    
    # 创建模型
    logger.info("创建模型...")
    model = DefectDetectionModel(
        code_embedding_dim=768,  # 多模态代码表示模型的输出维度
        hidden_size=args.hidden_size,
        num_defect_types=len(defect_types),
        num_severity_levels=len(severity_levels),
        dropout=args.dropout
    )
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # 创建训练器
    trainer = DefectDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        defect_types=defect_types,
        severity_levels=severity_levels
    )
    
    # 如果仅评估模型
    if args.eval_only:
        if args.checkpoint_path:
            trainer.load_checkpoint(args.checkpoint_path)
        else:
            trainer.load_checkpoint(best=True)
        
        # 在测试集上评估
        logger.info("在测试集上评估模型...")
        test_metrics = trainer.evaluate(test_loader, mode='test')
        
        # 打印测试指标
        logger.info("测试集评估结果:")
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test Binary F1: {test_metrics['binary_f1']:.4f}")
        logger.info(f"Test Multilabel F1: {test_metrics['multilabel_f1']:.4f}")
        logger.info(f"Test Severity F1: {test_metrics['severity_f1']:.4f}")
        
        return
    
    # 训练模型
    logger.info("开始训练模型...")
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping
    )
    
    # 绘制训练历史
    logger.info("绘制训练历史...")
    plot_training_history(history, args.results_dir)
    
    # 在测试集上评估
    logger.info("在测试集上评估模型...")
    test_metrics = trainer.evaluate(test_loader, mode='test')
    
    # 打印测试指标
    logger.info("测试集评估结果:")
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Binary F1: {test_metrics['binary_f1']:.4f}")
    logger.info(f"Test Multilabel F1: {test_metrics['multilabel_f1']:.4f}")
    logger.info(f"Test Severity F1: {test_metrics['severity_f1']:.4f}")

if __name__ == '__main__':
    main()