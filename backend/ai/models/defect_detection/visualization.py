import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Any, Optional

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = 'Confusion Matrix'
):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        output_path: 输出路径
        title: 图表标题
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar()
    
    # 设置刻度标记
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)
    
    # 在混淆矩阵中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    
    # 保存图形
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: str,
    class_names: Optional[List[str]] = None,
    title: str = 'ROC Curve'
):
    """
    绘制ROC曲线
    
    Args:
        y_true: 真实标签，对于多标签分类，形状为 (n_samples, n_classes)
        y_score: 预测概率，对于多标签分类，形状为 (n_samples, n_classes)
        output_path: 输出路径
        class_names: 类别名称，用于多标签分类
        title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    
    # 检查是否是多标签分类
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        # 多标签分类
        for i in range(y_true.shape[1]):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            class_name = class_names[i] if class_names else f'Class {i}'
            plt.plot(
                fpr, tpr,
                lw=2,
                label=f'{class_name} (AUC = {roc_auc:.2f})'
            )
    else:
        # 二进制分类
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})'
        )
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    
    # 保存图形
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(
    history: Dict[str, List[float]],
    output_dir: str
):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典，包含各种指标的列表
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制损失
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制二进制分类F1分数
    if 'train_binary_f1' in history and 'val_binary_f1' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_binary_f1'], label='Train Binary F1')
        plt.plot(history['val_binary_f1'], label='Validation Binary F1')
        plt.title('Binary Classification F1 Score', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'binary_f1_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 绘制多标签分类F1分数
    if 'train_multilabel_f1' in history and 'val_multilabel_f1' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_multilabel_f1'], label='Train Multilabel F1')
        plt.plot(history['val_multilabel_f1'], label='Validation Multilabel F1')
        plt.title('Multilabel Classification F1 Score', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'multilabel_f1_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 绘制严重程度分类F1分数
    if 'train_severity_f1' in history and 'val_severity_f1' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_severity_f1'], label='Train Severity F1')
        plt.plot(history['val_severity_f1'], label='Validation Severity F1')
        plt.title('Severity Classification F1 Score', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'severity_f1_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

def visualize_model_evaluation(
    metrics: Dict[str, Any],
    output_dir: str,
    defect_types: List[str] = None,
    severity_levels: List[str] = None
):
    """
    可视化模型评估结果
    
    Args:
        metrics: 评估指标字典
        output_dir: 输出目录
        defect_types: 缺陷类型列表
        severity_levels: 严重程度级别列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 默认缺陷类型和严重程度级别
    if defect_types is None:
        defect_types = [
            'security_vulnerability',
            'performance_issue',
            'logic_error',
            'exception_handling',
            'concurrency_issue'
        ]
    
    if severity_levels is None:
        severity_levels = ['low', 'medium', 'high']
    
    # 二进制分类混淆矩阵
    if 'binary_confusion_matrix' in metrics:
        plot_confusion_matrix(
            y_true=metrics['binary_targets'],
            y_pred=metrics['binary_preds'],
            class_names=['No Defect', 'Defect'],
            output_path=os.path.join(output_dir, 'binary_confusion_matrix.png'),
            title='Binary Classification Confusion Matrix'
        )
    
    # 二进制分类ROC曲线
    if 'binary_probs' in metrics and 'binary_targets' in metrics:
        plot_roc_curve(
            y_true=metrics['binary_targets'],
            y_score=metrics['binary_probs'],
            output_path=os.path.join(output_dir, 'binary_roc_curve.png'),
            title='Binary Classification ROC Curve'
        )
    
    # 多标签分类ROC曲线
    if 'multilabel_probs' in metrics and 'multilabel_targets' in metrics:
        plot_roc_curve(
            y_true=metrics['multilabel_targets'],
            y_score=metrics['multilabel_probs'],
            output_path=os.path.join(output_dir, 'multilabel_roc_curve.png'),
            class_names=defect_types,
            title='Multilabel Classification ROC Curve'
        )
    
    # 严重程度分类混淆矩阵
    if 'severity_confusion_matrix' in metrics:
        plot_confusion_matrix(
            y_true=metrics['severity_targets'],
            y_pred=metrics['severity_preds'],
            class_names=severity_levels,
            output_path=os.path.join(output_dir, 'severity_confusion_matrix.png'),
            title='Severity Classification Confusion Matrix'
        )