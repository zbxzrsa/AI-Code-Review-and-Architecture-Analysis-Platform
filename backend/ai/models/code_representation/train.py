import os
import argparse
import torch
from torch.utils.data import DataLoader

from data_utils import CodeDataset, get_dataloader
from model import MultimodalCodeRepresentation
from trainer import CodeRepresentationTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="训练多模态代码表示学习模型")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--train_file", type=str, default="train.jsonl", help="训练数据文件")
    parser.add_argument("--val_file", type=str, default="val.jsonl", help="验证数据文件")
    parser.add_argument("--test_file", type=str, default="test.jsonl", help="测试数据文件")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    
    # 模型参数
    parser.add_argument("--hidden_size", type=int, default=768, help="隐藏层维度")
    parser.add_argument("--graph_hidden_size", type=int, default=256, help="图编码器隐藏层维度")
    parser.add_argument("--graph_in_channels", type=int, default=300, help="图节点初始特征维度")
    parser.add_argument("--num_graph_layers", type=int, default=2, help="图编码器层数")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout比率")
    parser.add_argument("--freeze_text_encoder", action="store_true", help="是否冻结文本编码器")
    
    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="预热步数")
    parser.add_argument("--contrastive_weight", type=float, default=1.0, help="对比学习损失权重")
    parser.add_argument("--mlm_weight", type=float, default=1.0, help="掩码语言建模损失权重")
    parser.add_argument("--graph_weight", type=float, default=1.0, help="图结构预测损失权重")
    
    # 其他参数
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点保存目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default=None, help="训练设备")
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置设备
    device = args.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    train_dataloader = get_dataloader(
        os.path.join(args.data_dir, args.train_file),
        batch_size=args.batch_size,
        shuffle=True,
        split="train"
    )
    
    val_dataloader = get_dataloader(
        os.path.join(args.data_dir, args.val_file),
        batch_size=args.batch_size,
        shuffle=False,
        split="val"
    ) if args.val_file else None
    
    test_dataloader = get_dataloader(
        os.path.join(args.data_dir, args.test_file),
        batch_size=args.batch_size,
        shuffle=False,
        split="test"
    ) if args.test_file else None
    
    # 创建模型
    model = MultimodalCodeRepresentation(
        hidden_size=args.hidden_size,
        graph_hidden_size=args.graph_hidden_size,
        graph_in_channels=args.graph_in_channels,
        num_graph_layers=args.num_graph_layers,
        num_attention_heads=args.num_attention_heads,
        dropout_rate=args.dropout_rate,
        freeze_text_encoder=args.freeze_text_encoder
    )
    
    # 创建训练器
    trainer = CodeRepresentationTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        device=device
    )
    
    # 训练模型
    history = trainer.train(
        contrastive_weight=args.contrastive_weight,
        mlm_weight=args.mlm_weight,
        graph_weight=args.graph_weight
    )
    
    # 测试模型
    if test_dataloader is not None:
        test_loss, test_metrics = trainer.test(
            contrastive_weight=args.contrastive_weight,
            mlm_weight=args.mlm_weight,
            graph_weight=args.graph_weight
        )
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Metrics: {test_metrics}")

if __name__ == "__main__":
    main()