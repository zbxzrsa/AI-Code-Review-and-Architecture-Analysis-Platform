import os
import sys
import torch
import logging
import argparse
import numpy as np
from typing import Dict, List, Any

# 添加父目录到路径，以便导入多模态代码表示模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_representation.model import MultimodalCodeRepresentation

# 导入缺陷检测模块
from classifiers import DefectDetectionModel
from data_preprocessing import DefectDataset
from visualization import visualize_model_evaluation

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='代码缺陷检测示例')
    parser.add_argument('--code_file', type=str, required=True, help='要分析的代码文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='缺陷检测模型检查点路径')
    parser.add_argument('--code_repr_model_path', type=str, required=True, help='多模态代码表示模型路径')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    return parser.parse_args()

def load_model(model_path: str, device: str) -> DefectDetectionModel:
    """加载缺陷检测模型"""
    # 定义缺陷类型和严重程度级别
    defect_types = [
        'security_vulnerability',
        'performance_issue',
        'logic_error',
        'exception_handling',
        'concurrency_issue'
    ]
    severity_levels = ['low', 'medium', 'high']
    
    # 创建模型
    model = DefectDetectionModel(
        code_embedding_dim=768,  # 多模态代码表示模型的输出维度
        hidden_size=512,
        num_defect_types=len(defect_types),
        num_severity_levels=len(severity_levels),
        dropout=0.1
    )
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def get_code_embedding(code_file: str, code_repr_model_path: str, device: str) -> torch.Tensor:
    """获取代码嵌入"""
    # 加载多模态代码表示模型
    code_repr_model = MultimodalCodeRepresentation.load_from_checkpoint(code_repr_model_path)
    code_repr_model.to(device)
    code_repr_model.eval()
    
    # 读取代码文件
    with open(code_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # 获取代码嵌入
    with torch.no_grad():
        code_embedding = code_repr_model.get_code_embedding(code)
    
    return code_embedding

def analyze_code(code_file: str, model: DefectDetectionModel, code_repr_model_path: str, device: str) -> Dict[str, Any]:
    """分析代码文件，检测缺陷"""
    # 获取代码嵌入
    code_embedding = get_code_embedding(code_file, code_repr_model_path, device)
    code_embedding = code_embedding.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(code_embedding.unsqueeze(0))
    
    # 处理预测结果
    binary_prob = outputs['binary'].item()
    has_defect = binary_prob > 0.5
    
    multilabel_probs = outputs['multilabel'].cpu().numpy()[0]
    defect_types = [
        'security_vulnerability',
        'performance_issue',
        'logic_error',
        'exception_handling',
        'concurrency_issue'
    ]
    detected_defects = [defect_types[i] for i in range(len(defect_types)) if multilabel_probs[i] > 0.5]
    
    severity_probs = torch.nn.functional.softmax(outputs['severity'], dim=1).cpu().numpy()[0]
    severity_levels = ['low', 'medium', 'high']
    severity = severity_levels[np.argmax(severity_probs)]
    
    # 返回结果
    return {
        'has_defect': has_defect,
        'binary_prob': binary_prob,
        'detected_defects': detected_defects,
        'defect_probs': {defect_types[i]: float(multilabel_probs[i]) for i in range(len(defect_types))},
        'severity': severity,
        'severity_probs': {severity_levels[i]: float(severity_probs[i]) for i in range(len(severity_levels))}
    }

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载模型
    logger.info("加载缺陷检测模型...")
    model = load_model(args.model_path, args.device)
    
    # 分析代码
    logger.info(f"分析代码文件: {args.code_file}")
    results = analyze_code(args.code_file, model, args.code_repr_model_path, args.device)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存结果
    import json
    with open(os.path.join(args.output_dir, 'defect_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印结果
    logger.info("代码缺陷分析结果:")
    logger.info(f"是否存在缺陷: {'是' if results['has_defect'] else '否'} (概率: {results['binary_prob']:.4f})")
    
    if results['has_defect']:
        logger.info(f"检测到的缺陷类型: {', '.join(results['detected_defects']) if results['detected_defects'] else '无'}")
        logger.info(f"缺陷严重程度: {results['severity']}")
        
        # 打印详细概率
        logger.info("缺陷类型概率:")
        for defect_type, prob in results['defect_probs'].items():
            logger.info(f"  - {defect_type}: {prob:.4f}")
        
        logger.info("严重程度概率:")
        for severity, prob in results['severity_probs'].items():
            logger.info(f"  - {severity}: {prob:.4f}")
    
    logger.info(f"详细结果已保存到: {os.path.join(args.output_dir, 'defect_analysis.json')}")

if __name__ == '__main__':
    main()