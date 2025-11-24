# 代码缺陷检测分类器

基于多模态代码表示学习模型的代码缺陷检测分类器，支持二进制分类（有缺陷/无缺陷）、多标签分类（缺陷类型）和严重程度评估。

## 功能特点

- **多任务学习**：同时进行三种分类任务
  - 二进制分类：检测代码是否存在缺陷
  - 多标签分类：识别缺陷类型（安全漏洞、性能问题、逻辑错误等）
  - 严重程度评估：评估缺陷的严重程度（低、中、高）

- **基于多模态代码表示**：利用预训练的多模态代码表示模型，融合代码的文本、AST、CFG和DFG信息

- **完整的评估指标**：提供混淆矩阵、ROC曲线、精确率、召回率和F1分数等评估指标

## 项目结构

```
defect_detection/
├── data_collection.py     # 数据收集和标注
├── data_preprocessing.py  # 数据预处理和特征提取
├── classifiers.py         # 分类器模型定义
├── trainer.py             # 模型训练和评估
├── visualization.py       # 评估结果可视化
├── train.py               # 训练脚本
├── example.py             # 使用示例
└── README.md              # 项目说明
```

## 快速开始

### 1. 数据收集和标注

使用`data_collection.py`从GitHub收集代码缺陷数据：

```bash
python data_collection.py --output_dir ./data --query "fix bug security" --max_repos 50
```

### 2. 训练模型

使用`train.py`训练代码缺陷检测模型：

```bash
python train.py --data_dir ./data --code_repr_model_path ../code_representation/checkpoints/best.pt --epochs 20 --batch_size 32
```

### 3. 分析代码缺陷

使用训练好的模型分析代码文件：

```bash
python example.py --code_file /path/to/your/code.py --model_path ./checkpoints/best.pt --code_repr_model_path ../code_representation/checkpoints/best.pt
```

## 模型架构

- **基础编码器**：预训练的多模态代码表示模型
- **分类头**：
  - 二进制分类器：MLP + Sigmoid
  - 多标签分类器：MLP + Sigmoid
  - 严重程度分类器：MLP + Softmax

## 评估指标

模型提供以下评估指标：

- **二进制分类**：准确率、精确率、召回率、F1分数、ROC曲线、AUC
- **多标签分类**：准确率、精确率、召回率、F1分数、ROC曲线、AUC
- **严重程度评估**：准确率、精确率、召回率、F1分数、混淆矩阵

## 可视化结果

训练和评估过程会生成以下可视化结果：

- 训练和验证损失曲线
- 各任务的F1分数曲线
- 混淆矩阵
- ROC曲线

## 使用示例

```python
from classifiers import DefectDetectionModel
from code_representation.model import MultimodalCodeRepresentation

# 加载模型
code_repr_model = MultimodalCodeRepresentation.load_from_checkpoint("../code_representation/checkpoints/best.pt")
defect_model = DefectDetectionModel.load_from_checkpoint("./checkpoints/best.pt")

# 获取代码嵌入
code_embedding = code_repr_model.get_code_embedding(code_text)

# 预测缺陷
outputs = defect_model(code_embedding.unsqueeze(0))
has_defect = outputs['binary'].item() > 0.5
defect_types = [defect_types[i] for i in range(len(defect_types)) if outputs['multilabel'][0, i] > 0.5]
severity = severity_levels[torch.argmax(outputs['severity'], dim=1).item()]
```

## 依赖项

- PyTorch >= 1.7.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.3.0
- numpy >= 1.19.0
- tqdm >= 4.50.0