# 使用指南

## 快速开始

### 1. 环境设置

```bash
# 安装依赖
pip install -r requirements.txt
```

**注意**: RDKit的安装可能需要使用conda：
```bash
conda install -c conda-forge rdkit
```

### 2. 完整运行管道

```bash
# 运行完整的训练和筛选管道
python run_pipeline.py --all
```

### 3. 分步运行

#### 步骤1: 数据收集
```bash
python run_pipeline.py --collect
```
或直接运行：
```bash
python src/data_collection.py
```

此步骤将：
- 从ChEMBL数据库获取抗氧化相关分子数据
- 如果API不可用，将创建合成数据集
- 保存原始数据到 `data/raw/antioxidant_dataset.csv`

#### 步骤2: 特征提取
```bash
python run_pipeline.py --extract
```
或直接运行：
```bash
python src/feature_extraction.py
```

此步骤将提取：
- **Morgan指纹** (1024 bits)：结构相似性
- **分子描述符** (25个)：物理化学性质
- **药效团特征** (20个)：抗氧化相关的结构特征
  - 羟基、酚羟基数量
  - 儿茶酚结构
  - 黄酮骨架
  - 共轭系统等

#### 步骤3: 训练传统机器学习模型
```bash
python run_pipeline.py --train-traditional
```
或直接运行：
```bash
python src/traditional_models.py
```

训练的模型包括：
- **Random Forest**: 集成学习
- **XGBoost**: 梯度提升树
- **LightGBM**: 轻量级梯度提升
- **SVM (RBF)**: 径向基核支持向量机
- **SVM (Linear)**: 线性核支持向量机
- **Gradient Boosting**: 梯度提升
- **Logistic Regression**: 逻辑回归
- **K-Nearest Neighbors**: K近邻
- **Naive Bayes**: 朴素贝叶斯

#### 步骤4: 训练深度学习模型
```bash
python run_pipeline.py --train-deep
```
或直接运行：
```bash
python src/deep_models.py
```

深度学习模型包括：
- **ChemBERTa**: 基于RoBERTa的化学预训练模型（77M参数）
  - 在大规模SMILES数据上预训练
  - 超越了传统BERT在化学领域的表现
- **MolFormer**: IBM开发的分子Transformer
- **Deep NN**: 多层全连接神经网络

**注意**: 深度学习模型训练可能需要GPU，首次运行会下载预训练模型（较大）。

#### 步骤5: 槲皮素筛选
```bash
python run_pipeline.py --screen
```
或直接运行：
```bash
python src/quercetin_screening.py
```

此步骤将：
- 计算候选分子与槲皮素的结构相似度
- 使用训练好的模型预测抗氧化活性
- 集成多个模型的预测结果
- 生成筛选报告和可视化

#### 步骤6: 可视化
```bash
python run_pipeline.py --visualize
```
或直接运行：
```bash
python src/visualization.py
```

## 详细分析

使用Jupyter Notebook进行详细分析：

```bash
jupyter notebook notebooks/analysis.ipynb
```

## 项目结构

```
.
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据
│   │   └── antioxidant_dataset.csv
│   ├── processed/             # 处理后的数据
│   │   ├── features.npy       # 特征矩阵
│   │   ├── labels.npy         # 标签
│   │   ├── smiles.pkl         # SMILES列表
│   │   └── feature_names.pkl  # 特征名称
│   └── quercetin/             # 槲皮素相关数据
│
├── src/                       # 源代码
│   ├── data_collection.py     # 数据收集
│   ├── feature_extraction.py  # 特征提取
│   ├── traditional_models.py  # 传统ML模型
│   ├── deep_models.py         # 深度学习模型
│   ├── train_evaluate.py      # 训练评估框架
│   ├── quercetin_screening.py # 槲皮素筛选
│   └── visualization.py       # 可视化工具
│
├── models/                    # 保存的模型
│   ├── traditional/           # 传统模型
│   ├── chemberta/            # ChemBERTa模型
│   └── deep_nn.pth           # 深度神经网络
│
├── results/                   # 结果输出
│   ├── model_comparison.png   # 模型对比图
│   ├── feature_importance.csv # 特征重要性
│   ├── quercetin/            # 槲皮素筛选结果
│   │   ├── screening_results.csv
│   │   ├── active_candidates.csv
│   │   ├── top_20_candidates.png
│   │   └── screening_report.txt
│   └── visualizations/       # 可视化结果
│
├── notebooks/                 # Jupyter notebooks
│   └── analysis.ipynb        # 数据分析notebook
│
├── requirements.txt           # 依赖包
├── README.md                  # 项目说明
├── USAGE_GUIDE.md            # 本文件
└── run_pipeline.py           # 主运行脚本
```

## 模型对比

### 传统机器学习模型
- **优点**: 训练快速，可解释性强，不需要GPU
- **适用**: 中小规模数据集，需要特征重要性分析
- **推荐**: Random Forest, XGBoost

### 深度学习模型
- **优点**: 自动特征学习，预训练模型性能强
- **适用**: 大规模数据，有GPU资源
- **推荐**: ChemBERTa (超越BERT的化学模型)

## 槲皮素筛选说明

槲皮素 (Quercetin) 是一种天然黄酮类化合物，具有强抗氧化活性。

**分子式**: C15H10O7  
**SMILES**: `O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1ccc(O)c(O)c1`

### 抗氧化关键特征：
1. **多个酚羟基** (-OH): 提供氢原子，清除自由基
2. **儿茶酚结构**: 邻二酚结构，螯合金属离子
3. **黄酮骨架**: 共轭体系稳定自由基
4. **3位羟基**: 增强抗氧化活性

筛选策略：
1. **结构相似性**: Tanimoto系数 > 0.5
2. **模型预测**: 集成多模型预测概率 > 0.5
3. **关键特征**: 具有类似的药效团特征

## 性能指标说明

- **Accuracy**: 整体准确率
- **Precision**: 预测为活性的分子中真正活性的比例
- **Recall**: 真正活性分子中被正确预测的比例
- **F1 Score**: Precision和Recall的调和平均
- **ROC-AUC**: 受试者工作特征曲线下面积
- **MCC**: Matthews相关系数，适合不平衡数据

## 常见问题

### 1. ChEMBL API连接失败
如果无法连接ChEMBL数据库，程序会自动使用合成数据集。

### 2. 内存不足
可以减少数据集大小或特征维度：
- 减少分子指纹位数 (n_bits=512)
- 使用数据子集进行训练

### 3. 深度学习模型下载慢
首次运行会从HuggingFace下载预训练模型，可能需要较长时间。建议：
- 使用代理或镜像
- 或跳过深度学习训练 `--train-traditional`

### 4. GPU相关错误
如果没有GPU，深度学习模型会自动使用CPU，但速度较慢。

## 扩展开发

### 添加新模型
在 `src/traditional_models.py` 或 `src/deep_models.py` 中添加新模型类。

### 使用自己的数据
替换 `data/raw/antioxidant_dataset.csv`，确保包含以下列：
- `smiles`: SMILES字符串
- `label`: 0或1（非活性/活性）

### 调整超参数
修改各模型文件中的参数配置。

## 引用

如果使用了本项目，请引用相关工具和库：
- RDKit: https://www.rdkit.org/
- ChemBERTa: https://github.com/seyonechithrananda/bert-loves-chemistry
- ChEMBL: https://www.ebi.ac.uk/chembl/

## 许可证

MIT License

## 联系方式

如有问题，请提交Issue或联系项目维护者。

