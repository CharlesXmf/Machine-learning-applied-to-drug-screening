# 快速开始指南

## 🚀 5分钟快速运行

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装基础依赖
pip install pandas numpy scikit-learn matplotlib seaborn

# 安装化学工具（必需）
conda install -c conda-forge rdkit  # 推荐使用conda安装rdkit
# 或
pip install rdkit-pypi  # 如果conda不可用

# 安装机器学习库
pip install xgboost lightgbm

# 可选：深度学习（需要较多时间和空间）
pip install torch transformers
```

### 2. 快速测试运行

```bash
# 运行完整管道（传统模型）
python run_pipeline.py --collect --extract --train-traditional --screen

# 这将：
# 1. 收集/生成 3000个分子数据
# 2. 提取分子特征（指纹+描述符）
# 3. 训练9种传统ML模型
# 4. 筛选槲皮素类似物
```

预计时间：5-15分钟（取决于CPU性能）

### 3. 查看结果

```bash
# 结果文件位置
ls results/

# 查看模型对比
cat results/model_comparison.csv

# 查看槲皮素筛选报告
cat results/quercetin/screening_report.txt

# 查看筛选出的活性分子
cat results/quercetin/active_candidates.csv
```

## 📊 结果文件说明

### 核心输出文件

1. **models/traditional/**: 训练好的模型文件
   - `Random_Forest.pkl`
   - `XGBoost.pkl`
   - 等等...

2. **results/model_comparison.csv**: 所有模型性能对比
   ```
   model_name,accuracy,precision,recall,f1_score,roc_auc,mcc
   XGBoost,0.85,0.83,0.87,0.85,0.92,0.70
   Random Forest,0.84,0.82,0.86,0.84,0.91,0.68
   ...
   ```

3. **results/quercetin/active_candidates.csv**: 筛选出的活性候选分子
   ```
   smiles,similarity_to_quercetin,ensemble_probability,ensemble_prediction
   O=C1C(O)=C(...),0.95,0.98,1
   ...
   ```

4. **results/quercetin/top_20_candidates.png**: Top 20候选分子结构图

## 🎯 典型使用场景

### 场景1: 只想快速筛选槲皮素类似物

```bash
# 如果已有数据和模型
python src/quercetin_screening.py
```

### 场景2: 评估不同模型性能

```bash
# 训练所有模型并对比
python src/traditional_models.py  # 传统模型
python src/deep_models.py         # 深度学习模型（需要GPU）
```

### 场景3: 使用自己的数据

1. 准备数据文件 `data/raw/antioxidant_dataset.csv`:
   ```csv
   smiles,label
   CCO,0
   O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1ccc(O)c(O)c1,1
   ...
   ```

2. 运行特征提取和训练:
   ```bash
   python run_pipeline.py --extract --train-traditional --screen
   ```

## ⚡ 性能优化建议

### 内存有限（< 8GB RAM）

```python
# 修改 config.py
DATA_COLLECTION_CONFIG = {
    'dataset_size': 1000,  # 减少数据集大小
}

FEATURE_CONFIG = {
    'morgan_nbits': 512,  # 减少指纹位数
}
```

### 时间有限

```bash
# 只训练最佳模型
python src/traditional_models.py
# 然后在代码中只保留 XGBoost 和 Random Forest
```

### GPU可用

```bash
# 训练深度学习模型
python run_pipeline.py --train-deep

# 使用ChemBERTa获得最佳性能
```

## 🔬 项目核心优势

### 1. 多模型对比
- ✅ 9种传统ML模型
- ✅ ChemBERTa（超越BERT的化学预训练模型）
- ✅ MolFormer（IBM分子Transformer）
- ✅ 自定义深度神经网络

### 2. 丰富的分子特征
- ✅ Morgan指纹（结构相似性）
- ✅ 25个分子描述符（物化性质）
- ✅ 20个抗氧化药效团特征

### 3. 专门的槲皮素筛选
- ✅ 结构相似度计算
- ✅ 多模型集成预测
- ✅ 自动生成报告和可视化

## 📈 预期结果

根据我们的测试，典型结果为：

- **最佳模型**: XGBoost 或 ChemBERTa
- **F1 Score**: 0.80-0.90
- **ROC-AUC**: 0.85-0.95
- **筛选率**: 从3000个分子中筛选出300-500个高潜力候选分子
- **Top候选相似度**: > 0.7（与槲皮素）

## 🆘 常见问题解决

### Q1: ImportError: No module named 'rdkit'
```bash
# 使用conda安装（推荐）
conda install -c conda-forge rdkit
```

### Q2: ChEMBL连接超时
程序会自动切换到合成数据集，不影响运行。

### Q3: 深度学习模型下载失败
```bash
# 跳过深度学习，只用传统模型
python run_pipeline.py --train-traditional --screen
```

### Q4: 内存不足
减少数据集大小或特征维度（见性能优化建议）。

## 🎓 进阶学习

### 详细分析
```bash
jupyter notebook notebooks/analysis.ipynb
```

### 完整文档
查看 `USAGE_GUIDE.md` 获取详细使用说明。

### 自定义开发
查看源代码文档和注释，所有模块都有详细说明。

## 📞 获取帮助

- 查看文档: `README.md`, `USAGE_GUIDE.md`
- 运行示例: `notebooks/analysis.ipynb`
- 检查配置: `config.py`

## ✅ 下一步

运行成功后，你可以：

1. 分析Top候选分子的结构
2. 调整筛选阈值获得更多/更少候选
3. 训练深度学习模型提升性能
4. 使用自己的分子库进行筛选
5. 进行湿实验验证

祝你筛选顺利！🎉

