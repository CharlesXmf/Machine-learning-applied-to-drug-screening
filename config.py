"""
配置文件 - 集中管理项目参数
"""

import os

# ============================================================================
# 目录配置
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
QUERCETIN_DATA_DIR = os.path.join(DATA_DIR, 'quercetin')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRADITIONAL_MODELS_DIR = os.path.join(MODELS_DIR, 'traditional')
DEEP_MODELS_DIR = os.path.join(MODELS_DIR, 'deep')

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
QUERCETIN_RESULTS_DIR = os.path.join(RESULTS_DIR, 'quercetin')
VIZ_DIR = os.path.join(RESULTS_DIR, 'visualizations')

# ============================================================================
# 数据收集配置
# ============================================================================
DATA_COLLECTION_CONFIG = {
    'dataset_size': 3000,  # 目标数据集大小
    'active_ratio': 0.5,   # 活性分子比例
    'max_assays': 50,      # 最大实验数量
    'limit_per_assay': 100,  # 每个实验的最大活性数据
}

# ChEMBL搜索关键词
ANTIOXIDANT_KEYWORDS = [
    'oxidative stress',
    'antioxidant',
    'ROS',
    'reactive oxygen species',
    'superoxide dismutase',
    'catalase',
    'glutathione peroxidase',
    'DPPH',
    'ABTS',
    'FRAP'
]

# ============================================================================
# 特征提取配置
# ============================================================================
FEATURE_CONFIG = {
    # 分子指纹
    'morgan_radius': 2,
    'morgan_nbits': 1024,
    'use_morgan': True,
    'use_maccs': False,
    'use_rdkit_fp': False,
    
    # 分子描述符
    'use_descriptors': True,
    
    # 药效团特征
    'use_pharmacophore': True,
}

# ============================================================================
# 槲皮素配置
# ============================================================================
QUERCETIN_CONFIG = {
    'smiles': 'O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1ccc(O)c(O)c1',
    'name': 'Quercetin',
    'molecular_formula': 'C15H10O7',
    'molecular_weight': 302.24,
}

# 已知抗氧化分子SMILES（用于测试和验证）
KNOWN_ANTIOXIDANTS = {
    'Quercetin': 'O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1ccc(O)c(O)c1',
    'Naringenin': 'O=C1CC(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12',
    'Myricetin': 'O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1cc(O)c(O)c(O)c1',
    'Isorhamnetin': 'COc1cc(C2Oc3cc(O)cc(O)c3C(=O)C2O)ccc1O',
    'Caffeic acid': 'O=C(O)C=Cc1ccc(O)c(O)c1',
    'p-Coumaric acid': 'O=C(O)C=Cc1ccc(O)cc1',
    'Ferulic acid': 'COc1cc(C=CC(=O)O)ccc1O',
    'Catechin': 'OC1Cc2c(O)cc(O)cc2OC1c1ccc(O)c(O)c1',
    'Resveratrol': 'Oc1ccc(C=Cc2cc(O)cc(O)c2)cc1',
    'Gallic acid': 'O=C(O)c1cc(O)c(O)c(O)c1',
}

# ============================================================================
# 传统机器学习模型配置
# ============================================================================
TRADITIONAL_ML_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'val_size': 0.1,
    
    # Random Forest
    'rf_n_estimators': 200,
    'rf_max_depth': 20,
    'rf_min_samples_split': 5,
    
    # XGBoost
    'xgb_n_estimators': 200,
    'xgb_max_depth': 8,
    'xgb_learning_rate': 0.1,
    
    # LightGBM
    'lgb_n_estimators': 200,
    'lgb_max_depth': 8,
    'lgb_learning_rate': 0.1,
    
    # SVM
    'svm_C': 10.0,
    'svm_gamma': 'scale',
}

# ============================================================================
# 深度学习模型配置
# ============================================================================
DEEP_LEARNING_CONFIG = {
    # ChemBERTa
    'chemberta_model': 'DeepChem/ChemBERTa-77M-MLM',
    'chemberta_epochs': 5,
    'chemberta_batch_size': 16,
    'chemberta_max_length': 128,
    
    # MolFormer
    'molformer_model': 'ibm/MolFormer-XL-both-10pct',
    'molformer_epochs': 5,
    'molformer_batch_size': 16,
    
    # Deep NN
    'dnn_hidden_dim': 256,
    'dnn_epochs': 50,
    'dnn_batch_size': 64,
    'dnn_learning_rate': 0.001,
    
    # 通用
    'use_gpu': True,  # 如果可用则使用GPU
}

# 可用的预训练模型
AVAILABLE_PRETRAINED_MODELS = {
    'ChemBERTa-77M': 'DeepChem/ChemBERTa-77M-MLM',
    'ChemBERTa-10M': 'DeepChem/ChemBERTa-10M-MLM',
    'PubChem-BERT': 'seyonec/PubChem10M_SMILES_BPE_450k',
    'MolFormer-XL': 'ibm/MolFormer-XL-both-10pct',
    'MolFormer-Large': 'ibm/MolFormer-Large-both-10pct',
}

# ============================================================================
# 筛选配置
# ============================================================================
SCREENING_CONFIG = {
    'similarity_threshold': 0.5,  # 相似度阈值
    'probability_threshold': 0.5,  # 预测概率阈值
    'top_n_candidates': 20,        # 展示前N个候选分子
    'ensemble_method': 'mean',     # 集成方法: 'mean', 'voting'
}

# ============================================================================
# 可视化配置
# ============================================================================
VISUALIZATION_CONFIG = {
    'figure_dpi': 300,
    'figure_format': 'png',
    'mol_img_size': (250, 250),
    'mols_per_row': 5,
    'color_active': 'green',
    'color_inactive': 'red',
}

# ============================================================================
# 评估指标配置
# ============================================================================
METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc',
    'mcc',  # Matthews correlation coefficient
]

# ============================================================================
# 日志配置
# ============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'pipeline.log',
}

# ============================================================================
# 性能优化配置
# ============================================================================
PERFORMANCE_CONFIG = {
    'n_jobs': -1,  # 使用所有CPU核心
    'use_cache': True,
    'cache_dir': '.cache',
}

# ============================================================================
# 实验配置（用于超参数调优）
# ============================================================================
EXPERIMENT_CONFIG = {
    'cv_folds': 5,  # 交叉验证折数
    'random_search_iter': 20,  # 随机搜索迭代次数
    'grid_search': False,  # 是否使用网格搜索
}

