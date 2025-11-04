"""
传统机器学习模型模块
包括：Random Forest, XGBoost, SVM, LightGBM等
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
import pickle
import os
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

class TraditionalMLModels:
    """传统机器学习模型集合"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """初始化所有模型"""
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            
            'SVM (RBF)': SVC(
                C=10.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            
            'SVM (Linear)': SVC(
                C=1.0,
                kernel='linear',
                probability=True,
                random_state=self.random_state
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            ),
            
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            
            'Naive Bayes': GaussianNB()
        }
        
        return models
    
    def preprocess_data(self, X_train, X_test, scale=True):
        """数据预处理"""
        if scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, scaler
        else:
            return X_train, X_test, None
    
    def train_single_model(self, model_name: str, model, X_train, y_train, X_test, y_test):
        """训练单个模型"""
        print(f"\n训练 {model_name}...")
        
        # 某些模型需要标准化
        need_scaling = model_name in ['SVM (RBF)', 'SVM (Linear)', 'Logistic Regression', 'K-Nearest Neighbors']
        
        if need_scaling:
            X_train_proc, X_test_proc, scaler = self.preprocess_data(X_train, X_test, scale=True)
            self.scalers[model_name] = scaler
        else:
            X_train_proc, X_test_proc = X_train, X_test
        
        # 训练模型
        model.fit(X_train_proc, y_train)
        
        # 预测
        y_pred = model.predict(X_test_proc)
        y_pred_proba = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # 评估
        metrics = self.evaluate_model(y_test, y_pred, y_pred_proba)
        metrics['model_name'] = model_name
        
        # 保存模型
        self.models[model_name] = model
        
        return metrics
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
        """评估模型性能"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """训练所有模型"""
        print("="*60)
        print("开始训练传统机器学习模型")
        print("="*60)
        
        models = self.initialize_models()
        results = []
        
        for model_name, model in models.items():
            try:
                metrics = self.train_single_model(
                    model_name, model, X_train, y_train, X_test, y_test
                )
                results.append(metrics)
                
                print(f"\n{model_name} 性能:")
                print(f"  准确率: {metrics['accuracy']:.4f}")
                print(f"  精确率: {metrics['precision']:.4f}")
                print(f"  召回率: {metrics['recall']:.4f}")
                print(f"  F1分数: {metrics['f1_score']:.4f}")
                print(f"  MCC: {metrics['mcc']:.4f}")
                if 'roc_auc' in metrics:
                    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
                    
            except Exception as e:
                print(f"\n训练 {model_name} 时出错: {e}")
                continue
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def hyperparameter_tuning(self, model_type: str, X_train, y_train):
        """超参数调优"""
        print(f"\n对 {model_type} 进行超参数调优...")
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            },
            
            'SVM (RBF)': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        }
        
        if model_type not in param_grids:
            print(f"没有为 {model_type} 定义参数网格")
            return None
        
        base_models = {
            'Random Forest': RandomForestClassifier(random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=self.random_state)
        }
        
        model = base_models[model_type]
        param_grid = param_grids[model_type]
        
        # 网格搜索
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n最佳参数: {grid_search.best_params_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def get_feature_importance(self, model_name: str, feature_names: list = None, top_n: int = 20):
        """获取特征重要性"""
        if model_name not in self.models:
            print(f"模型 {model_name} 未找到")
            return None
        
        model = self.models[model_name]
        
        # 检查模型是否支持特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print(f"{model_name} 不支持特征重要性分析")
            return None
        
        # 创建DataFrame
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_models(self, output_dir: str = 'models/traditional'):
        """保存所有模型"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # 保存scalers
        if self.scalers:
            for scaler_name, scaler in self.scalers.items():
                scaler_path = os.path.join(output_dir, f"{scaler_name.replace(' ', '_')}_scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
        
        # 保存结果
        if len(self.results) > 0:
            self.results.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
        
        print(f"\n模型已保存到: {output_dir}")
    
    def load_model(self, model_path: str):
        """加载模型"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def plot_model_comparison(self, output_dir: str = 'results'):
        """绘制模型比较图"""
        if len(self.results) == 0:
            print("没有可用的结果进行绘图")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 性能指标比较
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if metric in self.results.columns:
                data = self.results.sort_values(metric, ascending=False)
                
                axes[idx].barh(data['model_name'], data[metric])
                axes[idx].set_xlabel(metric.upper())
                axes[idx].set_title(f'{metric.upper()} Comparison')
                axes[idx].set_xlim(0, 1)
                
                # 添加数值标签
                for i, v in enumerate(data[metric]):
                    axes[idx].text(v, i, f' {v:.3f}', va='center')
        
        # ROC-AUC比较（如果有）
        if 'roc_auc' in self.results.columns:
            data = self.results.sort_values('roc_auc', ascending=False)
            axes[5].barh(data['model_name'], data['roc_auc'])
            axes[5].set_xlabel('ROC-AUC')
            axes[5].set_title('ROC-AUC Comparison')
            axes[5].set_xlim(0, 1)
            
            for i, v in enumerate(data['roc_auc']):
                axes[5].text(v, i, f' {v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"模型比较图已保存到: {os.path.join(output_dir, 'model_comparison.png')}")
        plt.close()

def main():
    """主函数"""
    # 加载特征
    print("加载特征数据...")
    X = np.load('data/processed/features.npy')
    y = np.load('data/processed/labels.npy')
    
    with open('data/processed/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"数据形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    
    # 训练模型
    ml_models = TraditionalMLModels()
    results = ml_models.train_all_models(X_train, y_train, X_test, y_test)
    
    # 显示结果
    print("\n" + "="*60)
    print("所有模型性能对比:")
    print("="*60)
    print(results.to_string(index=False))
    
    # 保存模型
    ml_models.save_models()
    
    # 绘制对比图
    ml_models.plot_model_comparison()
    
    # 特征重要性分析（选择最佳模型）
    best_model = results.loc[results['f1_score'].idxmax(), 'model_name']
    print(f"\n最佳模型: {best_model}")
    
    importance = ml_models.get_feature_importance(best_model, feature_names, top_n=20)
    if importance is not None:
        print(f"\n{best_model} 的前20个重要特征:")
        print(importance.to_string(index=False))
        importance.to_csv('results/feature_importance.csv', index=False)
    
    print("\n" + "="*60)
    print("传统模型训练完成!")
    print("="*60)

if __name__ == '__main__':
    main()

