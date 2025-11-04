"""
模型训练和评估的统一框架
"""

import numpy as np
import pandas as pd
import argparse
import os
import pickle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from traditional_models import TraditionalMLModels
from deep_models import ChemBERTaModel, MolFormerModel, DeepNNClassifier

class ModelTrainingPipeline:
    """模型训练管道"""
    
    def __init__(self, data_dir='data/processed', output_dir='models', results_dir='results'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.results_dir = results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        self.all_results = {}
    
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        
        # 加载特征数据
        self.X = np.load(os.path.join(self.data_dir, 'features.npy'))
        self.y = np.load(os.path.join(self.data_dir, 'labels.npy'))
        
        # 加载SMILES
        with open(os.path.join(self.data_dir, 'smiles.pkl'), 'rb') as f:
            self.smiles = pickle.load(f)
        
        # 加载特征名称
        with open(os.path.join(self.data_dir, 'feature_names.pkl'), 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print(f"  特征形状: {self.X.shape}")
        print(f"  标签分布: {np.bincount(self.y)}")
        print(f"  SMILES数量: {len(self.smiles)}")
        
        return self.X, self.y, self.smiles
    
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """划分数据集"""
        print("\n划分数据集...")
        
        # 先划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test, \
        self.smiles_train, self.smiles_test = train_test_split(
            self.X, self.y, self.smiles,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )
        
        # 再从训练集中划分验证集
        self.X_train, self.X_val, self.y_train, self.y_val, \
        self.smiles_train, self.smiles_val = train_test_split(
            self.X_train, self.y_train, self.smiles_train,
            test_size=val_size,
            random_state=random_state,
            stratify=self.y_train
        )
        
        print(f"  训练集: {self.X_train.shape[0]} 样本")
        print(f"  验证集: {self.X_val.shape[0]} 样本")
        print(f"  测试集: {self.X_test.shape[0]} 样本")
    
    def train_traditional_models(self):
        """训练传统机器学习模型"""
        print("\n" + "="*60)
        print("训练传统机器学习模型")
        print("="*60)
        
        ml_models = TraditionalMLModels()
        results = ml_models.train_all_models(
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        
        # 保存模型
        ml_models.save_models(os.path.join(self.output_dir, 'traditional'))
        
        # 绘制对比图
        ml_models.plot_model_comparison(self.results_dir)
        
        # 保存结果
        self.all_results['traditional'] = results
        
        return results
    
    def train_deep_models(self, use_transformer=True, use_dnn=True):
        """训练深度学习模型"""
        print("\n" + "="*60)
        print("训练深度学习模型")
        print("="*60)
        
        deep_results = []
        
        # 1. ChemBERTa (Transformer模型)
        if use_transformer:
            try:
                print("\n训练 ChemBERTa...")
                chemberta = ChemBERTaModel()
                chemberta.train(
                    self.smiles_train, self.y_train,
                    self.smiles_val, self.y_val,
                    output_dir=os.path.join(self.output_dir, 'chemberta'),
                    epochs=3,  # 可以调整
                    batch_size=16
                )
                
                metrics = chemberta.evaluate(self.smiles_test, self.y_test)
                metrics['model_name'] = 'ChemBERTa'
                deep_results.append(metrics)
                
                print("\nChemBERTa 性能:")
                for k, v in metrics.items():
                    if k != 'model_name':
                        print(f"  {k}: {v:.4f}")
                
            except Exception as e:
                print(f"ChemBERTa训练失败: {e}")
        
        # 2. 深度神经网络
        if use_dnn:
            try:
                print("\n训练深度神经网络...")
                dnn = DeepNNClassifier(input_dim=self.X_train.shape[1])
                dnn.train(
                    self.X_train, self.y_train,
                    self.X_val, self.y_val,
                    epochs=30,
                    batch_size=64
                )
                
                metrics = dnn.evaluate(self.X_test, self.y_test)
                metrics['model_name'] = 'Deep NN'
                deep_results.append(metrics)
                
                print("\nDeep NN 性能:")
                for k, v in metrics.items():
                    if k != 'model_name':
                        print(f"  {k}: {v:.4f}")
                
                # 保存模型
                dnn.save(os.path.join(self.output_dir, 'deep_nn.pth'))
                
            except Exception as e:
                print(f"深度神经网络训练失败: {e}")
        
        # 保存结果
        if deep_results:
            deep_results_df = pd.DataFrame(deep_results)
            deep_results_df.to_csv(
                os.path.join(self.results_dir, 'deep_learning_results.csv'),
                index=False
            )
            self.all_results['deep'] = deep_results_df
        
        return deep_results
    
    def compare_all_models(self):
        """比较所有模型"""
        print("\n" + "="*60)
        print("所有模型性能对比")
        print("="*60)
        
        all_results_list = []
        
        # 合并传统模型结果
        if 'traditional' in self.all_results:
            all_results_list.append(self.all_results['traditional'])
        
        # 合并深度学习结果
        if 'deep' in self.all_results:
            all_results_list.append(self.all_results['deep'])
        
        if all_results_list:
            combined_results = pd.concat(all_results_list, ignore_index=True)
            
            # 按F1分数排序
            combined_results = combined_results.sort_values('f1_score', ascending=False)
            
            print("\n性能排名（按F1分数）:")
            print(combined_results[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string(index=False))
            
            # 保存综合结果
            combined_results.to_csv(
                os.path.join(self.results_dir, 'all_models_comparison.csv'),
                index=False
            )
            
            return combined_results
        
        return None
    
    def run_full_pipeline(self, train_traditional=True, train_deep=True):
        """运行完整的训练管道"""
        print("="*60)
        print("开始模型训练管道")
        print("="*60)
        
        # 加载数据
        self.load_data()
        
        # 划分数据
        self.split_data()
        
        # 训练传统模型
        if train_traditional:
            self.train_traditional_models()
        
        # 训练深度学习模型
        if train_deep:
            self.train_deep_models(use_transformer=True, use_dnn=True)
        
        # 比较所有模型
        results = self.compare_all_models()
        
        print("\n" + "="*60)
        print("训练管道完成！")
        print("="*60)
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练抗氧化分子筛选模型')
    parser.add_argument('--model', type=str, default='all',
                       choices=['traditional', 'deep', 'all'],
                       help='选择训练的模型类型')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='模型输出目录')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 创建训练管道
    pipeline = ModelTrainingPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        results_dir=args.results_dir
    )
    
    # 运行训练
    train_traditional = args.model in ['traditional', 'all']
    train_deep = args.model in ['deep', 'all']
    
    results = pipeline.run_full_pipeline(
        train_traditional=train_traditional,
        train_deep=train_deep
    )

if __name__ == '__main__':
    main()

