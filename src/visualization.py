"""
可视化工具模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

class MolecularVisualizer:
    """分子可视化工具"""
    
    def __init__(self, output_dir='results/visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_molecule_grid(self, smiles_list, labels=None, filename='molecules_grid.png',
                          mols_per_row=5, img_size=(250, 250)):
        """绘制分子结构网格"""
        mols = []
        legends = []
        
        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mols.append(mol)
                if labels:
                    legends.append(f"#{idx+1}: {labels[idx]}")
                else:
                    legends.append(f"#{idx+1}")
        
        if mols:
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=mols_per_row,
                subImgSize=img_size,
                legends=legends if labels else None
            )
            
            output_path = os.path.join(self.output_dir, filename)
            img.save(output_path)
            print(f"分子网格图已保存: {output_path}")
            
            return img
        
        return None
    
    def plot_feature_distribution(self, X, y, feature_names, top_n=10, filename='feature_distributions.png'):
        """绘制特征分布"""
        # 计算特征重要性（简单使用方差）
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-top_n:]
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, feature_idx in enumerate(top_indices):
            if idx >= 10:
                break
            
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature {feature_idx}'
            
            # 分别绘制活性和非活性样本的分布
            active_values = X[y == 1, feature_idx]
            inactive_values = X[y == 0, feature_idx]
            
            axes[idx].hist(active_values, bins=30, alpha=0.5, label='Active', color='green')
            axes[idx].hist(inactive_values, bins=30, alpha=0.5, label='Inactive', color='red')
            axes[idx].set_xlabel(feature_name)
            axes[idx].set_ylabel('Count')
            axes[idx].legend()
            axes[idx].set_title(f'{feature_name}')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"特征分布图已保存: {output_path}")
        plt.close()
    
    def plot_pca_visualization(self, X, y, labels=None, filename='pca_visualization.png'):
        """PCA降维可视化"""
        print("执行PCA降维...")
        
        # PCA降维到2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 分别绘制两类
        colors = ['red', 'green']
        class_names = ['Inactive', 'Active']
        
        for class_idx in [0, 1]:
            mask = y == class_idx
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=colors[class_idx],
                label=class_names[class_idx],
                alpha=0.6,
                s=50
            )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('PCA Visualization of Molecular Features')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"PCA可视化已保存: {output_path}")
        plt.close()
        
        return X_pca, pca
    
    def plot_tsne_visualization(self, X, y, perplexity=30, filename='tsne_visualization.png'):
        """t-SNE降维可视化"""
        print("执行t-SNE降维（可能需要一些时间）...")
        
        # t-SNE降维
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        X_tsne = tsne.fit_transform(X)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['red', 'green']
        class_names = ['Inactive', 'Active']
        
        for class_idx in [0, 1]:
            mask = y == class_idx
            ax.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                c=colors[class_idx],
                label=class_names[class_idx],
                alpha=0.6,
                s=50
            )
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('t-SNE Visualization of Molecular Features')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE可视化已保存: {output_path}")
        plt.close()
        
        return X_tsne
    
    def plot_interactive_3d_pca(self, X, y, smiles=None, filename='pca_3d.html'):
        """交互式3D PCA可视化"""
        print("创建3D PCA可视化...")
        
        # PCA降维到3D
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'PC3': X_pca[:, 2],
            'Class': ['Active' if label == 1 else 'Inactive' for label in y]
        })
        
        if smiles:
            df['SMILES'] = smiles
        
        # 创建3D散点图
        fig = px.scatter_3d(
            df, x='PC1', y='PC2', z='PC3',
            color='Class',
            color_discrete_map={'Active': 'green', 'Inactive': 'red'},
            hover_data=['SMILES'] if smiles else None,
            title='3D PCA Visualization',
            labels={
                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
            }
        )
        
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        print(f"3D PCA可视化已保存: {output_path}")
    
    def plot_molecular_properties_comparison(self, smiles_active, smiles_inactive,
                                            filename='properties_comparison.png'):
        """比较活性和非活性分子的物理化学性质"""
        print("比较分子性质...")
        
        def calculate_properties(smiles_list):
            """计算分子性质"""
            properties = {
                'MolWt': [],
                'LogP': [],
                'NumHDonors': [],
                'NumHAcceptors': [],
                'TPSA': [],
                'NumRotatableBonds': [],
                'NumAromaticRings': []
            }
            
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    properties['MolWt'].append(Descriptors.MolWt(mol))
                    properties['LogP'].append(Descriptors.MolLogP(mol))
                    properties['NumHDonors'].append(Descriptors.NumHDonors(mol))
                    properties['NumHAcceptors'].append(Descriptors.NumHAcceptors(mol))
                    properties['TPSA'].append(Descriptors.TPSA(mol))
                    properties['NumRotatableBonds'].append(Descriptors.NumRotatableBonds(mol))
                    properties['NumAromaticRings'].append(Descriptors.NumAromaticRings(mol))
            
            return properties
        
        # 计算性质
        props_active = calculate_properties(smiles_active)
        props_inactive = calculate_properties(smiles_inactive)
        
        # 绘制对比图
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, prop_name in enumerate(props_active.keys()):
            if idx >= 7:
                break
            
            axes[idx].hist(props_active[prop_name], bins=30, alpha=0.5,
                          label='Active', color='green', density=True)
            axes[idx].hist(props_inactive[prop_name], bins=30, alpha=0.5,
                          label='Inactive', color='red', density=True)
            axes[idx].set_xlabel(prop_name)
            axes[idx].set_ylabel('Density')
            axes[idx].set_title(f'{prop_name} Distribution')
            axes[idx].legend()
        
        # 最后一个子图显示统计摘要
        axes[7].axis('off')
        summary_text = "Property Statistics:\n\n"
        summary_text += "Active molecules:\n"
        summary_text += f"  Count: {len(smiles_active)}\n"
        summary_text += f"  Avg MW: {np.mean(props_active['MolWt']):.2f}\n"
        summary_text += f"  Avg LogP: {np.mean(props_active['LogP']):.2f}\n\n"
        summary_text += "Inactive molecules:\n"
        summary_text += f"  Count: {len(smiles_inactive)}\n"
        summary_text += f"  Avg MW: {np.mean(props_inactive['MolWt']):.2f}\n"
        summary_text += f"  Avg LogP: {np.mean(props_inactive['LogP']):.2f}\n"
        
        axes[7].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"性质对比图已保存: {output_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=['Inactive', 'Active'],
                             filename='confusion_matrix.png'):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {output_path}")
        plt.close()
    
    def plot_roc_curves(self, y_true, y_probs_dict, filename='roc_curves.png'):
        """绘制多个模型的ROC曲线"""
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, y_prob in y_probs_dict.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, lw=2,
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存: {output_path}")
        plt.close()

def main():
    """主函数 - 创建可视化示例"""
    import pickle
    
    print("="*60)
    print("创建可视化")
    print("="*60)
    
    visualizer = MolecularVisualizer()
    
    try:
        # 加载数据
        print("\n加载数据...")
        X = np.load('data/processed/features.npy')
        y = np.load('data/processed/labels.npy')
        
        with open('data/processed/smiles.pkl', 'rb') as f:
            smiles = pickle.load(f)
        
        with open('data/processed/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # PCA可视化
        visualizer.plot_pca_visualization(X, y)
        
        # t-SNE可视化
        # visualizer.plot_tsne_visualization(X[:1000], y[:1000])  # 仅用部分数据以节省时间
        
        # 3D交互式PCA
        visualizer.plot_interactive_3d_pca(X, y, smiles)
        
        # 特征分布
        visualizer.plot_feature_distribution(X, y, feature_names)
        
        # 分子性质对比
        smiles_active = [smiles[i] for i in range(len(smiles)) if y[i] == 1]
        smiles_inactive = [smiles[i] for i in range(len(smiles)) if y[i] == 0]
        
        visualizer.plot_molecular_properties_comparison(
            smiles_active[:100],
            smiles_inactive[:100]
        )
        
        # 分子网格
        top_active = smiles_active[:20]
        visualizer.plot_molecule_grid(
            top_active,
            labels=[f'Active #{i+1}' for i in range(len(top_active))],
            filename='top_active_molecules.png'
        )
        
        print("\n" + "="*60)
        print("可视化完成！")
        print("="*60)
        
    except Exception as e:
        print(f"可视化过程出错: {e}")
        print("请先运行数据收集和特征提取脚本")

if __name__ == '__main__':
    main()

