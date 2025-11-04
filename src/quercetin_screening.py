"""
槲皮素筛选模块
使用训练好的模型筛选具有槲皮素样抗氧化活性的分子
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit import DataStructs
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from feature_extraction import MolecularFeatureExtractor
from traditional_models import TraditionalMLModels

class QuercetinScreening:
    """槲皮素筛选器"""
    
    def __init__(self, models_dir='models', results_dir='results'):
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        os.makedirs(os.path.join(results_dir, 'quercetin'), exist_ok=True)
        
        # 槲皮素的SMILES和分子结构
        self.quercetin_smiles = "O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1ccc(O)c(O)c1"
        self.quercetin_mol = Chem.MolFromSmiles(self.quercetin_smiles)
        
        # 特征提取器
        self.feature_extractor = MolecularFeatureExtractor()
        
        # 加载模型
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """加载训练好的模型"""
        print("加载模型...")
        
        traditional_dir = os.path.join(self.models_dir, 'traditional')
        
        if os.path.exists(traditional_dir):
            # 加载所有pkl模型
            for file in os.listdir(traditional_dir):
                if file.endswith('.pkl') and 'scaler' not in file:
                    model_name = file.replace('.pkl', '').replace('_', ' ')
                    model_path = os.path.join(traditional_dir, file)
                    
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    
                    print(f"  加载: {model_name}")
        
        if len(self.models) == 0:
            print("警告: 未找到已训练的模型，请先运行训练脚本")
    
    def calculate_similarity_to_quercetin(self, smiles_list):
        """计算与槲皮素的相似度"""
        print("\n计算与槲皮素的结构相似度...")
        
        # 槲皮素的Morgan指纹
        quercetin_fp = AllChem.GetMorganFingerprintAsBitVect(
            self.quercetin_mol, radius=2, nBits=2048
        )
        
        similarities = []
        
        for smiles in tqdm(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                similarities.append(0.0)
                continue
            
            # 计算Tanimoto相似度
            mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            similarity = DataStructs.TanimotoSimilarity(quercetin_fp, mol_fp)
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def analyze_quercetin_features(self):
        """分析槲皮素的结构特征"""
        print("\n分析槲皮素的结构特征...")
        
        mol = self.quercetin_mol
        
        features = {
            '分子式': Chem.rdMolDescriptors.CalcMolFormula(mol),
            '分子量': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            '氢键供体数': Descriptors.NumHDonors(mol),
            '氢键受体数': Descriptors.NumHAcceptors(mol),
            '可旋转键数': Descriptors.NumRotatableBonds(mol),
            '芳香环数': Descriptors.NumAromaticRings(mol),
            'TPSA': Descriptors.TPSA(mol),
            '羟基数': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))),
            '酚羟基数': len(mol.GetSubstructMatches(Chem.MolFromSmarts('c[OH]'))),
            '儿茶酚结构数': len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1c(O)c(O)ccc1'))),
        }
        
        print("\n槲皮素结构特征:")
        for key, value in features.items():
            print(f"  {key}: {value}")
        
        # 绘制结构
        img = Draw.MolToImage(mol, size=(400, 400))
        img_path = os.path.join(self.results_dir, 'quercetin', 'quercetin_structure.png')
        img.save(img_path)
        print(f"\n槲皮素结构图已保存: {img_path}")
        
        return features
    
    def screen_candidates(self, candidate_smiles_list, threshold=0.5):
        """筛选候选分子"""
        print("\n" + "="*60)
        print("开始筛选候选分子")
        print("="*60)
        print(f"候选分子数量: {len(candidate_smiles_list)}")
        
        # 提取特征
        print("\n提取分子特征...")
        features, feature_names = self.feature_extractor.extract_combined_features(
            candidate_smiles_list,
            include_fingerprints=True,
            include_descriptors=True,
            include_pharmacophore=True
        )
        
        # 计算相似度
        similarities = self.calculate_similarity_to_quercetin(candidate_smiles_list)
        
        # 使用模型预测
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            print(f"\n使用 {model_name} 进行预测...")
            
            try:
                # 某些模型可能需要标准化
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features)[:, 1]
                    preds = (probs >= threshold).astype(int)
                else:
                    preds = model.predict(features)
                    probs = preds.astype(float)
                
                predictions[model_name] = preds
                probabilities[model_name] = probs
                
                print(f"  预测为活性的分子: {preds.sum()} / {len(preds)}")
                
            except Exception as e:
                print(f"  {model_name} 预测失败: {e}")
        
        # 集成预测（投票）
        if predictions:
            ensemble_probs = np.mean(list(probabilities.values()), axis=0)
            ensemble_preds = (ensemble_probs >= threshold).astype(int)
        else:
            ensemble_probs = np.zeros(len(candidate_smiles_list))
            ensemble_preds = np.zeros(len(candidate_smiles_list))
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'smiles': candidate_smiles_list,
            'similarity_to_quercetin': similarities,
            'ensemble_probability': ensemble_probs,
            'ensemble_prediction': ensemble_preds
        })
        
        # 添加各模型的预测
        for model_name, probs in probabilities.items():
            results_df[f'{model_name}_probability'] = probs
            results_df[f'{model_name}_prediction'] = predictions[model_name]
        
        # 按概率排序
        results_df = results_df.sort_values('ensemble_probability', ascending=False)
        
        # 筛选出预测为活性的分子
        active_candidates = results_df[results_df['ensemble_prediction'] == 1]
        
        print("\n" + "="*60)
        print("筛选结果")
        print("="*60)
        print(f"预测为活性的分子: {len(active_candidates)} / {len(results_df)}")
        print(f"高相似度分子 (>0.7): {(similarities > 0.7).sum()}")
        
        # 保存结果
        output_path = os.path.join(self.results_dir, 'quercetin', 'screening_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\n完整结果已保存: {output_path}")
        
        active_path = os.path.join(self.results_dir, 'quercetin', 'active_candidates.csv')
        active_candidates.to_csv(active_path, index=False)
        print(f"活性候选分子已保存: {active_path}")
        
        return results_df, active_candidates
    
    def visualize_top_candidates(self, results_df, top_n=20):
        """可视化Top候选分子"""
        print(f"\n可视化Top {top_n}候选分子...")
        
        top_candidates = results_df.head(top_n)
        
        # 转换为分子对象
        mols = []
        legends = []
        
        for idx, row in top_candidates.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                mols.append(mol)
                legends.append(
                    f"Prob: {row['ensemble_probability']:.3f}\n"
                    f"Sim: {row['similarity_to_quercetin']:.3f}"
                )
        
        # 绘制分子网格
        if mols:
            img = Draw.MolsToGridImage(
                mols[:min(len(mols), top_n)],
                molsPerRow=5,
                subImgSize=(250, 250),
                legends=legends[:min(len(legends), top_n)]
            )
            
            output_path = os.path.join(self.results_dir, 'quercetin', f'top_{top_n}_candidates.png')
            img.save(output_path)
            print(f"Top候选分子图已保存: {output_path}")
    
    def plot_screening_distribution(self, results_df):
        """绘制筛选结果分布"""
        print("\n绘制筛选结果分布...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 概率分布
        axes[0, 0].hist(results_df['ensemble_probability'], bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Ensemble Probability')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Probability Distribution')
        axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 0].legend()
        
        # 2. 相似度分布
        axes[0, 1].hist(results_df['similarity_to_quercetin'], bins=50, edgecolor='black', color='green')
        axes[0, 1].set_xlabel('Similarity to Quercetin')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Similarity Distribution')
        
        # 3. 概率 vs 相似度
        scatter = axes[1, 0].scatter(
            results_df['similarity_to_quercetin'],
            results_df['ensemble_probability'],
            c=results_df['ensemble_prediction'],
            cmap='RdYlGn',
            alpha=0.6
        )
        axes[1, 0].set_xlabel('Similarity to Quercetin')
        axes[1, 0].set_ylabel('Ensemble Probability')
        axes[1, 0].set_title('Probability vs Similarity')
        plt.colorbar(scatter, ax=axes[1, 0], label='Prediction')
        
        # 4. 预测分布（饼图）
        pred_counts = results_df['ensemble_prediction'].value_counts()
        axes[1, 1].pie(pred_counts, labels=['Inactive', 'Active'], autopct='%1.1f%%',
                      colors=['lightcoral', 'lightgreen'])
        axes[1, 1].set_title('Prediction Distribution')
        
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, 'quercetin', 'screening_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"分布图已保存: {output_path}")
        plt.close()
    
    def generate_report(self, results_df, active_candidates):
        """生成筛选报告"""
        print("\n生成筛选报告...")
        
        report = []
        report.append("="*60)
        report.append("槲皮素类似物筛选报告")
        report.append("="*60)
        report.append("")
        
        # 基本统计
        report.append("1. 基本统计")
        report.append(f"   总候选分子数: {len(results_df)}")
        report.append(f"   预测为活性: {len(active_candidates)} ({len(active_candidates)/len(results_df)*100:.1f}%)")
        report.append("")
        
        # 相似度统计
        report.append("2. 与槲皮素的相似度")
        report.append(f"   平均相似度: {results_df['similarity_to_quercetin'].mean():.3f}")
        report.append(f"   最高相似度: {results_df['similarity_to_quercetin'].max():.3f}")
        report.append(f"   相似度>0.7的分子: {(results_df['similarity_to_quercetin'] > 0.7).sum()}")
        report.append(f"   相似度>0.5的分子: {(results_df['similarity_to_quercetin'] > 0.5).sum()}")
        report.append("")
        
        # Top 10候选分子
        report.append("3. Top 10候选分子")
        report.append("")
        top10 = results_df.head(10)
        for idx, row in top10.iterrows():
            report.append(f"   #{idx+1}")
            report.append(f"   SMILES: {row['smiles']}")
            report.append(f"   活性概率: {row['ensemble_probability']:.3f}")
            report.append(f"   相似度: {row['similarity_to_quercetin']:.3f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        # 保存报告
        report_path = os.path.join(self.results_dir, 'quercetin', 'screening_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n报告已保存: {report_path}")

def main():
    """主函数"""
    print("="*60)
    print("槲皮素类似物筛选")
    print("="*60)
    
    # 创建筛选器
    screener = QuercetinScreening()
    
    # 分析槲皮素特征
    quercetin_features = screener.analyze_quercetin_features()
    
    # 加载候选分子（这里使用测试数据集作为示例）
    print("\n加载候选分子...")
    
    # 方案1: 从处理好的数据集中加载
    try:
        with open('data/processed/smiles.pkl', 'rb') as f:
            candidate_smiles = pickle.load(f)
        print(f"加载了 {len(candidate_smiles)} 个候选分子")
    except:
        # 方案2: 手动定义一些已知的抗氧化分子SMILES作为测试
        candidate_smiles = [
            "O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1ccc(O)c(O)c1",  # 槲皮素
            "O=C1CC(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12",  # 柚皮素
            "O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1cc(O)c(O)c(O)c1",  # 杨梅素
            "COc1cc(C2Oc3cc(O)cc(O)c3C(=O)C2O)ccc1O",  # 异鼠李素
            "O=C(O)C=Cc1ccc(O)c(O)c1",  # 咖啡酸
            "O=C(O)C=Cc1ccc(O)cc1",  # 对香豆酸
            "COc1cc(C=CC(=O)O)ccc1O",  # 阿魏酸
        ]
        print(f"使用测试数据: {len(candidate_smiles)} 个分子")
    
    # 筛选
    results_df, active_candidates = screener.screen_candidates(candidate_smiles)
    
    # 可视化
    screener.visualize_top_candidates(results_df, top_n=20)
    screener.plot_screening_distribution(results_df)
    
    # 生成报告
    screener.generate_report(results_df, active_candidates)
    
    print("\n" + "="*60)
    print("筛选完成！")
    print("="*60)

if __name__ == '__main__':
    main()

