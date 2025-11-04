"""
分子特征提取模块
支持多种分子表示方法：SMILES、分子指纹、分子描述符等
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import pickle
import os
from tqdm import tqdm
from typing import List, Dict, Tuple

class MolecularFeatureExtractor:
    """分子特征提取器"""
    
    def __init__(self):
        self.feature_names = []
        
    def smiles_to_mol(self, smiles: str):
        """SMILES字符串转换为分子对象"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except:
            return None
    
    def extract_fingerprints(self, smiles_list: List[str], 
                            fp_type='morgan', 
                            radius=2, 
                            n_bits=2048) -> np.ndarray:
        """
        提取分子指纹
        
        Args:
            smiles_list: SMILES字符串列表
            fp_type: 指纹类型 ('morgan', 'maccs', 'rdkit', 'atom_pair', 'topological')
            radius: Morgan指纹半径
            n_bits: 指纹位数
            
        Returns:
            特征矩阵
        """
        print(f"提取 {fp_type} 指纹 (n_bits={n_bits})...")
        
        fingerprints = []
        
        for smiles in tqdm(smiles_list):
            mol = self.smiles_to_mol(smiles)
            
            if mol is None:
                fingerprints.append(np.zeros(n_bits))
                continue
            
            try:
                if fp_type == 'morgan':
                    # Morgan指纹 (类似ECFP)
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                    
                elif fp_type == 'maccs':
                    # MACCS keys (167 bits)
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    n_bits = 167
                    
                elif fp_type == 'rdkit':
                    # RDKit指纹
                    fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
                    
                elif fp_type == 'atom_pair':
                    # Atom pair指纹
                    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
                    
                elif fp_type == 'topological':
                    # 拓扑指纹
                    fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
                    
                else:
                    raise ValueError(f"未知的指纹类型: {fp_type}")
                
                # 转换为numpy数组
                arr = np.zeros(n_bits)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fingerprints.append(arr)
                
            except Exception as e:
                fingerprints.append(np.zeros(n_bits))
        
        return np.array(fingerprints)
    
    def extract_descriptors(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        提取分子描述符
        
        Returns:
            (特征矩阵, 特征名称列表)
        """
        print("提取分子描述符...")
        
        # 常用的分子描述符
        descriptor_functions = {
            'MolWt': Descriptors.MolWt,
            'LogP': Descriptors.MolLogP,
            'NumHDonors': Descriptors.NumHDonors,
            'NumHAcceptors': Descriptors.NumHAcceptors,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'NumAromaticRings': Descriptors.NumAromaticRings,
            'NumAliphaticRings': Descriptors.NumAliphaticRings,
            'TPSA': Descriptors.TPSA,
            'NumHeteroatoms': Descriptors.NumHeteroatoms,
            'NumValenceElectrons': Descriptors.NumValenceElectrons,
            'NumRadicalElectrons': Descriptors.NumRadicalElectrons,
            'FractionCsp3': Descriptors.FractionCsp3,
            'NumSaturatedRings': Descriptors.NumSaturatedRings,
            'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles,
            'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles,
            'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles,
            'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles,
            'RingCount': Descriptors.RingCount,
            'MolMR': Descriptors.MolMR,
            'BertzCT': Descriptors.BertzCT,
            'BalabanJ': Descriptors.BalabanJ,
            'Chi0v': Descriptors.Chi0v,
            'Chi1v': Descriptors.Chi1v,
            'Kappa1': Descriptors.Kappa1,
            'Kappa2': Descriptors.Kappa2,
            'Kappa3': Descriptors.Kappa3,
        }
        
        feature_names = list(descriptor_functions.keys())
        descriptors_matrix = []
        
        for smiles in tqdm(smiles_list):
            mol = self.smiles_to_mol(smiles)
            
            if mol is None:
                descriptors_matrix.append(np.zeros(len(feature_names)))
                continue
            
            try:
                descriptor_values = []
                for name, func in descriptor_functions.items():
                    try:
                        value = func(mol)
                        # 处理NaN和Inf
                        if np.isnan(value) or np.isinf(value):
                            value = 0.0
                        descriptor_values.append(value)
                    except:
                        descriptor_values.append(0.0)
                
                descriptors_matrix.append(descriptor_values)
                
            except Exception as e:
                descriptors_matrix.append(np.zeros(len(feature_names)))
        
        return np.array(descriptors_matrix), feature_names
    
    def extract_pharmacophore_features(self, smiles_list: List[str]) -> np.ndarray:
        """
        提取药效团特征（与抗氧化相关的结构特征）
        """
        print("提取药效团特征...")
        
        features = []
        
        for smiles in tqdm(smiles_list):
            mol = self.smiles_to_mol(smiles)
            
            if mol is None:
                features.append(np.zeros(20))
                continue
            
            try:
                # 抗氧化相关的结构特征
                feature_dict = {
                    # 羟基数量（-OH）
                    'num_oh': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))),
                    
                    # 酚羟基数量
                    'num_phenolic_oh': len(mol.GetSubstructMatches(Chem.MolFromSmarts('c[OH]'))),
                    
                    # 儿茶酚结构 (邻二酚)
                    'num_catechol': len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1c(O)c(O)ccc1'))),
                    
                    # 没食子酰基结构
                    'num_galloyl': len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1c(O)c(O)c(O)cc1'))),
                    
                    # 黄酮骨架
                    'has_flavone': len(mol.GetSubstructMatches(Chem.MolFromSmarts('O=C1CC(Oc2ccccc12)c1ccccc1'))) > 0,
                    
                    # 黄酮醇骨架
                    'has_flavonol': len(mol.GetSubstructMatches(Chem.MolFromSmarts('O=C1C(O)=C(Oc2ccccc12)c1ccccc1'))) > 0,
                    
                    # 芳香环数量
                    'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                    
                    # 甲氧基数量（-OCH3）
                    'num_methoxy': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OD2]([#6])[CH3]'))),
                    
                    # 羧基数量（-COOH）
                    'num_carboxyl': len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)[OH]'))),
                    
                    # 酯基数量
                    'num_ester': len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)O[#6]'))),
                    
                    # 双键数量
                    'num_double_bonds': len([bond for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE]),
                    
                    # 共轭系统长度（简化）
                    'conjugation_score': len(mol.GetSubstructMatches(Chem.MolFromSmarts('*=*-*=*'))),
                    
                    # 氧原子数量
                    'num_oxygen': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#8]'))),
                    
                    # 氮原子数量
                    'num_nitrogen': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]'))),
                    
                    # 硫原子数量
                    'num_sulfur': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#16]'))),
                    
                    # 芳香杂环数量
                    'num_aromatic_heterocycles': Descriptors.NumAromaticHeterocycles(mol),
                    
                    # 分子对称性（简化）
                    'symmetry_score': len(Chem.GetSymmSSSR(mol)),
                    
                    # 刚性度（旋转键比例）
                    'rigidity': 1 - (Descriptors.NumRotatableBonds(mol) / max(mol.GetNumBonds(), 1)),
                    
                    # 平面性指标（芳香环比例）
                    'planarity': Descriptors.NumAromaticRings(mol) / max(Descriptors.RingCount(mol), 1) if Descriptors.RingCount(mol) > 0 else 0,
                    
                    # 疏水性/亲水性平衡
                    'hydrophilic_ratio': (Descriptors.NumHDonors(mol) + Descriptors.NumHAcceptors(mol)) / max(Descriptors.MolWt(mol), 1) * 100,
                }
                
                feature_vector = list(feature_dict.values())
                features.append(feature_vector)
                
            except Exception as e:
                features.append(np.zeros(20))
        
        return np.array(features, dtype=float)
    
    def extract_combined_features(self, smiles_list: List[str], 
                                  include_fingerprints=True,
                                  include_descriptors=True,
                                  include_pharmacophore=True) -> Tuple[np.ndarray, List[str]]:
        """
        提取组合特征
        
        Returns:
            (特征矩阵, 特征名称列表)
        """
        print("\n提取组合分子特征...")
        print(f"  包含指纹: {include_fingerprints}")
        print(f"  包含描述符: {include_descriptors}")
        print(f"  包含药效团特征: {include_pharmacophore}")
        
        all_features = []
        all_feature_names = []
        
        # 1. 分子指纹
        if include_fingerprints:
            morgan_fp = self.extract_fingerprints(smiles_list, fp_type='morgan', n_bits=1024)
            all_features.append(morgan_fp)
            all_feature_names.extend([f'Morgan_{i}' for i in range(1024)])
        
        # 2. 分子描述符
        if include_descriptors:
            descriptors, desc_names = self.extract_descriptors(smiles_list)
            all_features.append(descriptors)
            all_feature_names.extend(desc_names)
        
        # 3. 药效团特征
        if include_pharmacophore:
            pharma_features = self.extract_pharmacophore_features(smiles_list)
            all_features.append(pharma_features)
            pharma_names = [
                'num_oh', 'num_phenolic_oh', 'num_catechol', 'num_galloyl',
                'has_flavone', 'has_flavonol', 'num_aromatic_rings', 'num_methoxy',
                'num_carboxyl', 'num_ester', 'num_double_bonds', 'conjugation_score',
                'num_oxygen', 'num_nitrogen', 'num_sulfur', 'num_aromatic_heterocycles',
                'symmetry_score', 'rigidity', 'planarity', 'hydrophilic_ratio'
            ]
            all_feature_names.extend(pharma_names)
        
        # 合并所有特征
        combined_features = np.hstack(all_features)
        
        print(f"\n总特征维度: {combined_features.shape}")
        
        return combined_features, all_feature_names
    
    def process_dataset(self, input_csv: str, output_dir: str = 'data/processed'):
        """
        处理数据集，提取特征并保存
        """
        print("="*60)
        print("开始处理数据集")
        print("="*60)
        
        # 读取数据
        df = pd.read_csv(input_csv)
        print(f"\n加载数据: {len(df)} 条记录")
        
        # 提取SMILES和标签
        smiles_list = df['smiles'].tolist()
        labels = df['label'].values
        
        # 提取特征
        features, feature_names = self.extract_combined_features(
            smiles_list,
            include_fingerprints=True,
            include_descriptors=True,
            include_pharmacophore=True
        )
        
        # 保存特征
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为numpy数组
        np.save(os.path.join(output_dir, 'features.npy'), features)
        np.save(os.path.join(output_dir, 'labels.npy'), labels)
        
        # 保存特征名称
        with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(feature_names, f)
        
        # 保存SMILES
        with open(os.path.join(output_dir, 'smiles.pkl'), 'wb') as f:
            pickle.dump(smiles_list, f)
        
        # 保存为CSV（可选，用于检查）
        feature_df = pd.DataFrame(features, columns=feature_names)
        feature_df['label'] = labels
        feature_df['smiles'] = smiles_list
        feature_df.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
        
        print(f"\n特征已保存到: {output_dir}")
        print(f"  - features.npy: {features.shape}")
        print(f"  - labels.npy: {labels.shape}")
        print(f"  - feature_names.pkl: {len(feature_names)} 个特征")
        
        return features, labels, feature_names

def main():
    """主函数"""
    extractor = MolecularFeatureExtractor()
    
    # 处理数据集
    features, labels, feature_names = extractor.process_dataset(
        input_csv='data/raw/antioxidant_dataset.csv',
        output_dir='data/processed'
    )
    
    print("\n" + "="*60)
    print("特征提取完成!")
    print("="*60)

if __name__ == '__main__':
    main()

