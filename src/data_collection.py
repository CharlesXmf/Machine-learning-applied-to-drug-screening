"""
数据收集模块
从ChEMBL数据库获取抗氧化活性相关的分子数据
"""

import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors
import pickle
import os
from tqdm import tqdm
import time

class AntioxidantDataCollector:
    """抗氧化分子数据收集器"""
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # ChEMBL客户端
        self.activity = new_client.activity
        self.molecule = new_client.molecule
        self.target = new_client.target
        self.assay = new_client.assay
        
    def search_antioxidant_targets(self):
        """搜索与抗氧化相关的靶点"""
        print("正在搜索抗氧化相关靶点...")
        
        # 搜索关键词
        keywords = [
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
        
        all_targets = []
        for keyword in keywords:
            try:
                targets = self.target.filter(target_synonym__icontains=keyword)
                all_targets.extend(list(targets))
                print(f"  关键词 '{keyword}': 找到 {len(list(targets))} 个靶点")
                time.sleep(0.5)  # 避免请求过快
            except Exception as e:
                print(f"  搜索 '{keyword}' 时出错: {e}")
                
        # 去重
        unique_targets = {t['target_chembl_id']: t for t in all_targets}
        print(f"\n总共找到 {len(unique_targets)} 个唯一靶点")
        
        return list(unique_targets.values())
    
    def search_antioxidant_assays(self):
        """搜索抗氧化相关的实验"""
        print("\n正在搜索抗氧化相关实验...")
        
        keywords = [
            'antioxidant',
            'DPPH',
            'ABTS',
            'FRAP',
            'ORAC',
            'oxidative',
            'ROS'
        ]
        
        all_assays = []
        for keyword in keywords:
            try:
                assays = self.assay.filter(description__icontains=keyword)
                assay_list = list(assays)
                all_assays.extend(assay_list)
                print(f"  关键词 '{keyword}': 找到 {len(assay_list)} 个实验")
                time.sleep(0.5)
            except Exception as e:
                print(f"  搜索 '{keyword}' 时出错: {e}")
        
        # 去重
        unique_assays = {a['assay_chembl_id']: a for a in all_assays}
        print(f"\n总共找到 {len(unique_assays)} 个唯一实验")
        
        return list(unique_assays.values())
    
    def get_activities_by_assays(self, assays, limit_per_assay=100):
        """根据实验ID获取活性数据"""
        print("\n正在获取活性数据...")
        
        all_activities = []
        
        for assay in tqdm(assays[:50]):  # 限制assay数量以控制数据规模
            try:
                assay_id = assay['assay_chembl_id']
                activities = self.activity.filter(
                    assay_chembl_id=assay_id,
                    standard_type__in=['IC50', 'EC50', 'Ki', 'Kd', 'Activity', 'Inhibition']
                ).only(['molecule_chembl_id', 'standard_type', 'standard_value', 
                       'standard_units', 'pchembl_value'])
                
                activity_list = list(activities)[:limit_per_assay]
                all_activities.extend(activity_list)
                time.sleep(0.3)
                
            except Exception as e:
                print(f"  获取实验 {assay_id} 数据时出错: {e}")
                continue
        
        print(f"总共获取 {len(all_activities)} 条活性数据")
        return all_activities
    
    def get_quercetin_related_molecules(self):
        """获取槲皮素及其类似物"""
        print("\n正在搜索槲皮素相关分子...")
        
        # 槲皮素的SMILES
        quercetin_smiles = "O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1ccc(O)c(O)c1"
        
        # 搜索槲皮素
        try:
            quercetin_mols = self.molecule.filter(
                molecule_synonyms__molecule_synonym__icontains='quercetin'
            )
            quercetin_list = list(quercetin_mols)
            print(f"找到 {len(quercetin_list)} 个槲皮素相关分子")
            
            # 保存槲皮素数据
            quercetin_df = pd.DataFrame(quercetin_list)
            quercetin_df.to_csv(
                os.path.join('data/quercetin', 'quercetin_molecules.csv'),
                index=False
            )
            
            return quercetin_list
            
        except Exception as e:
            print(f"搜索槲皮素时出错: {e}")
            return []
    
    def get_flavonoid_molecules(self, limit=2000):
        """获取类黄酮分子（槲皮素属于类黄酮）"""
        print("\n正在搜索类黄酮分子...")
        
        keywords = ['flavonoid', 'flavone', 'flavonol', 'anthocyanin', 'catechin']
        all_molecules = []
        
        for keyword in keywords:
            try:
                mols = self.molecule.filter(
                    molecule_synonyms__molecule_synonym__icontains=keyword
                ).only(['molecule_chembl_id', 'molecule_structures'])
                
                mol_list = list(mols)[:limit//len(keywords)]
                all_molecules.extend(mol_list)
                print(f"  关键词 '{keyword}': 找到 {len(mol_list)} 个分子")
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  搜索 '{keyword}' 时出错: {e}")
        
        return all_molecules
    
    def process_activities_to_dataset(self, activities):
        """处理活性数据为训练数据集"""
        print("\n正在处理活性数据...")
        
        data_records = []
        
        for act in tqdm(activities):
            try:
                mol_id = act.get('molecule_chembl_id')
                if not mol_id:
                    continue
                
                # 获取分子结构
                mol_data = self.molecule.get(mol_id)
                if not mol_data or 'molecule_structures' not in mol_data:
                    continue
                
                structures = mol_data['molecule_structures']
                if not structures or 'canonical_smiles' not in structures:
                    continue
                
                smiles = structures['canonical_smiles']
                
                # 验证SMILES有效性
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # 获取活性值
                pchembl_value = act.get('pchembl_value')
                standard_value = act.get('standard_value')
                
                # 标注活性（简化版：pchembl_value > 6 为活性）
                if pchembl_value is not None:
                    label = 1 if float(pchembl_value) > 6.0 else 0
                elif standard_value is not None:
                    # IC50/EC50 < 10 μM 认为有活性
                    try:
                        val = float(standard_value)
                        units = act.get('standard_units', '')
                        if units == 'nM':
                            label = 1 if val < 10000 else 0
                        elif units == 'uM' or units == 'μM':
                            label = 1 if val < 10 else 0
                        else:
                            continue
                    except:
                        continue
                else:
                    continue
                
                data_records.append({
                    'molecule_chembl_id': mol_id,
                    'smiles': smiles,
                    'standard_type': act.get('standard_type'),
                    'standard_value': standard_value,
                    'standard_units': act.get('standard_units'),
                    'pchembl_value': pchembl_value,
                    'label': label,
                    'assay_chembl_id': act.get('assay_chembl_id')
                })
                
                time.sleep(0.1)
                
            except Exception as e:
                continue
        
        df = pd.DataFrame(data_records)
        print(f"\n成功处理 {len(df)} 条数据")
        print(f"活性分子: {(df['label']==1).sum()}, 非活性分子: {(df['label']==0).sum()}")
        
        return df
    
    def create_synthetic_dataset(self, size=3000):
        """
        创建合成数据集（用于演示）
        实际应用中应该使用真实的ChEMBL数据
        """
        print(f"\n创建合成数据集（{size}条记录）...")
        
        # 已知的抗氧化分子SMILES
        antioxidant_smiles = [
            "O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1ccc(O)c(O)c1",  # 槲皮素
            "O=C1CC(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12",  # 柚皮素
            "O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1cc(O)c(O)c(O)c1",  # 杨梅素
            "COc1cc(C2Oc3cc(O)cc(O)c3C(=O)C2O)ccc1O",  # 异鼠李素
            "CC1OC(Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)C(O)C(O)C1O",  # 槲皮苷
            "O=C(O)C=Cc1ccc(O)c(O)c1",  # 咖啡酸
            "O=C(O)C=Cc1ccc(O)cc1",  # 对香豆酸
            "COc1cc(C=CC(=O)O)ccc1O",  # 阿魏酸
            "CC(C)=CCc1c(O)cc(O)c2c1OC(c1ccc(O)cc1)CC2=O",  # 8-异戊烯基柚皮素
        ]
        
        # 生成更多变体
        from rdkit.Chem import AllChem
        from rdkit.Chem import MolStandardize
        
        all_smiles = []
        labels = []
        
        # 生成活性分子（标签=1）
        for base_smiles in antioxidant_smiles:
            mol = Chem.MolFromSmiles(base_smiles)
            if mol:
                # 添加原始分子
                all_smiles.append(Chem.MolToSmiles(mol))
                labels.append(1)
                
                # 生成类似物
                for _ in range(size // (len(antioxidant_smiles) * 3)):
                    try:
                        # 简单修改
                        modified_mol = Chem.RWMol(mol)
                        all_smiles.append(Chem.MolToSmiles(modified_mol))
                        labels.append(1)
                    except:
                        pass
        
        # 从ChEMBL随机获取一些非抗氧化分子（标签=0）
        print("添加负样本...")
        try:
            random_mols = self.molecule.filter(
                max_phase=4  # 已上市药物
            ).only(['molecule_structures'])
            
            count = 0
            for mol_data in random_mols:
                if count >= size // 2:
                    break
                    
                try:
                    if 'molecule_structures' in mol_data:
                        structures = mol_data['molecule_structures']
                        if structures and 'canonical_smiles' in structures:
                            smiles = structures['canonical_smiles']
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                all_smiles.append(smiles)
                                labels.append(0)
                                count += 1
                except:
                    continue
                    
                time.sleep(0.1)
                
        except Exception as e:
            print(f"获取负样本时出错: {e}")
        
        # 创建DataFrame
        df = pd.DataFrame({
            'smiles': all_smiles[:size],
            'label': labels[:size]
        })
        
        # 去重
        df = df.drop_duplicates(subset=['smiles'])
        
        # 添加molecule_chembl_id
        df['molecule_chembl_id'] = [f'MOL_{i:06d}' for i in range(len(df))]
        
        print(f"创建了 {len(df)} 条记录")
        print(f"活性分子: {(df['label']==1).sum()}, 非活性分子: {(df['label']==0).sum()}")
        
        return df
    
    def collect_full_dataset(self):
        """收集完整数据集"""
        print("="*60)
        print("开始收集抗氧化分子数据集")
        print("="*60)
        
        # 方法1: 从ChEMBL获取（如果API可用）
        try:
            # 搜索实验
            assays = self.search_antioxidant_assays()
            
            if len(assays) > 0:
                # 获取活性数据
                activities = self.get_activities_by_assays(assays)
                
                # 处理为数据集
                df = self.process_activities_to_dataset(activities)
                
                if len(df) < 1000:
                    print("\nChEMBL数据不足，补充合成数据...")
                    synthetic_df = self.create_synthetic_dataset(3000 - len(df))
                    df = pd.concat([df, synthetic_df], ignore_index=True)
            else:
                print("\n未找到足够的ChEMBL数据，使用合成数据集...")
                df = self.create_synthetic_dataset(3000)
                
        except Exception as e:
            print(f"\nChEMBL数据收集出错: {e}")
            print("使用合成数据集...")
            df = self.create_synthetic_dataset(3000)
        
        # 保存数据
        output_path = os.path.join(self.output_dir, 'antioxidant_dataset.csv')
        df.to_csv(output_path, index=False)
        print(f"\n数据集已保存到: {output_path}")
        
        # 保存统计信息
        stats = {
            'total_samples': len(df),
            'positive_samples': (df['label']==1).sum(),
            'negative_samples': (df['label']==0).sum(),
            'unique_molecules': df['molecule_chembl_id'].nunique()
        }
        
        with open(os.path.join(self.output_dir, 'dataset_stats.txt'), 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print("\n数据集统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 获取槲皮素数据
        os.makedirs('data/quercetin', exist_ok=True)
        quercetin_mols = self.get_quercetin_related_molecules()
        
        return df

def main():
    """主函数"""
    collector = AntioxidantDataCollector()
    dataset = collector.collect_full_dataset()
    
    print("\n" + "="*60)
    print("数据收集完成!")
    print("="*60)

if __name__ == '__main__':
    main()


