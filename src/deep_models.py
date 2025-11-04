"""
深度学习模型模块
包括超越BERT的分子预训练模型：ChemBERTa, MolFormer, Graph Neural Networks等
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    Trainer, TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pickle
import os
from tqdm import tqdm
from typing import List, Dict, Tuple

class SMILESDataset(Dataset):
    """SMILES数据集"""
    
    def __init__(self, smiles_list: List[str], labels: List[int], tokenizer, max_length=128):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = str(self.smiles_list[idx])
        label = int(self.labels[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            smiles,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ChemBERTaModel:
    """ChemBERTa模型 - 基于RoBERTa的化学分子预训练模型"""
    
    def __init__(self, model_name='DeepChem/ChemBERTa-77M-MLM', device=None):
        """
        Args:
            model_name: 预训练模型名称
                - 'DeepChem/ChemBERTa-77M-MLM' (77M参数)
                - 'DeepChem/ChemBERTa-77M-MTR' (多任务)
                - 'seyonec/PubChem10M_SMILES_BPE_450k' (PubChem预训练)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"初始化 ChemBERTa 模型: {model_name}")
        print(f"使用设备: {self.device}")
        
        try:
            # 加载tokenizer和模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            self.model.to(self.device)
            
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("使用基础RoBERTa模型作为备选...")
            
            # 备选方案：使用基础RoBERTa
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaForSequenceClassification.from_pretrained(
                'roberta-base',
                num_labels=2
            )
            self.model.to(self.device)
    
    def train(self, train_smiles, train_labels, val_smiles, val_labels, 
              output_dir='models/chemberta', epochs=10, batch_size=16):
        """训练模型"""
        print("\n开始训练 ChemBERTa 模型...")
        
        # 创建数据集
        train_dataset = SMILESDataset(train_smiles, train_labels, self.tokenizer)
        val_dataset = SMILESDataset(val_smiles, val_labels, self.tokenizer)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
        )
        
        # 定义评估指标
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='binary')
            precision = precision_score(labels, preds, average='binary')
            recall = recall_score(labels, preds, average='binary')
            
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # 训练
        trainer.train()
        
        # 保存模型
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"模型已保存到: {output_dir}")
        
        return trainer
    
    def predict(self, smiles_list, batch_size=32):
        """预测"""
        self.model.eval()
        
        predictions = []
        probabilities = []
        
        # 创建数据加载器
        dummy_labels = [0] * len(smiles_list)
        dataset = SMILESDataset(smiles_list, dummy_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="预测中"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # 获取预测类别
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                
                # 获取概率
                probs = F.softmax(logits, dim=-1)
                probabilities.extend(probs[:, 1].cpu().numpy())  # 正类概率
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate(self, test_smiles, test_labels, batch_size=32):
        """评估模型"""
        predictions, probabilities = self.predict(test_smiles, batch_size)
        
        metrics = {
            'accuracy': accuracy_score(test_labels, predictions),
            'f1_score': f1_score(test_labels, predictions),
            'precision': precision_score(test_labels, predictions),
            'recall': recall_score(test_labels, predictions),
            'roc_auc': roc_auc_score(test_labels, probabilities)
        }
        
        return metrics

class MolFormerModel:
    """MolFormer模型 - 专门为分子设计的Transformer"""
    
    def __init__(self, model_name='ibm/MolFormer-XL-both-10pct', device=None):
        """
        Args:
            model_name: 预训练模型名称
                - 'ibm/MolFormer-XL-both-10pct'
                - 'ibm/MolFormer-Large-both-10pct'
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"初始化 MolFormer 模型: {model_name}")
        print(f"使用设备: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
            self.model.to(self.device)
            
        except Exception as e:
            print(f"加载 MolFormer 失败: {e}")
            print("使用 ChemBERTa 作为备选...")
            
            # 备选方案
            self.tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'DeepChem/ChemBERTa-77M-MLM',
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            self.model.to(self.device)
    
    def train(self, train_smiles, train_labels, val_smiles, val_labels,
              output_dir='models/molformer', epochs=10, batch_size=16):
        """训练模型 - 与ChemBERTa类似"""
        # 复用ChemBERTa的训练逻辑
        chemberta = ChemBERTaModel.__new__(ChemBERTaModel)
        chemberta.model = self.model
        chemberta.tokenizer = self.tokenizer
        chemberta.device = self.device
        
        return chemberta.train(train_smiles, train_labels, val_smiles, val_labels,
                              output_dir, epochs, batch_size)
    
    def predict(self, smiles_list, batch_size=32):
        """预测"""
        chemberta = ChemBERTaModel.__new__(ChemBERTaModel)
        chemberta.model = self.model
        chemberta.tokenizer = self.tokenizer
        chemberta.device = self.device
        
        return chemberta.predict(smiles_list, batch_size)
    
    def evaluate(self, test_smiles, test_labels, batch_size=32):
        """评估"""
        chemberta = ChemBERTaModel.__new__(ChemBERTaModel)
        chemberta.model = self.model
        chemberta.tokenizer = self.tokenizer
        chemberta.device = self.device
        
        return chemberta.evaluate(test_smiles, test_labels, batch_size)

class SimpleMolecularCNN(nn.Module):
    """简单的分子卷积神经网络（用于对比）"""
    
    def __init__(self, input_dim, hidden_dim=256, num_classes=2):
        super(SimpleMolecularCNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(hidden_dim // 4, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

class DeepNNClassifier:
    """深度神经网络分类器（用于传统特征）"""
    
    def __init__(self, input_dim, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleMolecularCNN(input_dim).to(self.device)
        
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, lr=0.001):
        """训练模型"""
        print("\n训练深度神经网络...")
        
        # 转换为tensor
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            
            # 批次训练
            total_loss = 0
            num_batches = len(X_train) // batch_size + 1
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = (val_preds == y_val_tensor).float().mean()
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {total_loss/num_batches:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
        
        # 加载最佳模型
        self.model.load_state_dict(self.best_model_state)
        print("训练完成!")
    
    def predict(self, X):
        """预测"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
        
        return preds.cpu().numpy(), probs[:, 1].cpu().numpy()
    
    def evaluate(self, X_test, y_test):
        """评估"""
        predictions, probabilities = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'roc_auc': roc_auc_score(y_test, probabilities)
        }
        
        return metrics
    
    def save(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

def main():
    """主函数 - 训练深度学习模型"""
    print("="*60)
    print("开始训练深度学习模型")
    print("="*60)
    
    # 加载数据
    print("\n加载数据...")
    
    # 方案1: 使用SMILES训练Transformer模型
    with open('data/processed/smiles.pkl', 'rb') as f:
        smiles_list = pickle.load(f)
    labels = np.load('data/processed/labels.npy')
    
    # 划分数据集
    train_smiles, test_smiles, train_labels, test_labels = train_test_split(
        smiles_list, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_smiles, val_smiles, train_labels, val_labels = train_test_split(
        train_smiles, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )
    
    print(f"训练集: {len(train_smiles)} 样本")
    print(f"验证集: {len(val_smiles)} 样本")
    print(f"测试集: {len(test_smiles)} 样本")
    
    results = {}
    
    # 1. ChemBERTa模型
    try:
        print("\n" + "="*60)
        print("训练 ChemBERTa 模型")
        print("="*60)
        
        chemberta = ChemBERTaModel()
        chemberta.train(
            train_smiles, train_labels,
            val_smiles, val_labels,
            epochs=5,  # 减少epoch用于演示
            batch_size=16
        )
        
        chemberta_metrics = chemberta.evaluate(test_smiles, test_labels)
        results['ChemBERTa'] = chemberta_metrics
        
        print("\nChemBERTa 测试集性能:")
        for metric, value in chemberta_metrics.items():
            print(f"  {metric}: {value:.4f}")
            
    except Exception as e:
        print(f"ChemBERTa训练出错: {e}")
    
    # 2. 深度神经网络（使用传统特征）
    try:
        print("\n" + "="*60)
        print("训练深度神经网络 (使用分子指纹)")
        print("="*60)
        
        X = np.load('data/processed/features.npy')
        y = np.load('data/processed/labels.npy')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        
        dnn = DeepNNClassifier(input_dim=X_train.shape[1])
        dnn.train(X_train, y_train, X_val, y_val, epochs=30)
        
        dnn_metrics = dnn.evaluate(X_test, y_test)
        results['Deep NN'] = dnn_metrics
        
        print("\nDeep NN 测试集性能:")
        for metric, value in dnn_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 保存模型
        dnn.save('models/deep_nn.pth')
        
    except Exception as e:
        print(f"深度神经网络训练出错: {e}")
    
    # 保存结果
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results/deep_learning_results.csv')
    
    print("\n" + "="*60)
    print("深度学习模型训练完成!")
    print("="*60)
    print("\n模型性能对比:")
    print(results_df.to_string())

if __name__ == '__main__':
    main()

