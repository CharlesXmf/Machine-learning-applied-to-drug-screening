# BERT模型训练问题诊断和解决方案

## 快速诊断

运行诊断脚本：
```bash
python diagnose_bert_issue.py
```

## 常见问题及解决方案

### 1. 网络连接问题（最常见）

**症状：**
- 错误信息包含 "Connection", "timeout", "network"
- 无法从Hugging Face下载模型

**解决方案：**
```python
# 方法1: 使用离线模式（如果之前下载过）
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', local_files_only=True)

# 方法2: 手动下载模型
# 访问 https://huggingface.co/models 搜索并下载模型

# 方法3: 使用代理
import os
os.environ['HTTP_PROXY'] = 'http://your-proxy:port'
os.environ['HTTPS_PROXY'] = 'http://your-proxy:port'
```

### 2. GPU内存不足

**症状：**
- 错误信息包含 "out of memory", "CUDA"
- 训练过程中程序崩溃

**解决方案：**
```python
# 在代码中修改batch_size
chemberta.train(
    train['smiles'].tolist(),
    train['label'].tolist(),
    val['smiles'].tolist(),
    val['label'].tolist(),
    epochs=3,
    batch_size=4  # 从16改为4或更小
)

# 或者强制使用CPU
chemberta = ChemBERTaModel()
chemberta.device = 'cpu'  # 强制使用CPU
```

### 3. 模型不存在或名称错误

**症状：**
- 错误信息包含 "not found", "404"
- 模型名称错误

**解决方案：**
```python
# 检查模型是否存在
# 访问 https://huggingface.co/DeepChem/ChemBERTa-77M-MLM

# 使用备选模型（代码中已实现）
# 会自动尝试：
# 1. DeepChem/ChemBERTa-77M-MLM
# 2. seyonec/PubChem10M_SMILES_BPE_450k
# 3. bert-base-uncased
```

### 4. 依赖包版本问题

**症状：**
- ImportError
- 版本不兼容错误

**解决方案：**
```bash
# 更新transformers
pip install --upgrade transformers torch

# 检查版本
python -c "import transformers; print(transformers.__version__)"
python -c "import torch; print(torch.__version__)"
```

### 5. 数据格式问题

**症状：**
- 错误信息包含 "shape", "size"
- 数据形状不匹配

**解决方案：**
```python
# 检查数据
print(f"训练数据量: {len(train_smiles)}")
print(f"标签类别: {set(train_labels)}")

# 确保标签是0和1
assert set(train_labels).issubset({0, 1}), "标签必须是0或1"
```

### 6. Windows特定问题

**症状：**
- dataloader错误
- 多进程问题

**解决方案：**
```python
# 在TrainingArguments中添加
training_args = TrainingArguments(
    ...
    dataloader_num_workers=0,  # Windows上设为0
)
```

## 逐步排查步骤

### 步骤1: 检查依赖
```bash
pip list | grep -E "torch|transformers"
```

### 步骤2: 测试模型加载
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print("✓ Tokenizer加载成功")
```

### 步骤3: 测试小数据集
```python
# 只用前10条数据测试
small_train = train.head(10)
chemberta.train(
    small_train['smiles'].tolist(),
    small_train['label'].tolist(),
    val.head(5)['smiles'].tolist(),
    val.head(5)['label'].tolist(),
    epochs=1,
    batch_size=2
)
```

### 步骤4: 检查日志
查看 `models/chemberta/logs/` 目录下的日志文件

## 修改后的代码改进

代码已更新，包含：
1. ✅ 更详细的错误信息
2. ✅ 数据验证
3. ✅ 常见问题诊断
4. ✅ Windows兼容性改进
5. ✅ 内存不足检测

## 如果仍然失败

1. **使用基础BERT模型**（代码会自动fallback）
   - 如果ChemBERTa失败，会自动尝试PubChem BERT
   - 如果还是失败，会使用bert-base-uncased

2. **跳过BERT训练**（仅使用传统模型）
   - 代码会继续运行，只使用传统机器学习模型
   - 结果仍然有效

3. **查看完整错误日志**
   - 运行脚本时会打印完整的traceback
   - 根据具体错误信息查找解决方案

## 联系支持

如果问题仍然存在，请提供：
1. 完整的错误信息（traceback）
2. Python版本：`python --version`
3. PyTorch版本：`python -c "import torch; print(torch.__version__)"`
4. Transformers版本：`python -c "import transformers; print(transformers.__version__)"`
5. 操作系统信息


