# 数据处理指南

## 概述

本文档说明 nano-gpt 的数据处理流程和关键参数计算方法。

## 数据准备

### 运行数据准备脚本

```bash
# 准备预训练数据（FineWeb）
python -m data_prep.fineweb

# 准备微调数据（Dolly-15k）  
python -m data_prep.prepare_dolly
```

## 数据分片机制

### 分片参数

| 数据集 | 分片大小 | 总大小 | 分片数 |
|--------|----------|--------|--------|
| FineWeb（预训练） | 100M tokens | 10B tokens | 100个 |
| Dolly-15k（微调） | 10M tokens | 2.8M tokens | 1个 |

### 文件命名规则

- 预训练：`edufineweb_train_000000.npy`, `edufineweb_train_000001.npy`, ...
- 微调：`dolly_train_000000.npy`, `dolly_val_000000.npy`

## 训练步数计算

### 关键公式

```
每步处理的 tokens = batch_size × seq_length × num_gpus × grad_accumulation_steps
训练步数 = 总 tokens / 每步处理的 tokens
```

### 默认配置下的计算

使用默认参数：
- batch_size = 16
- seq_length = 1024
- grad_accumulation_steps = 32
- 每步处理 tokens = 16 × 1024 × 1 × 32 = 524,288 (0.5M)

### 快速参考

| 数据集 | 总 Tokens | 1 Epoch 步数 | 建议训练 Epochs |
|--------|-----------|--------------|-----------------|
| FineWeb | 10B | ~20,000 | 2-3 |
| Dolly-15k | 2.8M | ~6 | 1-2 |

## 内存优化建议

### GPU 内存不足时的调整策略

```bash
# 方案1：减小批次大小
--batch_size 8  # 内存使用减半，训练时间翻倍

# 方案2：减小序列长度
--seq_length 512  # 适用于较短文本

# 方案3：使用梯度检查点
--gradient_checkpointing  # 用计算换内存
```

### 批次大小与梯度累积的权衡

| batch_size | grad_accumulation | 内存使用 | 训练速度 |
|------------|-------------------|----------|----------|
| 32 | 16 | 高 | 快 |
| 16 | 32 | 中 | 中 |
| 8 | 64 | 低 | 慢 |

## 数据加载优化

### 多进程加载

```bash
# 增加数据加载进程
--num_workers 4  # 默认为0

# 预取数据
--prefetch_factor 2  # 每个worker预取2个批次
```

### 分布式训练的数据分片

分布式训练时，DataLoader 会自动：
1. 将数据分片均匀分配给各个进程
2. 确保每个进程看到不同的数据
3. 在 epoch 结束时同步

## 常见问题

### Q: 如何计算特定配置下的训练时间？

```python
# 示例计算
tokens_per_step = 16 * 1024 * 1 * 32  # 524,288
total_tokens = 2.8e6  # Dolly数据集
steps = total_tokens / tokens_per_step  # ~6步
time_per_step = 30  # 秒（根据GPU）
total_time = steps * time_per_step / 60  # ~3分钟
```

### Q: 如何处理不同长度的文本？

1. **填充策略**：短文本填充到 seq_length
2. **动态批处理**：相似长度的文本分组（未实现）
3. **截断策略**：超长文本截断到 seq_length

### Q: 验证集如何使用？

- 如果存在 `*_val_*.npy` 文件，自动作为验证集
- 否则使用第一个训练分片的 10% 作为验证
- 验证频率由 `--eval_interval` 控制

## 数据格式说明

### 预处理后的数据格式

所有数据保存为 NumPy 数组：
- 数据类型：`np.uint16`（节省空间）
- 形状：`(num_tokens,)`（一维数组）
- 内容：tokenized 后的 token IDs

### 自定义数据集

创建自定义数据集的步骤：

1. 准备文本文件
2. 使用 tiktoken 进行 tokenization
3. 保存为 `.npy` 格式
4. 放入指定目录

示例代码：
```python
import numpy as np
import tiktoken

# 初始化 tokenizer
enc = tiktoken.get_encoding("gpt2")

# 处理文本
tokens = enc.encode("你的文本内容")

# 保存
np.save("custom_train_000000.npy", np.array(tokens, dtype=np.uint16))
```

## 相关文档

- [训练指南](TRAINING.md) - 完整的训练流程
- [架构说明](ARCHITECTURE.md) - 模型架构细节