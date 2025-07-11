# 数据处理与计算说明

## 概述
本文档详细说明了nano-gpt项目中的数据处理方法、分片策略以及训练步数的计算方式。

## 数据分片（Shard）机制

### 1. 分片大小设置
在数据准备阶段，我们使用分片来管理大规模数据集：

```python
# 预训练数据集
shard_size = int(1e8)  # 100M tokens per shard

# 微调数据集（如Dolly-15k）
shard_size = int(1e7)  # 10M tokens per shard
```

### 2. 分片策略对比

| 数据集类型 | 分片大小 | 原因 | 效果 |
|-----------|---------|------|------|
| 预训练（edu_fineweb10B） | 100M tokens | 数据量巨大（10B tokens） | 100个分片文件，便于并行处理 |
| 微调（Dolly-15k） | 10M tokens | 数据量较小（2.8M tokens） | 1个分片文件，简化管理 |

### 3. 分片文件命名规则
- 预训练：`edufineweb_train_000.npy`, `edufineweb_train_001.npy`, ...
- 微调：`dolly_000.npy` 或 `dolly_train_000.npy`

### 4. DataLoader中的分片处理
```python
# 训练集使用除第一个外的所有分片
if split == "train" and len(shards) > 1:
    shards = shards[1:]

# 验证集使用第一个分片
elif split == "val" and len(shards) > 0:
    shards = shards[:1]
```

## 训练步数（Steps）计算

### 1. 基本概念
- **Token**: 文本的基本单位（通过tiktoken编码）
- **Batch**: 一次前向传递处理的样本数
- **Sequence Length**: 每个样本的token长度（固定为1024）
- **Gradient Accumulation**: 梯度累积次数

### 2. 每步处理的数据量

#### 预训练配置
```python
batch_size = 16
sequence_length = 1024
total_batch_size = 524288  # 0.5M tokens
grad_accum_steps = total_batch_size // (batch_size * sequence_length)
# grad_accum_steps = 524288 / (16 * 1024) = 32
```

每个step处理：
- 单次前向：16 × 1024 = 16,384 tokens
- 梯度累积32次：16,384 × 32 = 524,288 tokens
- **总计：0.5M tokens/step**

#### 微调配置（相同）
使用相同的配置以保持一致性。

### 3. Epoch与Steps的关系

#### 预训练（edu_fineweb10B）
- 数据集大小：10B tokens
- 每步处理：0.5M tokens
- 1 epoch = 10B / 0.5M = 20,000 steps
- 实际训练：19,073 steps（约0.95 epoch）

#### 微调（Dolly-15k）
- 数据集大小：2.8M tokens
- 每步处理：0.5M tokens
- 1 epoch = 2.8M / 0.5M ≈ 5-6 steps
- 推荐训练：6 steps（1 epoch）

### 4. 步数计算公式
```
steps_per_epoch = total_tokens / (batch_size × sequence_length × grad_accum_steps)
```

## 实际案例分析

### 案例1：Dolly-15k微调

**数据统计**：
```python
# 运行 data/prepare_dolly.py 后的输出
Total tokens: 2,859,954
```

**精确计算**：
- 1 epoch = 2,859,954 / 524,288 ≈ 5.46 steps
- 取整为6 steps确保覆盖所有数据

**训练配置**：
```bash
python model/finetune_dolly.py \
    --max_steps 6 \
    --checkpoint_interval 2 \
    --eval_interval 2
```

### 案例2：调整批次大小

如果GPU内存有限，可以调整批次大小：

**配置1：减小批次大小**
```bash
--batch_size 8  # 梯度累积变为64次
```
- 每步仍处理0.5M tokens
- 训练时间增加（更多梯度累积）
- 内存占用减少

**配置2：增大批次大小**
```bash
--batch_size 32  # 梯度累积变为16次
```
- 每步仍处理0.5M tokens
- 训练时间减少
- 需要更多GPU内存

## 常见问题

### Q1：为什么预训练和微调使用不同的分片大小？
**A**：预训练数据量巨大（10B tokens），使用100M分片可以：
- 避免单个文件过大
- 支持多进程并行读取
- 便于分布式训练

微调数据量小（2.8M tokens），使用10M分片即可满足需求。

### Q2：如何计算训练N个epoch需要多少步？
**A**：使用公式：
```
total_steps = N × (dataset_tokens / tokens_per_step)
```

例如，训练Dolly-15k 2个epoch：
```
total_steps = 2 × (2.8M / 0.5M) = 2 × 5.6 ≈ 12 steps
```

### Q3：验证集如何划分？
**A**：
- 如果有专门的验证集文件（如`dolly_val_*.npy`），直接使用
- 否则，使用第一个分片作为验证集（约10%的数据）
- 对于单分片数据集，训练和验证使用相同数据（注意过拟合）

### Q4：如何估算训练时间？
**A**：基于实际测试，RTX 3090上：
- 每步约需30-40秒
- Dolly-15k 1 epoch（6 steps）：约3-4分钟
- 预训练1 epoch（20k steps）：约7-10天

## 最佳实践

1. **数据准备**
   - 始终检查tokenization后的总token数
   - 确保分片大小合理（不要太大或太小）
   - 保留原始数据用于验证

2. **步数设置**
   - 微调通常1个epoch足够
   - 监控验证集loss避免过拟合
   - 保存多个检查点便于选择最佳模型

3. **内存优化**
   - 调整batch_size而非total_batch_size
   - 使用梯度累积保持有效批次大小
   - 考虑使用混合精度训练（bfloat16）

4. **分布式训练**
   - 确保所有进程看到相同的数据顺序
   - 合理划分数据避免重复
   - 使用DDP时注意同步检查点