# 微调指南

## 核心原则
微调的目标是让预训练模型适应特定任务，而不是从头学习。关键是**轻微调整**，避免破坏预训练知识。

## 使用方法

### 基本使用
```bash
# 从预训练模型开始微调
python model/finetune_dolly.py

# 查看所有可用参数
python model/finetune_dolly.py --help
```

### 断点续训
```bash
# 自动从最新检查点恢复
python model/finetune_dolly.py --auto_resume

# 从指定检查点恢复
python model/finetune_dolly.py --resume --checkpoint_path logs_finetune/model_00100.pt
```

### 主要命令行参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pretrained_checkpoint` | logs/model_19072.pt | 预训练模型路径 |
| `--data_dir` | dolly15k | 数据集目录 |
| `--batch_size` | 16 | 批次大小 |
| `--max_lr` | 6e-5 | 最大学习率 |
| `--warmup_steps` | 1 | 预热步数 |
| `--max_steps` | 6 | 最大训练步数（约1个epoch） |
| `--checkpoint_interval` | 2 | 检查点保存间隔 |
| `--keep_last_n_checkpoints` | 3 | 保留最近N个检查点 |
| `--eval_interval` | 2 | 验证评估间隔 |
| `--generate_interval` | 2 | 生成样本间隔 |

## 关键参数设置

### 1. 学习率
```python
max_lr = 6e-5  # 预训练的1/10到1/100
```
更小的学习率让权重调整更温和。

### 2. 训练步数
```python
# 只训练1个epoch
max_steps = int(1 * steps_per_epoch)
```
过多训练会导致灾难性遗忘。

### 3. 权重衰减
```python
weight_decay = 0.01  # 在AdamW中设置
```
防止权重偏离原始值太远，保持预训练知识。

### 4. 批次大小
```python
batch_size = 8-32  # 通常比预训练小
```
较小的批次使训练更稳定。

## 监控过拟合

```python
# 验证集loss上升 + 训练集loss下降 = 过拟合
if val_loss > best_val_loss and train_loss < prev_train_loss:
    print("Warning: Overfitting detected")
```

## 数据量分析

### 预训练 vs 微调
- **预训练**: 10B tokens × 1 epoch = 10B tokens
- **微调**: 2.8M tokens (Dolly-15k) = 预训练的0.028%

### 推荐的微调策略
| 策略 | Epochs | 总Tokens | 说明 |
|------|--------|----------|------|
| 保守 | 0.3 | 840K | 最小风险，快速验证 |
| 平衡 | 0.5 | 1.4M | 推荐，效果与风险平衡 |
| 充分 | 1.0 | 2.8M | 完整学习，注意过拟合 |
| 深度 | 2-3 | 5.6-8.4M | 仅在数据质量高时使用 |

## 数据准备
- Dolly-15k等指令数据集：`shard_size = 1e7` (10M tokens)
- 预训练数据集：`shard_size = 1e8` (100M tokens)

## 断点续训功能

### 支持的检查点类型
1. **完整检查点** - 包含模型权重、优化器状态、随机数状态等
2. **仅权重检查点** - 只包含模型权重，适用于从预训练模型开始

### 恢复策略
程序会按以下顺序尝试恢复：
1. 如果指定了 `--auto_resume` 或 `--resume`，首先尝试加载微调检查点
2. 如果微调检查点不存在或加载失败，尝试加载预训练模型
3. 如果都失败，使用随机初始化（会有警告）

### 检查点内容
保存的检查点包含：
- 模型权重 (`model`)
- 优化器状态 (`optimizer`)
- 训练步数 (`step`)
- 验证损失 (`val_loss`)
- 数据加载器状态 (`train_loader_state`)
- 随机数生成器状态 (PyTorch, CUDA, NumPy, Python)

## 快速检查清单
- [ ] 学习率是否足够小？
- [ ] 训练步数是否控制在1个epoch以内？
- [ ] 是否启用权重衰减？
- [ ] 是否定期评估验证集loss？
## 示例用法

### 1. 从头开始微调
```bash
# 使用默认预训练模型
python model/finetune_dolly.py

# 使用自定义预训练模型
python model/finetune_dolly.py --pretrained_checkpoint path/to/pretrained.pt
```

### 2. 调整训练参数
```bash
# 更小的学习率和更多训练步数
python model/finetune_dolly.py --max_lr 3e-5 --max_steps 1750 --warmup_steps 100

# 更大的批次大小（需要更多显存）
python model/finetune_dolly.py --batch_size 32
```

### 3. 断点续训示例
```bash
# 训练中断后，自动恢复
python model/finetune_dolly.py --auto_resume

# 从特定检查点恢复，调整学习率
python model/finetune_dolly.py --resume --checkpoint_path logs_finetune/model_00500.pt --max_lr 1e-5
```

### 4. 分布式训练
```bash
# 单机多卡
torchrun --standalone --nproc_per_node=4 model/finetune_dolly.py

# 多机多卡
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=1234 model/finetune_dolly.py
```

## 常见问题

### 1. 如何选择预训练模型？
- 确保预训练模型的配置与微调脚本中的 `GPTConfig` 匹配
- 可以使用 `train_gpt2.py` 训练的任何检查点

### 2. 内存不足怎么办？
- 减小 `batch_size`
- 增加 `grad_accum_steps`（在代码中调整）
- 使用更小的模型配置

### 3. 如何评估微调效果？
- 观察验证集损失趋势
- 查看生成样本质量
- 比较微调前后的生成结果

### 4. 如何避免过拟合？
- 使用较小的学习率
- 减少训练步数
- 增加权重衰减
- 监控验证集损失