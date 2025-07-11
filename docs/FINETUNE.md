# 微调指南

## 核心原则
微调的目标是让预训练模型适应特定任务，而不是从头学习。关键是**轻微调整**，避免破坏预训练知识。

## 关键参数设置

### 1. 学习率
```python
max_lr = 6e-5  # 预训练的1/10到1/100
```
更小的学习率让权重调整更温和。

### 2. 训练步数
```python
# 只训练0.2-0.5个epoch
max_steps = int(0.5 * steps_per_epoch)
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

## 快速检查清单
- [ ] 学习率是否足够小？
- [ ] 训练步数是否控制在1个epoch以内？
- [ ] 是否启用权重衰减？
- [ ] 是否定期评估验证集loss？
- [ ] 数据分片大小是否合适？