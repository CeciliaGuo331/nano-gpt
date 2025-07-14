# nano-gpt 训练指南

## 预训练

### 基础命令

```bash
# 使用默认参数开始预训练
python -m model.train_gpt2
```

### 参数说明

| 参数 (Parameter)            | 默认值 (Default Value) | 参数说明 (Description)                                  |
| :-------------------------- | :--------------------- | :------------------------------------------------------ |
| **检查点管理**              |                        |                                                         |
| `--resume`                  | `False`                | 一个开关选项。如果使用，则从指定的检查点恢复训练。      |
| `--checkpoint_path`         | `None`                 | 指定要从中恢复训练的检查点文件路径。                    |
| `--auto_resume`             | `False`                | 一个开关选项。如果使用，则自动从最新的检查点恢复。      |
| `--checkpoint_interval`     | `5000`                 | 每隔多少个训练步数（steps）保存一次检查点。             |
| `--keep_last_n_checkpoints` | `-1`                   | 只保留最后 N 个检查点。设置为 `-1` 表示保留所有检查点。 |
| **评估与生成**              |                        |                                                         |
| `--eval_interval`           | `250`                  | 每隔多少步在验证集上评估一次损失（loss）。              |
| `--hellaswag_interval`      | `250`                  | 每隔多少步在 HellaSwag 数据集上进行一次评估。           |
| `--generate_interval`       | `250`                  | 每隔多少步生成一次文本样本，用于直观地查看模型效果。    |

## 指令微调

### 前置条件

微调需要预训练好的模型存档点：

```bash
# 确保有预训练检查点
ls log/model_*.pt
```

### 基础命令

```bash
# 使用默认参数进行微调
python -m model.finetune_dolly

# 自动从最新检查点恢复
python -m model.finetune_dolly --auto_resume

# 从指定检查点恢复
python -m model.finetune_dolly --resume --checkpoint_path logs_finetune/model_00100.pt
```

### 参数说明

| 参数 (Parameter)            | 默认值 (Default Value) | 参数说明 (Description)                                |
| :-------------------------- | :--------------------- | :---------------------------------------------------- |
| **检查点管理**              |                        |                                                       |
| `--resume`                  | `False`                | 一个开关选项。如果使用，则从检查点恢复训练。          |
| `--checkpoint_path`         | `None`                 | 指定要从中恢复训练的检查点文件路径。                  |
| `--auto_resume`             | `False`                | 一个开关选项。如果使用，则自动从最新的检查点恢复。    |
| `--checkpoint_interval`     | `-1`                   | 每隔多少步保存一次检查点。-1 表示不按步数保存。       |
| `--keep_last_n_checkpoints` | `3`                    | 只保留最后 N 个检查点。设置为 -1 表示保留所有。       |
| **评估与生成**              |                        |                                                       |
| `--eval_interval`           | `-1`                   | 每隔多少步在验证集上评估一次损失。-1 表示每步都评估。 |
| `--generate_interval`       | `2`                    | 每隔多少步生成一次文本样本。                          |
| **模型与数据**              |                        |                                                       |
| `--pretrained_checkpoint`   | `log/model_19072.pt`   | 指定预训练模型的检查点文件路径。                      |
| `--data_dir`                | `dolly15k`             | 包含 Dolly-15k 数据集的目录。                         |
| **训练超参数**              |                        |                                                       |
| `--batch_size`              | `16`                   | 训练时的批量大小。                                    |
| `--max_lr`                  | `1e-6`                 | 训练过程中允许的最高学习率。                          |
| `--warmup_steps`            | `1`                    | 学习率预热（warmup）的步数。                          |
| `--max_steps`               | `3`                    | 训练的总步数。                                        |

### 关键参数设置

#### 1. 学习率

```python
max_lr = 1e-6
```

更小的学习率让权重调整更温和。

#### 2. 训练步数

```python
# 只训练1个epoch
max_steps = int(1 * steps_per_epoch)
```

过多训练会导致灾难性遗忘。

#### 3. 权重衰减

```python
# 在AdamW中设置
weight_decay = 0.01
```

防止权重偏离原始值太远，保持预训练知识。

#### 4. 批次大小

```python
# 通常比预训练小
batch_size = 8-32
```

较小的批次使训练更稳定。
