# nano-gpt 训练指南

本项目的模型训练分为两个核心阶段：**预训练 (Pre-training)** 和 **指令微调 (Instruction Fine-tuning)**。

两个阶段在工程实现上共享了许多通用实践（如使用 `torchrun` 进行分布式训练），但在训练目标、数据加载和损失计算方面存在关键差异。

## 1\. 预训练阶段 (Pre-training)

-   **目标**: 在大规模无标签文本（如 OpenWebText）上训练模型，使其学习通用的语言规律、语法结构和世界知识。
-   **脚本**: `model/train_gpt2.py`

### 执行命令

```bash
# 【单 GPU】使用默认参数开始预训练
python -m model.train_gpt2

# 【多 GPU】使用 torchrun 进行分布式训练 (以 8 卡为例)
torchrun --standalone --nproc_per_node=8 model/train_gpt2.py

# 【恢复训练】自动从最新的检查点恢复，并设置自定义批次大小
python -m model.train_gpt2 --auto_resume --batch_size 8

# 【恢复训练】从指定的检查点文件恢复
python -m model.train_gpt2 --resume --checkpoint_path /path/to/your/checkpoint.pt
```

### 参数说明

| 参数 (Parameter)            | 别名 (Alias) | 默认值 (Default) | 说明 (Description)                                             |
| :-------------------------- | :----------- | :--------------- | :------------------------------------------------------------- |
| **检查点管理**              |              |                  |                                                                |
| `--auto_resume`             |              | `False`          | (开关) 自动从日志目录中最新的检查点恢复训练。                  |
| `--resume`                  |              | `False`          | (开关) 从检查点恢复训练。需要与 `--checkpoint_path` 配合使用。 |
| `--checkpoint_path`         |              | `None`           | 用于恢复训练的检查点文件路径。                                 |
| `--checkpoint_interval`     |              | `5000`           | 每隔多少个训练步数 (steps) 保存一次检查点。                    |
| `--keep_last_n_checkpoints` |              | `-1`             | 保留最近的 N 个检查点。`-1` 表示全部保留。                     |
| **评估与生成**              |              |                  |                                                                |
| `--eval_interval`           |              | `250`            | 每隔 N 步在验证集上评估一次损失 (loss)。                       |
| `--hellaswag_interval`      |              | `250`            | 每隔 N 步在 HellaSwag 数据集上进行一次评估。                   |
| `--generate_interval`       |              | `250`            | 每隔 N 步生成一次示例文本，用于直观地查看模型效果。            |
| **训练超参数**              |              |                  |                                                                |
| `--max_steps`               |              | `19073`          | 最大训练步数 (约等于在 OpenWebText 上训练一个 epoch)。         |
| `--batch_size`              | `-B`         | `16`             | 单个 GPU 上的微批次大小 (Micro batch size)。                   |

---

## 2\. 指令微调阶段 (Instruction Fine-tuning)

-   **目标**: 在小规模、有标签的“指令-回答”数据对上进行微调，教会模型如何理解并遵循用户指令，以生成更有帮助的回答。
-   **脚本**: `model/finetune_dolly.py`

### 前置条件

指令微调必须基于一个已经完成预训练的模型。请确保你拥有一个预训练模型的检查点文件 (checkpoint)。

```bash
# 检查预训练检查点是否存在
ls log/model_*.pt
```

### 执行命令

```bash
# 【单 GPU】从默认的预训练检查点开始微调
python -m model.finetune_dolly

# 【多 GPU】使用 torchrun 进行分布式微调 (以 8 卡为例)
torchrun --standalone --nproc_per_node=8 model/finetune_dolly.py

# 【恢复训练】自动从最新的微调检查点恢复，并使用较小的学习率
python -m model.finetune_dolly --resume auto --lr 1e-5

# 【恢复训练】从指定的微调检查点恢复
python -m model.finetune_dolly --resume --checkpoint_path logs_finetune/model_xxxx.pt
```

### 参数说明

| 参数 (Parameter)            | 别名 (Alias) | 默认值 (Default)     | 说明 (Description)                                                                             |
| :-------------------------- | :----------- | :------------------- | :--------------------------------------------------------------------------------------------- |
| **模型与数据**              |              |                      |                                                                                                |
| `--pretrained_checkpoint`   |              | `log/model_19072.pt` | 指定作为起点的预训练模型检查点路径。                                                           |
| `--data_dir`                |              | `dolly15k`           | 包含指令微调数据集 (如 Dolly-15k) 的目录。                                                     |
| `--log_dir`                 |              | `logs_finetune`      | 用于存放微调日志和检查点的目录。                                                               |
| **检查点管理**              |              |                      |                                                                                                |
| `--resume`                  |              | `'off'`              | 是否从检查点恢复微调。可选值: `'auto'` (自动从最新恢复), `'off'` (不恢复)。                    |
| `--checkpoint_path`         |              | `None`               | 若不使用 `auto` 恢复，可手动指定检查点文件路径 (需同时设置 `--resume`)。                       |
| `--checkpoint_interval`     |              | `50`                 | 每隔 N 步保存一次检查点。                                                                      |
| `--keep_last_n_checkpoints` |              | `3`                  | 保留最近的 N 个检查点。`-1` 表示全部保留。                                                     |
| **评估与生成**              |              |                      |                                                                                                |
| `--eval_interval`           |              | `50`                 | 每隔 N 步在验证集上评估一次损失。                                                              |
| `--generate_interval`       |              | `50`                 | 每隔 N 步生成一次示例文本。                                                                    |
| **训练超参数**              |              |                      |                                                                                                |
| `--max_steps`               |              | `500`                | 最大训练步数。                                                                                 |
| `--batch_size`              | `-B`         | `2`                  | 单个 GPU 上的批次大小。微调时通常使用较小的批次。                                              |
| `--context_length`          | `-T`         | `1024`               | 模型支持的最大上下文长度。                                                                     |
| `--grad_accum_steps`        |              | `5`                  | 梯度累积步数。有效批次大小 (Effective Batch Size) = `batch_size * n_gpus * grad_accum_steps`。 |
| `--lr`                      |              | `3e-5`               | 训练过程中允许的最高学习率。                                                                   |
| `--warmup_steps`            |              | `100`                | 学习率从 0 线性增加到最高值的预热 (warmup) 步数。                                              |

### 关键参数设置建议

在指令微调中，参数的选择对最终效果至关重要，目标是在教会模型新能力的同时，避免其忘记预训练阶段学到的知识（即“灾难性遗忘”）。

1.  **学习率 (Learning Rate)**
    微调时应使用比预训练小得多的学习率（例如 `1e-5` 到 `3e-5`），这能让模型权重在预训练的基础上进行平滑、温和的调整。

2.  **训练步数 (Training Steps)**
    通常只需要在指令数据集上训练 1-3 个 epoch。过多的训练步数很容易导致模型过拟合指令数据，从而损害其泛化能力，引发灾难性遗忘。

3.  **权重衰减 (Weight Decay)**
    使用如 AdamW 优化器时，设置一个合理的权重衰减值（如 `0.01`）可以作为一种正则化手段，防止模型权重在微调时偏离其初始值太远，有助于保留预训练知识。

4.  **批次大小 (Batch Size)**
    微调通常使用比预训练更小的批次大小（如 2-16），这有助于训练过程的稳定。

## 3\. 关键训练技术

### 分布式数据并行 (Distributed Data Parallel, DDP)

两个训练脚本都通过 `torchrun` 原生支持 DDP。在 DDP 模式下，每个 GPU 进程都拥有模型的完整副本，并独立处理一部分数据。在反向传播计算出梯度后，所有进程间的梯度会进行一次 `all-reduce` 同步操作，以确保所有模型副本的权重更新保持一致。这使得训练任务能够水平扩展到多张 GPU，显著缩短训练时间。

### 损失遮罩 (Loss Masking)

这是指令微调区别于预训练的核心优化技术。为了让模型专注于学习“如何根据指令生成回答”，损失函数 **仅在指令的 `### Response:` 部分进行计算**。

实现上，输入给模型的 `targets` 张量中，所有非回答部分的 token（包括指令、上下文等）都被设置为一个特殊的忽略值 `-100`。PyTorch 的 `F.cross_entropy` 损失函数在计算时会自动忽略这些值为 `-100` 的 token。这种方法极大地提升了微调的效率和最终效果。
