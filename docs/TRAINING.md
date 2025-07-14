# 训练指南

本指南涵盖了 nano-gpt 的完整训练流程，包括预训练、微调和断点续训。

## 目录

- [预训练](#预训练)
- [指令微调](#指令微调)
- [断点续训](#断点续训)
- [分布式训练](#分布式训练)
- [参数说明](#参数说明)
- [最佳实践](#最佳实践)

## 预训练

### 基础命令

```bash
# 使用默认参数开始预训练
python -m model.train_gpt2

# 自定义参数
python -m model.train_gpt2 \
    --batch_size 16 \
    --max_lr 3e-4 \
    --max_steps 40000 \
    --checkpoint_interval 1000
```

### 主要参数

| 参数                 | 默认值  | 说明                 |
| -------------------- | ------- | -------------------- |
| `--batch_size`       | 16      | 每个 GPU 的批次大小  |
| `--total_batch_size` | 524,288 | 总批次大小（tokens） |
| `--max_lr`           | 6e-4    | 最大学习率           |
| `--max_steps`        | 40,000  | 总训练步数           |
| `--warmup_steps`     | 2,000   | 预热步数             |

## 指令微调

### 前置条件

微调需要预训练好的模型检查点：

```bash
# 确保有预训练检查点
ls log/model_*.pt
```

### 基础命令

```bash
# 从预训练模型开始微调
python -m model.finetune_dolly --pretrained_checkpoint log/model_40000.pt

# 自定义微调参数
python -m model.finetune_dolly \
    --pretrained_checkpoint log/model_40000.pt \
    --max_lr 3e-5 \
    --max_steps 1750 \
    --batch_size 8 \
    --eval_interval 50
```

### 微调特有参数

| 参数                      | 默认值 | 说明                     |
| ------------------------- | ------ | ------------------------ |
| `--pretrained_checkpoint` | 必需   | 预训练模型路径           |
| `--max_lr`                | 3e-5   | 微调学习率（比预训练低） |
| `--max_steps`             | 1,750  | 微调步数                 |
| `--warmup_steps`          | 100    | 预热步数                 |

## 断点续训

### 自动恢复

最简单的方式是使用自动恢复功能：

```bash
# 预训练自动恢复
python -m model.train_gpt2 --auto_resume

# 微调自动恢复
python -m model.finetune_dolly --auto_resume
```

### 手动指定检查点

```bash
# 从特定检查点恢复预训练
python -m model.train_gpt2 \
    --resume \
    --checkpoint_path log/model_20000.pt

# 从特定检查点恢复微调
python -m model.finetune_dolly \
    --resume \
    --checkpoint_path logs_finetune/model_00500.pt
```

### 检查点管理

```bash
# 控制检查点保存频率
--checkpoint_interval 1000  # 每1000步保存

# 限制保留的检查点数量
--keep_last_n_checkpoints 5  # 只保留最近5个

# 永久保存特定检查点
--permanent_checkpoints 10000,20000,30000
```

## 分布式训练

### 单机多卡

```bash
# 使用所有可用 GPU
torchrun --standalone --nproc_per_node=gpu -m model.train_gpt2

# 指定 GPU 数量
torchrun --standalone --nproc_per_node=4 -m model.train_gpt2
```

### 多机多卡

```bash
# 主节点
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    -m model.train_gpt2

# 工作节点
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    -m model.train_gpt2
```

## 参数说明

### 通用参数

```bash
# 设备相关
--device         # 运行设备 (cuda/cpu)
--compile        # 使用 torch.compile 加速

# 数据相关
--data_root      # 数据目录路径
--num_workers    # 数据加载线程数

# 日志相关
--log_dir        # 日志保存目录
--wandb_log      # 启用 W&B 日志
--wandb_project  # W&B 项目名
```

### 优化参数

```bash
# 学习率调度
--min_lr         # 最小学习率
--decay_lr       # 是否衰减学习率
--weight_decay   # 权重衰减

# 梯度相关
--grad_clip      # 梯度裁剪阈值
--grad_accumulation_steps  # 梯度累积步数
```

## 最佳实践

### 1. 内存优化

GPU 内存不足时的策略：

```bash
# 减小批次大小
--batch_size 8

# 使用梯度累积
--grad_accumulation_steps 8

# 使用混合精度
--dtype bfloat16
```

### 2. 训练监控

```bash
# 频繁保存检查点
--checkpoint_interval 500

# 频繁评估
--eval_interval 100

# 启用详细日志
--log_level debug
```

### 3. 实验管理

```bash
# 使用有意义的运行名称
--run_name "gpt2_lr3e-4_bs16"

# 保存配置
--save_config

# 使用不同的输出目录
--log_dir logs/experiment_001
```

### 4. 调试技巧

```bash
# 快速测试流程
--max_steps 10
--eval_interval 5

# 检查梯度
--log_gradients

# 使用小数据集
--data_root test_data/
```

## 常见问题

### Q: 如何估算训练时间？

基于 RTX 3090 的参考数据：

- 预训练：约 30-40 秒/步
- 微调：约 20-30 秒/步
- 40,000 步预训练：约 14-18 天

### Q: 如何选择学习率？

- 预训练：3e-4 到 6e-4
- 微调：3e-5 到 5e-5
- 继续训练：使用更小的学习率

### Q: 检查点多大？

- 124M 参数模型：约 500MB/检查点
- 包含优化器状态：约 1.5GB/检查点

### Q: 如何处理训练不稳定？

1. 降低学习率
2. 增加预热步数
3. 启用梯度裁剪
4. 检查数据质量

## 进阶技巧

### 学习率调度

```python
# 自定义学习率调度
--lr_scheduler cosine_with_restarts
--lr_restarts 3
```

### 数据增强

```python
# 动态序列长度
--dynamic_seq_length
--min_seq_length 128
--max_seq_length 1024
```

### 模型并行

```python
# 启用模型并行（实验性）
--model_parallel
--model_parallel_size 2
```
