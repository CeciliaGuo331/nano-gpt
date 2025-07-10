# train_gpt2.py 断点续训功能使用指南

## 功能概述

已为 `train_gpt2.py` 添加了完整的断点续训功能，支持在训练中断后从最近的检查点恢复训练。该功能保存了所有必要的训练状态，确保训练可以无缝继续。

## 新增的命令行参数

1. `--resume` - 从检查点恢复训练
2. `--checkpoint_path` - 指定要加载的检查点路径
3. `--auto_resume` - 自动从最新的检查点恢复
4. `--checkpoint_interval` - 设置保存检查点的间隔（默认5000步）
5. `--keep_last_n_checkpoints` - 保留最新的N个检查点（默认5个）

## 使用示例

### 1. 正常开始训练
```bash
python train_gpt2.py
```

### 2. 自动从最新检查点恢复
```bash
python train_gpt2.py --auto_resume
```

### 3. 从指定检查点恢复
```bash
python train_gpt2.py --resume --checkpoint_path log/model_10000.pt
```

### 4. 修改检查点保存间隔
```bash
python train_gpt2.py --checkpoint_interval 1000
```

### 5. DDP分布式训练恢复
```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py --auto_resume
```

## 保存的状态信息

每个检查点包含以下信息：
- 模型参数 (`model.state_dict()`)
- 优化器状态 (`optimizer.state_dict()`)
- 模型配置 (`config`)
- 当前训练步数 (`step`)
- 验证损失 (`val_loss`)
- 数据加载器状态（当前shard和位置）
- 所有随机数生成器状态（PyTorch、CUDA、NumPy、Python）

## 特性说明

1. **自动保存最新检查点链接**：系统会创建 `log/latest_checkpoint.pt` 软链接，始终指向最新的检查点。

2. **自动清理旧检查点**：使用 `--keep_last_n_checkpoints` 参数可以自动删除旧的检查点，节省磁盘空间。

3. **日志文件处理**：续训时不会清空现有的日志文件，新的训练日志会追加到原有日志后面。

4. **DDP同步**：在分布式训练中，所有进程会同步加载相同的检查点。

5. **完整状态恢复**：不仅恢复模型和优化器，还恢复数据加载器位置和随机种子，确保训练的可重现性。

## 注意事项

1. 检查点文件较大，建议定期清理或使用 `--keep_last_n_checkpoints` 参数。
2. 恢复训练时，确保数据文件没有变化，否则可能导致数据加载错误。
3. 如果修改了模型结构，旧的检查点将无法加载。

## 示例工作流

```bash
# 开始新训练，每1000步保存检查点，只保留最新的3个
python train_gpt2.py --checkpoint_interval 1000 --keep_last_n_checkpoints 3

# 训练中断后，自动从最新检查点恢复
python train_gpt2.py --auto_resume

# 查看保存的检查点
ls -la log/model_*.pt

# 查看最新检查点链接
ls -la log/latest_checkpoint.pt
```