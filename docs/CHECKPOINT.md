# Checkpoint 与断点续训指南

本项目实现了一套稳健的检查点 (Checkpoint) 系统，以支持灵活的权重加载和从意外中断处无缝恢复训练（断点续训）。

## 1\. 核心概念：两种检查点加载模式

理解两种主要的加载模式至关重要，它们服务于不同的训练目标：

### 模式一：恢复完整训练 (`--resume` 或 `--auto_resume`)

-   **目的**：从一次被中断的训练中无缝恢复，确保训练过程在数学上完全等价于未中断的原始训练。
-   **行为**：加载检查点中的 **所有状态**，包括模型权重、优化器状态（如动量）、数据加载器的读取位置、当前的训练步数以及所有相关的随机数生成器状态。
-   **使用场景**：
    -   `--auto_resume`: 自动查找并从最新的检查点恢复训练。
    -   `--resume --checkpoint_path <path>`: 从一个手动指定的检查点文件恢复。

### 模式二：仅加载预训练权重 (`--pretrained_checkpoint`)

-   **目的**：开始一次 **全新的训练**（例如指令微调），但使用一个已经训练好的模型权重作为起点。
-   **行为**：**只加载** 检查点中的模型权重 (`model` state_dict)，并 **忽略** 其他所有状态（如优化器、步数、学习率等）。优化器、学习率调度器等都将重新初始化。
-   **使用场景**：在微调脚本 (`finetune_dolly.py`) 中，此为默认行为。脚本会从 `--pretrained_checkpoint` 指定的路径加载权重，然后开始一次全新的微调任务。

## 2\. 检查点实现机制

### 2.1 检查点内容

一个完整的检查点 (`.pt` 文件) 是一个包含了恢复训练所需全部信息的 Python 字典。

```python
checkpoint = {
    "model": model.state_dict(),                    # 模型权重
    "optimizer": optimizer.state_dict(),             # 优化器状态
    "config": model.config,                         # 模型配置
    "step": step,                                   # 当前训练步数
    "val_loss": val_loss,                          # 验证集损失
    "train_loader_state": train_loader.get_state(), # 数据加载器状态
    "rng_state": torch.get_rng_state(),            # PyTorch 随机数状态
    "cuda_rng_state": torch.cuda.get_rng_state(),  # CUDA 随机数状态
    "numpy_rng_state": np.random.get_state(),      # NumPy 随机数状态
    "python_rng_state": random.getstate(),         # Python 随机数状态
}
```

### 2.2 保存机制

-   **保存时机**：当训练步数 `step > 0` 且 `step % checkpoint_interval == 0` 时，或在训练的最后一步，会自动保存检查点。
-   **保存位置**：文件以 `log/model_{step:05d}.pt` 格式命名，例如 `log/model_00500.pt`。
-   **原子操作**：为防止文件损坏，保存过程是原子化的。脚本会先将检查点写入临时文件 (`.tmp`)，写入成功后再将其重命名为最终文件名。
-   **最新检查点链接**：每次成功保存后，都会创建或更新一个名为 `latest_checkpoint.pt` 的符号链接，使其始终指向最新生成的检查点文件，这为 `--auto_resume` 提供了便利。
-   **自动清理**：当使用 `--keep_last_n_checkpoints N` 参数时，系统会自动删除旧的检查点，只保留最近的 N 个，以节省磁盘空间。

### 2.3 恢复机制

-   **查找检查点**：当使用 `--auto_resume` 时，脚本会优先通过 `log/latest_checkpoint.pt` 符号链接查找，如果链接不存在，则会按时间戳查找最新的 `model_*.pt` 文件。
-   **恢复状态**：加载检查点后，脚本会恢复所有相关状态，包括模型权重、优化器、数据加载器位置以及所有随机数生成器状态，以确保训练的可复现性。
-   **PyTorch 版本兼容性**：在恢复随机数生成器（RNG）状态时，代码包含了对旧版 PyTorch 的兼容性处理，会尝试多种方法加载状态，确保在不同版本间的兼容性。
-   **继续训练**：状态恢复后，训练将从 `checkpoint["step"] + 1` 步开始继续。

### 2.4 分布式训练支持

在分布式数据并行 (DDP) 训练模式下：

-   只有主进程 (`rank 0`) 负责保存检查点文件。
-   恢复时，主进程首先确定并加载检查点，然后将路径广播给所有其他进程，确保所有 GPU 进程从同一个检查点恢复，保持状态一致。

## 3\. 命令行参数与示例

### Checkpoint 相关参数

| 参数                        | 类型 | 默认值  | 说明                                                  |
| :-------------------------- | :--- | :------ | :---------------------------------------------------- |
| `--auto_resume`             | flag | `False` | 自动从最新的 checkpoint 恢复。                        |
| `--resume`                  | flag | `False` | 启用断点续训功能，需与 `--checkpoint_path` 配合使用。 |
| `--checkpoint_path`         | str  | `None`  | 指定要恢复的 checkpoint 文件路径。                    |
| `--checkpoint_interval`     | int  | `5000`  | 每 N 步保存一次 checkpoint。                          |
| `--keep_last_n_checkpoints` | int  | `-1`    | 只保留最新的 N 个 checkpoint，-1 表示保留所有。       |

### 评估相关参数

| 参数                   | 类型 | 默认值 | 说明                             |
| :--------------------- | :--- | :----- | :------------------------------- |
| `--eval_interval`      | int  | `250`  | 每 N 步评估一次验证损失。        |
| `--hellaswag_interval` | int  | `250`  | 每 N 步进行一次 HellaSwag 评估。 |
| `--generate_interval`  | int  | `250`  | 每 N 步生成一次文本样本。        |

### 使用示例

```bash
# 1. 基础训练（每5000步保存一次）
python -m model.train_gpt2

# 2. 自动从上一次中断的地方恢复训练
python -m model.train_gpt2 --auto_resume

# 3. 从一个指定的 checkpoint 文件恢复
python -m model.train_gpt2 --resume --checkpoint_path log/model_00100.pt

# 4. 频繁保存和评估，并只保留最近5个 checkpoint
python -m model.train_gpt2 --checkpoint_interval 100 --eval_interval 50 --keep_last_n_checkpoints 5

# 5. 完整的自动化测试配置
python -m model.train_gpt2 \
    --auto_resume \
    --checkpoint_interval 10 \
    --keep_last_n_checkpoints 3 \
    --eval_interval 10 \
    --hellaswag_interval 50 \
    --generate_interval 20
```

## 4\. 最佳实践建议

1.  **生产环境**:
    -   设置合理的 `checkpoint_interval`（如 1000-5000 步）以平衡性能和安全性。
    -   设置较小的 `--keep_last_n_checkpoints`（如 3-5）来节省磁盘空间。
    -   开启 `--auto_resume` 选项以进行断点续训。
2.  **测试与开发环境**:
    -   使用较小的 `interval` 值（如 `10` 或 `50`）来快速验证检查点功能的正确性。
3.  **长时间训练**:
    -   考虑将 `log` 目录挂载到持久化网络存储上，防止本地磁盘故障。
    -   定期手动备份重要的检查点文件到安全位置。
    -   监控磁盘空间使用情况。
4.  **问题排查**:
    -   如果恢复失败，首先检查 checkpoint 文件是否完整（大小是否正常）。
    -   确认训练环境的 PyTorch 版本与保存 checkpoint 时的版本没有重大不兼容。
    -   仔细查看日志文件中的错误信息，定位问题源头。
