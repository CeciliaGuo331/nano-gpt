# Checkpoint 和断点续训指南

## 1. 实现方法和逻辑

### 1.1 Checkpoint 保存机制

#### 保存逻辑
- **保存时机**：当 `step > 0` 且 `step % checkpoint_interval == 0` 或是最后一步时保存
- **保存位置**：`log/model_{step:05d}.pt` 格式，如 `log/model_00010.pt`
- **原子性保存**：先保存到临时文件 `.tmp`，成功后再重命名，避免文件损坏
- **符号链接**：自动创建/更新 `log/latest_checkpoint.pt` 指向最新的checkpoint

#### 保存内容
```python
checkpoint = {
    "model": model.state_dict(),                    # 模型权重
    "optimizer": optimizer.state_dict(),             # 优化器状态
    "config": model.config,                         # 模型配置
    "step": step,                                   # 当前训练步数
    "val_loss": val_loss,                          # 验证损失
    "train_loader_state": train_loader.get_state(), # 数据加载器状态
    "rng_state": torch.get_rng_state(),            # PyTorch随机数状态
    "cuda_rng_state": torch.cuda.get_rng_state(),  # CUDA随机数状态
    "numpy_rng_state": np.random.get_state(),      # NumPy随机数状态
    "python_rng_state": random.getstate(),         # Python随机数状态
}
```

#### 自动清理机制
- 当设置 `--keep_last_n_checkpoints N` 时，只保留最新的N个checkpoint
- 按文件创建时间排序，删除最旧的文件

### 1.2 断点恢复机制

#### 恢复流程
1. **查找checkpoint**：
   - 优先使用 `log/latest_checkpoint.pt` 符号链接
   - 如果符号链接不存在，按时间戳查找最新的 `model_*.pt` 文件

2. **加载checkpoint**：
   - 使用 `weights_only=False` 加载（因为包含Python对象）
   - 验证必需的键是否存在

3. **恢复状态**：
   - 模型权重和优化器状态
   - 数据加载器位置（包括当前shard和position）
   - 所有随机数生成器状态（确保训练完全可重现）

4. **继续训练**：
   - 从 `checkpoint["step"] + 1` 开始继续训练

#### PyTorch版本兼容性处理
对于RNG状态恢复，代码包含了多种兼容性处理：
```python
# 1. 尝试直接设置
torch.set_rng_state(rng_state)

# 2. 如果失败，尝试ByteTensor转换（旧版PyTorch）
rng_state_bytes = torch.ByteTensor(rng_state.size())
rng_state_bytes.copy_(rng_state)
torch.set_rng_state(rng_state_bytes)

# 3. 最后尝试使用Generator API
gen = torch.Generator()
gen.set_state(rng_state)
```

### 1.3 分布式训练支持

- 只有主进程（rank 0）负责保存checkpoint
- 恢复时，主进程先确定checkpoint路径，然后广播给所有进程
- 确保所有进程从相同的checkpoint恢复

## 2. 命令行参数说明

### Checkpoint相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint_interval` | int | 5000 | 每N步保存一次checkpoint |
| `--keep_last_n_checkpoints` | int | -1 | 只保留最新的N个checkpoint，-1表示保留所有 |
| `--resume` | flag | False | 启用断点续训功能 |
| `--checkpoint_path` | str | None | 指定要恢复的checkpoint路径 |
| `--auto_resume` | flag | False | 自动从最新的checkpoint恢复 |

### 评估相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--eval_interval` | int | 250 | 每N步评估一次验证损失 |
| `--hellaswag_interval` | int | 250 | 每N步进行一次HellaSwag评估 |
| `--generate_interval` | int | 250 | 每N步生成一次文本样本 |

### 使用示例

```bash
# 1. 基础训练（每5000步保存checkpoint）
python -m model.train_gpt2

# 2. 频繁保存checkpoint（用于测试）
python -m model.train_gpt2 --checkpoint_interval 100 --eval_interval 50

# 3. 自动恢复训练
python -m model.train_gpt2 --auto_resume

# 4. 从指定checkpoint恢复
python -m model.train_gpt2 --resume --checkpoint_path log/model_00100.pt

# 5. 限制checkpoint数量
python -m model.train_gpt2 --checkpoint_interval 1000 --keep_last_n_checkpoints 5

# 6. 完整的测试配置
python -m model.train_gpt2 \
    --auto_resume \
    --checkpoint_interval 10 \
    --keep_last_n_checkpoints 3 \
    --eval_interval 10 \
    --hellaswag_interval 50 \
    --generate_interval 20
```

## 3. 可移除的冗余逻辑

在代码审查时，以下部分可以考虑简化或移除：

### 3.1 RNG状态恢复的多重fallback（第652-686行）

当前代码有三层fallback尝试恢复RNG状态。在确定PyTorch版本后，可以只保留适用的方法：

```python
# 可以简化为单一方法（根据实际PyTorch版本）
if isinstance(rng_state, torch.Tensor):
    rng_state = rng_state.cpu().contiguous()
    if rng_state.dtype != torch.uint8:
        rng_state = rng_state.to(torch.uint8)
    torch.set_rng_state(rng_state)
```

### 3.2 符号链接创建的错误处理（第568-576行）

符号链接创建失败的处理可以简化，因为这不是关键功能：

```python
# 当前有try-except处理OSError
# 可以简化为只在成功时打印消息
```

### 3.3 checkpoint保存的双重val_loss处理（第747-751行）

当前代码尝试使用验证损失，如果不存在则使用训练损失。可以统一处理：

```python
# 可以在训练循环中维护last_val_loss变量
# 避免使用'val_loss_accum' in locals()这种检查
```

### 3.4 DataLoader状态验证（第322-339行）

DataLoader的load_state中有大量的边界检查，在生产环境中可以简化：

```python
# 可以添加一个strict参数
# strict=False时跳过详细验证
```

### 3.5 调试相关代码

确保移除所有调试打印语句：
- RNG状态类型的Debug打印
- 各种Warning打印（除非是真正的警告）

## 4. 最佳实践建议

1. **生产环境**：
   - 设置合理的 `checkpoint_interval`（如1000-5000步）
   - 使用 `--keep_last_n_checkpoints 3-5` 节省磁盘空间
   - 始终使用 `--auto_resume` 确保训练的连续性

2. **测试环境**：
   - 使用较小的interval值快速测试功能
   - 可以设置所有interval为相同值便于观察

3. **长时间训练**：
   - 考虑将checkpoint保存到持久化存储
   - 定期备份重要的checkpoint
   - 监控磁盘空间使用情况

4. **问题排查**：
   - 如果恢复失败，检查checkpoint文件完整性
   - 确保PyTorch版本一致性
   - 查看log文件中的错误信息