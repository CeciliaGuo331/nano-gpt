# nano-gpt 数据处理综合指南

高质量的数据是成功训练语言模型的基础。本文档将详细说明 `nano-gpt` 项目中涉及的数据集处理流程、关键参数计算方法、优化策略及常见问题。

## 1\. 数据准备流程

本项目支持多种数据集，每种都通过专门的脚本进行处理，以生成训练所需的、经过 Tokenization 的分片文件 (`.npy`)。

### 1.1 预训练数据: FineWeb-Edu

-   **来源**: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)，一个经过高质量过滤和去重的教育内容网络文本数据集。
-   **处理脚本**: `data_prep/fineweb.py`
-   **执行命令**:
    ```bash
    python -m data_prep.fineweb
    ```
-   **说明**: 该脚本没有可配置的命令行参数，它会默认下载 `sample-10BT` 数据集，并将其处理后存入 `edu_fineweb10B` 目录。
-   **处理流程**:
    1.  **下载**: 使用 `datasets` 库下载数据集。
    2.  **Tokenization**: 使用 `tiktoken` 的 `gpt2` 编码器进行并行化 Tokenization。每个文档前都添加 `<|endoftext|>` (EOT) 特殊 Token 作为分隔符。
    3.  **分片 (Sharding)**: 将 Tokenized 数据流切分为大小为 1 亿 Token 的分片。第一个分片被指定为验证集 (`val`)，其余为训练集 (`train`)。

### 1.2 指令微调数据: Dolly-15k

-   **来源**: [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)，一个由 Databricks 员工生成的高质量人工指令数据集。
-   **处理脚本**: `data_prep/prepare_dolly.py`
-   **执行命令**:

    ```bash
    # 使用默认参数
    python -m data_prep.prepare_dolly

    # 自定义输出目录和验证集比例
    python -m data_prep.prepare_dolly --output_dir my_dolly_shards --val_split_ratio 0.05
    ```

-   **命令行参数**:
    -   `--output_dir` (str): 保存处理后数据分片的目录。**默认值**: `dolly15k`。
    -   `--val_split_ratio` (float): 用于验证集的比例。**默认值**: `0.1` (10%)。
    -   `--shard_size` (int): 每个分片文件的 Token 数量。**默认值**: `10,000,000`。
-   **处理流程**:
    1.  **格式化**: 每个样本被统一格式化为包含 `### Instruction:`，`### Context:` (如果存在) 和 `### Response:` 的结构化字符串。
    2.  **Tokenization**: 使用 `gpt2` 编码器处理格式化后的字符串，并在每个样本的末尾添加 EOT Token。
    3.  **切分与保存**: 在 Tokenization 之前，整个数据集按比例随机切分为训练集和验证集。然后，每个子集被独立地 Tokenized 并保存为分片文件。

### 1.3 自定义指令数据

-   **来源**: 用户提供的 `.jsonl` 文件，每行包含 `instruction`, `context` (可选), `response` 字段。
-   **处理脚本**: `data_prep/prepare_custom_dataset.py`
-   **执行命令**:
    ```bash
    python -m data_prep.prepare_custom_dataset \
      --input_file <path_to_your.jsonl> \
      --output_dir <your_output_directory> \
      --val_split_ratio 0.05
    ```
-   **命令行参数**:
    -   `--input_file` (str): **(必需)** 输入的 `.jsonl` 数据集文件路径。
    -   `--output_dir` (str): **(必需)** 处理后 `.npy` 分片的输出目录。
    -   `--val_split_ratio` (float): 从数据集中划分为验证集的比例。**默认值**: `0.1` (10%)。
    -   `--seed` (int): 用于随机打乱和切分的种子。**默认值**: `1337`。
-   **处理流程**: 该脚本提供了一个通用框架，用于处理任何符合特定格式的自定义指令数据集。它执行与 `prepare_dolly.py` 类似的步骤：读取、随机打乱、按比例切分、格式化、Tokenization 并保存。

## 2\. 数据格式与分片机制

### 2.1 预处理后的数据格式

所有预处理后的数据都以优化的格式保存为 NumPy 数组：

-   **数据类型**: `np.uint16` (相比 `int32` 或 `int64` 极大节省磁盘空间)。
-   **形状**: `(num_tokens,)` (所有 Token ID 组成的一维数组)。
-   **内容**: 经过 Tokenizer 处理后的 Token ID 序列。

### 2.2 数据分片机制

为高效处理大规模数据集，数据被切分为多个分片。

-   **分片参数**:
    | 数据集 | 分片大小 | 总大小 | 分片数 |
    | :--- | :--- | :--- | :--- |
    | FineWeb (预训练) | 100M tokens | 10B tokens | 100 个 |
    | Dolly-15k (微调) | 10M tokens | 2.8M tokens | 1 个 |
-   **文件命名规则**:
    -   预训练: `edufineweb_train_000000.npy`, `edufineweb_train_000001.npy`, ...
    -   微调: `dolly_train_000000.npy`, `dolly_val_000000.npy`

## 3\. 训练步数计算

### 3.1 关键公式

理解训练步数的计算对于规划训练至关重要。

```
每步有效处理的 tokens = batch_size × seq_length × num_gpus × grad_accumulation_steps
训练步数 (Steps per Epoch) = 总 tokens / 每步有效处理的 tokens
```

### 3.2 默认配置下的计算

使用 `train.py` 的默认参数：

-   `batch_size` = 16
-   `seq_length` = 1024
-   `grad_accumulation_steps` = 32
-   每步处理 tokens = 16 × 1024 × 1 × 32 = 524,288 (约 0.5M tokens)

### 3.3 快速参考

根据上述计算，可以估算不同数据集每个 Epoch 所需的步数：
| 数据集 | 总 Tokens | 1 Epoch 步数 | 建议训练 Epochs |
| :--- | :--- | :--- | :--- |
| FineWeb | 10B | \~20,000 | 2-3 |
| Dolly-15k | 2.8M | \~6 | 1-2 |

## 4\. 优化策略

### 4.1 GPU 内存优化

当遇到 GPU 内存不足 (CUDA out of memory) 的问题时，可以尝试以下策略：

```bash
# 方案1：减小批次大小 (最常用)
# 效果：内存使用减半，但为保持等效批次大小，需将梯度累积步数加倍，导致训练时间增加。
--batch_size 8 --grad_accumulation_steps 64

# 方案2：减小序列长度
# 效果：适用于处理较短文本的场景，能显著降低内存占用。
--seq_length 512

# 方案3：使用梯度检查点 (Gradient Checkpointing)
# 效果：用额外的计算时间来换取内存空间，适用于模型非常大的情况。
--gradient_checkpointing
```

-   **批次大小与梯度累积的权衡**:
    | `batch_size` | `grad_accumulation_steps` | 内存使用 | 训练速度 |
    | :--- | :--- | :--- | :--- |
    | 32 | 16 | 高 | 快 |
    | 16 | 32 | 中 | 中 |
    | 8 | 64 | 低 | 慢 |

### 4.2 数据加载优化

为避免 GPU 等待数据 IO，可以优化 DataLoader：

```bash
# 增加数据加载的并行工作进程数 (默认为0，即主进程加载)
--num_workers 4

# 配合 num_workers 使用，让每个 worker 提前预取批次
--prefetch_factor 2
```

-   **分布式训练的数据分片**:
    在分布式数据并行 (DDP) 训练中，DataLoader 会自动为每个 GPU 进程分配不同的数据分片，确保每个进程看到不同的数据样本，并在每个 Epoch 结束时进行同步。

## 5\. 常见问题 (FAQ)

**Q: 如何计算特定配置下的预估训练时间？**
A:

```python
# 示例计算 (Dolly-15k)
tokens_per_step = 16 * 1024 * 1 * 32  # 524,288
total_tokens = 2.8e6
steps_per_epoch = total_tokens / tokens_per_step  # ~6 步

# 假设根据实验，每步耗时约 30 秒 (高度依赖GPU型号)
time_per_epoch_seconds = steps_per_epoch * 30
total_time_minutes = time_per_epoch_seconds / 60  # ~3 分钟
```

**Q: 如何处理不同长度的文本？**
A: 当前实现主要采用两种策略：

1.  **填充 (Padding)**: 较短的文本序列会用特殊 token 填充到统一的 `seq_length`。
2.  **截断 (Truncation)**: 超过 `seq_length` 的文本序列会被截断。

**Q: 验证集是如何工作的？**
A:

-   **自动检测**: 如果数据目录中存在 `*_val_*.npy` 文件，它们会自动被用作验证集。
-   **自动划分**: 如果没有找到验证集文件，脚本会自动将第一个训练分片 (`*_train_000000.npy`) 的前 10% 作为验证数据。
-   **验证频率**: 由 `--eval_interval` 参数控制，例如 `--eval_interval 100` 表示每 100 步进行一次验证。

## 6\. 创建自定义数据集

您可以按照以下步骤创建自己的 `.npy` 格式数据集：

1.  准备一个或多个纯文本文件。
2.  使用 `tiktoken` 将文本编码为 Token ID。
3.  将 Token ID 序列保存为 `np.uint16` 类型的 NumPy 数组。
4.  将 `.npy` 文件放入数据目录，并遵循 `_train_` 或 `_val_` 的命名规则。

**示例代码:**

```python
import numpy as np
import tiktoken

# 1. 初始化 Tokenizer
enc = tiktoken.get_encoding("gpt2")

# 2. 准备并处理文本
with open('my_text_data.txt', 'r') as f:
    text = f.read()
tokens = enc.encode(text)

# 3. 保存为 .npy 文件
print(f"总计 {len(tokens)} 个 tokens")
np.save("custom_train_000000.npy", np.array(tokens, dtype=np.uint16))
```

## 7\. 相关文档

-   [训练指南](TRAINING.md) - 完整的模型训练流程说明。
-   [架构说明](https://www.google.com/search?q=ARCHITECTURE.md) - 模型架构的详细介绍。
