# 系统与模型架构

## 1. 代码库结构

项目的顶层目录按功能模块划分，旨在实现高内聚、低耦合。以下是主要文件和目录的详细说明：

```
/
├── README.md
├── requirements.txt         # 项目 Python 依赖
│
├── data_prep/               # --- 数据准备模块 ---
│   ├── fineweb.py           # 脚本：下载并处理 FineWeb 预训练数据集
│   ├── prepare_dolly.py     # 脚本：下载并处理 Dolly-15k 指令微调数据集
│   └── prepare_custom_dataset.py # 脚本：处理用户自定义的指令数据集
│
├── model/                   # --- 模型与训练核心 ---
│   ├── train_gpt2.py        # 核心：GPT-2 模型定义、预训练主循环
│   └── finetune_dolly.py    # 核心：指令微调主循环
│
├── eval/                    # --- 评估模块 ---
│   └── hellaswag.py         # 脚本：在 Hellaswag 基准上评估模型
│
├── web/                     # --- API 与 Web 服务 ---
│   ├── app.py               # 核心：Flask 应用，实现 OpenAI 兼容 API
│   ├── gunicorn_config.py   # 配置：Gunicorn 生产服务器配置
│   ├── static/              # 前端：存放 CSS 和 JS 文件
│   └── templates/           # 前端：存放 index.html 模板
│
├── log/                     # --- 输出目录 (自动生成) ---
│   └── ...                  # 存放预训练的检查点和日志
│
├── logs_finetune/           # --- 输出目录 (自动生成) ---
│   └── ...                  # 存放微调后的检查点和日志
│
└── docs/                    # --- 项目文档 ---
    ├── ARCHITECTURE.md      # 本文档：系统与模型架构
    ├── DATA_PROCESSING.md   # 文档：数据处理流程
    ├── TRAINING.md          # 文档：模型训练指南
    ├── DEPLOYMENT.md        # 文档：API 部署指南
    └── CHECKPOINT.md        # 文档：检查点管理机制
```

-   **`data_prep/`**: 包含所有数据预处理逻辑。每个脚本负责一种数据集，从下载、格式化到最终 Tokenize 并保存为训练脚本可读的 `.npy` 格式。
-   **`model/`**: 项目的核心。`train_gpt2.py` 不仅定义了 GPT-2 模型本身（`GPT` 类），还包含了完整的预训练流程。`finetune_dolly.py` 则是一个专门化的版本，用于指令微调。
-   **`eval/`**: 用于在标准化基准上衡量模型性能。
-   **`web/`**: 负责将训练好的模型包装成一个可用的服务。`app.py` 是入口，它使用 Flask 框架创建了一个 Web 服务器，并实现了与 OpenAI API 兼容的 `/v1/chat/completions` 等端点。
-   **`gunicorn_config.py`**: 用于生产环境部署的 Gunicorn 服务器配置文件，定义了 worker 类型、数量和超时等关键参数。
-   **`log/` & `logs_finetune/`**: 这两个目录是在训练过程中自动创建的，分别用于存放预训练和微调产生的模型检查点（`.pt` 文件）和训练日志（`log.txt`）。

## 2. 模型实现 (`model/train_gpt2.py`)

### 2.1 `GPT` 顶层类

`GPT` 类通过 `nn.ModuleDict` 整合了模型的所有组件。

-   **组件**:
    -   `wte`: `nn.Embedding` - 词嵌入层。
    -   `wpe`: `nn.Embedding` - 位置嵌入层。
    -   `h`: `nn.ModuleList` - 包含 `n_layer` 个 `Block` 模块。
    -   `ln_f`: `nn.LayerNorm` - 最终的层归一化。
-   **输出层 (`lm_head`)**: 一个 `nn.Linear` 层，将 Transformer 输出映射到词汇表 logits。其权重与 `wte` 共享。
-   **前向传播**: `forward(idx)` 方法的流程是：`idx` -> `tok_emb` + `pos_emb` -> `h` (循环) -> `ln_f` -> `logits`。

### 2.2 `Block` 核心模块

`Block` 模块是 Transformer 的基本单元，采用 Pre-LN 架构。

-   **数据流**:
    ```python
    x = x + self.attn(self.ln_1(x)) # Multi-Head Causal Self-Attention
    x = x + self.mlp(self.ln_2(x))  # MLP
    ```

### 2.3 关键子模块

#### `CausalSelfAttention`

-   **QKV 投影**: 使用单个 `nn.Linear` 层将输入 `x` 投影到 `3 * n_embd` 维度，然后分割为 Q, K, V。
-   **注意力计算**: 核心计算由 `F.scaled_dot_product_attention(is_causal=True)` 执行，该函数利用 Flash Attention 实现内存和计算高效的因果自注意力。

#### `MLP`

-   **结构**: 两层全连接网络，采用“扩展-收缩”架构：
    1.  `c_fc`: `nn.Linear(n_embd, 4 * n_embd)`
    2.  `c_proj`: `nn.Linear(4 * n_embd, n_embd)`
-   **激活函数**: `nn.GELU(approximate="tanh")`。

## 3. 训练优化策略

### 3.1 权重初始化 (`_init_weights`)

-   **线性层**: 权重从 `N(0, 0.02)` 正态分布初始化。对于残差路径上的 `c_proj` 层，标准差额外乘以 `(2 * n_layer)^-0.5` 以稳定训练。
-   **嵌入层**: 权重从 `N(0, 0.02)` 正态分布初始化。

### 3.2 优化器配置 (`configure_optimizers`)

-   **优化器**: 使用 `torch.optim.AdamW`。
-   **权重衰减 (Weight Decay)**: 参数被分为两组。所有维度 `>= 2` 的参数（如 `nn.Linear` 和 `nn.Embedding` 的权重）应用权重衰减；所有维度 `< 2` 的参数（如偏置和 `LayerNorm` 参数）不应用权重衰减。
-   **Fused 优化器**: 在 CUDA 环境下，自动启用 `fused=True` 以提升性能。
