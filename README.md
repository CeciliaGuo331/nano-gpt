# NanoGPT: 从随机权重到指令遵循

## 项目简介

本项目基于 [build-nanogpt](https://github.com/karpathy/build-nanogpt) 提供的 PyTorch GPT-2 模型原型实现，通过预训练建立语言基础，并使用 dolly-15k 指令数据集进行监督微调，构建了具备指令遵循能力的对话助理。项目实现了完整的 OpenAI 兼容 API 接口，支持流式生成、高级采样参数，提供了功能丰富的 Web 前端界面，并可与 LobeChat 等主流 AI 应用无缝集成。

## 模型架构

### 架构概述

GPT-2 采用了仅解码器（decoder-only）的 Transformer 架构，其核心由 12 个相同的 Transformer 块 (`Block`) 堆叠而成。每个`Block`包含两个关键组件：

1. 多头因果自注意力层 (Multi-Head Causal Self-Attention)
2. 前馈神经网络层 (Feed-Forward Network, or MLP)

`Block`模块采用了预归一化 (Pre-Normalization / Pre-LN) 结构，在每个子模块前进行层归一化 (LayerNorm)，并通过残差连接 (Residual Connection) 将子模块输出与其输入相加。相比后归一化结构，Pre-LN 提供更稳定的梯度流动，显著改善训练收敛性。

### 数据流处理

每个 Transformer 块的前向传播遵循以下流程：

```python
# 第一个子层：自注意力
x_norm1 = ln_1(x)                    # 层归一化
attn_out = self_attention(x_norm1)   # 多头自注意力
x = x + attn_out                     # 残差连接

# 第二个子层：前馈网络
x_norm2 = ln_2(x)                    # 层归一化
mlp_out = mlp(x_norm2)               # 前馈网络
x = x + mlp_out                      # 残差连接
```

### 模块实现和优化方法

1. 因果自注意力模块（CausalSelfAttention）

这个模块实现了多头因果自注意力机制。

-   高效的 QKV 计算： 用单一的线性层 `c_attn` 将输入 `x` 从维度 `n_embd` 投影到 `3 * n_embd`，并将结果张量沿最后一个维度分割成 Q, K, V 三部分，避免多次矩阵运算。

-   多头并行处理：在得到 Q, K, V 后，通过 `view()` 和 `transpose()` 操作将嵌入维度 `C` 拆分为 `n_head` 个头，每个头的维度为 `hs = C / n_head`。这使得模型能从多个表示子空间并行学习特征。

-   因果掩码与 Flash Attention：

    -   核心计算使用 `F.scaled_dot_product_attention(is_causal=True)`

    -   `is_causal=True` 自动应用上三角掩码，确保每个位置只能访问当前及之前的 token，满足自回归生成要求

    -   底层调用 Flash Attention 算法，实现内存高效的线性复杂度注意力计算

-   输出投影：在所有头的输出被重新拼接回 `(B, T, C)` 的形状后，通过一个最终的线性层 `c_proj` 进行投影，得到该模块的最终输出。

2. 前馈神经网络模块（MLP）

这是一个标准的前馈神经网络模块，也称为 Position-wise Feed-Forward Network。

-   维度变换：采用经典的"扩展-收缩"架构

    -   `c_fc：n_embd → 4×n_embd`（维度扩展）

    -   `c_proj：4×n_embd → n_embd`（维度收缩）

-   激活函数：使用 GELU（Gaussian Error Linear Unit）替代传统 ReLU，其平滑特性和随机正则化效应提升训练稳定性和模型性能。

-   特征学习：扩展的中间维度为模型提供更大的参数空间，增强复杂特征变换的学习能力。

这种精心设计的架构在保持计算效率的同时，为模型提供了强大的序列建模和特征学习能力。

## 训练流程

### 阶段一：基础预训练 (Pre-training)

### 阶段二：指令微调 (Instruction Fine-tuning)

## 性能评估

## 部署与应用

## 快速开始

### 1. 环境设置

### 2. 数据准备

### 3. 模型训练与微调

### 4. 启动 Web 应用

## 项目文件结构

## 展望
