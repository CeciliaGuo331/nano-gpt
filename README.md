# nano-gpt

本仓库在 Andrej Karpathy 的 [build-nanogpt](https://github.com/karpathy/build-nanogpt) 项目基础上进行改进与扩展。项目旨在完整复现一个基于GPT-2架构的语言模型，并在此基础上，通过指令微调（Instruction Fine-tuning）技术，使其从一个只会进行语法正确的无意义续写的“基础模型”转变为一个具备初步指令遵循能力的“入门级助手”，最终将其部署为一个可实际交互的Web应用。

## ✨ 项目亮点

* **端到端全流程复现**：涵盖了从数据处理、模型预训练、指令微调、性能评估到最终部署的完整大型语言模型生命周期。
* **清晰的两阶段训练**：通过预训练和微调的鲜明对比，直观展示了“指令遵循”能力是如何通过微调注入给基础模型的。
* **健壮的工程实践**：代码中加入了断点续训（Checkpointing）功能的集成，保证了长时间训练的稳定性。
* **应用驱动**：项目最终成果并非仅停留在模型层面，而是通过**Flask**和Web前端，实现了一个可交互、可视化的原型应用。

## 🏛️ 模型架构

`nano-GPT` 项目借鉴了 Andrej Karpathy 对 GPT-2 的经典复现。其核心是一种仅解码器（Decoder-only）的 Transformer 架构。网络由12个解码器模块级联组成，每个模块内部均集成了一个多头自注意力机制（Multi-Head Self-Attention）与一个前馈神经网络（Feed-Forward Network），并通过残差连接和层归一化保证了训练的稳定性。

## 📚 训练流程与数据集

本项目的训练分为两个核心阶段：

1.  **基础预训练 (Pre-training)**
    * **数据集**: **FineWeb** 数据集（OpenWebText的一个高质量子集）。
    * **目标**: 在海量的通用英文文本上进行自监督学习（预测下一个词），让模型掌握英语的语法、句法、基本事实知识和文本连贯性，成为一个“通才”。

2.  **指令微调 (Instruction Fine-tuning)**
    * **数据集**: **Databricks Dolly 15k**，一个由数千名员工手写的、高质量、包含7种任务类型的指令数据集。
    * **目标**: 在预训练好的模型基础上，使用这个“教材”进行有监督的微调，教会模型如何理解并遵循人类的指令，使其从“通才”转变为“专才”。

## 📊 模型能力评估

* **定量评估**: 使用 **Hellaswag** 数据集作为常识推理能力的基准测试，对比模型在微调前后的性能变化。
* **定性评估**: 设计了一系列覆盖不同任务类型（如问答、总结、创意写作）的测试指令，直观展示模型在微调前后回答质量的“脱胎换骨”。

| 任务类型 | 微调前模型 (Base Model) | 微调后模型 (Finetuned Model) |
| :--- | :--- | :--- |
| **封闭式问答** | `Q: What is the capital of France? A: What is the capital of Spain? It is...` | `Q: What is the capital of France? A: The capital of France is Paris.` |
| **创意写作** | `Write a short poem about the moon.` -> *(生成与诗歌无关的零散句子)* | `Write a short poem about the moon.` -> *(生成一段关于月亮的押韵短诗)* |

## 🚀 部署应用

项目通过以下技术栈实现了一个简单的Web应用：

* **后端**: 使用 **Flask** 框架将微调好的PyTorch模型封装成一个高效的HTTP API接口。
* **前端**: 使用原生 **HTML, CSS, 和 JavaScript** 构建一个简洁的用户界面，用户可以在网页上输入指令，并实时获取模型的生成结果。

## 快速开始

### 1. 环境设置

```bash
# 克隆本仓库
git clone [你的仓库URL]
cd [你的仓库目录]

# 安装依赖
pip install -r requirements.txt
```

*(注: `requirements.txt` 应包含 `torch`, `numpy`, `flask`, `transformers`, `datasets` 等)*

### 2\. 数据准备

```bash
# 运行脚本下载并预处理FineWeb数据集用于预训练
python data/prepare_fineweb.py

# 运行脚本下载并格式化Dolly-15k数据集用于微调
python data/prepare_dolly.py
```

### 3\. 模型训练与微调

```bash
# 第一阶段：执行基础预训练
# (该过程耗时较长，建议在有GPU的服务器上运行)
python training/train.py --config=configs/pretrain_gpt2.py

# 第二阶段：执行指令微调
# (确保 --init_from 指向你预训练好的模型检查点)
python training/finetune.py --config=configs/finetune_dolly.py
```

### 4\. 启动Web应用

```bash
# 启动Flask后端服务器 (用于开发测试)
python app.py

# 在浏览器中打开 frontend/index.html 文件即可开始交互
```

## 📁 项目文件结构

```
.
├── app.py                  # Flask后端服务代码
├── configs/
│   ├── pretrain_gpt2.py    # 预训练配置文件
│   └── finetune_dolly.py   # 微调配置文件
├── data/
│   ├── prepare_fineweb.py  # FineWeb数据集处理脚本
│   └── prepare_dolly.py    # Dolly数据集处理脚本
├── evaluation/
│   ├── eval_hellaswag.py   # Hellaswag评估脚本
│   └── qualitative_tests.py # 定性评估脚本
├── frontend/
│   ├── index.html          # 前端页面
│   └── script.js           # 前端交互逻辑
├── training/
│   ├── train.py            # 预训练主脚本
│   ├── finetune.py         # 微调主脚本
│   └── model.py            # GPT模型定义
└── README.md               # 项目说明文档
```

## 展望

  * **模型扩展**：如算力充足，可尝试更大规模的GPT-2变体（如355M参数版本），探索模型规模对指令遵循能力的影响。
  * **数据集融合**：混合多种指令数据集（如Alpaca, COIG）进行微调，提升模型的泛化能力。
  * **评估体系**：引入更多维度的评估基准，对模型的综合能力进行更全面的考察。

## 致谢

  * 本项目深受 [Andrej Karpathy](https://github.com/karpathy) 的 `build-nanogpt` 教程启发。
  * 微调数据使用了由 [Databricks](https://www.databricks.com/) 贡献的 `dolly-v2-12b` 数据集。
