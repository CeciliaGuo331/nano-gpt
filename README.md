# NanoGPT: 从预训练到指令微调的完整实践

本项目是一个基于 PyTorch 的 GPT-2 实现，旨在提供一个从数据准备、模型预训练、指令微调到最终部署的完整、端到端的开源解决方案。通过本项目，您可以深入了解一个小型语言模型（SLM）是如何从随机权重开始，一步步学习通用知识，并最终被塑造成一个能听懂人类指令的对话助理的。

我们最终提供了一个与 OpenAI API 完全兼容的服务接口，可以无缝集成到 LobeChat 等现代 AI 应用中。

## ✨ 主要功能

-   **完整的端到端流程**: 涵盖从数据处理到生产部署的每一个环节，代码清晰易懂。
-   **强大的训练能力**: 内置断点续训和分布式训练 (DDP) 支持，能够进行稳定、高效的大规模训练。
-   **有效的指令微调**: 验证了在 GPT-2 这样的经典架构上进行指令微调的有效性，并提供了完整的实现。
-   **OpenAI 兼容 API**: 提供高性能、支持流式生成的 API 接口，可作为任何 OpenAI 兼容客户端的后端服务。
-   **Web 交互界面**: 自带一个简洁的 Web UI，方便快速进行模型效果的演示和测试。

## 🚀 快速开始

### 1. 环境设置

确保您已安装 Python 3.10+。然后，克隆仓库并安装所需的依赖项：

```bash
# 克隆项目仓库
git clone https://github.com/your-username/nano-gpt.git
cd nano-gpt

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

本项目包含预训练和微调两个阶段，需要准备相应的数据集。

-   **预训练数据 (FineWeb-Edu)**: 下载并处理 FineWeb-Edu 数据集用于模型预训练。
    ```bash
    python -m data_prep.fineweb
    ```
-   **指令微调数据 (Dolly-15k)**: 下载并处理 Dolly-15k 数据集用于指令微调。
    ```bash
    python -m data_prep.prepare_dolly
    ```

### 3. 模型训练与微调

-   **第一步：预训练**: 在准备好的 FineWeb 数据上进行预训练，建立模型的基础语言能力。
    ```bash
    # 运行预训练脚本
    python -m model.train_gpt2
    ```
-   **第二步：指令微调**: 使用预训练好的模型权重，在 Dolly-15k 数据集上进行微调，使其学会遵循指令。
    ```bash
    # 运行微调脚本，它会自动加载最新的预训练检查点
    python -m model.finetune_dolly
    ```

### 4. 启动 Web 应用

训练完成后，您可以启动后端服务，它将提供 API 接口和 Web UI。

```bash
# 使用 gunicorn 在生产模式下启动
gunicorn --config gunicorn_config.py web.app:app
```

服务启动后，您可以：
-   在浏览器中打开 `http://127.0.0.1:5001` 访问 Web 界面。
-   将 `http://127.0.0.1:5001/v1` 作为 API 端点配置到 LobeChat 或其他 OpenAI 客户端中。

## ⚙️ API 使用示例

您可以使用 `curl` 或任何 HTTP 客户端与模型的 API 进行交互。请确保将 `YOUR_API_KEY` 替换为您在环境中设置的密钥。

```bash
curl http://127.0.0.1:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "model_00599.pt",
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms."
      }
    ],
    "stream": false
  }'
```

## 📚 详细文档

如果您想深入了解本项目的技术细节、设计思路和实现方法，我们提供了更详细的文档：

-   **[系统架构 (`docs/ARCHITECTURE.md`)]**: 深入了解模型的核心组件和系统设计。
-   **[数据处理 (`docs/DATA_PROCESSING.md`)]**: 查看关于预训练和微调数据的详细处理流程。
-   **[模型训练 (`docs/TRAINING.md`)]**: 获取关于训练策略、超参数和关键技术的详细信息。
-   **[部署指南 (`docs/DEPLOYMENT.md`)]**: 了解生产环境部署的细节和最佳实践。
-   **[检查点管理 (`docs/CHECKPOINT.md`)]**: 详细说明检查点的保存、恢复和管理机制。

## 📂 项目文件结构

```
/
├── data_prep/         # 数据准备脚本 (FineWeb, Dolly, 自定义)
├── model/             # 模型核心与训练脚本 (预训练, 微调)
├── eval/              # 评估脚本 (Hellaswag)
├── web/               # Web 应用 (Flask后端, API实现, 前端模板)
├── log/               # 预训练模型检查点和日志
├── logs_finetune/     # 微调模型检查点和日志
├── gunicorn_config.py # Gunicorn 生产服务器配置
├── requirements.txt   # Python 依赖项
└── README.md          # 本文档
```

## 🤝 如何贡献

我们欢迎任何形式的贡献！如果您有任何想法、建议或发现了 Bug，请随时提交 Pull Request 或创建 Issue。

## 📄 许可证

本项目采用 MIT 许可证。详情请见 `LICENSE` 文件。

