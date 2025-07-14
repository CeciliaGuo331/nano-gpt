# nano-gpt

基于 GPT-2 架构的教育性语言模型实现，支持预训练、指令微调和 Web 部署。

## ✨ 特性

- **完整的 GPT-2 实现** - 包含多头注意力、位置编码等核心组件
- **两阶段训练流程** - 支持基础预训练和指令微调
- **断点续训** - 训练中断后可从检查点恢复
- **Web API 服务** - Flask 实现的推理 API
- **分布式训练** - 支持多 GPU 并行训练

## 🚀 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/yourusername/nano-gpt.git
cd nano-gpt

# 安装依赖
pip install -r requirements.txt
```

### 重要提示

所有脚本需要从项目根目录以模块方式运行：

```bash
# ✅ 正确方式
python -m model.train_gpt2

# ❌ 错误方式
python model/train_gpt2.py
```

### 基础使用

```bash
# 1. 准备数据
python -m data_prep.fineweb      # 预训练数据
python -m data_prep.prepare_dolly # 微调数据

# 2. 预训练模型
python -m model.train_gpt2

# 3. 指令微调
python -m model.finetune_dolly --pretrained_checkpoint log/model_40000.pt

# 4. 启动服务
python -m web.app
```

## 📚 文档

- [训练指南](docs/TRAINING.md) - 预训练、微调、断点续训
- [数据处理](docs/DATA_PROCESSING.md) - 数据准备和分片机制
- [部署指南](docs/DEPLOYMENT.md) - 本地部署和生产环境配置
- [架构说明](docs/ARCHITECTURE.md) - 模型架构和设计决策
- [脚本运行指南](docs/SCRIPT_RUNNING_GUIDE.md) - 详细的运行说明

## 📁 项目结构

```
nano-gpt/
├── model/              # 模型实现
│   ├── train_gpt2.py   # 预训练脚本
│   └── finetune_dolly.py # 微调脚本
├── data_prep/          # 数据处理
├── eval/               # 评估模块
├── web/                # Web 服务
└── docs/               # 项目文档
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。

## 📄 许可证

MIT License

## 🙏 致谢

- 受 [Andrej Karpathy](https://github.com/karpathy) 的 build-nanogpt 项目启发
- 使用 [Databricks Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) 数据集进行微调
