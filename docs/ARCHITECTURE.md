# nano-GPT 架构文档

## 系统概览

```mermaid
graph TB
    subgraph "用户层"
        A[Web UI<br/>待实现]
    end
    
    subgraph "服务层"
        B[Flask API<br/>serve.py]
    end
    
    subgraph "推理层"
        C[推理引擎<br/>inference.py]
    end
    
    subgraph "模型层"
        D[GPT-2 模型<br/>train_gpt2.py]
    end
    
    subgraph "数据层"
        E1[预训练数据<br/>fineweb.py]
        E2[微调数据<br/>prepare_dolly.py<br/>待实现]
    end
    
    A -.->|HTTP| B
    B --> C
    C --> D
    D --> E1
    D --> E2
    
    style A fill:#fee
    style E2 fill:#fee
```

## 核心组件

### 模型架构 (train_gpt2.py)

```mermaid
graph TD
    GPT[GPT主模型] --> TB[12个Transformer块]
    TB --> CA[因果自注意力<br/>Flash Attention]
    TB --> FF[前馈网络<br/>4x扩展]
    TB --> LN[层归一化]
    
    CA --> MH[12个注意力头]
    FF --> GELU[GELU激活]
```

**关键参数**：
- 模型大小：124M参数
- 词汇表：50,257 tokens
- 嵌入维度：768
- 上下文长度：1024

### 数据流

```mermaid
graph LR
    subgraph "训练流程"
        T1[原始文本] --> T2[Tokenizer<br/>GPT-2 BPE]
        T2 --> T3[二进制文件<br/>.npy格式]
        T3 --> T4[DataLoader]
        T4 --> T5[模型训练]
    end
    
    subgraph "推理流程"
        I1[用户输入] --> I2[API请求]
        I2 --> I3[模型推理]
        I3 --> I4[采样生成<br/>Top-k/Temperature]
        I4 --> I5[返回结果]
    end
```

## 项目结构

```
nano-gpt/
├── model/
│   ├── train_gpt2.py      # 模型定义+预训练
│   └── finetune_dolly.py  # 指令微调(待实现)
├── data_prep/
│   ├── fineweb.py         # 预训练数据处理
│   └── prepare_dolly.py   # 微调数据处理(待实现)
├── platform/
│   ├── serve.py           # Flask API服务
│   └── inference.py       # 推理接口
├── eval/
│   └── hellaswag.py       # 模型评估
├── docs/                  # 项目文档
└── log/                   # 训练日志和检查点
```

## 关键特性

### 训练优化
- **混合精度训练** (AMP)
- **梯度累积**
- **分布式训练** (DDP)
- **检查点续训**

### 推理优化
- **KV缓存**
- **Top-k采样**
- **温度控制**

## 当前状态

✅ **已完成**
- GPT-2基础模型实现
- 预训练流程
- 推理引擎
- API服务框架

❌ **待实现**
- Dolly数据集准备
- 指令微调脚本
- Web前端界面
- 完整的错误处理

## 技术栈

- **深度学习**: PyTorch
- **分词器**: tiktoken
- **Web框架**: Flask
- **数据处理**: NumPy, datasets

---
*最后更新: 2025-07-11*