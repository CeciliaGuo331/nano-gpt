# 脚本运行指南

## 为什么使用模块运行方式？

项目使用相对导入确保代码的模块化和可扩展性。使用 `python -m` 运行脚本可以：

- 正确解析相对导入
- 保持模块路径一致性
- 避免导入错误

## 各模块命令参考

### 数据准备

```bash
python -m data_prep.fineweb          # 预训练数据
python -m data_prep.prepare_dolly    # 微调数据
```

### 模型训练

```bash
python -m model.train_gpt2           # 预训练
python -m model.finetune_dolly       # 微调
```

### Web 服务

```bash
python -m web.app                  # API 服务
python -m web.inference              # 推理测试
```

## 常用参数组合

### 内存受限环境

```bash
python -m model.train_gpt2 --batch_size 8 --gradient_checkpointing
```

### 快速实验

```bash
python -m model.train_gpt2 --max_steps 100 --eval_interval 50
```

### 分布式训练

```bash
torchrun --nproc_per_node=4 -m model.train_gpt2
```

## 环境变量

```bash
# CUDA 设备选择
CUDA_VISIBLE_DEVICES=0,1 python -m model.train_gpt2

# 模型路径
MODEL_CHECKPOINT=path/to/model.pt python -m web.app

# 组合使用
CUDA_VISIBLE_DEVICES=0 PORT=8080 python -m web.app
```

## 调试技巧

### 1. 检查导入

```bash
python -c "from model.train_gpt2 import GPT; print('Import successful')"
```

### 2. 验证路径

```bash
python -c "import sys; print(sys.path)"
```

### 3. 测试小批量

```bash
python -m model.train_gpt2 --max_steps 10 --batch_size 2
```

## IDE 配置

### VS Code

在 `.vscode/launch.json` 中：

```json
{
  "name": "Train GPT2",
  "type": "python",
  "request": "launch",
  "module": "model.train_gpt2",
  "cwd": "${workspaceFolder}",
  "args": ["--max_steps", "100"]
}
```

### PyCharm

- Working directory: 项目根目录
- Module name: `model.train_gpt2`
- Parameters: `--max_steps 100`

## 故障排查

### ModuleNotFoundError

确保：

1. 在项目根目录运行
2. 使用 `-m` 参数
3. 模块路径使用点号分隔

### ImportError

检查：

1. 相对导入语法是否正确
2. `__init__.py` 文件是否存在
3. Python 路径是否包含项目根目录

## 相关文档

- [训练指南](TRAINING.md) - 详细的训练参数
- [部署指南](DEPLOYMENT.md) - 生产环境配置
