# 模型部署指南

本文档详细介绍如何部署训练好的 nanoGPT 模型。

## 前置要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（推荐，用于 GPU 加速）
- 训练好的模型 checkpoint 文件

## 部署方式

### 1. 本地推理脚本

最简单的使用方式是通过 `inference.py` 脚本进行本地推理。

#### 基本使用

```python
from inference import load_model_from_checkpoint, generate_text

# 加载模型
model = load_model_from_checkpoint('log/latest_checkpoint.pt')

# 生成文本
text = generate_text(
    model,
    prompt="Once upon a time",
    max_length=200,
    temperature=0.8,
    top_k=50
)
print(text)
```

#### 参数说明

- `checkpoint_path`: 模型 checkpoint 文件路径
- `device`: 运行设备（'cuda' 或 'cpu'）
- `prompt`: 输入提示词
- `max_length`: 生成的最大 token 数量
- `temperature`: 控制随机性（0.1-1.0，越高越随机）
- `top_k`: Top-K 采样，只从概率最高的 K 个词中选择

### 2. Flask Web API

通过 `serve.py` 部署为 REST API 服务。

#### 安装依赖

```bash
pip install flask tiktoken
```

#### 启动服务

```bash
# 使用默认设置
python serve.py

# 自定义配置
MODEL_CHECKPOINT=log/model_40000.pt PORT=8080 python serve.py
```

#### API 端点

**POST /generate**

生成文本的主要端点。

请求示例：
```json
{
    "prompt": "In the year 2050",
    "max_length": 150,
    "temperature": 0.7,
    "top_k": 40
}
```

响应示例：
```json
{
    "prompt": "In the year 2050",
    "generated_text": "In the year 2050, the world had changed dramatically...",
    "tokens_generated": 150
}
```

**GET /health**

健康检查端点。

响应示例：
```json
{
    "status": "healthy",
    "device": "cuda"
}
```

### 3. 生产环境部署

#### 使用 Gunicorn

安装 Gunicorn：
```bash
pip install gunicorn
```

启动服务：
```bash
gunicorn -w 4 -b 0.0.0.0:5000 serve:app
```

#### Docker 部署

创建 `Dockerfile`：

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制代码和模型
COPY *.py ./
COPY log/ ./log/

# 暴露端口
EXPOSE 5000

# 启动服务
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "serve:app"]
```

创建 `requirements.txt`：
```
torch>=2.0.0
tiktoken
flask
gunicorn
numpy
```

构建和运行：
```bash
# 构建镜像
docker build -t nanogpt-api .

# 运行容器
docker run -p 5000:5000 --gpus all nanogpt-api
```

#### Kubernetes 部署

创建 `deployment.yaml`：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nanogpt-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nanogpt-api
  template:
    metadata:
      labels:
        app: nanogpt-api
    spec:
      containers:
      - name: nanogpt-api
        image: nanogpt-api:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "4Gi"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_CHECKPOINT
          value: "/model/checkpoint.pt"
        volumeMounts:
        - name: model-volume
          mountPath: /model
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: nanogpt-service
spec:
  selector:
    app: nanogpt-api
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

### 4. 性能优化

#### 批处理

修改 `serve.py` 支持批量请求：

```python
@app.route('/batch_generate', methods=['POST'])
def batch_generate():
    """批量生成文本"""
    data = request.json
    prompts = data.get('prompts', [])
    
    # 批量处理
    results = []
    for prompt in prompts:
        # 这里可以优化为真正的批处理
        result = generate_single(prompt, **data.get('params', {}))
        results.append(result)
    
    return jsonify({'results': results})
```

#### 缓存

使用 Redis 缓存常见请求：

```python
import redis
import hashlib
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cache_key(prompt, params):
    """生成缓存键"""
    content = f"{prompt}:{json.dumps(params, sort_keys=True)}"
    return hashlib.md5(content.encode()).hexdigest()

def generate_with_cache(prompt, **params):
    """带缓存的生成函数"""
    cache_key = get_cache_key(prompt, params)
    
    # 尝试从缓存获取
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 生成新结果
    result = generate_text(model, prompt, **params)
    
    # 存入缓存（1小时过期）
    redis_client.setex(cache_key, 3600, json.dumps(result))
    
    return result
```

### 5. 监控和日志

#### 添加日志

```python
import logging
from flask import Flask, request, jsonify
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.before_request
def log_request():
    """记录请求信息"""
    request.start_time = time.time()
    logger.info(f"Request: {request.method} {request.path}")

@app.after_request
def log_response(response):
    """记录响应信息"""
    duration = time.time() - request.start_time
    logger.info(f"Response: {response.status_code} ({duration:.3f}s)")
    return response
```

#### Prometheus 监控

```python
from prometheus_client import Counter, Histogram, generate_latest

# 定义指标
request_count = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
request_duration = Histogram('api_request_duration_seconds', 'API request duration')
generation_tokens = Counter('generated_tokens_total', 'Total generated tokens')

@app.route('/metrics')
def metrics():
    """Prometheus 指标端点"""
    return generate_latest()
```

### 6. 安全建议

1. **API 认证**
```python
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/generate', methods=['POST'])
@require_api_key
def generate():
    # ... 原有代码
```

2. **请求限流**
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.headers.get('X-API-Key', 'anonymous'),
    default_limits=["100 per hour"]
)

@app.route('/generate', methods=['POST'])
@limiter.limit("10 per minute")
def generate():
    # ... 原有代码
```

3. **输入验证**
```python
def validate_generate_request(data):
    """验证生成请求参数"""
    if not data.get('prompt'):
        return False, "Prompt is required"
    
    max_length = data.get('max_length', 100)
    if max_length > 1000:
        return False, "max_length cannot exceed 1000"
    
    temperature = data.get('temperature', 0.8)
    if not 0.1 <= temperature <= 1.0:
        return False, "temperature must be between 0.1 and 1.0"
    
    return True, None
```

## 故障排除

### 常见问题

1. **内存不足**
   - 降低批处理大小
   - 使用量化技术
   - 增加交换空间

2. **GPU 不可用**
   - 检查 CUDA 安装
   - 确认 PyTorch GPU 版本
   - 检查 nvidia-smi 输出

3. **生成质量差**
   - 检查 checkpoint 是否正确加载
   - 调整 temperature 和 top_k 参数
   - 确认使用了正确的 tokenizer

### 性能调优

1. **使用 TorchScript**
```python
# 导出模型
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "model_scripted.pt")

# 加载使用
model = torch.jit.load("model_scripted.pt")
```

2. **使用 ONNX**
```python
# 导出到 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,
    do_constant_folding=True
)
```

3. **使用 TensorRT（NVIDIA GPU）**
```python
import torch_tensorrt

# 编译模型
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=[1, 512])],
    enabled_precisions={torch.float, torch.half}
)
```

## 总结

本指南涵盖了从简单的本地推理到生产级部署的各种方案。根据你的具体需求选择合适的部署方式：

- **开发测试**：使用 inference.py 脚本
- **小规模服务**：使用 Flask + Gunicorn
- **生产环境**：使用 Docker/Kubernetes + 监控 + 缓存
- **高性能场景**：使用模型优化技术（量化、TensorRT 等）

记得根据实际负载调整配置，并做好监控和日志记录。