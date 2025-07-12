# 部署指南

## 概述

本指南介绍如何部署训练好的 nano-gpt 模型。

## 快速开始

### 本地部署

```bash
# 启动 Flask 服务
python -m web.serve

# 指定模型和端口
MODEL_CHECKPOINT=log/model_40000.pt PORT=8080 python -m web.serve
```

服务启动后访问 `http://localhost:5000`

## API 接口

### POST /generate

生成文本的主要接口。

**请求示例：**
```json
{
    "prompt": "Once upon a time",
    "max_length": 150,
    "temperature": 0.8,
    "top_k": 50
}
```

**响应示例：**
```json
{
    "prompt": "Once upon a time",
    "generated_text": "Once upon a time, there was a small village...",
    "tokens_generated": 150
}
```

**参数说明：**
- `prompt`: 输入文本（必需）
- `max_length`: 最大生成长度（默认：100）
- `temperature`: 随机性控制，0.1-1.0（默认：0.8）
- `top_k`: Top-K 采样（默认：50）

### GET /health

健康检查接口。

**响应示例：**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cuda"
}
```

## 生产部署

### 使用 Gunicorn

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动服务（4个工作进程）
gunicorn -w 4 -b 0.0.0.0:5000 web.serve:app

# 带超时设置
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 web.serve:app
```

### Docker 部署

**Dockerfile:**
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制代码
COPY model/ ./model/
COPY web/ ./web/
COPY eval/ ./eval/

# 复制模型文件
COPY log/ ./log/

# 暴露端口
EXPOSE 5000

# 启动服务
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "web.serve:app"]
```

**构建和运行：**
```bash
# 构建镜像
docker build -t nano-gpt-api .

# 运行容器
docker run -p 5000:5000 --gpus all nano-gpt-api

# 使用环境变量
docker run -p 5000:5000 --gpus all \
    -e MODEL_CHECKPOINT=/app/log/model_40000.pt \
    nano-gpt-api
```

### 使用 Nginx 反向代理

**nginx.conf:**
```nginx
upstream app {
    server localhost:5000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

## 性能优化

### 1. 模型加载优化

```python
# 在 web/serve.py 中预加载模型
model = load_model_from_checkpoint(checkpoint_path)
model.eval()  # 设置为评估模式
```

### 2. 批处理请求

对于高并发场景，可以实现请求批处理：

```python
# 累积多个请求
batch_prompts = ["prompt1", "prompt2", "prompt3"]
# 批量生成
results = batch_generate(model, batch_prompts)
```

### 3. 缓存常见请求

使用 Redis 缓存：

```bash
# 安装 Redis
pip install redis

# 在代码中使用缓存
import redis
cache = redis.Redis(host='localhost', port=6379)
```

## 监控和日志

### 基础日志

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### 请求追踪

```python
@app.before_request
def log_request():
    logger.info(f"Request: {request.method} {request.path}")

@app.after_request  
def log_response(response):
    logger.info(f"Response: {response.status_code}")
    return response
```

## 安全建议

### 1. API 密钥认证

```python
API_KEY = os.environ.get('API_KEY')

@app.before_request
def check_api_key():
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({'error': 'Invalid API key'}), 401
```

### 2. 请求限流

```bash
# 安装限流库
pip install flask-limiter

# 使用限流
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/generate', methods=['POST'])
@limiter.limit("10 per minute")
def generate():
    # 处理请求
```

### 3. 输入验证

```python
def validate_request(data):
    if not data.get('prompt'):
        return False, "Prompt is required"
    
    if len(data['prompt']) > 1000:
        return False, "Prompt too long"
        
    if data.get('max_length', 100) > 500:
        return False, "max_length too large"
        
    return True, None
```

## 故障排查

### 常见问题

1. **GPU 不可用**
   ```bash
   # 检查 CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **内存不足**
   - 使用更小的模型
   - 减少并发工作进程数
   - 启用模型量化

3. **响应超时**
   - 增加 Gunicorn 超时时间
   - 减小 max_length 参数
   - 使用异步处理

### 性能监控

```bash
# CPU 和内存使用
htop

# GPU 使用
nvidia-smi -l 1

# 网络连接
netstat -an | grep 5000
```

## 部署清单

- [ ] 选择合适的硬件（GPU 推荐）
- [ ] 安装必要的依赖
- [ ] 配置环境变量
- [ ] 设置日志和监控
- [ ] 配置反向代理
- [ ] 实施安全措施
- [ ] 进行负载测试
- [ ] 准备回滚方案

## 相关文档

- [训练指南](TRAINING.md) - 模型训练
- [API 参考](API_REFERENCE.md) - 详细 API 文档
- [故障排查](TROUBLESHOOTING.md) - 更多问题解决