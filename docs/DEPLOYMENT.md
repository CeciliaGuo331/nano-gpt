# 部署指南

## 快速开始

### 本地开发

该方式使用 Flask 内置的开发服务器，非常适合在本地进行代码调试和测试。

```bash
# 启动 Flask 开发服务 (使用默认的测试 API Key)
python -m web.app

# 你也可以通过环境变量指定模型和端口
MODEL_CHECKPOINT=log/model_40000.pt
python -m web.app
```

服务启动后，你可以通过浏览器访问 `http://localhost:5002` 查看前端页面，并通过 API 工具调用接口。

### 生产部署

该方式使用 `gunicorn` 服务器，它是一个功能强大的 WSGI HTTP 服务器，比 Flask 自带的服务器更稳定、性能更好，适合在生产环境或需要对外提供服务时使用。我们推荐结合 `gevent` worker 来处理高并发 I/O。

1.  **安装 Gunicorn 和 Gevent**:

    ```bash
    pip install gunicorn gevent
    ```

2.  **配置 Gunicorn**:
    项目根目录下已提供 `gunicorn_config.py` 文件，其中包含了 Gunicorn 的基本配置，例如绑定地址、worker 类型和数量、超时设置等。模型加载是按需进行的，每个 worker 会独立加载和缓存模型。

3.  **设置生产环境 API Key 并启动服务**:

    ```bash
    # 设置一个安全的 API Key，然后启动服务
    export MY_API_KEY='your_super_secret_production_api_key'
    gunicorn -c gunicorn_config.py web.app:app
    ```

服务启动后，你可以通过浏览器访问 `http://localhost:5001` (或 `gunicorn_config.py` 中配置的端口) 查看前端页面，并通过 API 工具调用接口。

## Web 界面功能

### 参数配置

Web 界面支持完整的 OpenAI 兼容参数：

-   **最大长度 (Max Tokens)**: 生成文本的最大 token 数量，范围 1-512
-   **随机度 (Temperature)**: 控制生成的随机性，0.1-2.0，值越高越随机
-   **词汇限制 (Top-K)**: 从概率最高的 K 个词中采样，1-100
-   **核采样 (Top-P)**: 累积概率阈值，0.0-1.0，控制词汇多样性
-   **话题惩罚 (Presence Penalty)**: 减少重复话题，-2.0 到 2.0
-   **重复惩罚 (Frequency Penalty)**: 根据词频减少重复，-2.0 到 2.0

### 快捷操作

-   **Ctrl+Enter**: 快速生成文本
-   **自动检测**: API 地址自动设置为当前访问地址
-   **响应式设计**: 支持手机、平板、桌面各种屏幕

## API 接口

### 认证 (Authentication)

所有 API 请求都必须包含有效的 API Key。请在 HTTP 请求头中提供该 Key。

| Header          | 类型     | 描述                         |
| :-------------- | :------- | :--------------------------- |
| `Authorization` | `string` | Bearer token 格式的 API 密钥 |

### GET /v1/models

获取可用模型列表。

**示例命令**

```bash
curl -X GET http://localhost:5002/v1/models \
     -H "Authorization: Bearer a_default_key_for_testing"
```

**成功响应示例**

```json
{
    "object": "list",
    "data": [
        {
            "id": "latest_checkpoint.pt",
            "object": "model",
            "owned_by": "user",
            "permission": []
        }
    ]
}
```

### POST /v1/chat/completions

文本生成接口，完全兼容 OpenAI Chat Completions API。

**请求 Body 参数**

| 参数                | 类型      | 是否必需 | 默认值  | 描述                                       |
| :------------------ | :-------- | :------- | :------ | :----------------------------------------- |
| `model`             | `string`  | **是**   | -       | 要使用的模型名称                           |
| `messages`          | `array`   | **是**   | -       | 对话消息数组                               |
| `max_tokens`        | `integer` | 否       | `150`   | 最大生成 token 数，范围：1-512             |
| `temperature`       | `float`   | 否       | `0.7`   | 随机度，范围：0.1-2.0                      |
| `top_k`             | `integer` | 否       | `50`    | Top-K 采样，范围：1-100                    |
| `top_p`             | `float`   | 否       | `0.9`   | 核采样阈值，范围：0.0-1.0                  |
| `presence_penalty`  | `float`   | 否       | `0.0`   | 话题重复惩罚，范围：-2.0 到 2.0            |
| `frequency_penalty` | `float`   | 否       | `0.0`   | 词频重复惩罚，范围：-2.0 到 2.0            |
| `stream`            | `boolean` | 否       | `false` | 是否使用流式响应（支持 LobeChat 等客户端） |

**示例命令**

```bash
# 基础聊天请求
curl -X POST http://localhost:5002/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer a_default_key_for_testing" \
     -d '{
           "model": "latest_checkpoint.pt",
           "messages": [
             {
               "role": "user",
               "content": "Hello, world!"
             }
           ],
           "max_tokens": 100,
           "temperature": 0.7
         }'

# 使用高级采样参数
curl -X POST http://localhost:5002/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer a_default_key_for_testing" \
     -d '{
           "model": "latest_checkpoint.pt",
           "messages": [
             {
               "role": "user",
               "content": "写一篇关于人工智能的文章"
             }
           ],
           "max_tokens": 200,
           "temperature": 0.8,
           "top_k": 50,
           "top_p": 0.95,
           "presence_penalty": 0.1,
           "frequency_penalty": 0.1
         }'

# 流式响应（适用于 LobeChat）
curl -X POST http://localhost:5002/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer a_default_key_for_testing" \
     -d '{
           "model": "latest_checkpoint.pt",
           "messages": [{"role": "user", "content": "你好"}],
           "stream": true
         }'
```

**成功响应示例** (`200 OK`)

```json
{
    "id": "chatcmpl-5b5e6aaa-bae1-4761-8ba5-f6dd94785aea",
    "object": "chat.completion",
    "created": 1752905271,
    "model": "latest_checkpoint.pt",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm here to help you with any questions or tasks you might have. How can I assist you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 4,
        "completion_tokens": 23,
        "total_tokens": 27
    }
}
```

### GET /v1

连通性检查端点。

```bash
curl -X GET http://localhost:5002/v1
```

### GET /health

健康检查端点。

```bash
curl -X GET http://localhost:5002/health
```

**失败响应示例** (`401 Unauthorized`)

```json
{
    "error": {
        "message": "Incorrect API key provided.",
        "type": "invalid_request_error",
        "code": "invalid_api_key"
    }
}
```

## 采样参数详解

### Temperature (随机度)

-   **范围**: 0.1 - 2.0
-   **效果**: 控制生成的随机性
-   **建议**:
    -   0.1-0.5: 更确定性，适合事实性内容
    -   0.7-1.0: 平衡创造性和连贯性
    -   1.2-2.0: 高创造性，可能不太连贯

### Top-K (词汇限制)

-   **范围**: 1 - 100
-   **效果**: 只从概率最高的 K 个词中采样
-   **建议**:
    -   1-10: 非常保守
    -   20-50: 平衡多样性
    -   50+: 高多样性

### Top-P (核采样)

-   **范围**: 0.0 - 1.0
-   **效果**: 从累积概率达到 P 的词汇集合中采样
-   **建议**:
    -   0.1-0.5: 保守选择
    -   0.8-0.95: 推荐范围
    -   0.95+: 高多样性

### Presence Penalty (话题惩罚)

-   **范围**: -2.0 到 2.0
-   **效果**: 对已出现的词汇施加固定惩罚
-   **建议**:
    -   负值: 鼓励重复，增强一致性
    -   0: 无惩罚
    -   正值: 减少重复，增加新颖性

### Frequency Penalty (重复惩罚)

-   **范围**: -2.0 到 2.0
-   **效果**: 根据词汇出现频率施加惩罚
-   **建议**:
    -   负值: 鼓励常用词
    -   0: 无惩罚
    -   正值: 减少高频词，促进词汇多样性

## 网络访问配置

### 局域网访问

默认配置支持局域网访问，确保防火墙允许 5002 端口：

```bash
# macOS
sudo pfctl -f /etc/pf.conf

# Linux
sudo ufw allow 5002
```

### 与 LobeChat 集成

1. 启动服务：`python -m web.app`
2. 在 LobeChat 中添加自定义模型提供商
3. API 地址：`http://server-ip:5002/v1`
4. API Key：`a_default_key_for_testing`（或自定义）
5. 支持流式响应，实时显示生成内容

## 故障排除

### 常见问题

1. **CORS 错误**: 已内置 CORS 支持，无需额外配置
2. **模型加载慢**: 首次加载需要时间，后续使用缓存
3. **内存不足**: 调整模型参数或使用更小的模型
4. **生成质量差**: 调整采样参数，特别是 temperature 和 top_p

### 性能优化

-   使用模型缓存避免重复加载
-   合理设置 max_tokens 控制生成长度
-   在生产环境使用 waitress 而非 Flask 开发服务器
