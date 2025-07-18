# 部署指南

## 快速开始

### 本地开发

该方式使用 Flask 内置的开发服务器，非常适合在本地进行代码调试和测试。

```bash
# 启动 Flask 开发服务 (使用默认的测试 API Key)
python web/app.py

# 你也可以通过环境变量指定模型和端口
MODEL_CHECKPOINT=log/model_40000.pt python web/app.py
```

服务启动后，你可以通过浏览器访问 `http://localhost:5002` 查看前端页面，并通过 API 工具调用接口。

### 生产部署 (推荐)

该方式使用 `waitress` 服务器，它比 Flask 自带的服务器更稳定、性能更好，适合在生产环境或需要对外提供服务时使用。

1.  **安装 waitress**:

    ```bash
    pip install waitress
    ```

2.  **创建 `run.py` 启动脚本**:
    在你的项目根目录下创建一个 `run.py` 文件，并填入以下内容：

    ```python
    # run.py
    from waitress import serve
    from web.app import app, load_model

    print("正在加载模型，请稍候...")
    with app.app_context():
        load_model(app)
    print("模型加载完毕。")

    print("启动 Waitress 服务器，监听 http://0.0.0.0:5002")
    serve(app, host='0.0.0.0', port=5002)
    ```

3.  **设置生产环境 API Key 并启动服务**:

    ```bash
    # 设置一个安全的 API Key，然后启动服务
    export MY_API_KEY='your_super_secret_production_api_key'
    python run.py
    ```

## API 接口

### 认证 (Authentication)

所有对 `/generate` 接口的请求都必须包含一个有效的 API Key。请在 HTTP 请求头中提供该 Key。

| Header      | 类型     | 描述                |
| :---------- | :------- | :------------------ |
| `X-API-Key` | `string` | 你的 API 访问密钥。 |

### POST /generate

文本生成接口。

**请求 Body 参数**

| 参数          | 类型      | 是否必需 | 默认值 | 描述                                                             |
| :------------ | :-------- | :------- | :----- | :--------------------------------------------------------------- |
| `prompt`      | `string`  | **是**   | `''`   | 作为生成起点的提示文本，不能为空。                               |
| `max_length`  | `integer` | 否       | `150`  | 要生成的最大 token 数量。有效范围：1 到 512。                    |
| `temperature` | `float`   | 否       | `0.7`  | 控制生成文本的随机性。值越高，随机性越强。有效范围：0.0 到 2.0。 |
| `top_k`       | `integer` | 否       | `50`   | 在每一步生成时，只从概率最高的 `k` 个词中进行采样。              |

**示例命令**

```bash
# 将 a_default_key_for_testing 替换为你的 API Key
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
```

**成功响应示例** (`200 OK`)

```json
{
  "status": "success",
  "data": {
    "prompt": "你好，世界",
    "generated_text": "你好，世界，这是一个美好的日子..."
  }
}
```

**失败响应示例** (`401 Unauthorized`)

```json
{
  "status": "error",
  "error": {
    "type": "auth_error",
    "message": "Unauthorized. Invalid or missing API Key."
  }
}
```
