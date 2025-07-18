# /web/app.py
from flask import Flask, request, jsonify, render_template, current_app, Response
from functools import wraps
import torch
import tiktoken
from model.train_gpt2 import GPT
import os

# --- 应用初始化 ---
app = Flask(__name__, template_folder='templates', static_folder='static')


# --- 模型加载函数 ---
def load_model(flask_app):
    """
    加载模型和 tokenizer，并将其附加到 Flask app 实例。
    """
    print("开始为当前 worker 加载模型...")
    # 在 macOS 上，通常没有专用的 CUDA GPU，所以强制使用 CPU
    device = 'cpu'
    print(f"检测到 macOS，强制使用设备: {device}")
    
    checkpoint_path = os.environ.get('MODEL_CHECKPOINT', 'logs_finetune/latest_checkpoint.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型 checkpoint 未找到: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    
    enc = tiktoken.get_encoding('gpt2')
    
    flask_app.config['MODEL'] = model
    flask_app.config['TOKENIZER'] = enc
    flask_app.config['DEVICE'] = device
    
    print(f"模型加载成功，使用设备: {device}")

# --- API Key 认证 ---

# 1. 从环境变量获取有效的 API Key，如果不存在，则使用一个默认值用于测试。
#    在生产环境中，你应该通过 `export MY_API_KEY='your_real_api_key'` 来设置它。
VALID_API_KEY = os.environ.get("MY_API_KEY", "a_default_key_for_testing")

# 2. 创建一个装饰器来检查 API Key
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 约定 API Key 通过 X-API-Key 这个请求头传递
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != VALID_API_KEY:
            return jsonify({
                "status": "error",
                "error": {"type": "auth_error", "message": "Unauthorized. Invalid or missing API Key."}
            }), 401 # 401 Unauthorized
        return f(*args, **kwargs)
    return decorated_function


# --- 生产级 API 端点 ---

@app.route('/generate', methods=['POST'])
@require_api_key  # <-- 在这里应用 API Key 认证装饰器
def generate():
    """
    生成文本的生产级 API 端点，包含输入验证和统一的响应格式。
    """
    # 检查模型是否已加载
    if 'MODEL' not in current_app.config:
        return jsonify({
            "status": "error",
            "error": {"type": "server_error", "message": "Model is not loaded in this worker."}
        }), 503 # 503 Service Unavailable

    # 1. 检查 content type 是否为 json
    if not request.is_json:
        return jsonify({
            "status": "error",
            "error": {"type": "invalid_request", "message": "Content-Type must be application/json"}
        }), 400 # 400 Bad Request

    data = request.get_json()

    # 2. 验证必需的字段和类型
    if not data or 'prompt' not in data or not isinstance(data.get('prompt'), str) or not data['prompt'].strip():
        return jsonify({
            "status": "error",
            "error": {"type": "validation_error", "message": "Field 'prompt' is required and must be a non-empty string."}
        }), 400

    # 3. 获取并验证可选字段
    try:
        prompt = data['prompt']
        # 提供默认值，并进行类型和范围校验
        max_length = int(data.get('max_length', 150))
        temperature = float(data.get('temperature', 0.7))
        top_k = int(data.get('top_k', 50))

        if not (0 < max_length <= 512):
            raise ValueError("Field 'max_length' must be an integer between 1 and 512.")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("Field 'temperature' must be a float between 0.0 and 2.0.")

    except (ValueError, TypeError) as e:
        return jsonify({
            "status": "error",
            "error": {"type": "validation_error", "message": str(e)}
        }), 400

    # --- 模型推理核心逻辑 ---
    try:
        model = current_app.config['MODEL']
        enc = current_app.config['TOKENIZER']
        device = current_app.config['DEVICE']
        
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        tokens = tokens.unsqueeze(0)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                tokens, 
                max_new_tokens=max_length, 
                temperature=temperature, 
                top_k=top_k
            )
            generated_text = enc.decode(generated_tokens[0].tolist())
        
        # 4. 返回统一的成功响应
        return jsonify({
            "status": "success",
            "data": {
                "prompt": prompt,
                "generated_text": generated_text
            }
        })
    
    except Exception as e:
        # 捕获推理过程中可能发生的任何其他错误
        print(f"推理过程中发生错误: {e}")
        return jsonify({
            "status": "error",
            "error": {"type": "server_error", "message": "An unexpected error occurred during model inference."}
        }), 500 # 500 Internal Server Error


@app.route('/favicon.ico')
def favicon():
    # 返回一个 204 No Content 响应，告诉浏览器这里没有图标，从而优雅地结束请求。
    return Response(status=204)

@app.route("/")
def hello():
    # 假设你的 web/templates/ 目录下有一个 index.html 文件
    return render_template('index.html') 

# --- 应用启动 (用于本地开发) ---
if __name__ == '__main__':
    print("以开发模式启动 Flask 服务器...")
    with app.app_context():
        load_model(app)
    app.run(host='0.0.0.0', port=5002, debug=True)
