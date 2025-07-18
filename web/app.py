# /web/app.py
import os
import glob
import logging
import time
import uuid
from functools import wraps

import torch
import tiktoken
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS

from model.train_gpt2 import GPT, GPTConfig

# --- 应用初始化 ---
app = Flask(__name__, template_folder='templates', static_folder='static')
# 将日志级别设置为DEBUG以捕获所有信息
app.logger.setLevel(logging.DEBUG) 
# 使用最明确、最宽松的CORS配置
CORS(app, 
     resources={r"/*": {"origins": "*"}}, 
     allow_headers="*", 
     methods="*",
     supports_credentials=True)

# !!! 关键改动 1: 添加一个全局的请求前处理器来记录所有进入的请求 !!!
@app.before_request
def log_each_request():
    """
    在每个请求被分派到视图函数之前，记录其详细信息。
    这是最底层的日志点，可以帮助我们捕获被 CORS 预检等机制拦截的请求。
    """
    app.logger.debug("--- [Global Before Request Log] ---")
    app.logger.debug(f"Request From: {request.remote_addr}")
    app.logger.debug(f"Request Path: {request.path}")
    app.logger.debug(f"Request Method: {request.method}")
    app.logger.debug(f"Request Headers:\n{request.headers}")
    app.logger.debug("--- [End Global Before Request Log] ---")

# !!! 关键改动 2: 添加一个全局的响应后处理器来记录响应头 !!!
@app.after_request
def after_request_func(response):
    """
    在每个响应发送回客户端之前，记录其详细信息。
    这对于调试CORS响应头至关重要。
    """
    app.logger.debug("--- [Global After Request Log] ---")
    app.logger.debug(f"Response for {request.path}:")
    app.logger.debug(f"Status: {response.status}")
    app.logger.debug(f"Headers:\n{response.headers}")
    app.logger.debug("--- [End Global After Request Log] ---")
    return response

# --- 模型缓存和目录配置 ---
MODEL_CACHE = {}
MODELS_DIR = ["./logs_finetune", "./log", "./log_backup"]

# --- API Key 认证 ---
VALID_API_KEY = os.environ.get("MY_API_KEY", "a_default_key_for_testing")

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 预检请求由 Flask-CORS 自动处理，此处不再需要检查 OPTIONS
        
        auth_header = request.headers.get('Authorization')
        api_key = None
        if auth_header and auth_header.startswith('Bearer '):
            api_key = auth_header.split(' ')[1]
        else:
            # 兼容旧的 X-API-Key
            api_key = request.headers.get('X-API-Key')

        if not api_key or api_key != VALID_API_KEY:
            app.logger.warning(f"API Key authentication failed for {request.path}. Provided Key: '{api_key}'")
            return jsonify({
                "error": {"message": "Incorrect API key provided.", "type": "invalid_request_error", "code": "invalid_api_key"}
            }), 401
        
        return f(*args, **kwargs)
    return decorated_function

# --- 模型加载逻辑 ---
def find_model_full_path(model_name):
    for directory in MODELS_DIR:
        if not os.path.isdir(directory): continue
        search_pattern = os.path.join(directory, '**', model_name)
        found_paths = glob.glob(search_pattern, recursive=True)
        if found_paths: return found_paths[0]
    return None

def get_model(model_name):
    if model_name in MODEL_CACHE: return MODEL_CACHE[model_name]
    app.logger.info(f"从磁盘加载模型 '{model_name}'...")
    checkpoint_path = find_model_full_path(model_name)
    if not checkpoint_path: raise FileNotFoundError(f"在配置的目录中未找到模型文件: {model_name}")
    app.logger.info(f"找到模型路径: {checkpoint_path}")
    device = 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    enc = tiktoken.get_encoding('gpt2')
    model_assets = {'model': model, 'tokenizer': enc, 'device': device}
    MODEL_CACHE[model_name] = model_assets
    app.logger.info(f"模型 '{model_name}' 加载并缓存成功")
    return model_assets

# --- OpenAI 兼容 API 端点 ---

# !!! 关键改动 3: 恢复对连通性检查端点的 API Key 认证 !!!
@app.route('/v1', methods=['GET', 'POST', 'HEAD'])
@require_api_key
def v1_health_check():
    """
    为 LobeChat 等客户端提供一个简单的连通性检查端点。
    """
    app.logger.info("--- [Connectivity Check Endpoint Hit & Authenticated] ---")
    app.logger.info(f"Method: {request.method}")
    app.logger.info("--- [End Connectivity Check Log] ---")
    
    # 返回一个 OpenAI 风格的空列表，以满足客户端的格式期望
    return jsonify({"object": "list", "data": []})

@app.route('/v1/models', methods=['GET'])
@require_api_key
def list_models():
    """
    提供与 OpenAI /v1/models 兼容的模型列表。
    """
    try:
        model_basenames = set()
        for directory in MODELS_DIR:
            if not os.path.isdir(directory):
                app.logger.warning(f"模型目录 '{directory}' 未找到，将被跳过。")
                continue
            
            search_pattern = os.path.join(directory, '**', '*.pt')
            for path in glob.glob(search_pattern, recursive=True):
                model_basenames.add(os.path.basename(path))
        
        model_list = [{"id": name, "object": "model", "owned_by": "user", "permission": []} for name in sorted(list(model_basenames))]
        app.logger.info(f"返回模型列表: {len(model_list)} 个模型")
        return jsonify({"object": "list", "data": model_list})
    except Exception as e:
        app.logger.error(f"列出模型时出错: {e}", exc_info=True)
        return jsonify({"error": {"message": "无法列出模型。", "type": "server_error"}}), 500

@app.route('/v1/chat/completions', methods=['POST'])
@require_api_key
def chat_completions():
    """
    提供与 OpenAI /v1/chat/completions 兼容的聊天端点。
    """
    try:
        data = request.get_json()
        model_name = data.get('model')
        messages = data.get('messages')
        
        if not model_name or not messages:
            return jsonify({"error": {"message": "请求参数 'model' 和 'messages' 是必需的。", "type": "invalid_request_error"}}), 400

        prompt = ""
        if isinstance(messages, list) and len(messages) > 0:
            last_message = messages[-1]
            if last_message.get('role') == 'user':
                prompt = last_message.get('content', '')

        if not prompt:
            return jsonify({"error": {"message": "未找到有效的用户输入。", "type": "invalid_request_error"}}), 400

        max_tokens = int(data.get('max_tokens', 150))
        temperature = float(data.get('temperature', 0.7))
        
        assets = get_model(model_name)
        tokens = assets['tokenizer'].encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=assets['device']).unsqueeze(0)
        
        with torch.no_grad():
            generated_tokens = assets['model'].generate(tokens, max_new_tokens=max_tokens, temperature=temperature)
            generated_text = assets['tokenizer'].decode(generated_tokens[0].tolist())

        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(tokens[0]),
                "completion_tokens": len(generated_tokens[0]) - len(tokens[0]),
                "total_tokens": len(generated_tokens[0])
            }
        }
        return jsonify(response)

    except FileNotFoundError as e:
        return jsonify({"error": {"message": str(e), "type": "invalid_request_error", "code": "model_not_found"}}), 404
    except Exception as e:
        app.logger.error(f"推理过程中发生错误: {e}", exc_info=True)
        return jsonify({"error": {"message": "服务器内部错误。", "type": "server_error"}}), 500

# --- Web 界面路由 ---
@app.route("/")
def hello():
    app.logger.info("--- [View Function] Rendering index.html ---")
    return render_template('index.html') 

# --- 应用启动 ---
if __name__ == '__main__':
    print("以开发模式启动 OpenAI 兼容的 Flask 服务器...")
    app.run(host='0.0.0.0', port=5002, debug=True)
