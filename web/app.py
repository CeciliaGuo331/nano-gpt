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
app.logger.setLevel(logging.INFO)
CORS(app) 

# --- 模型缓存和目录配置 ---
MODEL_CACHE = {}
MODELS_DIR = ["./logs_finetune", "./log", "./log_backup"]

# --- API Key 认证 ---
VALID_API_KEY = os.environ.get("MY_API_KEY", "a_default_key_for_testing")

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return Response(status=200)
        
        auth_header = request.headers.get('Authorization')
        api_key = None
        if auth_header and auth_header.startswith('Bearer '):
            api_key = auth_header.split(' ')[1]
        else:
            api_key = request.headers.get('X-API-Key')

        if not api_key or api_key != VALID_API_KEY:
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

@app.route('/v1/models', methods=['GET'])
@require_api_key
def list_models():
    """
    提供与 OpenAI /v1/models 兼容的模型列表。
    """
    try:
        model_basenames = set()
        for directory in MODELS_DIR:
            if not os.path.isdir(directory): continue
            search_pattern = os.path.join(directory, '**', '*.pt')
            for path in glob.glob(search_pattern, recursive=True):
                model_basenames.add(os.path.basename(path))
        
        model_list = [{"id": name, "object": "model", "owned_by": "user", "permission": []} for name in sorted(list(model_basenames))]
        app.logger.info(f"返回模型列表: {model_list}")
        return jsonify({"object": "list", "data": model_list})
    except Exception as e:
        app.logger.error(f"列出模型时出错: {e}")
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

# --- Web 界面路由 (已修复) ---
@app.route("/")
def hello():
    """
    !!! 关键改动: 恢复渲染 index.html !!!
    """
    return render_template('index.html') 

# --- 应用启动 ---
if __name__ == '__main__':
    print("以开发模式启动 OpenAI 兼容的 Flask 服务器...")
    app.run(host='0.0.0.0', port=5002, debug=True)
