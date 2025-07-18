# /web/app.py
from flask import Flask, request, jsonify, render_template, current_app, Response
from functools import wraps
import torch
import tiktoken
from model.train_gpt2 import GPT
import os
import logging

# --- 应用初始化 ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.logger.setLevel(logging.INFO)

# --- 模型缓存和目录配置 ---
MODEL_CACHE = {}
MODELS_DIR = "logs_finetune" # 存放所有 .pt 模型文件的目录

# --- API Key 认证 ---
VALID_API_KEY = os.environ.get("MY_API_KEY", "a_default_key_for_testing")

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != VALID_API_KEY:
            return jsonify({
                "status": "error",
                "error": {"type": "auth_error", "message": "Unauthorized. Invalid or missing API Key."}
            }), 401
        return f(*args, **kwargs)
    return decorated_function

# --- 模型加载与缓存逻辑 ---
def get_model(model_name):
    """
    从缓存中获取模型，如果不存在则从磁盘加载并缓存。
    """
    # 检查缓存
    if model_name in MODEL_CACHE:
        app.logger.info(f"从缓存返回模型 '{model_name}'")
        return MODEL_CACHE[model_name]
    
    # 从磁盘加载
    app.logger.info(f"从磁盘加载模型 '{model_name}'...")
    device = 'cpu'
    checkpoint_path = os.path.join(MODELS_DIR, model_name)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件未找到: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    enc = tiktoken.get_encoding('gpt2')
    
    # 存入缓存
    model_assets = {'model': model, 'tokenizer': enc, 'device': device}
    MODEL_CACHE[model_name] = model_assets
    app.logger.info(f"模型 '{model_name}' 加载并缓存成功")
    return model_assets

# --- API 端点 ---

@app.route('/models', methods=['GET'])
@require_api_key
def list_models():
    """
    列出在 MODELS_DIR 目录中所有可用的模型文件。
    """
    try:
        if not os.path.isdir(MODELS_DIR):
             app.logger.warning(f"模型目录 '{MODELS_DIR}' 未找到。")
             return jsonify({"status": "success", "data": {"models": []}})
        
        model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')])
        return jsonify({"status": "success", "data": {"models": model_files}})
    except Exception as e:
        app.logger.error(f"列出模型时出错: {e}")
        return jsonify({"status": "error", "error": {"type": "server_error", "message": "无法列出模型。" }}), 500

@app.route('/generate', methods=['POST'])
@require_api_key
def generate():
    """
    根据指定的模型生成文本。
    """
    if not request.is_json:
        return jsonify({"status": "error", "error": {"type": "invalid_request", "message": "Content-Type must be application/json"}}), 400

    data = request.get_json()
    
    # 验证模型名称
    model_name = data.get('model_name')
    if not model_name or not isinstance(model_name, str):
        return jsonify({"status": "error", "error": {"type": "validation_error", "message": "Field 'model_name' is required."}}), 400

    # 验证其他参数...
    try:
        prompt = data.get('prompt', '')
        if not prompt.strip():
            raise ValueError("Field 'prompt' must be a non-empty string.")
        max_length = int(data.get('max_length', 150))
        temperature = float(data.get('temperature', 0.7))
        top_k = int(data.get('top_k', 50))
    except (ValueError, TypeError) as e:
        return jsonify({"status": "error", "error": {"type": "validation_error", "message": str(e)}}), 400

    # --- 模型推理核心逻辑 ---
    try:
        assets = get_model(model_name)
        model = assets['model']
        enc = assets['tokenizer']
        device = assets['device']
        
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated_tokens = model.generate(tokens, max_new_tokens=max_length, temperature=temperature, top_k=top_k)
            generated_text = enc.decode(generated_tokens[0].tolist())
        
        return jsonify({"status": "success", "data": {"prompt": prompt, "generated_text": generated_text}})
    
    except FileNotFoundError as e:
        return jsonify({"status": "error", "error": {"type": "not_found", "message": str(e)}}), 404
    except Exception as e:
        app.logger.error(f"推理过程中发生错误: {e}", exc_info=True)
        return jsonify({"status": "error", "error": {"type": "server_error", "message": "An unexpected error occurred during model inference."}}), 500

@app.route('/favicon.ico')
def favicon():
    return Response(status=204)

@app.route("/")
def hello():
    return render_template('index.html') 

# --- 应用启动 (用于本地开发) ---
if __name__ == '__main__':
    # 注意：模型现在是按需加载的，所以启动时不再预加载任何模型。
    print("以开发模式启动 Flask 服务器...")
    app.run(host='0.0.0.0', port=5002, debug=True)
