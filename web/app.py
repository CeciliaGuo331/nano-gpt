# /web/app.py
import os
import glob
import logging
from functools import wraps

import torch
import tiktoken
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS # 关键改动 1: 导入 CORS

from model.train_gpt2 import GPT, GPTConfig

# --- 应用初始化 ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.logger.setLevel(logging.INFO)
# 关键改动 2: 为整个应用启用 CORS，允许所有来源的跨域请求
CORS(app) 

# --- 模型缓存和目录配置 ---
MODEL_CACHE = {}
# 关键改动 3: 将目录配置改为列表，以支持多个位置
MODELS_DIR = ["./logs_finetune", "./log", "./log_backup"]  # 存放所有 .pt 模型文件的目录

# --- API Key 认证 ---
VALID_API_KEY = os.environ.get("MY_API_KEY", "a_default_key_for_testing")

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 对 OPTIONS 预检请求直接放行
        if request.method == 'OPTIONS':
            return f(*args, **kwargs)
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != VALID_API_KEY:
            return jsonify({
                "status": "error",
                "error": {"type": "auth_error", "message": "Unauthorized. Invalid or missing API Key."}
            }), 401
        return f(*args, **kwargs)
    return decorated_function

# --- 新增辅助函数：用于在所有目录中查找模型文件的完整路径 ---
def find_model_full_path(model_name):
    """在所有配置的目录中递归查找模型文件的完整路径"""
    for directory in MODELS_DIR:
        if not os.path.isdir(directory):
            continue
        # 使用 glob 递归搜索
        search_pattern = os.path.join(directory, '**', model_name)
        found_paths = glob.glob(search_pattern, recursive=True)
        if found_paths:
            return found_paths[0]  # 返回找到的第一个匹配项
    return None # 如果在所有目录中都找不到，返回 None

# --- 模型加载与缓存逻辑 (已更新) ---
def get_model(model_name):
    """
    从缓存中获取模型，如果不存在则从磁盘加载并缓存。
    """
    if model_name in MODEL_CACHE:
        app.logger.info(f"从缓存返回模型 '{model_name}'")
        return MODEL_CACHE[model_name]
    
    app.logger.info(f"从磁盘加载模型 '{model_name}'...")
    
    # 关键改动 4: 使用新的辅助函数查找模型的完整路径
    checkpoint_path = find_model_full_path(model_name)
    
    if not checkpoint_path:
        # 如果 find_model_full_path 返回 None，说明文件不存在
        raise FileNotFoundError(f"在配置的目录中未找到模型文件: {model_name}")
    
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

# --- API 端点 (已更新) ---

@app.route('/models', methods=['GET'])
@require_api_key
def list_models():
    """
    列出在所有 MODELS_DIR 目录中递归找到的所有可用模型文件。
    """
    try:
        model_basenames = set()
        # 关键改动 5: 遍历所有配置的目录
        for directory in MODELS_DIR:
            if not os.path.isdir(directory):
                app.logger.warning(f"模型目录 '{directory}' 未找到，将被跳过。")
                continue
            
            # 使用 glob 递归查找所有 .pt 文件
            search_pattern = os.path.join(directory, '**', '*.pt')
            model_full_paths = glob.glob(search_pattern, recursive=True)
            
            # 将找到的模型的 *文件名* 添加到集合中以去重
            for path in model_full_paths:
                model_basenames.add(os.path.basename(path))

        # 返回排序后的列表，保持接口格式不变
        return jsonify({"status": "success", "data": {"models": sorted(list(model_basenames))}})
    except Exception as e:
        app.logger.error(f"列出模型时出错: {e}")
        return jsonify({"status": "error", "error": {"type": "server_error", "message": "无法列出模型。" }}), 500

@app.route('/generate', methods=['POST'])
@require_api_key
def generate():
    """
    根据指定的模型生成文本。
    (此函数逻辑未改变，但现在依赖于更新后的 get_model)
    """
    if not request.is_json:
        return jsonify({"status": "error", "error": {"type": "invalid_request", "message": "Content-Type must be application/json"}}), 400

    data = request.get_json()
    
    model_name = data.get('model_name')
    if not model_name or not isinstance(model_name, str):
        return jsonify({"status": "error", "error": {"type": "validation_error", "message": "Field 'model_name' is required."}}), 400

    try:
        prompt = data.get('prompt', '')
        if not prompt.strip():
            raise ValueError("Field 'prompt' must be a non-empty string.")
        max_length = int(data.get('max_length', 150))
        temperature = float(data.get('temperature', 0.7))
        top_k = int(data.get('top_k', 50))
    except (ValueError, TypeError) as e:
        return jsonify({"status": "error", "error": {"type": "validation_error", "message": str(e)}}), 400

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
    # 确保在运行前安装 flask-cors: pip install flask-cors
    print("以开发模式启动 Flask 服务器...")
    app.run(host='0.0.0.0', port=5002, debug=True)
