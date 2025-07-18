# /web/app.py
import os
import glob
import logging
import time
import uuid
import json
from functools import wraps

import torch
import tiktoken
from flask import Flask, request, jsonify, render_template, Response
# from flask_cors import CORS  # 使用手动CORS处理

from model.train_gpt2 import GPT, GPTConfig

# --- 应用初始化 ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.logger.setLevel(logging.INFO)

# --- CORS处理 ---
@app.before_request
def handle_preflight():
    app.logger.info(f"收到请求: {request.method} {request.url} 来自 {request.remote_addr}")
    if request.method == "OPTIONS":
        app.logger.info(f"处理预检请求: {request.url}")
        response = Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Max-Age"] = "86400"
        return response

@app.after_request  
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"  
    response.headers["Access-Control-Allow-Methods"] = "*"
    return response 

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
    if model_name in MODEL_CACHE: 
        app.logger.info(f"从缓存加载模型 '{model_name}'")
        return MODEL_CACHE[model_name]
    
    app.logger.info(f"从磁盘加载模型 '{model_name}'...")
    checkpoint_path = find_model_full_path(model_name)
    if not checkpoint_path: 
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
    app.logger.info(f"模型 '{model_name}' 加载并缓存成功，缓存大小: {len(MODEL_CACHE)}")
    return model_assets

# --- OpenAI 兼容 API 端点 ---

@app.route('/v1', methods=['GET', 'POST', 'HEAD'])
def v1_health_check():
    app.logger.info(f"连通性检查请求: {request.method} {request.url}")
    app.logger.info(f"请求头: {dict(request.headers)}")
    
    # 返回简单的健康状态，符合大多数OpenAI客户端期望
    response = jsonify({
        "status": "ok",
        "version": "v1"
    })
    app.logger.info(f"返回响应: {response.get_data(as_text=True)}")
    return response

# 添加更多可能的连通性检查端点
@app.route('/', methods=['HEAD'])
def head_root():
    return Response(status=200)

@app.route('/v1/engines', methods=['GET'])
def list_engines():
    return jsonify({"data": [], "object": "list"})

@app.route('/v1/health', methods=['GET'])
def v1_health():
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route('/v1/models', methods=['GET'])
@require_api_key
def list_models():
    """
    提供与 OpenAI /v1/models 兼容的模型列表。
    """
    try:
        app.logger.info(f"模型列表请求来自: {request.remote_addr}")
        model_basenames = set()
        for directory in MODELS_DIR:
            if not os.path.isdir(directory): 
                app.logger.warning(f"模型目录 '{directory}' 未找到，将被跳过。")
                continue
            search_pattern = os.path.join(directory, '**', '*.pt')
            for path in glob.glob(search_pattern, recursive=True):
                model_basenames.add(os.path.basename(path))
        
        model_list = [{"id": name, "object": "model", "owned_by": "user", "permission": []} for name in sorted(list(model_basenames))]
        app.logger.info(f"返回模型列表: {len(model_list)} 个模型给 {request.remote_addr}")
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
        app.logger.info(f"聊天请求来自: {request.remote_addr}")
        app.logger.info(f"请求头: {dict(request.headers)}")
        
        data = request.get_json()
        app.logger.info(f"请求数据: {data}")
        
        model_name = data.get('model')
        messages = data.get('messages')
        stream = data.get('stream', False)  # 检查是否需要流式响应
        
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
        top_k = int(data.get('top_k', 50))
        top_p = float(data.get('top_p', 0.9))
        presence_penalty = float(data.get('presence_penalty', 0.0))
        frequency_penalty = float(data.get('frequency_penalty', 0.0))
        
        # 确保参数在合理范围内，避免数值问题
        temperature = max(0.1, min(temperature, 2.0))
        top_k = max(1, min(top_k, 100))
        top_p = max(0.0, min(top_p, 1.0))
        presence_penalty = max(-2.0, min(presence_penalty, 2.0))
        frequency_penalty = max(-2.0, min(frequency_penalty, 2.0))
        
        assets = get_model(model_name)
        tokens = assets['tokenizer'].encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=assets['device']).unsqueeze(0)
        
        # 根据请求类型返回不同格式的响应
        if stream:
            # 真正的流式生成
            def generate_stream():
                import json
                import time
                
                # 用于收集流式生成的tokens
                stream_tokens = []
                
                def stream_callback(token_id, step):
                    """流式生成回调函数"""
                    stream_tokens.append(token_id)
                    # 解码当前token
                    try:
                        token_text = assets['tokenizer'].decode([token_id])
                        # 发送增量内容
                        chunk_data = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token_text},
                                "finish_reason": None
                            }]
                        }
                        return f"data: {json.dumps(chunk_data)}\n\n"
                    except:
                        # 如果单个token无法解码，跳过
                        return ""
                
                # 开始流式生成
                start_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(start_chunk)}\n\n"
                
                # 执行真正的流式生成
                with torch.no_grad():
                    current_tokens = tokens.clone()
                    for step in range(max_tokens):
                        # 生成单个token
                        generated_tokens = assets['model'].generate(
                            current_tokens,
                            max_new_tokens=1,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            stream=True,
                            stream_callback=None  # 我们手动处理流式输出
                        )
                        
                        # 获取新生成的token
                        new_token = generated_tokens[0, -1:].tolist()
                        if new_token:
                            new_token_id = new_token[0]
                            stream_tokens.append(new_token_id)
                            
                            # 尝试解码新token
                            try:
                                token_text = assets['tokenizer'].decode([new_token_id])
                                chunk_data = {
                                    "id": f"chatcmpl-{uuid.uuid4()}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": token_text},
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                            except:
                                # 某些token可能无法单独解码，累积几个再试
                                if len(stream_tokens) >= 3:
                                    try:
                                        accumulated_text = assets['tokenizer'].decode(stream_tokens[-3:])
                                        chunk_data = {
                                            "id": f"chatcmpl-{uuid.uuid4()}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": accumulated_text},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                        stream_tokens = []  # 清空已发送的tokens
                                    except:
                                        pass
                        
                        # 更新当前token序列
                        current_tokens = generated_tokens
                        
                        # 检查是否生成了结束token或达到最大长度
                        if step >= max_tokens - 1:
                            break
                
                # 发送结束chunk
                final_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(tokens[0]),
                        "completion_tokens": len(stream_tokens),
                        "total_tokens": len(tokens[0]) + len(stream_tokens)
                    }
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(
                generate_stream(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': '*',
                    'Access-Control-Allow-Methods': '*'
                }
            )
        else:
            # 非流式生成（原有逻辑）
            with torch.no_grad():
                generated_tokens = assets['model'].generate(
                    tokens, 
                    max_new_tokens=max_tokens, 
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty
                )
            # 只解码新生成的token，不包括输入的prompt
            new_tokens = generated_tokens[0][len(tokens[0]):]
            generated_text = assets['tokenizer'].decode(new_tokens.tolist())

            # 清理生成的文本，确保格式正确
            generated_text = generated_text.strip()
            
            # 如果生成的文本为空，提供默认回复
            if not generated_text:
                generated_text = "抱歉，我无法生成有意义的回复。请尝试不同的提示词。"
            
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
                    "completion_tokens": len(new_tokens),
                    "total_tokens": len(tokens[0]) + len(new_tokens)
                }
            }
            
            app.logger.info(f"生成参数 - max_tokens: {max_tokens}, temperature: {temperature}, top_k: {top_k}, top_p: {top_p}, presence_penalty: {presence_penalty}, frequency_penalty: {frequency_penalty}")
            app.logger.info(f"生成响应 - prompt: '{prompt}', response: '{generated_text}'")
            app.logger.info(f"响应长度: {len(generated_text)} 字符")
            app.logger.info(f"流式响应: {stream}")
            
            # 返回非流式响应（原有格式）
            response_obj = jsonify(response)
            response_obj.headers['Content-Type'] = 'application/json; charset=utf-8'
            response_obj.headers['Content-Length'] = str(len(response_obj.get_data()))
            return response_obj

    except FileNotFoundError as e:
        return jsonify({"error": {"message": str(e), "type": "invalid_request_error", "code": "model_not_found"}}), 404
    except Exception as e:
        app.logger.error(f"推理过程中发生错误: {e}", exc_info=True)
        return jsonify({"error": {"message": "服务器内部错误。", "type": "server_error"}}), 500

# --- Web 界面路由 ---
@app.route("/")
def hello():
    return render_template('index.html') 

# --- 应用启动 ---
if __name__ == '__main__':
    print("以开发模式启动 OpenAI 兼容的 Flask 服务器...")
    app.run(host='0.0.0.0', port=5002, debug=True)
