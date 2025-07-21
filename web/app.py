# /web/app.py
import os
import glob
import logging
import time
import uuid
import json
import platform
import psutil
import torch

from functools import wraps

import tiktoken
from flask import Flask, request, jsonify, render_template, Response
# from flask_cors import CORS  # 使用手动CORS处理

from model.train_gpt2 import GPT, GPTConfig

# --- 应用初始化 ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.logger.setLevel(logging.INFO)

# --- 全局变量用于存储系统信息 ---
SYSTEM_INFO = {}

def collect_system_info():
    """收集服务器硬件和系统信息"""
    info = {}
    # CPU 信息
    info['cpu_count'] = psutil.cpu_count(logical=True)
    info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
    
    # 尝试通过 psutil 获取 CPU 频率
    cpu_freq_str = 'N/A'
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            # psutil.cpu_freq() 返回的频率单位是 MHz
            current_freq_mhz = cpu_freq.current
            if current_freq_mhz >= 1000:
                cpu_freq_str = f"{current_freq_mhz / 1000:.2f} GHz"
            else:
                cpu_freq_str = f"{current_freq_mhz:.0f} MHz"
        else:
            app.logger.warning("psutil.cpu_freq() 返回 None，无法获取 CPU 频率。")
    except Exception as e:
        app.logger.warning(f"无法通过 psutil 获取 CPU 频率信息: {e}")
        cpu_freq_str = 'N/A'
    
    info['cpu_freq'] = cpu_freq_str
    
    # 内存信息
    mem = psutil.virtual_memory()
    info['total_memory_gb'] = round(mem.total / (1024**3), 2)
    info['available_memory_gb'] = round(mem.available / (1024**3), 2)
    
    # GPU 信息 (如果可用)
    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        info['gpu_memory_allocated_gb'] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        info['gpu_memory_cached_gb'] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
    else:
        info['gpu_count'] = 0
        info['gpu_name'] = 'N/A'
    
    # 操作系统信息
    info['os_system'] = platform.system()
    info['os_release'] = platform.release()
    info['os_version'] = platform.version()
    info['python_version'] = platform.python_version()
    
    app.logger.info(f"收集到的系统信息: {info}")
    return info

# 在应用启动时收集系统信息
SYSTEM_INFO = collect_system_info()

@app.route('/v1/system_info', methods=['GET'])
def get_system_info():
    """返回服务器硬件和系统信息"""
    return jsonify(SYSTEM_INFO)

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
MODELS_DIR = ["./logs_finetune", "./log"]

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
    # 移除模型名称中的备注，以便找到实际文件
    original_model_name = model_name.replace(" (预训练)", "").replace(" (微调)", "")
    
    if original_model_name in MODEL_CACHE: 
        app.logger.info(f"从缓存加载模型 '{original_model_name}'")
        return MODEL_CACHE[original_model_name]
    
    app.logger.info(f"从磁盘加载模型 '{original_model_name}'...")
    checkpoint_path = find_model_full_path(original_model_name)
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
        model_info = {} # 使用字典来存储模型信息，键为模型名称，值为带备注的名称
        for directory in MODELS_DIR:
            if not os.path.isdir(directory): 
                app.logger.warning(f"模型目录 '{directory}' 未找到，将被跳过。")
                continue
            search_pattern = os.path.join(directory, '**', '*.pt')
            for path in glob.glob(search_pattern, recursive=True):
                model_basename = os.path.basename(path)
                if directory == "./log":
                    display_name = f"{model_basename} (预训练)"
                elif directory == "./logs_finetune":
                    display_name = f"{model_basename} (微调)"
                else:
                    display_name = model_basename # 默认情况，不添加备注
                model_info[model_basename] = display_name # 存储原始名称和显示名称的映射
        
        # 构建模型列表，使用带备注的名称作为id
        model_list = [{"id": display_name, "object": "model", "owned_by": "user", "permission": []} for original_name, display_name in sorted(model_info.items())]
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
                start_time = time.time()
                
                # 发送开始chunk
                start_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None
                    }],
                    "usage": {
                        "prompt_tokens": len(tokens[0]),
                        "completion_tokens": 0,
                        "total_tokens": len(tokens[0]),
                        "elapsed_time": 0.0,
                        "tokens_per_second": 0.0
                    }
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
                            
                            # 计算性能指标
                            current_time = time.time()
                            elapsed_time = current_time - start_time
                            completion_tokens_count = len(stream_tokens)
                            tokens_per_second = completion_tokens_count / elapsed_time if elapsed_time > 0 else 0

                            # 尝试解码新token
                            try:
                                token_text = assets['tokenizer'].decode([new_token_id])
                                chunk_data = {
                                    "id": f"chatcmpl-{uuid.uuid4()}",
                                    "object": "chat.completion.chunk",
                                    "created": int(current_time),
                                    "model": model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": token_text},
                                        "finish_reason": None
                                    }],
                                    "usage": {
                                        "prompt_tokens": len(tokens[0]),
                                        "completion_tokens": completion_tokens_count,
                                        "total_tokens": len(tokens[0]) + completion_tokens_count,
                                        "elapsed_time": round(elapsed_time, 4),
                                        "tokens_per_second": round(tokens_per_second, 2)
                                    }
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
                                            "created": int(current_time),
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": accumulated_text},
                                                "finish_reason": None
                                            }],
                                            "usage": {
                                                "prompt_tokens": len(tokens[0]),
                                                "completion_tokens": completion_tokens_count,
                                                "total_tokens": len(tokens[0]) + completion_tokens_count,
                                                "elapsed_time": round(elapsed_time, 4),
                                                "tokens_per_second": round(tokens_per_second, 2)
                                            }
                                        }
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                        # stream_tokens = []  # 清空已发送的tokens
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
                        "total_tokens": len(tokens[0]) + len(stream_tokens),
                        "elapsed_time": round(time.time() - start_time, 4),
                        "tokens_per_second": round(len(stream_tokens) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0, 2)
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
            start_time = time.time()
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
            end_time = time.time()
            
            # 只解码新生成的token，不包括输入的prompt
            new_tokens = generated_tokens[0][len(tokens[0]):]
            generated_text = assets['tokenizer'].decode(new_tokens.tolist())

            # 清理生成的文本，确保格式正确
            generated_text = generated_text.strip()
            
            # 如果生成的文本为空，提供默认回复
            if not generated_text:
                generated_text = "抱歉，我无法生成有意义的回复。请尝试不同的提示词。"
            
            elapsed_time = end_time - start_time
            completion_tokens_count = len(new_tokens)
            tokens_per_second = completion_tokens_count / elapsed_time if elapsed_time > 0 else 0

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
                    "completion_tokens": completion_tokens_count,
                    "total_tokens": len(tokens[0]) + completion_tokens_count,
                    "elapsed_time": round(elapsed_time, 4),
                    "tokens_per_second": round(tokens_per_second, 2)
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
