# /web/app.py
from flask import Flask, request, jsonify, render_template, current_app, Response
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

# --- API 端点 ---
@app.route('/generate', methods=['POST'])
def generate():
    """生成文本的 API 端点"""
    try:
        model = current_app.config['MODEL']
        enc = current_app.config['TOKENIZER']
        device = current_app.config['DEVICE']

        data = request.json
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.8)
        top_k = data.get('top_k', 50)
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
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
        
        return jsonify({
            'prompt': prompt,
            'generated_text': generated_text,
            'tokens_generated': len(generated_tokens[0]) - len(tokens[0])
        })
    
    except Exception as e:
        print(f"生成过程中发生错误: {e}")
        return jsonify({'error': str(e)}), 500

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
