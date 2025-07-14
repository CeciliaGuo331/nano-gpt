from flask import Flask, request, jsonify, render_template 
import torch
import tiktoken
from model.train_gpt2 import GPT
import os

app = Flask(__name__)

# 全局变量存储模型
model = None
enc = None
device = None

def load_model():
    """加载模型（只在启动时执行一次）"""
    global model, enc, device
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = os.environ.get('MODEL_CHECKPOINT', 'logs_finetune/latest_checkpoint.pt')
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only = False)
    config = checkpoint['config']
    
    # 创建并加载模型
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    
    # 初始化 tokenizer
    enc = tiktoken.get_encoding('gpt2')
    
    print(f"模型加载成功，使用设备: {device}")

@app.route('/generate', methods=['POST'])
def generate():
    """生成文本的 API 端点"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.8)
        top_k = data.get('top_k', 50)
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # 编码输入
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        tokens = tokens.unsqueeze(0)
        
        # 生成文本
        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = model(tokens)
                logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat((tokens, next_token), dim=1)
        
        # 解码结果
        generated_text = enc.decode(tokens[0].tolist())
        
        return jsonify({
            'prompt': prompt,
            'generated_text': generated_text,
            'tokens_generated': len(tokens[0]) - len(enc.encode(prompt))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查端点"""
    return jsonify({'status': 'healthy', 'device': device})

@app.route("/")
def hello():
    return render_template('index.html') 

if __name__ == '__main__':
    # 启动时加载模型
    load_model()
    
    # 启动服务
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug = True)