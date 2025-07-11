import torch
import tiktoken
from model import GPT, GPTConfig

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """加载保存的模型用于推理"""
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从 checkpoint 中获取模型配置
    config = checkpoint['config']
    
    # 创建模型
    model = GPT(config)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model'])
    
    # 设置为评估模式
    model.eval()
    model.to(device)
    
    return model

def generate_text(model, prompt, max_length=100, temperature=0.8, top_k=50, device='cuda'):
    """使用模型生成文本"""
    # 获取 tokenizer
    enc = tiktoken.get_encoding('gpt2')
    
    # 编码输入文本
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0)  # 添加 batch 维度
    
    # 生成文本
    with torch.no_grad():
        for _ in range(max_length):
            # 获取模型预测
            logits, _ = model(tokens)
            logits = logits[:, -1, :]  # 只取最后一个位置的输出
            
            # 应用温度
            logits = logits / temperature
            
            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # 应用 softmax 获取概率
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # 采样下一个 token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到序列中
            tokens = torch.cat((tokens, next_token), dim=1)
    
    # 解码生成的文本
    generated_tokens = tokens[0].tolist()
    generated_text = enc.decode(generated_tokens)
    
    return generated_text

def main():
    # 使用示例
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    checkpoint_path = 'log/latest_checkpoint.pt'  # 或者指定具体的 checkpoint 文件
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    # 生成文本
    prompt = "The future of artificial intelligence is"
    generated_text = generate_text(
        model, 
        prompt, 
        max_length=100,
        temperature=0.8,
        top_k=50,
        device=device
    )
    
    print(f"输入: {prompt}")
    print(f"生成: {generated_text}")

if __name__ == "__main__":
    main()