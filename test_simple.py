"""
简化版的生成参数测试
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import tiktoken

# 必须先导入train_gpt2模块，让Python知道这个模块
import model.train_gpt2 as train_gpt2
from model.train_gpt2 import GPT, GPTConfig

def test_generation_simple():
    """简化的测试函数"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")
    
    # 加载模型
    print("加载模型...")
    model = GPT(GPTConfig(vocab_size=50304))
    checkpoint = torch.load("logs_finetune/model_final.pt", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    # 测试不同的temperature
    temperatures = [0.8, 0.6, 0.4]
    prompt = "### Instruction:\nWhat is machine learning?\n\n### Response:\n"
    
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"Temperature: {temp}")
        print(f"Prompt: {prompt}")
        
        # Tokenize
        tokens = enc.encode(prompt)
        tokens = [50256] + tokens
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate
        xgen = tokens
        with torch.no_grad():
            for _ in range(80):  # 生成80个token
                logits, _ = model(xgen)
                logits = logits[:, -1, :] / temp
                
                # Top-k=30
                v, _ = torch.topk(logits, min(30, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == 50256:  # <|endoftext|>
                    break
                    
                xgen = torch.cat((xgen, next_token), dim=1)
        
        # Decode
        output = enc.decode(xgen[0].tolist())
        print(f"Response: {output}")

if __name__ == "__main__":
    test_generation_simple()