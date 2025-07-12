"""
测试不同生成参数对模型输出质量的影响
"""

import os
import torch
import tiktoken

# 直接导入train_gpt2作为模块，让Python识别这个名称
import model.train_gpt2
from model.train_gpt2 import GPT, GPTConfig

def test_generation(model_path, temperature=0.8, top_k=40):
    """测试模型生成质量"""
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    print(f"生成参数: temperature={temperature}, top_k={top_k}")
    
    model = GPT(GPTConfig(vocab_size=50304))
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)  # 添加weights_only=False
    
    # 处理不同的checkpoint格式
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # 测试prompts
    prompts = [
        "### Instruction:\nWhat is machine learning?\n\n### Response:\n",
        "### Instruction:\nExplain quantum computing in simple terms.\n\n### Response:\n",
        "### Instruction:\nWrite a Python function to reverse a string.\n\n### Response:\n",
    ]
    
    print("\n" + "="*60)
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i}: {prompt}")
        
        # Tokenize
        tokens = enc.encode(prompt)
        tokens = [50256] + tokens  # prepend <|endoftext|>
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # 生成
        xgen = tokens
        max_new_tokens = 100
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = model(xgen)
                logits = logits[:, -1, :] / temperature
                
                # Top-k采样
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 检查是否遇到结束标记
                if next_token.item() == 50256:  # <|endoftext|>
                    break
                    
                xgen = torch.cat((xgen, next_token), dim=1)
        
        # 解码并打印
        output = enc.decode(xgen[0].tolist())
        print(f"Response: {output}")
        print("-" * 60)

if __name__ == "__main__":
    # 测试最终模型with不同参数
    model_path = "logs_finetune/model_final.pt"
    
    print("\n测试1: 原始参数 (temperature=0.8, top_k=40)")
    test_generation(model_path, temperature=0.8, top_k=40)
    
    print("\n\n测试2: 降低temperature (temperature=0.6, top_k=40)")
    test_generation(model_path, temperature=0.6, top_k=40)
    
    print("\n\n测试3: 更保守的参数 (temperature=0.5, top_k=30)")
    test_generation(model_path, temperature=0.5, top_k=30)
    
    print("\n\n测试4: 极保守参数 (temperature=0.4, top_k=20)")
    test_generation(model_path, temperature=0.4, top_k=20)
    
    # 也可以测试step 2的模型
    print("\n\n测试5: Step 2模型，保守参数 (temperature=0.5, top_k=30)")
    test_generation("logs_finetune/model_00002.pt", temperature=0.5, top_k=30)