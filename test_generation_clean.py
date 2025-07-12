"""
优雅的模型加载和测试脚本
"""

import os
import torch
import tiktoken
from model.train_gpt2 import GPT, GPTConfig

def load_model_weights_only(checkpoint_path, device="cuda"):
    """只加载模型权重，避免模块依赖问题"""
    # 创建模型
    model = GPT(GPTConfig(vocab_size=50304))
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 只加载model state dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        # 如果是旧格式的checkpoint（直接是state_dict）
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def test_temperature_effects(model_path="logs_finetune/model_final.pt"):
    """测试不同temperature对生成质量的影响"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")
    
    print(f"加载模型: {model_path}")
    model = load_model_weights_only(model_path, device)
    
    # 测试用例
    test_cases = [
        {
            "name": "机器学习定义",
            "prompt": "### Instruction:\nWhat is machine learning?\n\n### Response:\n"
        },
        {
            "name": "Python函数",
            "prompt": "### Instruction:\nWrite a Python function to reverse a string.\n\n### Response:\n"
        }
    ]
    
    # 测试不同的temperature
    temperatures = [0.8, 0.5, 0.3]
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"测试: {test_case['name']}")
        print(f"{'='*80}")
        
        for temp in temperatures:
            print(f"\nTemperature = {temp}:")
            print("-" * 40)
            
            # 生成文本
            generated = generate_text(
                model, 
                enc, 
                test_case['prompt'], 
                temperature=temp,
                top_k=20,  # 使用较小的top_k以获得更集中的输出
                max_tokens=80,
                device=device
            )
            
            # 只显示生成的部分（去掉prompt）
            response_only = generated[len(test_case['prompt']):]
            print(response_only.strip())

def generate_text(model, tokenizer, prompt, temperature=0.8, top_k=40, max_tokens=100, device="cuda"):
    """生成文本的核心函数"""
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    tokens = [50256] + tokens  # prepend <|endoftext|>
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    generated = tokens
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(generated)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if we hit <|endoftext|>
            if next_token.item() == 50256:
                break
                
            generated = torch.cat((generated, next_token), dim=1)
    
    # Decode and return
    return tokenizer.decode(generated[0].tolist())

def compare_checkpoints():
    """比较不同checkpoint的生成质量"""
    checkpoints = [
        "logs_finetune/model_00001.pt",
        "logs_finetune/model_00002.pt", 
        "logs_finetune/model_final.pt"
    ]
    
    prompt = "### Instruction:\nWhat is machine learning?\n\n### Response:\n"
    
    print("\n" + "="*80)
    print("比较不同训练步骤的模型")
    print("="*80)
    
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            print(f"\n{ckpt}:")
            print("-" * 40)
            
            model = load_model_weights_only(ckpt)
            enc = tiktoken.get_encoding("gpt2")
            
            generated = generate_text(
                model, enc, prompt, 
                temperature=0.5, top_k=20, max_tokens=60
            )
            
            response = generated[len(prompt):]
            print(response.strip())

if __name__ == "__main__":
    # 主要测试：不同temperature的效果
    test_temperature_effects()
    
    # 可选：比较不同checkpoint
    print("\n\n是否要比较不同训练步骤的模型? (y/n): ", end="")
    if input().lower() == 'y':
        compare_checkpoints()