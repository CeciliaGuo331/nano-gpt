"""
微调GPT-2模型使用Dolly-15k数据集
基于train_gpt2.py修改，专门用于指令微调
"""

import os
import math
import time
import glob
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

# 导入预训练脚本中的模型定义
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_gpt2 import GPT, GPTConfig

# -----------------------------------------------------------------------------
# 微调专用配置

# 数据配置
data_dir = "dolly15k"
batch_size = 16  # 微调使用较小的batch size
total_batch_size = 524288  # 0.5M tokens
assert total_batch_size % (batch_size * 1024) == 0
grad_accum_steps = total_batch_size // (batch_size * 1024)

# 优化器配置
max_lr = 6e-5  # 微调使用更小的学习率（预训练的1/10）
min_lr = max_lr * 0.1
warmup_steps = 50  # 更短的warmup
weight_decay = 0.01  # 权重衰减，防止过拟合

# 训练配置
# Dolly-15k约2.8M tokens，batch_size=16，每个batch约3200 tokens
# 1 epoch ≈ 2.8M / 3200 ≈ 875 steps
# 训练0.5个epoch
total_steps = 875  # 1 epoch
max_steps = 1 * total_steps  # 1 epoch

# 评估配置
val_loss_every = 20
val_max_steps = 20
generate_every = 100

# 检查点配置
checkpoint_interval = 100
keep_last_n_checkpoints = 3
auto_resume = True
pretrained_checkpoint = "logs/model_19073.pt"  # 预训练模型路径

# 系统配置
device = "cuda" if torch.cuda.is_available() else "cpu"
compile_model = False  # torch.compile 在某些系统上可能有问题
seed = 1337
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# -----------------------------------------------------------------------------
# 数据加载器

class DataLoaderLite:
    def __init__(self, data_root, B, T, split):
        self.B = B
        self.T = T
        assert split in {"train", "val"}
        
        # 获取所有分片文件
        shards = sorted(glob.glob(os.path.join(data_root, f"dolly_{split}_*.npy")))
        if len(shards) == 0:
            # 如果没有train/val区分，尝试加载所有分片
            shards = sorted(glob.glob(os.path.join(data_root, "dolly_*.npy")))
            if split == "val" and len(shards) > 0:
                # 没有专门的验证集，使用第一个分片作为验证集
                shards = shards[:1]
            elif split == "train" and len(shards) > 1:
                # 使用剩余的分片作为训练集
                shards = shards[1:]
            elif split == "train" and len(shards) == 1:
                # 只有一个分片，都用作训练集
                shards = shards
        
        if len(shards) == 0:
            raise ValueError(f"No data files found in {data_root}")
            
        print(f"Found {len(shards)} shards for {split} split")
        
        # 加载并连接所有分片
        all_tokens = []
        for shard in shards:
            tokens = np.load(shard).astype(np.int32)
            all_tokens.append(tokens)
        
        self.tokens = np.concatenate(all_tokens) if len(all_tokens) > 0 else np.array([], dtype=np.int32)
        print(f"Loaded {len(self.tokens):,} tokens for {split} split")
        
        # 状态
        self.current_position = 0
        
    def get_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        if len(buf) < B * T + 1:
            # 如果到达末尾，从头开始
            self.current_position = 0
            buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        
        buf = torch.tensor(buf, dtype=torch.long)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        # 推进位置
        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
            
        return x.to(device), y.to(device)
    
    def get_state(self):
        return {
            "current_position": self.current_position,
        }
    
    def load_state(self, state):
        self.current_position = state["current_position"]

# -----------------------------------------------------------------------------
# 训练循环

def main():
    # 设置DDP
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
    
    # 创建日志目录
    log_dir = "logs_finetune"
    if master_process:
        os.makedirs(log_dir, exist_ok=True)
        print(f"Logging to {log_dir}")
        
    # 加载预训练模型
    print(f"Loading pretrained model from {pretrained_checkpoint}")
    checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    model = GPT(GPTConfig())
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    
    if compile_model:
        model = torch.compile(model)
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model
    
    # 创建数据加载器
    train_loader = DataLoaderLite(data_dir, batch_size, 1024, "train")
    val_loader = DataLoaderLite(data_dir, batch_size, 1024, "val")
    
    # 如果验证集为空，使用训练集的一部分
    if len(val_loader.tokens) == 0:
        print("No validation set found, using part of training set")
        val_loader = train_loader
    
    # 优化器
    optimizer = raw_model.configure_optimizers(
        weight_decay=weight_decay, 
        learning_rate=max_lr, 
        device_type=device.split(":")[0]
    )
    
    # 学习率调度
    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    
    # 初始化tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # 训练循环
    start_step = 0
    if auto_resume:
        # 尝试找到最新的微调检查点
        checkpoints = sorted(glob.glob(os.path.join(log_dir, "model_*.pt")))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"Resuming from {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            raw_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint["step"] + 1
            train_loader.load_state(checkpoint["train_loader_state"])
    
    # 评估函数
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        val_loader.current_position = 0
        losses = []
        for _ in range(val_max_steps):
            x, y = val_loader.get_batch()
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)
    
    # 生成函数
    @torch.no_grad()
    def generate_samples():
        model.eval()
        prompts = [
            "### Instruction:\nWhat is machine learning?\n\n### Response:\n",
            "### Instruction:\nExplain quantum computing in simple terms.\n\n### Response:\n",
            "### Instruction:\nWrite a Python function to reverse a string.\n\n### Response:\n",
        ]
        
        for i, prompt in enumerate(prompts):
            if master_process:
                print(f"\nPrompt {i}: {prompt}")
            
            tokens = enc.encode(prompt)
            tokens = [50256] + tokens  # prepend <|endoftext|>
            tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            # 生成
            xgen = tokens
            max_new_tokens = 100
            temperature = 0.8
            top_k = 40
            
            for _ in range(max_new_tokens):
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    logits, _ = model(xgen)
                logits = logits[:, -1, :] / temperature
                
                # 可选top-k采样
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                xgen = torch.cat((xgen, next_token), dim=1)
            
            if master_process:
                output = enc.decode(xgen[0].tolist())
                print(f"Response: {output}")
        
        model.train()
    
    # 训练开始
    if master_process:
        print(f"\n{'='*50}")
        print(f"Starting fine-tuning from step {start_step}")
        print(f"Total steps: {max_steps}")
        print(f"Batch size: {batch_size}, Gradient accumulation: {grad_accum_steps}")
        print(f"Effective batch size: {batch_size * grad_accum_steps}")
        print(f"Learning rate: {max_lr} -> {min_lr}")
        print(f"Weight decay: {weight_decay}")
        print(f"{'='*50}\n")
    
    # 训练前生成样本对比
    if master_process and start_step == 0:
        print("\n=== Pre-finetuning samples ===")
        generate_samples()
    
    # 主训练循环
    for step in range(start_step, max_steps):
        t0 = time.time()
        
        # 学习率调度
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # 梯度累积
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.get_batch()
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / grad_accum_steps
            
            loss_accum += loss.detach()
            loss.backward()
        
        # 梯度裁剪
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 参数更新
        optimizer.step()
        
        # 同步
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 日志
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (batch_size * 1024 * grad_accum_steps) / dt
        
        if master_process and step % 10 == 0:
            print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | "
                  f"norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        
        # 验证
        if step > 0 and step % val_loss_every == 0:
            val_loss = estimate_loss()
            if master_process:
                print(f"validation loss: {val_loss:.4f}")
        
        # 生成样本
        if master_process and step > 0 and step % generate_every == 0:
            print(f"\n=== Generation samples at step {step} ===")
            generate_samples()
        
        # 保存检查点
        if master_process and step > 0 and step % checkpoint_interval == 0:
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "train_loader_state": train_loader.get_state(),
                "config": raw_model.config,
            }
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # 清理旧检查点
            if keep_last_n_checkpoints > 0:
                checkpoints = sorted(glob.glob(os.path.join(log_dir, "model_*.pt")))
                if len(checkpoints) > keep_last_n_checkpoints:
                    for old_checkpoint in checkpoints[:-keep_last_n_checkpoints]:
                        os.remove(old_checkpoint)
                        print(f"Removed old checkpoint: {old_checkpoint}")
    
    # 训练结束
    if master_process:
        print("\n=== Fine-tuning completed ===")
        print("Final generation samples:")
        generate_samples()
        
        # 保存最终模型
        final_checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(), 
            "step": max_steps,
            "config": raw_model.config,
        }
        final_path = os.path.join(log_dir, "model_final.pt")
        torch.save(final_checkpoint, final_path)
        print(f"\nSaved final model to {final_path}")
    
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()