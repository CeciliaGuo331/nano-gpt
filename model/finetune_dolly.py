"""
微调GPT-2模型使用Dolly-15k数据集。

该脚本是 train_gpt2.py 的一个专门化版本，针对指令微调任务进行了深度优化。
它融合了预训练脚本中成熟的工程实践和为微调任务定制的核心逻辑。

核心特性:
1.  **指令微调数据加载器**: 实现基于样本的加载、动态填充和批次内随机化。
2.  **损失遮罩 (Loss Masking)**: 仅在模型生成的回应部分计算损失，提升训练效率。
3.  **职责分离的检查点机制**: 清晰地区分“恢复训练”和“加载预训练权重”两种操作。
4.  **内存优化与健壮性**: 包含内存优化参数，并修复了所有已知的bug。
"""

import os
import math
import time
import glob
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken
import argparse
import random
import inspect
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from torch.serialization import safe_globals

# 导入预训练脚本中的模型定义
try:
    from .train_gpt2 import GPT, GPTConfig
except ImportError:
    from train_gpt2 import GPT, GPTConfig

# -----------------------------------------------------------------------------
# 数据加载器

class DataLoaderLite:
    """
    为指令微调任务设计的数据加载器。
    它将数据视为独立的样本列表，而非连续的token流。
    """
    def __init__(self, data_root: str, B: int, T: int, split: str, 
                 master_process: bool, device: str, vocab_size: int):
        self.B = B
        self.T = T
        self.device = device
        self.master_process = master_process
        self.vocab_size = vocab_size
        assert split in {"train", "val"}

        enc = tiktoken.get_encoding('gpt2')
        self.eot_token_id = enc._special_tokens['<|endoftext|>']
        self.pad_token_id = -100
        self.response_template = "### Response:"
        self.response_template_tokens = enc.encode(self.response_template)

        shards = sorted([os.path.join(data_root, s) for s in os.listdir(data_root) if split in s])
        assert len(shards) > 0, f"no shards found for split {split}"
        if self.master_process:
            print(f"found {len(shards)} shards for split {split}")

        tokens_stream = np.concatenate([np.load(shard).astype(np.int32) for shard in shards])
        
        eot_indices = np.where(tokens_stream == self.eot_token_id)[0]
        samples = np.split(tokens_stream, eot_indices + 1)
        
        self.samples = [s for s in samples if 0 < len(s) <= self.T]
        if self.master_process:
            print(f"Loaded and split into {len(self.samples):,} valid samples for {split} split.")
        
        self.reset()

    def reset(self):
        self.current_position = 0
        random.shuffle(self.samples)

    def _find_subsequence(self, main_list: List[int], sub_list: List[int]) -> int:
        len_sub = len(sub_list)
        for i in range(len(main_list) - len_sub + 1):
            if main_list[i : i + len_sub] == sub_list:
                return i + len_sub
        return -1

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.current_position + self.B >= len(self.samples):
            self.reset()

        batch_samples = self.samples[self.current_position : self.current_position + self.B]
        self.current_position += self.B

        max_len_in_batch = max(len(s) for s in batch_samples)
        
        x_list, y_list = [], []
        for sample in batch_samples:
            sample_list = sample.tolist()
            padded_sample = np.full(max_len_in_batch, 0, dtype=np.int32)
            padded_sample[:len(sample_list)] = sample_list
            
            x = padded_sample[:-1].copy()
            y = padded_sample[1:].copy()
            
            y[:] = self.pad_token_id
            response_start_idx = self._find_subsequence(sample_list, self.response_template_tokens)
            if response_start_idx != -1:
                response_end_idx = len(sample_list) - 1
                y[response_start_idx-1 : response_end_idx-1] = sample[response_start_idx : response_end_idx]
            
            x_list.append(x)
            y_list.append(y)

        x = torch.tensor(np.array(x_list), dtype=torch.long, device=self.device)
        y = torch.tensor(np.array(y_list), dtype=torch.long, device=self.device)
        
        if x.min() < 0 or x.max() >= self.vocab_size:
            raise ValueError(f"DataLoader detected invalid token ID in batch x. "
                             f"Min: {x.min()}, Max: {x.max()}, Vocab size: {self.vocab_size}")
        
        return x, y

    def get_state(self) -> Dict[str, Any]:
        return {"current_position": self.current_position}

    def load_state(self, state: Dict[str, Any]):
        self.current_position = state.get("current_position", 0)

# -----------------------------------------------------------------------------
# 检查点管理函数

def save_checkpoint(step, model, optimizer, train_loader, val_loss, checkpoint_path, keep_last_n, master_process):
    """安全地保存一个完整的检查点用于恢复训练"""
    if not master_process: return
    temp_path = checkpoint_path + ".tmp"
    try:
        rng_state = torch.get_rng_state()
        if rng_state.dtype != torch.uint8:
            rng_state = rng_state.to(torch.uint8)
        
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": model.config, "step": step, "val_loss": val_loss,
            "train_loader_state": train_loader.get_state(),
            "rng_state": rng_state, "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
        }
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
            if cuda_rng_state.dtype != torch.uint8:
                cuda_rng_state = cuda_rng_state.to(torch.uint8)
            checkpoint["cuda_rng_state"] = cuda_rng_state
            
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        latest_path = os.path.join(os.path.dirname(checkpoint_path), "latest_checkpoint.pt")
        if os.path.exists(latest_path) or os.path.islink(latest_path): os.remove(latest_path)
        os.symlink(os.path.basename(checkpoint_path), latest_path)

        if keep_last_n > 0:
            checkpoints = sorted(glob.glob(os.path.join(os.path.dirname(checkpoint_path), "model_*.pt")), key=os.path.getmtime)
            for old_ckpt in checkpoints[:-keep_last_n]: os.remove(old_ckpt)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def load_finetune_checkpoint(path, model, optimizer, train_loader, device, master_process):
    """加载一个完整的微调检查点以恢复训练"""
    with safe_globals([GPTConfig]):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_loader.load_state(checkpoint['train_loader_state'])
    
    try:
        if 'rng_state' in checkpoint:
            rng_state = checkpoint['rng_state'].cpu().to(torch.uint8)
            torch.set_rng_state(rng_state)

        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            cuda_rng_state = checkpoint['cuda_rng_state'].cpu().to(torch.uint8)
            torch.cuda.set_rng_state(cuda_rng_state)

        if 'python_rng_state' in checkpoint: random.setstate(checkpoint['python_rng_state'])
        if 'numpy_rng_state' in checkpoint: np.random.set_state(checkpoint['numpy_rng_state'])
        
    except Exception as e:
        if master_process: print(f"Warning: Could not fully restore RNG state. Error: {e}")

    start_step = checkpoint['step'] + 1
    if master_process: print(f"Resumed from finetuning checkpoint at step {checkpoint['step']}")
    return start_step

def load_pretrained_weights(path, model, device, master_process):
    """只加载预训练模型的权重，忽略其他所有状态"""
    try:
        with safe_globals([GPTConfig]):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        if master_process: print(f"Successfully loaded pretrained weights from {path}")
    except Exception as e:
        if master_process: print(f"Error loading pretrained weights: {e}")

# -----------------------------------------------------------------------------
# 主函数

def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on Dolly-15k dataset.")
    parser.add_argument("--data_dir", type=str, default="dolly15k", help="数据目录")
    parser.add_argument("--log_dir", type=str, default="logs_finetune", help="日志和检查点目录")
    parser.add_argument("--pretrained_checkpoint", type=str, default="log/model_19072.pt", help="预训练模型路径")
    parser.add_argument("--resume", type=str, choices=['auto', 'off'], default='off', help="是否从检查点恢复训练。'off'表示默认从预训练模型开始，'auto'表示从最新的微调检查点恢复。")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="保存检查点的步数间隔")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=3, help="保留最近的检查点数量")
    parser.add_argument("--max_steps", type=int, default=500, help="最大训练步数")
    parser.add_argument("--batch_size", "-B", type=int, default=2, help="批次大小")
    parser.add_argument("--context_length", "-T", type=int, default=1024, help="最大上下文长度")
    parser.add_argument("--grad_accum_steps", type=int, default=5, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=3e-5, help="最大学习率")
    parser.add_argument("--warmup_steps", type=int, default=100, help="学习率预热步数")
    parser.add_argument("--eval_interval", type=int, default=50, help="验证步数间隔")
    parser.add_argument("--generate_interval", type=int, default=50, help="生成示例文本的步数间隔")
    args = parser.parse_args()

    # --- 系统与DDP设置 ---
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(1337 + ddp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337 + ddp_rank)

    if master_process:
        os.makedirs(args.log_dir, exist_ok=True)
        print(f"Running on device: {device}, World size: {ddp_world_size}")
        print(f"Logging to {args.log_dir}")
        print("Arguments:", vars(args))
    log_file = os.path.join(args.log_dir, f"log.txt")

    # --- 模型与优化器初始化 ---
    vocab_size = 50304
    model_config = GPTConfig(vocab_size=vocab_size)
    model = GPT(model_config)
    
    if not hasattr(GPT, 'generate'):
        @torch.no_grad()
        def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
            self.eval()
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            self.train()
            return idx
        GPT.generate = generate
        if master_process:
            print("Dynamically added 'generate' method to GPT class.")

    model.to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    optimizer = raw_model.configure_optimizers(weight_decay=0.01, learning_rate=args.lr, device_type=device_type)

    # --- 数据加载器 ---
    train_loader = DataLoaderLite(args.data_dir, args.batch_size, args.context_length, "train", master_process, device, vocab_size)
    val_loader = DataLoaderLite(args.data_dir, args.batch_size, args.context_length, "val", master_process, device, vocab_size)
    torch.set_float32_matmul_precision("high")

    # --- 学习率调度器 ---
    def get_lr(it):
        if it < args.warmup_steps:
            return args.lr * (it + 1) / args.warmup_steps
        if it > args.max_steps:
            return args.lr * 0.1
        decay_ratio = (it - args.warmup_steps) / (args.max_steps - args.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.lr * 0.1 + coeff * (args.lr * 0.9)
    
    # --- 评估与生成函数 ---
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = torch.zeros(20, device=device)
        val_loader.reset()
        for k in range(20):
            x, y = val_loader.get_batch()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[k] = loss.item()
        model.train()
        if ddp:
            dist.all_reduce(losses, op=dist.ReduceOp.AVG)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return losses.mean().item()

    @torch.no_grad()
    def generate_samples():
        if not master_process: return
        model.eval()
        enc = tiktoken.get_encoding("gpt2")
        prompts = [
            "### Instruction:\nWhat is machine learning?\n\n### Response:\n",
            "### Instruction:\nExplain quantum computing in simple terms.\n\n### Response:\n",
        ]
        for prompt in prompts:
            print(f"\n--- Prompt ---\n{prompt}", end="")
            tokens = enc.encode(prompt)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            generated_tokens = raw_model.generate(tokens, max_new_tokens=100, temperature=0.7, top_k=50)
            generated_text = enc.decode(generated_tokens[0, len(tokens[0]):].tolist())
            print(generated_text)
        model.train()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 检查点恢复 ---
    start_step = 0
    if args.resume == 'auto':
        latest_ckpt = os.path.join(args.log_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_ckpt):
            try:
                start_step = load_finetune_checkpoint(latest_ckpt, raw_model, optimizer, train_loader, device, master_process)
            except Exception as e:
                print(f"Failed to load finetuning checkpoint, starting from scratch. Error: {e}")
                start_step = 0

    if start_step == 0 and os.path.exists(args.pretrained_checkpoint):
        load_pretrained_weights(args.pretrained_checkpoint, raw_model, device, master_process)

    # --- 主训练循环 ---
    if master_process: print(f"\n🚀 Starting fine-tuning from step {start_step}...")
    val_loss = float('inf')
    
    if start_step == 0 and master_process: generate_samples()

    for step in range(start_step, args.max_steps):
        t0 = time.time()
        
        if step > 0 and step % args.eval_interval == 0:
            val_loss = estimate_loss()
            if master_process:
                print(f"\nValidation loss: {val_loss:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"step {step} val_loss {val_loss:.4f}\n")

        optimizer.zero_grad()
        loss_accum = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = train_loader.get_batch()
            if ddp: model.require_backward_grad_sync = (_ == args.grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        
        if ddp: dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        lr = get_lr(step)
        for param_group in optimizer.param_groups: param_group["lr"] = lr
        optimizer.step()
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        dt = time.time() - t0
        
        if master_process:
            tokens_per_sec = (args.batch_size * args.context_length * args.grad_accum_steps) / dt
            log_str = (f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | "
                       f"norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            print(log_str)
            with open(log_file, "a") as f:
                f.write(f"step {step} train_loss {loss_accum.item():.6f}\n")
        
        if step > 0 and step % args.generate_interval == 0:
            generate_samples()
        
        # 【关键修正】移除在循环最后一步强制保存的逻辑
        if args.checkpoint_interval > 0 and step > 0 and step % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.log_dir, f"model_{step:05d}.pt")
            save_checkpoint(step, raw_model, optimizer, train_loader, val_loss, 
                            checkpoint_path, args.keep_last_n_checkpoints, master_process)
    
    # --- 训练结束后的最终评估与保存 ---
    if master_process:
        print("\n🎉 Fine-tuning completed!")
        
        # 执行最终评估
        print("Performing final evaluation...")
        final_val_loss = estimate_loss()
        print(f"Final validation loss: {final_val_loss:.4f}")
        with open(log_file, "a") as f:
            f.write(f"final val_loss {final_val_loss:.4f}\n")
            
        # 生成最终样本
        print("Generating final samples...")
        generate_samples()
        
        # 【关键修正】保存最终模型，使用正确的步数和文件名格式
        final_step = args.max_steps - 1
        final_checkpoint_path = os.path.join(args.log_dir, f"model_{final_step:05d}.pt")
        save_checkpoint(final_step, raw_model, optimizer, train_loader, final_val_loss, 
                        final_checkpoint_path, args.keep_last_n_checkpoints, master_process)
    
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()