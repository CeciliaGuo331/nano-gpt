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
import argparse
import random
import inspect

# 导入预训练脚本中的模型定义
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_gpt2 import GPT, GPTConfig

# -----------------------------------------------------------------------------
# 数据加载器

class DataLoaderLite:
    def __init__(self, data_root, B, T, split, device):
        self.B = B
        self.T = T
        self.device = device
        assert split in {"train", "val"}
        
        # get the shard filenames - 与 train_gpt2.py 保持一致
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        
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
            
        return x.to(self.device), y.to(self.device)
    
    def get_state(self):
        return {
            "current_position": self.current_position,
        }
    
    def load_state(self, state):
        self.current_position = state["current_position"]

# -----------------------------------------------------------------------------
# 辅助函数

def clean_old_checkpoints(log_dir, keep_n):
    """Keep only the last N checkpoints"""
    if keep_n <= 0:
        return  # Don't clean if keep_n is invalid

    checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not checkpoints:
        return  # No checkpoints to clean

    checkpoints.sort(key=os.path.getctime)

    if len(checkpoints) > keep_n:
        for checkpoint in checkpoints[:-keep_n]:
            try:
                os.remove(checkpoint)
                print(f"Removed old checkpoint: {checkpoint}")
            except OSError as e:
                print(f"Warning: Failed to remove checkpoint {checkpoint}: {e}")


def save_checkpoint(step, model, optimizer, train_loader, val_loss, checkpoint_path, keep_last_n_checkpoints, master_process=True):
    """Save a checkpoint with all necessary state"""
    # Create a temporary file first to avoid corruption
    temp_path = checkpoint_path + ".tmp"

    try:
        # 确保 RNG 状态是正确的格式
        rng_state = torch.get_rng_state()
        if rng_state.dtype != torch.uint8:
            rng_state = rng_state.to(torch.uint8)
        
        cuda_rng_state = None
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
            if cuda_rng_state.dtype != torch.uint8:
                cuda_rng_state = cuda_rng_state.to(torch.uint8)
        
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": model.config,
            "step": step,
            "val_loss": val_loss,
            "train_loader_state": train_loader.get_state(),
            "rng_state": rng_state,
            "cuda_rng_state": cuda_rng_state,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        # Save to temporary file first
        torch.save(checkpoint, temp_path)

        # Atomically rename to final path
        os.rename(temp_path, checkpoint_path)

        # save a symlink to the latest checkpoint
        log_dir = os.path.dirname(checkpoint_path)
        latest_path = os.path.join(log_dir, "latest_checkpoint.pt")
        try:
            if os.path.exists(latest_path) or os.path.islink(latest_path):
                os.remove(latest_path)
            os.symlink(os.path.basename(checkpoint_path), latest_path)
        except OSError as e:
            if master_process:
                print(f"Warning: Failed to create latest checkpoint symlink: {e}")
        if master_process:
            print(f"Saved checkpoint to {checkpoint_path}")

        # Clean old checkpoints
        if keep_last_n_checkpoints > 0:
            clean_old_checkpoints(log_dir, keep_last_n_checkpoints)

    except Exception as e:
        # Clean up temp file if save failed
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        if master_process:
            print(f"Error: Failed to save checkpoint: {e}")
        raise


def load_checkpoint(checkpoint_path, model, optimizer, train_loader, device, master_process=True):
    """Load a checkpoint and restore all state"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含 Python 对象的 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        # 如果失败，尝试只加载权重
        if master_process:
            print(f"Warning: Failed to load full checkpoint, trying weights-only mode: {e}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # 对于纯权重文件，创建一个最小的checkpoint结构
        if "model" not in checkpoint:
            checkpoint = {"model": checkpoint, "step": -1}
        return checkpoint.get("step", -1) + 1

    # Load model weights
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        # 兼容旧格式或纯权重文件
        model.load_state_dict(checkpoint)
        return 0

    # Load optimizer state if available
    if "optimizer" in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            if master_process:
                print(f"Warning: Could not load optimizer state: {e}")

    # Load train loader state if available
    if "train_loader_state" in checkpoint and train_loader is not None:
        try:
            train_loader.load_state(checkpoint["train_loader_state"])
        except Exception as e:
            if master_process:
                print(f"Warning: Could not load train loader state: {e}")

    # Restore random states if available
    if "rng_state" in checkpoint:
        try:
            rng_state = checkpoint["rng_state"]
            if isinstance(rng_state, torch.Tensor):
                rng_state = rng_state.cpu().contiguous()
                if rng_state.dtype != torch.uint8:
                    rng_state = rng_state.to(torch.uint8)
            torch.set_rng_state(rng_state)
        except Exception as e:
            if master_process:
                print(f"Warning: Could not restore PyTorch RNG state: {e}")
    
    # 恢复 CUDA RNG 状态
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        try:
            cuda_rng_state = checkpoint["cuda_rng_state"]
            if isinstance(cuda_rng_state, torch.Tensor):
                cuda_rng_state = cuda_rng_state.cpu().contiguous()
                if cuda_rng_state.dtype != torch.uint8:
                    cuda_rng_state = cuda_rng_state.to(torch.uint8)
            torch.cuda.set_rng_state(cuda_rng_state)
        except Exception as e:
            if master_process:
                print(f"Warning: Could not restore CUDA RNG state: {e}")
    
    # Restore numpy and python random states
    if "numpy_rng_state" in checkpoint:
        try:
            np.random.set_state(checkpoint["numpy_rng_state"])
        except Exception as e:
            if master_process:
                print(f"Warning: Could not restore numpy RNG state: {e}")
                
    if "python_rng_state" in checkpoint:
        try:
            random.setstate(checkpoint["python_rng_state"])
        except Exception as e:
            if master_process:
                print(f"Warning: Could not restore python RNG state: {e}")

    start_step = checkpoint.get("step", -1) + 1  # Resume from next step
    if master_process:
        print(f"Resumed from checkpoint at step {checkpoint.get('step', 'unknown')}")
    return start_step

# -----------------------------------------------------------------------------
# 训练循环

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on Dolly-15k dataset")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Automatically resume from latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=2,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=3,
        help="Keep only the last N checkpoints, -1 to keep all",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=2,
        help="Evaluate validation loss every N steps",
    )
    parser.add_argument(
        "--generate_interval",
        type=int,
        default=2,
        help="Generate text samples every N steps",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="log/model_19072.pt",
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dolly15k",
        help="Directory containing the Dolly dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=6e-6,  # 降低10倍，从6e-5到6e-6
        help="Maximum learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=3,  # 增加warmup步数
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=6,
        help="Maximum number of training steps (default: ~1 epoch for Dolly-15k)",
    )
    args = parser.parse_args()

    # 微调专用配置
    # 数据配置
    data_dir = args.data_dir
    batch_size = args.batch_size  # 微调使用较小的batch size
    total_batch_size = 524288  # 0.5M tokens
    assert total_batch_size % (batch_size * 1024) == 0
    grad_accum_steps = total_batch_size // (batch_size * 1024)

    # 优化器配置
    max_lr = args.max_lr  # 微调使用更小的学习率（预训练的1/10）
    min_lr = max_lr * 0.1
    warmup_steps = args.warmup_steps  # 更短的warmup
    weight_decay = 0.01  # 权重衰凊，防止过拟合

    # 训练配置
    # Dolly-15k约2.8M tokens，每个step处理0.5M tokens
    # 1 epoch ≈ 2.8M / 0.5M ≈ 5-6 steps
    max_steps = args.max_steps

    # 评估配置
    val_loss_every = args.eval_interval
    val_max_steps = 20
    generate_every = args.generate_interval

    # 检查点配置
    checkpoint_interval = args.checkpoint_interval
    keep_last_n_checkpoints = args.keep_last_n_checkpoints
    auto_resume = args.auto_resume
    pretrained_checkpoint = args.pretrained_checkpoint

    # 系统配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model = False  # torch.compile 在某些系统上可能有问题
    seed = 1337
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
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
        
    # 创建模型
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    
    if compile_model:
        model = torch.compile(model)
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model
    
    # 创建数据加载器
    print(f"Loading data from: {data_dir}")
    train_loader = DataLoaderLite(data_dir, batch_size, 1024, "train", device)
    val_loader = DataLoaderLite(data_dir, batch_size, 1024, "val", device)
    
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
    
    # Print optimizer info if master process
    if master_process:
        param_dict = {pn: p for pn, p in raw_model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.split(":")[0] == "cuda"
        print(f"using fused AdamW: {use_fused}")
    
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
    
    # 确定起始步骤和加载检查点
    start_step = 0
    
    # 首先尝试恢复微调检查点
    if args.resume or args.auto_resume:
        checkpoint_path = args.checkpoint_path
        if args.auto_resume and checkpoint_path is None:
            # Find the latest checkpoint
            latest_symlink = os.path.join(log_dir, "latest_checkpoint.pt")
            if os.path.islink(latest_symlink) and os.path.exists(latest_symlink):
                # Prefer using the latest symlink if it exists
                checkpoint_path = os.path.join(log_dir, os.readlink(latest_symlink))
                if master_process:
                    print(
                        f"Auto-resuming from latest checkpoint (via symlink): {checkpoint_path}"
                    )
            else:
                # Fallback to finding the latest by timestamp
                checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
                if checkpoints:
                    checkpoint_path = max(checkpoints, key=os.path.getctime)
                    if master_process:
                        print(f"Auto-resuming from latest checkpoint: {checkpoint_path}")

        # In DDP, ensure all processes use the same checkpoint
        if ddp:
            # Broadcast checkpoint path from master to all processes
            if master_process:
                # Convert checkpoint path to bytes for broadcasting
                path_bytes = checkpoint_path.encode("utf-8") if checkpoint_path else b""
                path_length = torch.tensor(
                    [len(path_bytes)], dtype=torch.int64, device=device
                )
            else:
                path_length = torch.tensor([0], dtype=torch.int64, device=device)

            # Broadcast length first
            dist.broadcast(path_length, 0)

            # Prepare buffer for path
            if not master_process:
                path_bytes = torch.zeros(
                    path_length.item(), dtype=torch.uint8, device=device
                )
            else:
                path_bytes = torch.tensor(
                    list(path_bytes), dtype=torch.uint8, device=device
                )

            # Broadcast the actual path
            dist.broadcast(path_bytes, 0)

            # Decode on non-master processes
            if not master_process and path_length.item() > 0:
                checkpoint_path = bytes(path_bytes.cpu().numpy()).decode("utf-8")
            elif not master_process:
                checkpoint_path = None

        if checkpoint_path:
            try:
                start_step = load_checkpoint(
                    checkpoint_path, raw_model, optimizer, train_loader, device, master_process
                )
            except (FileNotFoundError, RuntimeError, KeyError) as e:
                if master_process:
                    print(f"Error loading checkpoint: {e}")
                    print("Will try loading pretrained model instead...")
                start_step = 0
    
    # 如果没有恢复微调检查点，加载预训练模型
    if start_step == 0 and pretrained_checkpoint:
        if master_process:
            print(f"Loading pretrained model from {pretrained_checkpoint}")
        try:
            # 尝试加载完整的检查点
            pretrained_start_step = load_checkpoint(
                pretrained_checkpoint, raw_model, None, None, device, master_process
            )
            if master_process:
                print(f"Loaded pretrained model from step {pretrained_start_step - 1}")
        except Exception as e:
            if master_process:
                print(f"Error loading pretrained model: {e}")
                print("Starting with random initialization...")
    
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
                
                # 检查是否遇到结束标记
                if next_token.item() == 50256:  # <|endoftext|> token
                    break
                    
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
        print(f"Tokens per step: {batch_size * 1024 * grad_accum_steps:,}")
        print(f"Total tokens in dataset: ~2.8M")
        print(f"Approximate epochs: {(max_steps * batch_size * 1024 * grad_accum_steps) / 2.8e6:.1f}")
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
        last_step = step == max_steps - 1
        if master_process and step > 0 and (step % checkpoint_interval == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            # Use the last validation loss if available, otherwise use current training loss
            val_loss_to_save = val_loss if 'val_loss' in locals() else loss_accum.item()
            try:
                save_checkpoint(
                    step,
                    raw_model,
                    optimizer,
                    train_loader,
                    val_loss_to_save,
                    checkpoint_path,
                    keep_last_n_checkpoints,
                    master_process,
                )
            except Exception as e:
                print(f"Warning: Failed to save checkpoint at step {step}: {e}")
                print("Training will continue, but checkpoint was not saved.")
    
    # 训练结束
    if master_process:
        print("\n=== Fine-tuning completed ===")
        print("Final generation samples:")
        generate_samples()
        
        # 保存最终模型
        final_path = os.path.join(log_dir, "model_final.pt")
        try:
            save_checkpoint(
                max_steps,
                raw_model,
                optimizer,
                train_loader,
                val_loss if 'val_loss' in locals() else loss_accum.item(),
                final_path,
                keep_last_n_checkpoints,
                master_process,
            )
        except Exception as e:
            print(f"Warning: Failed to save final checkpoint: {e}")
    
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()