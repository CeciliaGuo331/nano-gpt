"""
完成gpt-2模型的预训练
"""

import os
import math
import time
import inspect
import argparse
import glob
import random
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from eval.hellaswag import render_example, iterate_examples

# -----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Note: These prints are commented out to avoid dependency on master_process
        # If needed, the caller can print these stats
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        # print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=0.9, 
                 presence_penalty=0.0, frequency_penalty=0.0, stream=False, stream_callback=None):
        """
        从一个初始的 token 序列 idx (形状: B, T) 开始，生成后续的 token。
        
        参数:
        - idx: 输入token序列 (B, T)
        - max_new_tokens: 生成的最大token数
        - temperature: 温度参数，控制随机性
        - top_k: Top-K采样，从概率最高的k个token中采样
        - top_p: 核采样，从累积概率达到p的token集合中采样
        - presence_penalty: 存在惩罚，减少重复话题
        - frequency_penalty: 频率惩罚，根据频率减少重复词汇
        - stream: 是否启用流式生成
        - stream_callback: 流式生成的回调函数，接收每个新生成的token
        """
        # 确保模型处于评估模式
        self.eval()
        
        # 记录已生成的token及其频率，用于惩罚计算
        generated_tokens = {}  # {token_id: count}
        original_length = idx.size(1)
        
        for step in range(max_new_tokens):
            # 如果当前序列长度超过了模型的上下文长度，就截取最后 block_size 个 token
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # 前向传播，获取 logits
            logits, _ = self(idx_cond)
            
            # 只关注最后一个时间步的 logits
            logits = logits[:, -1, :] / temperature
            
            # 应用presence_penalty和frequency_penalty
            if presence_penalty != 0.0 or frequency_penalty != 0.0:
                # 获取已生成的token（不包括原始输入）
                if idx.size(1) > original_length:
                    generated_sequence = idx[0, original_length:].tolist()
                    
                    # 统计频率
                    for token_id in generated_sequence:
                        generated_tokens[token_id] = generated_tokens.get(token_id, 0) + 1
                    
                    # 应用惩罚
                    for token_id, count in generated_tokens.items():
                        if token_id < logits.size(-1):  # 确保token_id在词汇表范围内
                            # presence_penalty: 对出现过的token施加固定惩罚
                            if presence_penalty != 0.0:
                                logits[0, token_id] -= presence_penalty
                            
                            # frequency_penalty: 根据出现频率施加惩罚
                            if frequency_penalty != 0.0:
                                logits[0, token_id] -= frequency_penalty * count
            
            # 应用top_k截断
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 应用top_p核采样
            if top_p < 1.0:
                # 获取排序后的logits和索引
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # 计算累积概率
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 找到累积概率超过top_p的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个超过阈值的token（移位以保留边界）
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 创建掩码并应用
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # 应用 softmax 得到概率分布
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # 从概率分布中采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 将新采样的 token 拼接到序列的末尾
            idx = torch.cat((idx, idx_next), dim=1)
            
            # 如果启用流式生成，调用回调函数
            if stream and stream_callback:
                new_token_id = idx_next[0, 0].item()
                stream_callback(new_token_id, step)

        # 恢复到训练模式
        self.train()
        return idx
# -----------------------------------------------------------------------------
import tiktoken
import numpy as np


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    def get_state(self):
        """Return the current state for checkpointing"""
        return {
            "current_shard": self.current_shard,
            "current_position": self.current_position,
        }

    def load_state(self, state):
        """Load state from checkpoint"""
        if "current_shard" not in state or "current_position" not in state:
            raise KeyError("DataLoader state missing required keys")

        self.current_shard = state["current_shard"]
        self.current_position = state["current_position"]

        # Validate shard index
        if self.current_shard < 0 or self.current_shard >= len(self.shards):
            if master_process:
                print(
                    f"Warning: Invalid shard index {self.current_shard}, resetting to 0"
                )
            self.current_shard = 0
            self.current_position = self.B * self.T * self.process_rank

        self.tokens = load_tokens(self.shards[self.current_shard])

        # Validate position
        if self.current_position < 0 or self.current_position >= len(self.tokens):
            if master_process:
                print(
                    f"Warning: Invalid position {self.current_position}, resetting to start of shard"
                )
            self.current_position = self.B * self.T * self.process_rank


# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (
        mask[..., 1:]
    ).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train GPT-2")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--auto_resume", action="store_true", help="Automatically resume from latest checkpoint")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=-1, help="Keep only the last N checkpoints, -1 to keep all")
    parser.add_argument("--eval_interval", type=int, default=250, help="Evaluate validation loss every N steps")
    parser.add_argument("--hellaswag_interval", type=int, default=250, help="Evaluate HellaSwag every N steps")
    parser.add_argument("--generate_interval", type=int, default=250, help="Generate text samples every N steps")
    parser.add_argument("--max_steps", type=int, default=19073, help="Max training steps")
    parser.add_argument("--batch_size", "-B", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    # run the training loop
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")

    total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
    B = args.batch_size  # micro batch size
    T = 1024  # sequence length
    assert (
        total_batch_size % (B * T * ddp_world_size) == 0
    ), "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
    )
    val_loader = DataLoaderLite(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
    )

    torch.set_float32_matmul_precision("high")

    # create model
    model = GPT(GPTConfig(vocab_size=50304))
    # model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
    model.to(device)
    use_compile = (
        False  # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    )
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = args.max_steps
    # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
    

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)


    # optimize!
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device_type=device_type
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
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")

    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    # Only clear log file if starting fresh, not resuming
    if not (args.resume or args.auto_resume):
        with open(log_file, "w") as f:  # open for writing to clear the file
            pass


    # checkpoint loading functionality
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
                    if master_process:
                        print(f"Removed old checkpoint: {checkpoint}")
                except OSError as e:
                    if master_process:
                        print(f"Warning: Failed to remove checkpoint {checkpoint}: {e}")


    def save_checkpoint(step, model, optimizer, train_loader, val_loss, checkpoint_path):
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
            if args.keep_last_n_checkpoints > 0:
                clean_old_checkpoints(log_dir, args.keep_last_n_checkpoints)

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


    def load_checkpoint(checkpoint_path, model, optimizer, train_loader, device):
        """Load a checkpoint and restore all state"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含 Python 对象的 checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")

        # Verify checkpoint has all required keys
        required_keys = [
            "model",
            "optimizer",
            "train_loader_state",
            "step",
            "rng_state",
            "numpy_rng_state",
            "python_rng_state",
        ]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise KeyError(f"Checkpoint missing required keys: {missing_keys}")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_loader.load_state(checkpoint["train_loader_state"])

        # Restore random states
        rng_state = checkpoint["rng_state"]
        
        # PyTorch 的 set_rng_state 在新版本中对类型要求严格
        # 我们需要确保它是正确的格式
        if isinstance(rng_state, torch.Tensor):
            # 确保是 CPU 上的连续张量
            rng_state = rng_state.cpu().contiguous()
            # 确保是 uint8 类型
            if rng_state.dtype != torch.uint8:
                rng_state = rng_state.to(torch.uint8)
            
            # 尝试不同的方法设置状态
            try:
                # 方法1: 直接设置
                torch.set_rng_state(rng_state)
            except TypeError:
                try:
                    # 方法2: 使用 torch.ByteTensor (旧版兼容)
                    import torch as torch_module
                    if hasattr(torch_module, 'ByteTensor'):
                        # 转换为 ByteTensor
                        rng_state_bytes = torch_module.ByteTensor(rng_state.size())
                        rng_state_bytes.copy_(rng_state)
                        torch.set_rng_state(rng_state_bytes)
                    else:
                        raise
                except:
                    # 方法3: 使用 Generator API
                    gen = torch.Generator()
                    gen.set_state(rng_state)
                    # 获取种子并重新设置
                    seed = gen.initial_seed()
                    torch.manual_seed(seed)
                    if master_process:
                        print("Warning: RNG state partially restored using seed")
        
        # 恢复 CUDA RNG 状态
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            cuda_rng_state = checkpoint["cuda_rng_state"]
            if isinstance(cuda_rng_state, torch.Tensor):
                cuda_rng_state = cuda_rng_state.cpu().contiguous()
                if cuda_rng_state.dtype != torch.uint8:
                    cuda_rng_state = cuda_rng_state.to(torch.uint8)
            
            try:
                torch.cuda.set_rng_state(cuda_rng_state)
            except Exception as e:
                if master_process:
                    print(f"Warning: Could not restore CUDA RNG state: {e}")
        
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])

        start_step = checkpoint["step"] + 1  # Resume from next step
        if master_process:
            print(f"Resumed from checkpoint at step {checkpoint['step']}")
        return start_step


    # Determine the starting step
    start_step = 0
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
                    checkpoint_path, raw_model, optimizer, train_loader, device
                )
            except (FileNotFoundError, RuntimeError, KeyError) as e:
                if master_process:
                    print(f"Error loading checkpoint: {e}")
                    print("Starting from scratch...")
                start_step = 0
        else:
            if args.resume and master_process:
                print("Warning: No checkpoint specified or found, starting from scratch")

    for step in range(start_step, max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        # once in a while evaluate our validation loss
        if step % args.eval_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        
        # checkpoint saving logic (moved outside validation block)
        if master_process and step > 0 and (step % args.checkpoint_interval == 0 or last_step):
            # optionally write model checkpoints
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            # Use the last validation loss if available, otherwise use current training loss
            val_loss_to_save = val_loss_accum.item() if 'val_loss_accum' in locals() else loss_accum.item()
            try:
                save_checkpoint(
                    step,
                    raw_model,
                    optimizer,
                    train_loader,
                    val_loss_to_save,
                    checkpoint_path,
                )
            except Exception as e:
                print(f"Warning: Failed to save checkpoint at step {step}: {e}")
                print("Training will continue, but checkpoint was not saved.")

        # once in a while evaluate hellaswag
        if (step % args.hellaswag_interval == 0 or last_step) and (not use_compile):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(
                    num_correct_norm, dtype=torch.long, device=device
                )
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # once in a while generate from the model (except step 0, which is noise)
        if ((step > 0 and step % args.generate_interval == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen)  # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :]  # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # added after video, this field is also used by the forward pass.
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()  # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        tokens_processed = (
            train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        )
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()
