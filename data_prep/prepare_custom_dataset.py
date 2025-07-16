"""
预处理自定义的 JSONL 格式指令数据集。

该脚本会读取一个 .jsonl 文件，将其随机打乱并按比例切分为训练集和验证集，
然后将每个集合格式化、token化，并分别保存为训练脚本所需的 .npy 格式。
"""

import os
import json
import argparse
import numpy as np
import tiktoken
import random
from multiprocessing import Pool
from tqdm import tqdm
from typing import List, Dict

def format_prompt(sample: dict) -> str:
    """
    根据样本字典，将其格式化为统一的 prompt 字符串。
    """
    instruction = sample.get('instruction', '')
    context = sample.get('context', '')
    response = sample.get('response', '')

    if context:
        return f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

def tokenize_sample(sample: dict):
    """
    对单个样本进行格式化和 token 化。
    """
    # 初始化 tokenizer 和 EOT token
    # 在每个子进程中单独初始化是安全的
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    
    # 格式化并编码
    full_prompt = format_prompt(sample)
    tokens = enc.encode_ordinary(full_prompt)
    tokens.append(eot) # 在每个样本的末尾添加结束符
    
    # 转换为 numpy 数组
    return np.array(tokens, dtype=np.uint16)

def process_and_save_split(samples: List[Dict], split_name: str, output_dir: str):
    """
    使用多进程处理一个数据集子集（如训练集或验证集）并保存。
    """
    if not samples:
        print(f"No samples for split '{split_name}', skipping.")
        return

    # 使用多进程进行并行处理
    num_processes = os.cpu_count() // 2 or 1
    print(f"Tokenizing {len(samples)} samples for '{split_name}' split using {num_processes} processes...")
    
    with Pool(num_processes) as pool:
        all_tokens_list = list(tqdm(pool.imap(tokenize_sample, samples, chunksize=16), total=len(samples), desc=f"Tokenizing {split_name}"))

    # 合并所有 token 数组
    all_tokens = np.concatenate(all_tokens_list)
    print(f"Total tokens in '{split_name}' split: {len(all_tokens):,}")

    # 保存为单个 .npy 分片文件
    output_filename = os.path.join(output_dir, f"{split_name}_00000.npy")
    np.save(output_filename, all_tokens)
    print(f"'{split_name}' split saved to: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and split a custom .jsonl dataset.")
    parser.add_argument("--input_file", type=str, required=True, help="输入的 .jsonl 数据集文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="处理后 .npy 分片的输出目录")
    parser.add_argument("--val_split_ratio", type=float, default=0.1, help="从数据集中划分为验证集的比例 (例如 0.1 表示 10%%)")
    parser.add_argument("--seed", type=int, default=1337, help="用于随机打乱和切分的种子，以保证可复现性")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 读取所有样本
    samples = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")
    
    print(f"Read {len(samples)} total samples from {args.input_file}")

    # 随机打乱数据集
    random.seed(args.seed)
    random.shuffle(samples)
    print("Shuffled dataset.")

    # 切分训练集和验证集
    val_size = int(len(samples) * args.val_split_ratio)
    if val_size == 0 and args.val_split_ratio > 0:
        print(f"Warning: Validation split ratio {args.val_split_ratio} is too small for the dataset size {len(samples)}. "
              f"At least one sample is needed. Treating all data as training data.")
    
    if val_size > 0:
        train_samples = samples[val_size:]
        val_samples = samples[:val_size]
        print(f"Splitting dataset: {len(train_samples)} training samples, {len(val_samples)} validation samples.")
    else:
        train_samples = samples
        val_samples = []
        print("No validation split created.")

    # 分别处理和保存训练集和验证集
    process_and_save_split(train_samples, "train", args.output_dir)
    process_and_save_split(val_samples, "val", args.output_dir)
    
    print(f"\n✅ Successfully processed dataset.")
    print(f"Output saved to directory: {args.output_dir}")

if __name__ == '__main__':
    main()