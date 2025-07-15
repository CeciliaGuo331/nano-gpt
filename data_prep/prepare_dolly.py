"""
Dolly-15k dataset preprocessing and tokenization script.
https://huggingface.co/datasets/databricks/databricks-dolly-15k

This script downloads the Dolly-15k dataset, formats it for instruction-following,
tokenizes the text using `tiktoken`, and saves the tokenized data into shards
on disk.

Example usage:
$ python prepare_dolly.py
$ python prepare_dolly.py --output_dir my_dolly_shards --val_split_ratio 0.05
"""

import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess and tokenize the Dolly-15k dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dolly15k",
        help="Directory to save the processed data shards."
    )
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.1,
        help="Ratio of the dataset to use for the validation set (e.g., 0.1 for 10%)."
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=int(1e7),
        help="Number of tokens per shard file."
    )
    return parser.parse_args()

def format_dolly_sample(sample):
    """
    Formats a single Dolly sample into a structured prompt for training.
    The format includes instruction, context (if available), and response.
    """
    instruction = sample['instruction']
    context = sample['context']
    response = sample['response']
    
    if context:
        return f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

def tokenize(sample):
    """
    Tokenizes a single formatted document string and returns a numpy array of uint16 tokens.
    """
    # Initialize tokenizer and EOT token within the function for multiprocessing safety
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    
    # Format the sample into a single string
    text = format_dolly_sample(sample)
    
    # Tokenize the text. `encode_ordinary` does not add any special tokens.
    tokens = enc.encode_ordinary(text)
    # Manually append the end-of-text token to mark the end of a sequence.
    tokens.append(eot)
    
    tokens_np = np.array(tokens, dtype=np.uint16)
    # The original assertion is removed because dtype=np.uint16 handles the range check.
    return tokens_np

def process_and_save_split(dataset, split_name, output_dir, shard_size):
    """
    Tokenizes a dataset split using multiprocessing and saves it to sharded .npy files.
    
    This function follows a two-stage process:
    1. Tokenize all documents in parallel and collect them.
    2. Concatenate all tokens and write them to disk in fixed-size shards.
    """
    print(f"\nProcessing '{split_name}' split with {len(dataset)} samples...")
    
    # --- Stage 1: Tokenization using multiprocessing ---
    # Use half of the available CPU cores, with a minimum of 1.
    nprocs = max(1, os.cpu_count() // 2)
    
    all_tokens_list = []
    with mp.Pool(nprocs) as pool:
        # Create a progress bar for the tokenization process
        pbar = tqdm(total=len(dataset), desc=f"Tokenizing {split_name}", unit="samples")
        for tokens_np in pool.imap(tokenize, dataset, chunksize=16):
            all_tokens_list.append(tokens_np)
            pbar.update(1)
        pbar.close()

    # --- Stage 2: Concatenate and write to shards ---
    if not all_tokens_list:
        print(f"No tokens generated for '{split_name}' split. Skipping.")
        return

    # Concatenate all token arrays into one large array
    all_tokens = np.concatenate(all_tokens_list)
    total_tokens = len(all_tokens)
    print(f"Total tokens in '{split_name}' split: {total_tokens:,}")

    # Calculate the number of shards
    num_shards = (total_tokens + shard_size - 1) // shard_size
    print(f"Writing {num_shards} shard(s)...")

    # Write the concatenated tokens to sharded files
    for shard_index in tqdm(range(num_shards), desc=f"Writing {split_name} shards", unit="shards"):
        start_idx = shard_index * shard_size
        end_idx = min(start_idx + shard_size, total_tokens)
        
        shard_tokens = all_tokens[start_idx:end_idx]
        
        filename = os.path.join(output_dir, f"dolly_{split_name}_{shard_index:06d}.npy")
        np.save(filename, shard_tokens)
    
    print(f"Successfully saved {num_shards} shards for '{split_name}' split.")

def main():
    """Main function to orchestrate the dataset preparation."""
    args = parse_arguments()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. Load Dataset ---
    print("Loading Dolly-15k dataset from Hugging Face Hub...")
    # Using 'split="train"' as it's the only split available
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    # --- 2. Shuffle and Split Dataset ---
    # Shuffle the dataset for a random and representative split. Using a seed for reproducibility.
    print("Shuffling and splitting the dataset...")
    split_ds = ds.train_test_split(test_size=args.val_split_ratio, shuffle=True, seed=42)
    ds_train = split_ds['train']
    ds_val = split_ds['test']
    
    print(f"Dataset split:")
    print(f" - Train samples: {len(ds_train):,}")
    print(f" - Validation samples: {len(ds_val):,}")
    
    # --- 3. Process both splits ---
    process_and_save_split(ds_train, "train", args.output_dir, args.shard_size)
    process_and_save_split(ds_val, "val", args.output_dir, args.shard_size)
    
    print(f"\n✅ Done! All shards have been saved to '{args.output_dir}'")

if __name__ == "__main__":
    main()