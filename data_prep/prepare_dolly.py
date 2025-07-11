"""
Dolly-15k dataset preprocessing
https://huggingface.co/datasets/databricks/databricks-dolly-15k
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python prepare_dolly.py
Will save shards to the local directory "dolly15k".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
local_dir = "dolly15k"
shard_size = int(1e7)  # 10M tokens per shard, more suitable for fine-tuning
val_split_ratio = 0.1  # 10% for validation

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
print("Loading Dolly-15k dataset...")
ds = load_dataset("databricks/databricks-dolly-15k", split="train")

# Split into train and validation
print(f"Splitting dataset: {int((1-val_split_ratio)*100)}% train, {int(val_split_ratio*100)}% validation")
val_size = int(len(ds) * val_split_ratio)
train_size = len(ds) - val_size
ds_train = ds.select(range(train_size))
ds_val = ds.select(range(train_size, len(ds)))
print(f"Train samples: {len(ds_train)}, Validation samples: {len(ds_val)}")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token

def format_dolly_sample(sample):
    """Format a single Dolly sample into a training prompt."""
    instruction = sample['instruction']
    context = sample['context']
    response = sample['response']
    
    # Format the prompt
    if context:
        prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    
    return prompt

def tokenize(sample):
    """Tokenizes a single document and returns a numpy array of uint16 tokens."""
    # Format the sample
    text = format_dolly_sample(sample)
    
    # Tokenize
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    """Write tokens to a numpy file."""
    np.save(filename, tokens_np)

def process_split(dataset, split_name):
    """Process and save a dataset split."""
    print(f"\nProcessing {split_name} split...")
    nprocs = max(1, os.cpu_count()//2)
    
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"{split_name} shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                filename = os.path.join(DATA_CACHE_DIR, f"dolly_{split_name}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                if remainder > 0:
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                if progress_bar is not None:
                    progress_bar.close()
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                if remainder < len(tokens):
                    all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                    token_count = len(tokens)-remainder
                else:
                    token_count = 0
        
        # write any remaining tokens as the last shard
        if token_count > 0:
            filename = os.path.join(DATA_CACHE_DIR, f"dolly_{split_name}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
            if progress_bar is not None:
                progress_bar.close()
            shard_index += 1
        
        print(f"{split_name}: Saved {shard_index} shards, ~{((shard_index-1) * shard_size + token_count):,} tokens")

# Process both splits
process_split(ds_train, "train")
process_split(ds_val, "val")

print(f"\nDone! All shards saved to {DATA_CACHE_DIR}")