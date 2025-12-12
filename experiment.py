"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
seed = 1337
# device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# device_type = 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# ---------------------------------------------------------------------
def benchmark_generation(model, x, max_new_tokens_list, num_samples=5, use_cache=True, temperature=0.7, top_k=50):
    results = []
    
    for max_new_tokens in tqdm(max_new_tokens_list, desc="Benchmarking different token lengths"):
        # Warmup run (not counted)
        with torch.no_grad():
            _ = model.generate(x, 100, use_cache=use_cache, temperature=temperature, top_k=top_k)
        

        # Actual benchmark runs
        run_times, qkvtimes, kvtimes, attentimes = [], [], [], []
        for k in range(num_samples):
            t0 = time.time()
            with torch.no_grad():
                y = model.generate(x, max_new_tokens, use_cache=use_cache, 
                                 temperature=temperature, top_k=top_k)
            t1 = time.time()
            run_times.append(t1 - t0)
        
            total_qkv = total_kv_write = total_attn = 0.0
            for i, blk in enumerate(model.transformer.h):
                a = blk.attn
                total_qkv += a.qkv_time
                total_kv_write += a.kv_write_time
                total_attn += a.attn_time
                print(
                    f"[Timing] Layer {i}: QKV {a.qkv_time*1000:.3f} ms, "
                    f"KV-Write {a.kv_write_time*1000:.3f} ms, "
                    f"Attn {a.attn_time*1000:.3f} ms"
                )
            total = total_qkv + total_kv_write + total_attn
            qkvtimes.append(total_qkv)
            kvtimes.append(total_kv_write)
            attentimes.append(total_attn)
         
        # Calculate statistics
        mean_time = np.mean(run_times)
        std_time = np.std(run_times)
        tokens_per_sec = max_new_tokens / mean_time
        mean_time_qkv = np.mean(qkvtimes)
        std_time_qkv = np.std(qkvtimes)
        mean_time_kv = np.mean(kvtimes)
        std_time_kv = np.std(kvtimes)
        mean_time_attn = np.mean(attentimes)
        std_time_attn = np.std(attentimes)
        
        results.append({
            'max_new_tokens': max_new_tokens,
            'use_cache': use_cache,
            'mean_time': mean_time,
            # 'std_time': std_time,
            'mean_time_qkv': mean_time_qkv,
            # 'std_time_qkv': std_time_qkv,
            # 'mean_time_kv': mean_time_kv,
            # 'std_time_kv': std_time_kv,
            'mean_time_attn': mean_time_attn,
            # 'std_time_attn': std_time_attn,
            'tokens_per_sec': tokens_per_sec,
            'num_samples': num_samples,
            # 'kv_memory': model.kv_memory,
        })
    
    return pd.DataFrame(results)

def run_comparison(model, x, max_new_tokens_list, num_samples=5, temperature=0.7, top_k=50):
    # Run with cache
    df_cache = benchmark_generation(model, x, max_new_tokens_list, 
                                   num_samples=num_samples, use_cache=True,
                                   temperature=temperature, top_k=top_k)
    
    # Run without cache
    df_no_cache = benchmark_generation(model, x, max_new_tokens_list, 
                                      num_samples=num_samples, use_cache=False,
                                      temperature=temperature, top_k=top_k)
    
    # Combine results
    df = pd.concat([df_cache, df_no_cache]).sort_values(['max_new_tokens', 'use_cache'])
    return df

def plot_results(df):
    plt.figure(figsize=(12, 6))
    
    # Plot mean time
    plt.subplot(1, 2, 1)
    for use_cache, group in df.groupby('use_cache'):
        plt.errorbar(group['max_new_tokens'], group['mean_time'], 
                    yerr=group['std_time'], 
                    label=f"use_cache={use_cache}",
                    fmt='-o', capsize=5)
    plt.xlabel('Max New Tokens')
    plt.ylabel('Time (seconds)')
    plt.title('Generation Time Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot tokens per second
    plt.subplot(1, 2, 2)
    for use_cache, group in df.groupby('use_cache'):
        plt.plot(group['max_new_tokens'], group['tokens_per_sec'], 
                '-o', label=f"use_cache={use_cache}")
    plt.xlabel('Max New Tokens')
    plt.ylabel('Tokens per Second')
    plt.title('Generation Speed Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
# Configuration
max_new_tokens_list = [10, 20, 40, 80, 160]
num_samples = 1  # Number of runs per configuration
temperature = 0.7
top_k = 50

# Run comparison
df = run_comparison(model, x, max_new_tokens_list, num_samples, temperature, top_k)

# Print results table
print("\nBenchmark Results:")
print(df)

# Plot results
# plot_results(df)