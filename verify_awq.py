#!/usr/bin/env python3
"""
验证 AWQ 反量化、Softmax 和采样逻辑
对比 Python 官方实现 - 简化版
"""
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import numpy as np
import json
from safetensors import safe_open

def check_awq_weights():
    """检查 AWQ 权重格式"""
    print("\n=== 检查 AWQ 权重格式 ===")
    
    model_path = "../models/Qwen/Qwen3-4B-AWQ/model.safetensors"
    
    with safe_open(model_path, framework="pt") as f:
        keys = list(f.keys())
        print(f"Total tensors: {len(keys)}")
        
        # 找到所有 q_proj 相关的 key
        q_keys = [k for k in keys if 'q_proj' in k.lower()]
        print(f"\nQ_proj keys: {q_keys[:5]}")
        
        # 检查 q_proj weight
        for key in keys:
            if 'q_proj' in key and 'qweight' in key:
                print(f"\n{key}:")
                tensor = f.get_tensor(key)
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                print(f"  Data[:20]: {tensor.flatten()[:20].tolist()}")
                
            if 'q_proj' in key and 'scales' in key:
                print(f"\n{key}:")
                tensor = f.get_tensor(key)
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                print(f"  Data[:20]: {tensor.flatten()[:20].tolist()}")
                
            if 'q_proj' in key and 'qzeros' in key:
                print(f"\n{key}:")
                tensor = f.get_tensor(key)
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                print(f"  Data[:20]: {tensor.flatten()[:20].tolist()}")

def check_embed_tokens():
    """检查 embedding tokens"""
    print("\n=== 检查 Embedding Tokens ===")
    
    model_path = "../models/Qwen/Qwen3-4B-AWQ/model.safetensors"
    
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            if 'embed_tokens' in key:
                print(f"\n{key}:")
                tensor = f.get_tensor(key)
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                print(f"  Data[0][:10]: {tensor[0][:10].tolist()}")

def check_lm_head():
    """检查 lm_head 权重"""
    print("\n=== 检查 lm_head 权重 ===")
    
    model_path = "../models/Qwen/Qwen3-4B-AWQ/model.safetensors"
    
    found = False
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            if 'lm_head' in key.lower():
                found = True
                print(f"\n{key}:")
                tensor = f.get_tensor(key)
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                if 'qweight' in key:
                    print(f"  Data[:20]: {tensor.flatten()[:20].tolist()}")
                elif 'scales' in key:
                    print(f"  Data[:10]: {tensor.flatten()[:10].tolist()}")
    
    if not found:
        print("No lm_head weight found!")

def verify_tokenizer():
    """验证 tokenizer"""
    print("\n=== 验证 Tokenizer ===")
    
    import tokenizers
    
    # 加载 tokenizer
    with open("../models/Qwen/Qwen3-4B-AWQ/tokenizer.json") as f:
        tok_data = json.load(f)
    
    print(f"Tokenizer type: {tok_data.get('model', {}).get('type', 'unknown')}")
    
    # 加载 vocab
    with open("../models/Qwen/Qwen3-4B-AWQ/vocab.json") as f:
        vocab = json.load(f)
    
    print(f"Vocab size: {len(vocab)}")
    
    # 检查 "Hello" 的 token
    with open("../models/Qwen/Qwen3-4B-AWQ/tokenizer.json") as f:
        content = f.read()
    
    # 使用 tokenizers 库
    from tokenizers import Tokenizer
    
    tokenizer = Tokenizer.from_file("../models/Qwen/Qwen3-4B-AWQ/tokenizer.json")
    
    # 编码 "Hello"
    enc = tokenizer.encode("Hello")
    print(f"\n'Hello' encoding:")
    print(f"  IDs: {enc.ids}")
    print(f"  Tokens: {enc.tokens}")
    
    # 解码一些 token
    print(f"\nDecode test:")
    for tok_id in [9707, 1481, 61379]:
        dec = tokenizer.decode([tok_id])
        print(f"  ID {tok_id} -> '{dec}'")
    
    # 打印 top tokens
    print(f"\nDecode first 10 token IDs:")
    for i in range(10):
        dec = tokenizer.decode([i])
        print(f"  ID {i} -> '{dec}'")

if __name__ == "__main__":
    print("=" * 60)
    print("AWQ 模型验证")
    print("=" * 60)
    
    check_awq_weights()
    check_embed_tokens()
    check_lm_head()
    verify_tokenizer()
