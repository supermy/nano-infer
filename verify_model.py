#!/usr/bin/env python3
"""
验证 AWQ 反量化、Softmax 和采样逻辑
对比 Python 官方实现
"""
import torch
import numpy as np
import json

def load_model():
    """加载 Qwen3-4B-AWQ 模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = "../models/Qwen/Qwen3-4B-AWQ"
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    return model, tokenizer

def verify_awq_dequant():
    """验证 AWQ 反量化"""
    print("\n=== 验证 AWQ 反量化 ===")
    
    # 加载模型
    model, tokenizer = load_model()
    
    # 获取第一层的 q_proj 权重
    state_dict = model.state_dict()
    
    # 检查权重名称
    print("\n权重名称示例:")
    for key in list(state_dict.keys())[:10]:
        print(f"  {key}")
    
    # 获取 q_proj 权重
    q_weight = state_dict.get("model.layers.0.self_attn.q_proj.qweight", None)
    q_scales = state_dict.get("model.layers.0.self_attn.q_proj.scales", None)
    q_zeros = state_dict.get("model.layers.0.self_attn.q_proj.qzeros", None)
    
    if q_weight is not None:
        print(f"\nQ_proj 权重:")
        print(f"  qweight shape: {q_weight.shape}, dtype: {q_weight.dtype}")
        print(f"  qweight[0][:10]: {q_weight[0][:10]}")
        
    if q_scales is not None:
        print(f"  scales shape: {q_scales.shape}, dtype: {q_scales.dtype}")
        print(f"  scales[0][:10]: {q_scales[0][:10]}")
        
    if q_zeros is not None:
        print(f"  qzeros shape: {q_zeros.shape}, dtype: {q_zeros.dtype}")
        print(f"  qzeros[0][:10]: {q_zeros[0][:10]}")
    
    return model, tokenizer

def verify_inference():
    """验证推理过程"""
    print("\n=== 验证推理 ===")
    
    model, tokenizer = load_model()
    
    # 测试输入
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    print(f"\nInput: {prompt}")
    print(f"Input IDs: {input_ids}")
    
    # 推理
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
    print(f"\nLogits shape: {logits.shape}")
    
    # 获取最后一个 token 的 logits
    last_logits = logits[0, -1, :]
    print(f"Last token logits shape: {last_logits.shape}")
    
    # Top 10
    top10 = torch.topk(last_logits, 10)
    print(f"\nTop 10 tokens:")
    for i, (idx, score) in enumerate(zip(top10.indices, top10.values)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. ID={idx.item():>6}, score={score.item():.4f}, token='{token}'")
    
    # 生成几个 token
    print("\n=== 生成测试 ===")
    with torch.no_grad():
        # 只生成 5 个 token
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,  # greedy
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    output_text = tokenizer.decode(generated[0])
    print(f"Generated: {output_text}")
    
    # 对比每个 token
    print("\nGenerated tokens:")
    for i, tok_id in enumerate(generated[0]):
        token = tokenizer.decode([tok_id])
        print(f"  {i}: ID={tok_id.item()}, token='{token}'")

def verify_embed_tokens():
    """验证 embedding tokens"""
    print("\n=== 验证 Embedding Tokens ===")
    
    model, tokenizer = load_model()
    
    # 获取 embed_tokens
    embed_tokens = model.model.embed_tokens.weight.data
    print(f"Embed tokens shape: {embed_tokens.shape}")
    print(f"Embed tokens dtype: {embed_tokens.dtype}")
    print(f"Embed tokens[0][:10]: {embed_tokens[0][:10]}")
    
    # 查找 "Hello" 的 token ID
    hello_id = tokenizer.encode("Hello")
    print(f"\n'Hello' token IDs: {hello_id}")
    
    # 获取 "Hello" 的 embedding
    hello_emb = embed_tokens[hello_id[0]]
    print(f"Hello embedding: {hello_emb[:10]}")

if __name__ == "__main__":
    print("=" * 60)
    print("Python 官方实现验证")
    print("=" * 60)
    
    # 运行验证
    verify_awq_dequant()
    verify_embed_tokens()
    verify_inference()
