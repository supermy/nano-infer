#!/usr/bin/env python3
"""
生成 token ID 查找表供 C 代码使用
"""
import json
from tokenizers import Tokenizer

# 加载 tokenizer
tokenizer = Tokenizer.from_file("../models/Qwen/Qwen3-4B-AWQ/tokenizer.json")

# 测试一些常见字符串
test_strings = [
    "Hello",
    "你好",
    "world",
    "The",
    "the",
    " a",
    " be",
    "Hello world",
    "This is a test",
    "AI",
    " artificial",
    " intelligence",
]

# 生成 token ID 查找表
token_lookup = {}
for text in test_strings:
    enc = tokenizer.encode(text)
    token_lookup[text] = enc.ids

print("Token lookup table:")
for text, ids in token_lookup.items():
    print(f"  '{text}' -> {ids}")

# 保存为 JSON
with open("token_lookup.json", "w") as f:
    json.dump(token_lookup, f, ensure_ascii=False)

print("\nSaved to token_lookup.json")

# 生成完整的 vocab 映射表（token -> id）
print("\nGenerating full vocab mapping...")
vocab = {}
for text, ids in token_lookup.items():
    if ids:
        vocab[text] = ids[0]

# 保存
with open("vocab_map.json", "w") as f:
    json.dump(vocab, f, ensure_ascii=False)

print(f"Saved {len(vocab)} entries to vocab_map.json")
