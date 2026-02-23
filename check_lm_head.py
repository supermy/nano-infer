import json
import struct

with open('/Users/moyong/project/ai/models/Qwen/Qwen3-4B-AWQ/model.safetensors', 'rb') as f:
    header_size = struct.unpack('<Q', f.read(8))[0]
    header = f.read(header_size).decode('utf-8')
    meta = json.loads(header)
    
    # Check for lm_head
    print("Looking for lm_head tensors:")
    for name in meta.keys():
        if 'lm_head' in name.lower():
            print(f"  {name}")
    
    # Check if there's a separate lm_head
    print("\nChecking for separate lm_head weight file...")
    import os
    awq_dir = "/Users/moyong/project/ai/models/Qwen/Qwen3-4B-AWQ"
    for file in os.listdir(awq_dir):
        if 'lm_head' in file.lower():
            print(f"  Found: {file}")
