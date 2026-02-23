import json
import struct

with open('/Users/moyong/project/ai/models/Qwen/Qwen3-4B-AWQ/model.safetensors', 'rb') as f:
    header_size = struct.unpack('<Q', f.read(8))[0]
    header = f.read(header_size).decode('utf-8')
    meta = json.loads(header)
    
    # Check for q_norm and k_norm
    print("Looking for q_norm and k_norm:")
    for name in meta.keys():
        if 'q_norm' in name or 'k_norm' in name:
            print(f"  {name}: shape={meta[name]['shape']}, dtype={meta[name]['dtype']}")
    
    # Check all layer 0 tensors
    print("\nAll layer 0 tensors:")
    for name in meta.keys():
        if 'layers.0' in name:
            print(f"  {name}")
