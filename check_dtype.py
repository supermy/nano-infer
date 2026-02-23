import json
import struct

with open('/Users/moyong/project/ai/models/Qwen/Qwen3-4B-AWQ/model.safetensors', 'rb') as f:
    header_size = struct.unpack('<Q', f.read(8))[0]
    header = f.read(header_size).decode('utf-8')
    meta = json.loads(header)
    
    # Check embed_tokens data_offsets
    for name, info in meta.items():
        if 'embed_tokens' in name:
            print(f'{name}:')
            print(f'  dtype={info["dtype"]}, shape={info["shape"]}')
            print(f'  data_offsets={info["data_offsets"]}')
    
    # Check first few tensors
    print('\nFirst 5 tensors:')
    for i, (name, info) in enumerate(meta.items()):
        if i >= 5:
            break
        print(f'  {name}: data_offsets={info.get("data_offsets", "N/A")}')
