# Nano-Infer

A lightweight CPU inference engine for Qwen3-4B models written in pure C.

## Features

### Core Features
- **Pure C Implementation** - No external dependencies beyond standard C library
- **AWQ Quantization** - 4-bit quantization support with on-the-fly dequantization
- **FP8 Support** - E4M3 and E5M2 8-bit floating point quantization
- **Memory Efficient** - Uses mmap for weight loading with minimal memory footprint
- **Cross-Platform** - Works on macOS (Apple Silicon/Intel) and Linux (x86_64/ARM64)
- **OpenMP Support** - Multi-threaded inference for better performance
- **Safetensors Format** - Direct loading of HuggingFace safetensors weights

### KV Cache & Attention
- **KV Cache** - Full KV cache implementation with append/get operations
- **Prefix Caching** - Cache common prompt prefixes for faster repeated inference
- **Page Attention** - Paged KV cache management for efficient memory usage
- **Radix Attention** - Efficient attention for repeated prefixes using radix tree

### Performance Optimizations
- **SIMD Support** - AVX/AVX2/AVX512 for x86, NEON for ARM
- **CPU Offloading** - Layer-wise offloading to disk for memory-constrained systems
- **Architecture Detection** - Automatic CPU feature detection and optimization

## Supported Models

- Qwen3-4B (BF16/FP16)
- Qwen3-4B-AWQ (4-bit quantized)
- Qwen3-4B-Instruct-FP8 (8-bit quantized)

## Build

### Prerequisites

- GCC or Clang compiler
- CMake (optional)
- OpenMP (optional, for multi-threading)

### Using Make

```bash
make
```

### Using CMake

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
./qwen3-infer [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <path>` | Model directory path | `../models/Qwen/Qwen3-4B/` |
| `-p, --prompt <text>` | Input prompt for generation | - |
| `-t, --temperature <f>` | Sampling temperature | 0.7 |
| `-k, --top-k <n>` | Top-k sampling | 20 |
| `-P, --top-p <f>` | Top-p nucleus sampling | 0.9 |
| `-l, --max-length <n>` | Maximum generation length | 512 |
| `-s, --seed <n>` | Random seed | - |
| `-i, --interactive` | Interactive chat mode | - |
| `-v, --verbose` | Verbose output | - |
| `-h, --help` | Show help message | - |

### Examples

Single prompt generation:
```bash
./qwen3-infer -m /path/to/model -p "Hello, how are you?"
```

Interactive chat mode:
```bash
./qwen3-infer -i -m /path/to/model
```

With custom parameters:
```bash
./qwen3-infer -m /path/to/model -p "Tell me a story" -t 0.8 -k 40 -l 1024
```

## Project Structure

```
nano-infer/
├── include/
│   ├── awq.h           # AWQ dequantization functions
│   ├── common.h        # Common types and utilities
│   ├── config.h        # Model configuration structures
│   ├── fp8.h           # FP8 quantization support
│   ├── kv_cache.h      # KV cache, prefix cache, page attention, radix attention
│   ├── model.h         # Model and KV cache structures
│   ├── offload.h       # CPU offloading configuration
│   ├── safetensors.h   # Safetensors file reader
│   ├── simd.h          # SIMD optimization interface
│   └── tokenizer.h     # Tokenizer interface
├── src/
│   ├── awq.c           # AWQ implementation
│   ├── config.c        # Config loading from JSON
│   ├── fp8.c           # FP8 quantization implementation
│   ├── kv_cache.c      # KV cache, prefix cache, page attention, radix attention
│   ├── main.c          # Main entry point
│   ├── model.c         # Model loading and inference
│   ├── offload.c       # CPU offloading implementation
│   ├── safetensors.c   # Safetensors parsing with mmap
│   ├── simd.c          # SIMD optimizations (AVX/AVX2/AVX512/NEON)
│   └── tokenizer.c     # Tokenizer implementation
├── CMakeLists.txt
├── Makefile
└── README.md
```

## Technical Details

### Weight Loading
- Uses memory-mapped files (mmap) for efficient weight loading
- Supports safetensors format (single file and sharded)
- Zero-copy tensor access via mmap pointers

### Quantization
- **AWQ**: 4-bit quantization with group-wise scales and zeros
- **FP8**: E4M3 and E5M2 formats with row-wise or tensor-wise scaling
- Dequantization performed on-the-fly during inference
- Supports BF16/FP16 to FP32 conversion

### Attention
- Grouped Query Attention (GQA) support
- RoPE positional embeddings
- Q/K normalization for AWQ models

### KV Cache System
- **Basic KV Cache**: Per-layer cache with append/get operations
- **Prefix Cache**: Hash-based caching for common prompt prefixes with LRU eviction
- **Page Attention**: Memory pool with block tables for multi-sequence support
- **Radix Attention**: Trie-based prefix matching for efficient reuse

### CPU Offloading
- Automatic layer management between CPU and disk
- Configurable memory thresholds
- Transparent layer loading on demand

## Performance

### SIMD Optimizations
- **x86_64**: AVX, AVX2, AVX-512 support with FMA
- **ARM64**: NEON support for Apple Silicon and ARM Linux
- Automatic detection and selection at runtime

### Platform-Specific
- **Apple Silicon**: `-mcpu=apple-m1` optimization
- **Linux x86**: `-march=native` with AVX/AVX2
- **Linux ARM**: `-march=armv8-a` with NEON
- OpenMP parallelization for matrix operations

## License

MIT License

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the Qwen model series
- [AWQ](https://github.com/mit-han-lab/llm-awq) for the quantization method
