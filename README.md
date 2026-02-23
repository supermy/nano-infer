# Nano-Infer

A lightweight CPU inference engine for Qwen3-4B models written in pure C.

## Features

- **Pure C Implementation** - No external dependencies beyond standard C library
- **Multiple Quantization Support** - Supports FP16, AWQ, and FP8 quantized models
- **Memory Efficient** - Uses mmap for weight loading with minimal memory footprint
- **Cross-Platform** - Works on macOS (Apple Silicon) and Linux with architecture-specific optimizations
- **OpenMP Support** - Multi-threaded inference for better performance

## Supported Models

- Qwen3-4B (FP16/BF16)
- Qwen3-4B-AWQ (4-bit quantized)
- Qwen3-4B-Instruct-FP8

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
│   ├── model.h         # Model and KV cache structures
│   ├── safetensors.h   # Safetensors file reader
│   └── tokenizer.h     # Tokenizer interface
├── src/
│   ├── awq.c           # AWQ implementation
│   ├── config.c        # Config loading from JSON
│   ├── main.c          # Main entry point
│   ├── model.c         # Model loading and inference
│   ├── safetensors.c   # Safetensors parsing with mmap
│   └── tokenizer.c     # Tokenizer implementation
├── CMakeLists.txt
├── Makefile
└── README.md
```

## Technical Details

### Weight Loading
- Uses memory-mapped files (mmap) for efficient weight loading
- Supports safetensors format
- Lazy loading of model layers

### Quantization
- **AWQ**: 4-bit quantization with group-wise scales and zeros
- **FP8**: 8-bit floating point support
- Dequantization performed on-the-fly during inference

### KV Cache
- Per-layer KV cache management
- Supports variable sequence lengths

## Performance

Optimized for Apple Silicon (M1/M2/M3) with ARM NEON support:
- Architecture-specific compiler flags
- OpenMP parallelization for matrix operations

## License

MIT License

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the Qwen model series
- [AWQ](https://github.com/mit-han-lab/llm-awq) for the quantization method
