应用场景:
    端侧内存资源受限下小模型AWQ推理引擎；

### 已实现功能
- 纯 C 实现，无外部依赖
- AWQ 4-bit 量化支持
- mmap 高效权重加载
- OpenMP 多线程
- Safetensors 格式支持
- 支持模型路径
../models/Qwen/Qwen3-4B/
../models/Qwen/Qwen3-4B-AWQ/
../models/Qwen/Qwen3-4B-Instruct-2507-FP8

### 规划功能
- Prefix Caching - 缓存公共前缀
- Radix Attention - 高效前缀重复注意力
- Page Attention - 分页 KV Cache 管理
- KV Cache 优化 - 完整的 KV Cache 实现
- CPU Offloading - 分层卸载配置
- SIMD 支持 - AVX/AVX2/AVX512 优化
- FP8 支持 - 8-bit 浮点量化