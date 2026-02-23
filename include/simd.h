#ifndef SIMD_H
#define SIMD_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef enum {
    SIMD_NONE = 0,
    SIMD_SSE2 = 1,
    SIMD_SSE41 = 2,
    SIMD_AVX = 3,
    SIMD_AVX2 = 4,
    SIMD_AVX512 = 5,
    SIMD_NEON = 6,
} SIMDLevel;

SIMDLevel simd_detect_level(void);
const char* simd_level_name(SIMDLevel level);
bool simd_has_fma(void);
bool simd_has_avx512_vnni(void);

void simd_matmul_f32(const float* a, const float* b, float* c,
                     int m, int n, int k, SIMDLevel level);

void simd_matmul_awq(const float* input,
                     const uint8_t* qweight,
                     const uint16_t* scales,
                     const uint8_t* qzeros,
                     float* output,
                     int in_features, int out_features, int group_size,
                     SIMDLevel level);

void simd_softmax_f32(float* x, int n, SIMDLevel level);
void simd_rms_norm_f32(float* output, const float* input, const float* weight,
                       int hidden_size, float eps, SIMDLevel level);
void simd_silu_f32(float* x, int n, SIMDLevel level);
void simd_rope_f32(float* q, float* k, int seq_len, int num_heads,
                   int num_kv_heads, int head_dim, float rope_theta,
                   SIMDLevel level);

void simd_dequantize_awq(const uint8_t* qweight,
                         const float* scales,
                         const uint8_t* qzeros,
                         float* output,
                         int num_groups, int group_size, int out_features,
                         SIMDLevel level);

void simd_attention_score(const float* q, const float* k, float* scores,
                          int num_heads, int num_kv_heads, int head_dim,
                          int q_len, int kv_len, SIMDLevel level);

void simd_attention_reduce(const float* scores, const float* v, float* output,
                           int num_heads, int num_kv_heads, int head_dim,
                           int q_len, int kv_len, SIMDLevel level);

void simd_vec_add_f32(float* a, const float* b, int n, SIMDLevel level);
void simd_vec_mul_f32(float* a, const float* b, int n, SIMDLevel level);
void simd_vec_scale_f32(float* a, float scale, int n, SIMDLevel level);

void simd_memcpy_f32(float* dst, const float* src, int n, SIMDLevel level);
void simd_memset_f32(float* dst, float value, int n, SIMDLevel level);

float simd_reduce_sum_f32(const float* x, int n, SIMDLevel level);
float simd_reduce_max_f32(const float* x, int n, SIMDLevel level);
float simd_dot_product_f32(const float* a, const float* b, int n, SIMDLevel level);

#ifdef __AVX512F__
void simd_matmul_f32_avx512(const float* a, const float* b, float* c, int m, int n, int k);
void simd_softmax_f32_avx512(float* x, int n);
void simd_rms_norm_f32_avx512(float* output, const float* input, const float* weight,
                               int hidden_size, float eps);
#endif

#ifdef __AVX2__
void simd_matmul_f32_avx2(const float* a, const float* b, float* c, int m, int n, int k);
void simd_softmax_f32_avx2(float* x, int n);
void simd_rms_norm_f32_avx2(float* output, const float* input, const float* weight,
                             int hidden_size, float eps);
#endif

#ifdef __ARM_NEON
void simd_matmul_f32_neon(const float* a, const float* b, float* c, int m, int n, int k);
void simd_softmax_f32_neon(float* x, int n);
void simd_rms_norm_f32_neon(float* output, const float* input, const float* weight,
                             int hidden_size, float eps);
#endif

#endif
