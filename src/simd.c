#include "simd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#define HAS_AVX512 1
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX2 1
#endif

#if defined(__AVX__)
#include <immintrin.h>
#define HAS_AVX 1
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define HAS_NEON 1
#endif

SIMDLevel simd_detect_level(void) {
#if HAS_AVX512
    return SIMD_AVX512;
#elif HAS_AVX2
    return SIMD_AVX2;
#elif HAS_AVX
    return SIMD_AVX;
#elif HAS_NEON
    return SIMD_NEON;
#else
    return SIMD_NONE;
#endif
}

const char* simd_level_name(SIMDLevel level) {
    switch (level) {
        case SIMD_NONE:   return "None";
        case SIMD_SSE2:   return "SSE2";
        case SIMD_SSE41:  return "SSE4.1";
        case SIMD_AVX:    return "AVX";
        case SIMD_AVX2:   return "AVX2";
        case SIMD_AVX512: return "AVX-512";
        case SIMD_NEON:   return "NEON";
        default:          return "Unknown";
    }
}

bool simd_has_fma(void) {
#if defined(__FMA__)
    return true;
#else
    return false;
#endif
}

bool simd_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return true;
#else
    return false;
#endif
}

void simd_matmul_f32(const float* a, const float* b, float* c,
                     int m, int n, int k, SIMDLevel level) {
    (void)level;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }
}

void simd_matmul_awq(const float* input,
                     const uint8_t* qweight,
                     const uint16_t* scales,
                     const uint8_t* qzeros,
                     float* output,
                     int in_features, int out_features, int group_size,
                     SIMDLevel level) {
    (void)level;
    
    int num_groups = (in_features + group_size - 1) / group_size;
    int oc_int32 = out_features / 8;
    
    memset(output, 0, out_features * sizeof(float));
    
    const uint32_t* qw_int32 = (const uint32_t*)qweight;
    const uint32_t* qz_int32 = (const uint32_t*)qzeros;
    
    for (int i = 0; i < in_features; i++) {
        int g = i / group_size;
        
        const uint32_t* qw_row = qw_int32 + i * oc_int32;
        const uint16_t* scale_row = scales + g * out_features;
        const uint32_t* qz_row = qz_int32 + g * oc_int32;
        
        for (int j_int = 0; j_int < oc_int32; j_int++) {
            uint32_t w_packed = qw_row[j_int];
            uint32_t z_packed = qz_row[j_int];
            
            for (int k = 0; k < 8; k++) {
                int j = j_int * 8 + k;
                uint8_t w_val = (w_packed >> (k * 4)) & 0x0F;
                uint8_t z_val = (z_packed >> (k * 4)) & 0x0F;
                
                float scale = (float)scale_row[j];
                float w = ((float)w_val - (float)z_val) * scale;
                output[j] += input[i] * w;
            }
        }
    }
}

static inline float fp16_to_fp32(uint16_t fp16) {
    uint32_t sign = (fp16 >> 15) & 1;
    uint32_t exp = (fp16 >> 10) & 0x1F;
    uint32_t mant = fp16 & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            return (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
        }
    } else if (exp == 31) {
        return sign ? -INFINITY : INFINITY;
    } else {
        return (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15.0f);
    }
}

void simd_softmax_f32(float* x, int n, SIMDLevel level) {
    (void)level;
    
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

void simd_rms_norm_f32(float* output, const float* input, const float* weight,
                        int hidden_size, float eps, SIMDLevel level) {
    (void)level;
    
    float variance = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        variance += input[i] * input[i];
    }
    variance /= hidden_size;
    variance += eps;
    
    float inv_std = 1.0f / sqrtf(variance);
    
    for (int i = 0; i < hidden_size; i++) {
        output[i] = weight[i] * input[i] * inv_std;
    }
}

void simd_silu_f32(float* x, int n, SIMDLevel level) {
    (void)level;
    
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void simd_rope_f32(float* q, float* k, int seq_len, int num_heads,
                   int num_kv_heads, int head_dim, float rope_theta,
                   SIMDLevel level) {
    (void)level;
    
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < head_dim; d += 2) {
            float freq = 1.0f / powf(rope_theta, (float)d / head_dim);
            float angle = (float)s * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            
            for (int h = 0; h < num_heads; h++) {
                int idx = s * num_heads * head_dim + h * head_dim + d;
                float q0 = q[idx];
                float q1 = q[idx + 1];
                q[idx] = q0 * cos_val - q1 * sin_val;
                q[idx + 1] = q0 * sin_val + q1 * cos_val;
            }
            
            for (int h = 0; h < num_kv_heads; h++) {
                int idx = s * num_kv_heads * head_dim + h * head_dim + d;
                float k0 = k[idx];
                float k1 = k[idx + 1];
                k[idx] = k0 * cos_val - k1 * sin_val;
                k[idx + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
}

void simd_dequantize_awq(const uint8_t* qweight,
                         const float* scales,
                         const uint8_t* qzeros,
                         float* output,
                         int num_groups, int group_size, int out_features,
                         SIMDLevel level) {
    (void)level;
    
    int oc_int32 = out_features / 8;
    
    for (int g = 0; g < num_groups; g++) {
        const uint32_t* qw_row = (const uint32_t*)(qweight + g * group_size * oc_int32 * sizeof(uint32_t));
        const uint32_t* qz_row = (const uint32_t*)(qzeros + g * oc_int32 * sizeof(uint32_t));
        const float* scale_row = scales + g * out_features;
        
        for (int i = 0; i < group_size; i++) {
            for (int j_int = 0; j_int < oc_int32; j_int++) {
                uint32_t w_packed = qw_row[i * oc_int32 + j_int];
                uint32_t z_packed = qz_row[j_int];
                
                for (int k = 0; k < 8; k++) {
                    int j = j_int * 8 + k;
                    uint8_t w_val = (w_packed >> (k * 4)) & 0x0F;
                    uint8_t z_val = (z_packed >> (k * 4)) & 0x0F;
                    
                    float scale = scale_row[j];
                    output[(g * group_size + i) * out_features + j] = 
                        ((float)w_val - (float)z_val) * scale;
                }
            }
        }
    }
}

void simd_attention_score(const float* q, const float* k, float* scores,
                          int num_heads, int num_kv_heads, int head_dim,
                          int q_len, int kv_len, SIMDLevel level) {
    (void)level;
    
    int groups_per_head = num_heads / num_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for (int s = 0; s < q_len; s++) {
        for (int h = 0; h < num_heads; h++) {
            int kv_head = h / groups_per_head;
            
            for (int t = 0; t < kv_len; t++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q[s * num_heads * head_dim + h * head_dim + d] *
                             k[t * num_kv_heads * head_dim + kv_head * head_dim + d];
                }
                scores[s * num_heads * kv_len + h * kv_len + t] = score * scale;
            }
        }
    }
}

void simd_attention_reduce(const float* scores, const float* v, float* output,
                           int num_heads, int num_kv_heads, int head_dim,
                           int q_len, int kv_len, SIMDLevel level) {
    (void)level;
    
    int groups_per_head = num_heads / num_kv_heads;
    
    for (int s = 0; s < q_len; s++) {
        for (int h = 0; h < num_heads; h++) {
            int kv_head = h / groups_per_head;
            
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int t = 0; t < kv_len; t++) {
                    sum += scores[s * num_heads * kv_len + h * kv_len + t] *
                           v[t * num_kv_heads * head_dim + kv_head * head_dim + d];
                }
                output[s * num_heads * head_dim + h * head_dim + d] = sum;
            }
        }
    }
}

void simd_vec_add_f32(float* a, const float* b, int n, SIMDLevel level) {
    (void)level;
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

void simd_vec_mul_f32(float* a, const float* b, int n, SIMDLevel level) {
    (void)level;
    for (int i = 0; i < n; i++) {
        a[i] *= b[i];
    }
}

void simd_vec_scale_f32(float* a, float scale, int n, SIMDLevel level) {
    (void)level;
    for (int i = 0; i < n; i++) {
        a[i] *= scale;
    }
}

void simd_memcpy_f32(float* dst, const float* src, int n, SIMDLevel level) {
    (void)level;
    memcpy(dst, src, n * sizeof(float));
}

void simd_memset_f32(float* dst, float value, int n, SIMDLevel level) {
    (void)level;
    for (int i = 0; i < n; i++) {
        dst[i] = value;
    }
}

float simd_reduce_sum_f32(const float* x, int n, SIMDLevel level) {
    (void)level;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}

float simd_reduce_max_f32(const float* x, int n, SIMDLevel level) {
    (void)level;
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    return max_val;
}

float simd_dot_product_f32(const float* a, const float* b, int n, SIMDLevel level) {
    (void)level;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

#ifdef __AVX512F__
void simd_matmul_f32_avx512(const float* a, const float* b, float* c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            __m512 sum = _mm512_setzero_ps();
            for (int l = 0; l < k; l += 16) {
                int remaining = k - l;
                if (remaining >= 16) {
                    __m512 va = _mm512_loadu_ps(a + i * k + l);
                    __m512 vb = _mm512_loadu_ps(b + j * k + l);
                    sum = _mm512_fmadd_ps(va, vb, sum);
                } else {
                    for (int ll = l; ll < k; ll++) {
                        sum[0] += a[i * k + ll] * b[j * k + ll];
                    }
                }
            }
            c[i * n + j] = _mm512_reduce_add_ps(sum);
        }
    }
}

void simd_softmax_f32_avx512(float* x, int n) {
    __m512 max_vec = _mm512_set1_ps(x[0]);
    for (int i = 0; i < n; i += 16) {
        int remaining = n - i;
        if (remaining >= 16) {
            __m512 v = _mm512_loadu_ps(x + i);
            max_vec = _mm512_max_ps(max_vec, v);
        } else {
            for (int j = 0; j < remaining; j++) {
                if (x[i + j] > max_vec[0]) max_vec[0] = x[i + j];
            }
        }
    }
    float max_val = _mm512_reduce_max_ps(max_vec);
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    __m512 sum_broadcast = _mm512_set1_ps(sum);
    for (int i = 0; i < n; i += 16) {
        int remaining = n - i;
        if (remaining >= 16) {
            __m512 v = _mm512_loadu_ps(x + i);
            v = _mm512_div_ps(v, sum_broadcast);
            _mm512_storeu_ps(x + i, v);
        } else {
            for (int j = 0; j < remaining; j++) {
                x[i + j] /= sum;
            }
        }
    }
}

void simd_rms_norm_f32_avx512(float* output, const float* input, const float* weight,
                               int hidden_size, float eps) {
    __m512 sum_vec = _mm512_setzero_ps();
    for (int i = 0; i < hidden_size; i += 16) {
        int remaining = hidden_size - i;
        if (remaining >= 16) {
            __m512 v = _mm512_loadu_ps(input + i);
            sum_vec = _mm512_fmadd_ps(v, v, sum_vec);
        } else {
            for (int j = 0; j < remaining; j++) {
                sum_vec[0] += input[i + j] * input[i + j];
            }
        }
    }
    
    float variance = _mm512_reduce_add_ps(sum_vec) / hidden_size + eps;
    float inv_std = 1.0f / sqrtf(variance);
    __m512 inv_std_vec = _mm512_set1_ps(inv_std);
    
    for (int i = 0; i < hidden_size; i += 16) {
        int remaining = hidden_size - i;
        if (remaining >= 16) {
            __m512 in = _mm512_loadu_ps(input + i);
            __m512 w = _mm512_loadu_ps(weight + i);
            __m512 out = _mm512_mul_ps(_mm512_mul_ps(w, in), inv_std_vec);
            _mm512_storeu_ps(output + i, out);
        } else {
            for (int j = 0; j < remaining; j++) {
                output[i + j] = weight[i + j] * input[i + j] * inv_std;
            }
        }
    }
}
#endif

#ifdef __AVX2__
void simd_matmul_f32_avx2(const float* a, const float* b, float* c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            __m256 sum = _mm256_setzero_ps();
            for (int l = 0; l < k; l += 8) {
                int remaining = k - l;
                if (remaining >= 8) {
                    __m256 va = _mm256_loadu_ps(a + i * k + l);
                    __m256 vb = _mm256_loadu_ps(b + j * k + l);
                    sum = _mm256_fmadd_ps(va, vb, sum);
                } else {
                    for (int ll = l; ll < k; ll++) {
                        float vals[8] = {0};
                        vals[0] = a[i * k + ll] * b[j * k + ll];
                        __m256 v = _mm256_loadu_ps(vals);
                        sum = _mm256_add_ps(sum, v);
                    }
                }
            }
            float vals[8];
            _mm256_storeu_ps(vals, sum);
            c[i * n + j] = vals[0] + vals[1] + vals[2] + vals[3] + 
                           vals[4] + vals[5] + vals[6] + vals[7];
        }
    }
}

void simd_softmax_f32_avx2(float* x, int n) {
    __m256 max_vec = _mm256_set1_ps(x[0]);
    for (int i = 0; i < n; i += 8) {
        int remaining = n - i;
        if (remaining >= 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            max_vec = _mm256_max_ps(max_vec, v);
        } else {
            for (int j = 0; j < remaining; j++) {
                if (x[i + j] > max_vec[0]) {
                    max_vec = _mm256_set1_ps(x[i + j]);
                }
            }
        }
    }
    
    float vals[8];
    _mm256_storeu_ps(vals, max_vec);
    float max_val = vals[0];
    for (int i = 1; i < 8; i++) {
        if (vals[i] > max_val) max_val = vals[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    __m256 sum_broadcast = _mm256_set1_ps(sum);
    for (int i = 0; i < n; i += 8) {
        int remaining = n - i;
        if (remaining >= 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            v = _mm256_div_ps(v, sum_broadcast);
            _mm256_storeu_ps(x + i, v);
        } else {
            for (int j = 0; j < remaining; j++) {
                x[i + j] /= sum;
            }
        }
    }
}

void simd_rms_norm_f32_avx2(float* output, const float* input, const float* weight,
                             int hidden_size, float eps) {
    __m256 sum_vec = _mm256_setzero_ps();
    for (int i = 0; i < hidden_size; i += 8) {
        int remaining = hidden_size - i;
        if (remaining >= 8) {
            __m256 v = _mm256_loadu_ps(input + i);
            sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
        } else {
            for (int j = 0; j < remaining; j++) {
                float vals[8] = {input[i + j] * input[i + j], 0, 0, 0, 0, 0, 0, 0};
                __m256 v = _mm256_loadu_ps(vals);
                sum_vec = _mm256_add_ps(sum_vec, v);
            }
        }
    }
    
    float vals[8];
    _mm256_storeu_ps(vals, sum_vec);
    float variance = (vals[0] + vals[1] + vals[2] + vals[3] + 
                      vals[4] + vals[5] + vals[6] + vals[7]) / hidden_size + eps;
    float inv_std = 1.0f / sqrtf(variance);
    __m256 inv_std_vec = _mm256_set1_ps(inv_std);
    
    for (int i = 0; i < hidden_size; i += 8) {
        int remaining = hidden_size - i;
        if (remaining >= 8) {
            __m256 in = _mm256_loadu_ps(input + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            __m256 out = _mm256_mul_ps(_mm256_mul_ps(w, in), inv_std_vec);
            _mm256_storeu_ps(output + i, out);
        } else {
            for (int j = 0; j < remaining; j++) {
                output[i + j] = weight[i + j] * input[i + j] * inv_std;
            }
        }
    }
}
#endif

#ifdef __ARM_NEON
void simd_matmul_f32_neon(const float* a, const float* b, float* c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int l = 0; l < k; l += 4) {
                int remaining = k - l;
                if (remaining >= 4) {
                    float32x4_t va = vld1q_f32(a + i * k + l);
                    float32x4_t vb = vld1q_f32(b + j * k + l);
                    sum = vfmaq_f32(sum, va, vb);
                } else {
                    for (int ll = l; ll < k; ll++) {
                        sum = vaddq_f32(sum, vdupq_n_f32(a[i * k + ll] * b[j * k + ll]));
                    }
                }
            }
            float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
            c[i * n + j] = vget_lane_f32(sum2, 0) + vget_lane_f32(sum2, 1);
        }
    }
}

void simd_softmax_f32_neon(float* x, int n) {
    float32x4_t max_vec = vdupq_n_f32(x[0]);
    for (int i = 0; i < n; i += 4) {
        int remaining = n - i;
        if (remaining >= 4) {
            float32x4_t v = vld1q_f32(x + i);
            max_vec = vmaxq_f32(max_vec, v);
        } else {
            for (int j = 0; j < remaining; j++) {
                if (x[i + j] > vgetq_lane_f32(max_vec, 0)) {
                    max_vec = vdupq_n_f32(x[i + j]);
                }
            }
        }
    }
    
    float32x2_t max2 = vpmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
    max2 = vpmax_f32(max2, max2);
    float max_val = vget_lane_f32(max2, 0);
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    float32x4_t sum_broadcast = vdupq_n_f32(sum);
    for (int i = 0; i < n; i += 4) {
        int remaining = n - i;
        if (remaining >= 4) {
            float32x4_t v = vld1q_f32(x + i);
            v = vdivq_f32(v, sum_broadcast);
            vst1q_f32(x + i, v);
        } else {
            for (int j = 0; j < remaining; j++) {
                x[i + j] /= sum;
            }
        }
    }
}

void simd_rms_norm_f32_neon(float* output, const float* input, const float* weight,
                             int hidden_size, float eps) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (int i = 0; i < hidden_size; i += 4) {
        int remaining = hidden_size - i;
        if (remaining >= 4) {
            float32x4_t v = vld1q_f32(input + i);
            sum_vec = vfmaq_f32(sum_vec, v, v);
        } else {
            for (int j = 0; j < remaining; j++) {
                sum_vec = vaddq_f32(sum_vec, vdupq_n_f32(input[i + j] * input[i + j]));
            }
        }
    }
    
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
    sum2 = vpadd_f32(sum2, sum2);
    float variance = vget_lane_f32(sum2, 0) / hidden_size + eps;
    float inv_std = 1.0f / sqrtf(variance);
    float32x4_t inv_std_vec = vdupq_n_f32(inv_std);
    
    for (int i = 0; i < hidden_size; i += 4) {
        int remaining = hidden_size - i;
        if (remaining >= 4) {
            float32x4_t in = vld1q_f32(input + i);
            float32x4_t w = vld1q_f32(weight + i);
            float32x4_t out = vmulq_f32(vmulq_f32(w, in), inv_std_vec);
            vst1q_f32(output + i, out);
        } else {
            for (int j = 0; j < remaining; j++) {
                output[i + j] = weight[i + j] * input[i + j] * inv_std;
            }
        }
    }
}
#endif
