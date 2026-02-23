#include "model.h"
#include "config.h"
#include "safetensors.h"
#include "awq.h"
#include "kv_cache.h"
#include "simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static SIMDLevel g_simd_level = SIMD_NONE;

void model_init_simd(void) {
    g_simd_level = simd_detect_level();
    printf("SIMD Level: %s\n", simd_level_name(g_simd_level));
}

static void rms_norm(float* output, const float* input, const float* weight, int hidden_size, float eps) {
    simd_rms_norm_f32(output, input, weight, hidden_size, eps, g_simd_level);
}

static void silu(float* x, int n) {
    simd_silu_f32(x, n, g_simd_level);
}

static void softmax(float* x, int n) {
    simd_softmax_f32(x, n, g_simd_level);
}

static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = (uint32_t)bf16 << 16;
    float f32;
    memcpy(&f32, &f32_bits, sizeof(float));
    return f32;
}

static inline float fp16_to_f32(uint16_t fp16) {
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

static float* convert_bf16_to_f32(const uint16_t* bf16_data, size_t num_elements) {
    float* f32_data = (float*)malloc(num_elements * sizeof(float));
    for (size_t i = 0; i < num_elements; i++) {
        f32_data[i] = bf16_to_f32(bf16_data[i]);
    }
    return f32_data;
}

static void awq_matmul(
    const float* input,
    const uint8_t* qweight,
    const uint16_t* scales_fp16,
    const uint8_t* qzeros,
    float* output,
    int in_features,
    int out_features,
    int group_size
) {
    int num_groups = (in_features + group_size - 1) / group_size;
    int oc_int32 = out_features / 8;
    
    memset(output, 0, out_features * sizeof(float));
    
    const uint32_t* qw_int32 = (const uint32_t*)qweight;
    const uint32_t* qz_int32 = (const uint32_t*)qzeros;
    
    for (int i = 0; i < in_features; i++) {
        int g = i / group_size;
        
        const uint32_t* qw_row = qw_int32 + i * oc_int32;
        const uint16_t* scale_row = scales_fp16 + g * out_features;
        const uint32_t* qz_row = qz_int32 + g * oc_int32;
        
        for (int j_int = 0; j_int < oc_int32; j_int++) {
            uint32_t w_packed = qw_row[j_int];
            uint32_t z_packed = qz_row[j_int];
            
            for (int k = 0; k < 8; k++) {
                int j = j_int * 8 + k;
                uint8_t w_val = (w_packed >> (k * 4)) & 0x0F;
                uint8_t z_val = (z_packed >> (k * 4)) & 0x0F;
                
                float scale = fp16_to_f32(scale_row[j]);
                float w = ((float)w_val - (float)z_val) * scale;
                output[j] += input[i] * w;
            }
        }
    }
}

static void matmul(
    const float* input,
    const float* weight,
    float* output,
    int in_features,
    int out_features
) {
    for (int j = 0; j < out_features; j++) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[i] * weight[j * in_features + i];
        }
        output[j] = sum;
    }
}

static void apply_rope_single(float* vec, int pos, int head_dim, float rope_theta) {
    for (int d = 0; d < head_dim; d += 2) {
        float freq = 1.0f / powf(rope_theta, (float)d / head_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        float v0 = vec[d];
        float v1 = vec[d + 1];
        vec[d] = v0 * cos_val - v1 * sin_val;
        vec[d + 1] = v0 * sin_val + v1 * cos_val;
    }
}

static void apply_rope(float* q, float* k, int seq_len, int num_heads, int num_kv_heads, int head_dim, float rope_theta) {
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

static void awq_attention(
    Qwen3Model* model,
    const float* hidden_states,
    float* output,
    int seq_len,
    int layer_idx
) {
    int hidden_size = model->config.hidden_size;
    int num_heads = model->config.num_attention_heads;
    int num_kv_heads = model->config.num_key_value_heads;
    int head_dim = model->config.head_dim;
    int group_size = model->config.quant_group_size;
    float eps = model->config.rms_norm_eps;
    float rope_theta = model->config.rope_theta;
    
    AWQWeight* q_awq = &model->q_proj_awq[layer_idx];
    AWQWeight* k_awq = &model->k_proj_awq[layer_idx];
    AWQWeight* v_awq = &model->v_proj_awq[layer_idx];
    AWQWeight* o_awq = &model->o_proj_awq[layer_idx];
    
    float* q = (float*)malloc(seq_len * num_heads * head_dim * sizeof(float));
    float* k = (float*)malloc(seq_len * num_kv_heads * head_dim * sizeof(float));
    float* v = (float*)malloc(seq_len * num_kv_heads * head_dim * sizeof(float));
    float* attn_out = (float*)malloc(seq_len * num_heads * head_dim * sizeof(float));
    
    for (int s = 0; s < seq_len; s++) {
        awq_matmul(hidden_states + s * hidden_size,
                   q_awq->qweight, q_awq->scales_fp16, q_awq->qzeros,
                   q + s * num_heads * head_dim,
                   hidden_size, num_heads * head_dim, group_size);
        
        awq_matmul(hidden_states + s * hidden_size,
                   k_awq->qweight, k_awq->scales_fp16, k_awq->qzeros,
                   k + s * num_kv_heads * head_dim,
                   hidden_size, num_kv_heads * head_dim, group_size);
        
        awq_matmul(hidden_states + s * hidden_size,
                   v_awq->qweight, v_awq->scales_fp16, v_awq->qzeros,
                   v + s * num_kv_heads * head_dim,
                   hidden_size, num_kv_heads * head_dim, group_size);
    }
    
    if (model->q_norm_weights && model->k_norm_weights) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < num_heads; h++) {
                float* q_head = q + s * num_heads * head_dim + h * head_dim;
                rms_norm(q_head, q_head, model->q_norm_weights + layer_idx * head_dim, head_dim, eps);
            }
            for (int h = 0; h < num_kv_heads; h++) {
                float* k_head = k + s * num_kv_heads * head_dim + h * head_dim;
                rms_norm(k_head, k_head, model->k_norm_weights + layer_idx * head_dim, head_dim, eps);
            }
        }
    }
    
    apply_rope(q, k, seq_len, num_heads, num_kv_heads, head_dim, rope_theta);
    
    memset(attn_out, 0, seq_len * num_heads * head_dim * sizeof(float));
    
    int groups_per_head = num_heads / num_kv_heads;
    
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < num_heads; h++) {
            int kv_head = h / groups_per_head;
            
            float* scores = (float*)malloc(seq_len * sizeof(float));
            
            for (int t = 0; t <= s; t++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q[s * num_heads * head_dim + h * head_dim + d] *
                             k[t * num_kv_heads * head_dim + kv_head * head_dim + d];
                }
                scores[t] = score / sqrtf((float)head_dim);
            }
            
            softmax(scores, s + 1);
            
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int t = 0; t <= s; t++) {
                    sum += scores[t] * v[t * num_kv_heads * head_dim + kv_head * head_dim + d];
                }
                attn_out[s * num_heads * head_dim + h * head_dim + d] = sum;
            }
            
            free(scores);
        }
    }
    
    for (int s = 0; s < seq_len; s++) {
        awq_matmul(attn_out + s * num_heads * head_dim,
                   o_awq->qweight, o_awq->scales_fp16, o_awq->qzeros,
                   output + s * hidden_size,
                   num_heads * head_dim, hidden_size, group_size);
    }
    
    free(q);
    free(k);
    free(v);
    free(attn_out);
}

static void attention(
    Qwen3Model* model,
    const float* hidden_states,
    float* output,
    int seq_len,
    int layer_idx
) {
    int hidden_size = model->config.hidden_size;
    int num_heads = model->config.num_attention_heads;
    int num_kv_heads = model->config.num_key_value_heads;
    int head_dim = model->config.head_dim;
    
    float* q_proj = model->q_proj_weight[layer_idx];
    float* k_proj = model->k_proj_weight[layer_idx];
    float* v_proj = model->v_proj_weight[layer_idx];
    float* o_proj = model->o_proj_weight[layer_idx];
    
    float* q = (float*)malloc(seq_len * num_heads * head_dim * sizeof(float));
    float* k = (float*)malloc(seq_len * num_kv_heads * head_dim * sizeof(float));
    float* v = (float*)malloc(seq_len * num_kv_heads * head_dim * sizeof(float));
    float* attn_out = (float*)malloc(seq_len * num_heads * head_dim * sizeof(float));
    
    for (int s = 0; s < seq_len; s++) {
        matmul(hidden_states + s * hidden_size, q_proj,
               q + s * num_heads * head_dim, hidden_size, num_heads * head_dim);
        matmul(hidden_states + s * hidden_size, k_proj,
               k + s * num_kv_heads * head_dim, hidden_size, num_kv_heads * head_dim);
        matmul(hidden_states + s * hidden_size, v_proj,
               v + s * num_kv_heads * head_dim, hidden_size, num_kv_heads * head_dim);
    }
    
    memset(attn_out, 0, seq_len * num_heads * head_dim * sizeof(float));
    
    int groups_per_head = num_heads / num_kv_heads;
    
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < num_heads; h++) {
            int kv_head = h / groups_per_head;
            
            float* scores = (float*)malloc(seq_len * sizeof(float));
            
            for (int t = 0; t <= s; t++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q[s * num_heads * head_dim + h * head_dim + d] *
                             k[t * num_kv_heads * head_dim + kv_head * head_dim + d];
                }
                scores[t] = score / sqrtf((float)head_dim);
            }
            
            softmax(scores, s + 1);
            
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int t = 0; t <= s; t++) {
                    sum += scores[t] * v[t * num_kv_heads * head_dim + kv_head * head_dim + d];
                }
                attn_out[s * num_heads * head_dim + h * head_dim + d] = sum;
            }
            
            free(scores);
        }
    }
    
    for (int s = 0; s < seq_len; s++) {
        matmul(attn_out + s * num_heads * head_dim, o_proj,
               output + s * hidden_size, num_heads * head_dim, hidden_size);
    }
    
    free(q);
    free(k);
    free(v);
    free(attn_out);
}

static void awq_mlp(
    Qwen3Model* model,
    const float* hidden_states,
    float* output,
    int seq_len,
    int layer_idx
) {
    int hidden_size = model->config.hidden_size;
    int intermediate_size = model->config.intermediate_size;
    int group_size = model->config.quant_group_size;
    
    AWQWeight* gate_awq = &model->gate_proj_awq[layer_idx];
    AWQWeight* up_awq = &model->up_proj_awq[layer_idx];
    AWQWeight* down_awq = &model->down_proj_awq[layer_idx];
    
    float* gate = (float*)malloc(intermediate_size * sizeof(float));
    float* up = (float*)malloc(intermediate_size * sizeof(float));
    
    for (int s = 0; s < seq_len; s++) {
        awq_matmul(hidden_states + s * hidden_size,
                   gate_awq->qweight, gate_awq->scales_fp16, gate_awq->qzeros,
                   gate, hidden_size, intermediate_size, group_size);
        
        awq_matmul(hidden_states + s * hidden_size,
                   up_awq->qweight, up_awq->scales_fp16, up_awq->qzeros,
                   up, hidden_size, intermediate_size, group_size);
        
        silu(gate, intermediate_size);
        
        for (int i = 0; i < intermediate_size; i++) {
            gate[i] *= up[i];
        }
        
        awq_matmul(gate,
                   down_awq->qweight, down_awq->scales_fp16, down_awq->qzeros,
                   output + s * hidden_size, intermediate_size, hidden_size, group_size);
    }
    
    free(gate);
    free(up);
}

static void mlp_single(Qwen3Model* model, const float* hidden_states, float* output, int layer_idx) {
    int hidden_size = model->config.hidden_size;
    int intermediate_size = model->config.intermediate_size;
    
    float* gate_proj = model->gate_proj_weight[layer_idx];
    float* up_proj = model->up_proj_weight[layer_idx];
    float* down_proj = model->down_proj_weight[layer_idx];
    
    float* gate = (float*)malloc(intermediate_size * sizeof(float));
    float* up = (float*)malloc(intermediate_size * sizeof(float));
    
    matmul(hidden_states, gate_proj, gate, hidden_size, intermediate_size);
    matmul(hidden_states, up_proj, up, hidden_size, intermediate_size);
    
    silu(gate, intermediate_size);
    
    for (int i = 0; i < intermediate_size; i++) {
        gate[i] *= up[i];
    }
    
    matmul(gate, down_proj, output, intermediate_size, hidden_size);
    
    free(gate);
    free(up);
}

static void attention_single(Qwen3Model* model, const float* hidden_states, float* output, int pos, int layer_idx) {
    int hidden_size = model->config.hidden_size;
    int num_heads = model->config.num_attention_heads;
    int num_kv_heads = model->config.num_key_value_heads;
    int head_dim = model->config.head_dim;
    float rope_theta = model->config.rope_theta;
    
    float* q_proj = model->q_proj_weight[layer_idx];
    float* k_proj = model->k_proj_weight[layer_idx];
    float* v_proj = model->v_proj_weight[layer_idx];
    float* o_proj = model->o_proj_weight[layer_idx];
    
    float* q = (float*)malloc(num_heads * head_dim * sizeof(float));
    float* k = (float*)malloc(num_kv_heads * head_dim * sizeof(float));
    float* v = (float*)malloc(num_kv_heads * head_dim * sizeof(float));
    
    matmul(hidden_states, q_proj, q, hidden_size, num_heads * head_dim);
    matmul(hidden_states, k_proj, k, hidden_size, num_kv_heads * head_dim);
    matmul(hidden_states, v_proj, v, hidden_size, num_kv_heads * head_dim);
    
    for (int h = 0; h < num_heads; h++) {
        apply_rope_single(q + h * head_dim, pos, head_dim, rope_theta);
    }
    for (int h = 0; h < num_kv_heads; h++) {
        apply_rope_single(k + h * head_dim, pos, head_dim, rope_theta);
    }
    
    if (model->kv_cache) {
        kv_cache_append(model->kv_cache, layer_idx, pos, k, v);
    }
    
    int kv_len = pos + 1;
    float* k_cached = (float*)malloc(kv_len * num_kv_heads * head_dim * sizeof(float));
    float* v_cached = (float*)malloc(kv_len * num_kv_heads * head_dim * sizeof(float));
    
    for (int t = 0; t <= pos; t++) {
        kv_cache_get(model->kv_cache, layer_idx, t, 
                     k_cached + t * num_kv_heads * head_dim,
                     v_cached + t * num_kv_heads * head_dim);
    }
    
    int groups_per_head = num_heads / num_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    
    float* attn_out = (float*)calloc(num_heads * head_dim, sizeof(float));
    
    for (int h = 0; h < num_heads; h++) {
        int kv_head = h / groups_per_head;
        
        float* scores = (float*)malloc(kv_len * sizeof(float));
        for (int t = 0; t < kv_len; t++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q[h * head_dim + d] * k_cached[t * num_kv_heads * head_dim + kv_head * head_dim + d];
            }
            scores[t] = score * scale;
        }
        
        softmax(scores, kv_len);
        
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int t = 0; t < kv_len; t++) {
                sum += scores[t] * v_cached[t * num_kv_heads * head_dim + kv_head * head_dim + d];
            }
            attn_out[h * head_dim + d] = sum;
        }
        
        free(scores);
    }
    
    matmul(attn_out, o_proj, output, num_heads * head_dim, hidden_size);
    
    free(q);
    free(k);
    free(v);
    free(k_cached);
    free(v_cached);
    free(attn_out);
}

static void awq_mlp_single(Qwen3Model* model, const float* hidden_states, float* output, int layer_idx) {
    int hidden_size = model->config.hidden_size;
    int intermediate_size = model->config.intermediate_size;
    int group_size = model->config.quant_group_size;
    
    AWQWeight* gate_awq = &model->gate_proj_awq[layer_idx];
    AWQWeight* up_awq = &model->up_proj_awq[layer_idx];
    AWQWeight* down_awq = &model->down_proj_awq[layer_idx];
    
    float* gate = (float*)malloc(intermediate_size * sizeof(float));
    float* up = (float*)malloc(intermediate_size * sizeof(float));
    
    awq_matmul(hidden_states, gate_awq->qweight, gate_awq->scales_fp16, gate_awq->qzeros,
               gate, hidden_size, intermediate_size, group_size);
    awq_matmul(hidden_states, up_awq->qweight, up_awq->scales_fp16, up_awq->qzeros,
               up, hidden_size, intermediate_size, group_size);
    
    silu(gate, intermediate_size);
    
    for (int i = 0; i < intermediate_size; i++) {
        gate[i] *= up[i];
    }
    
    awq_matmul(gate, down_awq->qweight, down_awq->scales_fp16, down_awq->qzeros,
               output, intermediate_size, hidden_size, group_size);
    
    free(gate);
    free(up);
}

static void awq_attention_single(Qwen3Model* model, const float* hidden_states, float* output, int pos, int layer_idx) {
    int hidden_size = model->config.hidden_size;
    int num_heads = model->config.num_attention_heads;
    int num_kv_heads = model->config.num_key_value_heads;
    int head_dim = model->config.head_dim;
    int group_size = model->config.quant_group_size;
    float eps = model->config.rms_norm_eps;
    float rope_theta = model->config.rope_theta;
    
    AWQWeight* q_awq = &model->q_proj_awq[layer_idx];
    AWQWeight* k_awq = &model->k_proj_awq[layer_idx];
    AWQWeight* v_awq = &model->v_proj_awq[layer_idx];
    AWQWeight* o_awq = &model->o_proj_awq[layer_idx];
    
    float* q = (float*)malloc(num_heads * head_dim * sizeof(float));
    float* k = (float*)malloc(num_kv_heads * head_dim * sizeof(float));
    float* v = (float*)malloc(num_kv_heads * head_dim * sizeof(float));
    
    awq_matmul(hidden_states, q_awq->qweight, q_awq->scales_fp16, q_awq->qzeros,
               q, hidden_size, num_heads * head_dim, group_size);
    awq_matmul(hidden_states, k_awq->qweight, k_awq->scales_fp16, k_awq->qzeros,
               k, hidden_size, num_kv_heads * head_dim, group_size);
    awq_matmul(hidden_states, v_awq->qweight, v_awq->scales_fp16, v_awq->qzeros,
               v, hidden_size, num_kv_heads * head_dim, group_size);
    
    if (model->q_norm_weights && model->k_norm_weights) {
        for (int h = 0; h < num_heads; h++) {
            rms_norm(q + h * head_dim, q + h * head_dim, model->q_norm_weights + layer_idx * head_dim, head_dim, eps);
        }
        for (int h = 0; h < num_kv_heads; h++) {
            rms_norm(k + h * head_dim, k + h * head_dim, model->k_norm_weights + layer_idx * head_dim, head_dim, eps);
        }
    }
    
    for (int h = 0; h < num_heads; h++) {
        apply_rope_single(q + h * head_dim, pos, head_dim, rope_theta);
    }
    for (int h = 0; h < num_kv_heads; h++) {
        apply_rope_single(k + h * head_dim, pos, head_dim, rope_theta);
    }
    
    if (model->kv_cache) {
        kv_cache_append(model->kv_cache, layer_idx, pos, k, v);
    }
    
    int kv_len = pos + 1;
    float* k_cached = (float*)malloc(kv_len * num_kv_heads * head_dim * sizeof(float));
    float* v_cached = (float*)malloc(kv_len * num_kv_heads * head_dim * sizeof(float));
    
    for (int t = 0; t <= pos; t++) {
        kv_cache_get(model->kv_cache, layer_idx, t, 
                     k_cached + t * num_kv_heads * head_dim,
                     v_cached + t * num_kv_heads * head_dim);
    }
    
    int groups_per_head = num_heads / num_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    
    float* attn_out = (float*)calloc(num_heads * head_dim, sizeof(float));
    
    for (int h = 0; h < num_heads; h++) {
        int kv_head = h / groups_per_head;
        
        float* scores = (float*)malloc(kv_len * sizeof(float));
        for (int t = 0; t < kv_len; t++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q[h * head_dim + d] * k_cached[t * num_kv_heads * head_dim + kv_head * head_dim + d];
            }
            scores[t] = score * scale;
        }
        
        softmax(scores, kv_len);
        
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int t = 0; t < kv_len; t++) {
                sum += scores[t] * v_cached[t * num_kv_heads * head_dim + kv_head * head_dim + d];
            }
            attn_out[h * head_dim + d] = sum;
        }
        
        free(scores);
    }
    
    awq_matmul(attn_out, o_awq->qweight, o_awq->scales_fp16, o_awq->qzeros,
               output, num_heads * head_dim, hidden_size, group_size);
    
    free(q);
    free(k);
    free(v);
    free(k_cached);
    free(v_cached);
    free(attn_out);
}

static void mlp(
    Qwen3Model* model,
    const float* hidden_states,
    float* output,
    int seq_len,
    int layer_idx
) {
    int hidden_size = model->config.hidden_size;
    int intermediate_size = model->config.intermediate_size;
    
    float* gate_proj = model->gate_proj_weight[layer_idx];
    float* up_proj = model->up_proj_weight[layer_idx];
    float* down_proj = model->down_proj_weight[layer_idx];
    
    float* gate = (float*)malloc(intermediate_size * sizeof(float));
    float* up = (float*)malloc(intermediate_size * sizeof(float));
    
    for (int s = 0; s < seq_len; s++) {
        matmul(hidden_states + s * hidden_size, gate_proj, gate, hidden_size, intermediate_size);
        matmul(hidden_states + s * hidden_size, up_proj, up, hidden_size, intermediate_size);
        
        silu(gate, intermediate_size);
        
        for (int i = 0; i < intermediate_size; i++) {
            gate[i] *= up[i];
        }
        
        matmul(gate, down_proj, output + s * hidden_size, intermediate_size, hidden_size);
    }
    
    free(gate);
    free(up);
}

static float* forward(Qwen3Model* model, const int* input_ids, int seq_len) {
    int hidden_size = model->config.hidden_size;
    int vocab_size = model->config.vocab_size;
    int num_layers = model->config.num_hidden_layers;
    float eps = model->config.rms_norm_eps;
    bool is_awq = (model->config.quant_method == QUANT_AWQ);
    
    float* hidden_states = (float*)malloc(seq_len * hidden_size * sizeof(float));
    
    for (int s = 0; s < seq_len; s++) {
        int token_id = input_ids[s];
        if (token_id >= 0 && token_id < vocab_size) {
            if (model->embed_is_bf16 && model->embed_tokens_bf16) {
                for (int h = 0; h < hidden_size; h++) {
                    uint16_t bf16_val = model->embed_tokens_bf16[token_id * hidden_size + h];
                    hidden_states[s * hidden_size + h] = bf16_to_f32(bf16_val);
                }
            } else if (model->embed_tokens) {
                memcpy(hidden_states + s * hidden_size, 
                       model->embed_tokens + token_id * hidden_size, 
                       hidden_size * sizeof(float));
            } else {
                memset(hidden_states + s * hidden_size, 0, hidden_size * sizeof(float));
            }
        } else {
            memset(hidden_states + s * hidden_size, 0, hidden_size * sizeof(float));
        }
    }
    
    float* layer_input = (float*)malloc(seq_len * hidden_size * sizeof(float));
    float* attn_output = (float*)malloc(seq_len * hidden_size * sizeof(float));
    float* mlp_output = (float*)malloc(seq_len * hidden_size * sizeof(float));
    
    for (int layer = 0; layer < num_layers; layer++) {
        memcpy(layer_input, hidden_states, seq_len * hidden_size * sizeof(float));
        
        for (int s = 0; s < seq_len; s++) {
            rms_norm(hidden_states + s * hidden_size, hidden_states + s * hidden_size,
                     model->input_layernorm_weights + layer * hidden_size,
                     hidden_size, eps);
        }
        
        if (is_awq) {
            awq_attention(model, hidden_states, attn_output, seq_len, layer);
        } else {
            attention(model, hidden_states, attn_output, seq_len, layer);
        }
        
        for (int s = 0; s < seq_len; s++) {
            for (int i = 0; i < hidden_size; i++) {
                hidden_states[s * hidden_size + i] = layer_input[s * hidden_size + i] + attn_output[s * hidden_size + i];
            }
        }
        
        memcpy(layer_input, hidden_states, seq_len * hidden_size * sizeof(float));
        
        for (int s = 0; s < seq_len; s++) {
            rms_norm(hidden_states + s * hidden_size, hidden_states + s * hidden_size,
                     model->post_attention_layernorm_weights + layer * hidden_size,
                     hidden_size, eps);
        }
        
        if (is_awq) {
            awq_mlp(model, hidden_states, mlp_output, seq_len, layer);
        } else {
            mlp(model, hidden_states, mlp_output, seq_len, layer);
        }
        
        for (int s = 0; s < seq_len; s++) {
            for (int i = 0; i < hidden_size; i++) {
                hidden_states[s * hidden_size + i] = layer_input[s * hidden_size + i] + mlp_output[s * hidden_size + i];
            }
        }
    }
    
    free(layer_input);
    free(attn_output);
    free(mlp_output);
    
    // Apply final layer norm
    if (model->final_layernorm_weight) {
        for (int s = 0; s < seq_len; s++) {
            rms_norm(hidden_states + s * hidden_size, hidden_states + s * hidden_size,
                     model->final_layernorm_weight, hidden_size, eps);
        }
    }
    
    return hidden_states;
}

static int sample_top_k(float* logits, int vocab_size, int top_k, float temperature) {
    if (top_k <= 0 || top_k > vocab_size) {
        top_k = vocab_size;
    }
    
    int* indices = (int*)malloc(vocab_size * sizeof(int));
    for (int i = 0; i < vocab_size; i++) {
        indices[i] = i;
    }
    
    for (int i = 0; i < top_k; i++) {
        int max_idx = i;
        for (int j = i + 1; j < vocab_size; j++) {
            if (logits[indices[j]] > logits[indices[max_idx]]) {
                max_idx = j;
            }
        }
        int temp = indices[i];
        indices[i] = indices[max_idx];
        indices[max_idx] = temp;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        logits[indices[i]] = expf(logits[indices[i]] / temperature);
        sum += logits[indices[i]];
    }
    
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        cumsum += logits[indices[i]] / sum;
        if (r < cumsum) {
            int result = indices[i];
            free(indices);
            return result;
        }
    }
    
    int result = indices[top_k - 1];
    free(indices);
    return result;
}

static void* get_tensor_ptr(SafetensorsReader** readers, int num_readers, const char* name, size_t* size) {
    for (int r = 0; r < num_readers; r++) {
        void* ptr = NULL;
        if (safetensors_get_tensor_ptr(readers[r], name, &ptr, size) && ptr) {
            return ptr;
        }
    }
    return NULL;
}

static bool has_awq_weights(SafetensorsReader** readers, int num_readers) {
    for (int r = 0; r < num_readers; r++) {
        for (size_t t = 0; t < readers[r]->num_tensors; t++) {
            if (strstr(readers[r]->tensor_names[t], "qweigh")) {
                return true;
            }
        }
    }
    return false;
}

static bool load_awq_weight(SafetensorsReader** readers, int num_readers, 
                            AWQWeight* awq, const char* proj_name, int layer_idx,
                            int in_features, int out_features) {
    char name[256];
    size_t size;
    
    snprintf(name, sizeof(name), "model.layers.%d.%s.qweight", layer_idx, proj_name);
    awq->qweight = (uint8_t*)get_tensor_ptr(readers, num_readers, name, &size);
    
    snprintf(name, sizeof(name), "model.layers.%d.%s.scales", layer_idx, proj_name);
    awq->scales_fp16 = (uint16_t*)get_tensor_ptr(readers, num_readers, name, &size);
    
    snprintf(name, sizeof(name), "model.layers.%d.%s.qzeros", layer_idx, proj_name);
    awq->qzeros = (uint8_t*)get_tensor_ptr(readers, num_readers, name, &size);
    
    awq->in_features = in_features;
    awq->out_features = out_features;
    
    return (awq->qweight && awq->scales_fp16 && awq->qzeros);
}

Qwen3Model* model_load(const char* model_dir, const ModelConfig* config) {
    Qwen3Model* model = (Qwen3Model*)calloc(1, sizeof(Qwen3Model));
    if (!model) return NULL;
    
    memcpy(&model->config, config, sizeof(ModelConfig));
    model->num_layers_total = config->num_hidden_layers;
    
    char path[512];
    
    SafetensorsReader* reader1 = NULL;
    SafetensorsReader* reader2 = NULL;
    SafetensorsReader* reader3 = NULL;
    
    // Try single file first (common for AWQ models)
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    reader1 = safetensors_open(path);
    
    // If single file not found, try sharded files
    if (!reader1) {
        snprintf(path, sizeof(path), "%s/model-00001-of-00003.safetensors", model_dir);
        reader1 = safetensors_open(path);
        
        snprintf(path, sizeof(path), "%s/model-00002-of-00003.safetensors", model_dir);
        reader2 = safetensors_open(path);
        
        snprintf(path, sizeof(path), "%s/model-00003-of-00003.safetensors", model_dir);
        reader3 = safetensors_open(path);
    }
    
    if (!reader1 && !reader2 && !reader3) {
        fprintf(stderr, "Failed to open any model files\n");
        free(model);
        return NULL;
    }
    
    int num_readers = 0;
    if (reader1) num_readers++;
    if (reader2) num_readers++;
    if (reader3) num_readers++;
    
    model->readers = (SafetensorsReader**)malloc(num_readers * sizeof(SafetensorsReader*));
    model->num_readers = 0;
    if (reader1) model->readers[model->num_readers++] = reader1;
    if (reader2) model->readers[model->num_readers++] = reader2;
    if (reader3) model->readers[model->num_readers++] = reader3;
    
    printf("Loaded %d safetensors reader(s)\n", model->num_readers);
    
    int hidden_size = config->hidden_size;
    int vocab_size = config->vocab_size;
    int num_layers = config->num_hidden_layers;
    int intermediate_size = config->intermediate_size;
    int num_heads = config->num_attention_heads;
    int num_kv_heads = config->num_key_value_heads;
    int head_dim = config->head_dim;
    
    bool has_awq = has_awq_weights(model->readers, model->num_readers);
    
    if (has_awq) {
        printf("Detected AWQ quantized weights, loading in quantized form...\n");
        model->config.quant_method = QUANT_AWQ;
        model->config.quant_bits = 4;
        model->config.quant_group_size = 128;
    }
    
    size_t embed_size;
    uint16_t* embed_bf16 = (uint16_t*)get_tensor_ptr(model->readers, model->num_readers, 
                                                  "model.embed_tokens.weight", &embed_size);
    if (embed_bf16) {
        size_t num_embed = embed_size / sizeof(uint16_t);
        model->embed_tokens_bf16 = embed_bf16;
        model->embed_is_bf16 = true;
        printf("Loaded embed_tokens in BF16 (mmap, no conversion): %zu elements\n", num_embed);
    }
    
    model->input_layernorm_weights = (float*)malloc(num_layers * hidden_size * sizeof(float));
    model->post_attention_layernorm_weights = (float*)malloc(num_layers * hidden_size * sizeof(float));
    
    size_t norm_size;
    uint16_t* norm_bf16 = (uint16_t*)get_tensor_ptr(model->readers, model->num_readers,
                                                            "model.norm.weight", &norm_size);
    if (norm_bf16) {
        model->final_layernorm_weight = convert_bf16_to_f32(norm_bf16, norm_size / sizeof(uint16_t));
    }
    
    model->lm_head_weight = (float*)get_tensor_ptr(model->readers, model->num_readers,
                                                    "lm_head.weight", NULL);
    
    if (has_awq) {
        model->q_proj_awq = (AWQWeight*)calloc(num_layers, sizeof(AWQWeight));
        model->k_proj_awq = (AWQWeight*)calloc(num_layers, sizeof(AWQWeight));
        model->v_proj_awq = (AWQWeight*)calloc(num_layers, sizeof(AWQWeight));
        model->o_proj_awq = (AWQWeight*)calloc(num_layers, sizeof(AWQWeight));
        model->gate_proj_awq = (AWQWeight*)calloc(num_layers, sizeof(AWQWeight));
        model->up_proj_awq = (AWQWeight*)calloc(num_layers, sizeof(AWQWeight));
        model->down_proj_awq = (AWQWeight*)calloc(num_layers, sizeof(AWQWeight));
        
        for (int l = 0; l < num_layers; l++) {
            load_awq_weight(model->readers, model->num_readers, &model->q_proj_awq[l],
                           "self_attn.q_proj", l, hidden_size, num_heads * head_dim);
            load_awq_weight(model->readers, model->num_readers, &model->k_proj_awq[l],
                           "self_attn.k_proj", l, hidden_size, num_kv_heads * head_dim);
            load_awq_weight(model->readers, model->num_readers, &model->v_proj_awq[l],
                           "self_attn.v_proj", l, hidden_size, num_kv_heads * head_dim);
            load_awq_weight(model->readers, model->num_readers, &model->o_proj_awq[l],
                           "self_attn.o_proj", l, num_heads * head_dim, hidden_size);
            
            load_awq_weight(model->readers, model->num_readers, &model->gate_proj_awq[l],
                           "mlp.gate_proj", l, hidden_size, intermediate_size);
            load_awq_weight(model->readers, model->num_readers, &model->up_proj_awq[l],
                           "mlp.up_proj", l, hidden_size, intermediate_size);
            load_awq_weight(model->readers, model->num_readers, &model->down_proj_awq[l],
                           "mlp.down_proj", l, intermediate_size, hidden_size);
        }
    } else {
        model->q_proj_weight = (float**)malloc(num_layers * sizeof(float*));
        model->k_proj_weight = (float**)malloc(num_layers * sizeof(float*));
        model->v_proj_weight = (float**)malloc(num_layers * sizeof(float*));
        model->o_proj_weight = (float**)malloc(num_layers * sizeof(float*));
        model->gate_proj_weight = (float**)malloc(num_layers * sizeof(float*));
        model->up_proj_weight = (float**)malloc(num_layers * sizeof(float*));
        model->down_proj_weight = (float**)malloc(num_layers * sizeof(float*));
        
        for (int l = 0; l < num_layers; l++) {
            model->q_proj_weight[l] = (float*)malloc(num_heads * head_dim * hidden_size * sizeof(float));
            model->k_proj_weight[l] = (float*)malloc(num_kv_heads * head_dim * hidden_size * sizeof(float));
            model->v_proj_weight[l] = (float*)malloc(num_kv_heads * head_dim * hidden_size * sizeof(float));
            model->o_proj_weight[l] = (float*)malloc(hidden_size * num_heads * head_dim * sizeof(float));
            model->gate_proj_weight[l] = (float*)malloc(intermediate_size * hidden_size * sizeof(float));
            model->up_proj_weight[l] = (float*)malloc(intermediate_size * hidden_size * sizeof(float));
            model->down_proj_weight[l] = (float*)malloc(hidden_size * intermediate_size * sizeof(float));
        }
    }
    
    printf("Loading layer norm weights...\n");
    for (int l = 0; l < num_layers; l++) {
        char name[256];
        size_t size;
        
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
        uint16_t* ptr_bf16 = (uint16_t*)get_tensor_ptr(model->readers, model->num_readers, name, &size);
        if (ptr_bf16) {
            for (int i = 0; i < hidden_size; i++) {
                model->input_layernorm_weights[l * hidden_size + i] = bf16_to_f32(ptr_bf16[i]);
            }
        }
        
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
        ptr_bf16 = (uint16_t*)get_tensor_ptr(model->readers, model->num_readers, name, &size);
        if (ptr_bf16) {
            for (int i = 0; i < hidden_size; i++) {
                model->post_attention_layernorm_weights[l * hidden_size + i] = bf16_to_f32(ptr_bf16[i]);
            }
        }
    }
    
    if (has_awq) {
        model->q_norm_weights = (float*)malloc(num_layers * head_dim * sizeof(float));
        model->k_norm_weights = (float*)malloc(num_layers * head_dim * sizeof(float));
        
        printf("Loading q_norm and k_norm weights...\n");
        for (int l = 0; l < num_layers; l++) {
            char name[256];
            size_t size;
            
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", l);
            uint16_t* ptr_bf16 = (uint16_t*)get_tensor_ptr(model->readers, model->num_readers, name, &size);
            if (ptr_bf16) {
                for (int i = 0; i < head_dim; i++) {
                    model->q_norm_weights[l * head_dim + i] = bf16_to_f32(ptr_bf16[i]);
                }
            }
            
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", l);
            ptr_bf16 = (uint16_t*)get_tensor_ptr(model->readers, model->num_readers, name, &size);
            if (ptr_bf16) {
                for (int i = 0; i < head_dim; i++) {
                    model->k_norm_weights[l * head_dim + i] = bf16_to_f32(ptr_bf16[i]);
                }
            }
        }
    }
    
    if (!has_awq) {
        printf("Loading full precision weights...\n");
        for (int l = 0; l < num_layers; l++) {
            char name[256];
            size_t size;
            
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weigh", l);
            float* ptr = (float*)get_tensor_ptr(model->readers, model->num_readers, name, &size);
            if (ptr) memcpy(model->q_proj_weight[l], ptr, size);
            
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weigh", l);
            ptr = (float*)get_tensor_ptr(model->readers, model->num_readers, name, &size);
            if (ptr) memcpy(model->k_proj_weight[l], ptr, size);
            
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weigh", l);
            ptr = (float*)get_tensor_ptr(model->readers, model->num_readers, name, &size);
            if (ptr) memcpy(model->v_proj_weight[l], ptr, size);
            
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weigh", l);
            ptr = (float*)get_tensor_ptr(model->readers, model->num_readers, name, &size);
            if (ptr) memcpy(model->o_proj_weight[l], ptr, size);
            
            snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weigh", l);
            ptr = (float*)get_tensor_ptr(model->readers, model->num_readers, name, &size);
            if (ptr) memcpy(model->gate_proj_weight[l], ptr, size);
            
            snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weigh", l);
            ptr = (float*)get_tensor_ptr(model->readers, model->num_readers, name, &size);
            if (ptr) memcpy(model->up_proj_weight[l], ptr, size);
            
            snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weigh", l);
            ptr = (float*)get_tensor_ptr(model->readers, model->num_readers, name, &size);
            if (ptr) memcpy(model->down_proj_weight[l], ptr, size);
        }
    }
    
    printf("Model loaded successfully\n");
    return model;
}

void model_free(Qwen3Model* model) {
    if (!model) return;
    
    if (model->input_layernorm_weights) free(model->input_layernorm_weights);
    if (model->post_attention_layernorm_weights) free(model->post_attention_layernorm_weights);
    if (model->q_norm_weights) free(model->q_norm_weights);
    if (model->k_norm_weights) free(model->k_norm_weights);
    
    if (model->config.quant_method == QUANT_AWQ) {
        if (model->q_proj_awq) free(model->q_proj_awq);
        if (model->k_proj_awq) free(model->k_proj_awq);
        if (model->v_proj_awq) free(model->v_proj_awq);
        if (model->o_proj_awq) free(model->o_proj_awq);
        if (model->gate_proj_awq) free(model->gate_proj_awq);
        if (model->up_proj_awq) free(model->up_proj_awq);
        if (model->down_proj_awq) free(model->down_proj_awq);
    } else {
        for (int i = 0; i < model->num_layers_total; i++) {
            if (model->q_proj_weight && model->q_proj_weight[i]) free(model->q_proj_weight[i]);
            if (model->k_proj_weight && model->k_proj_weight[i]) free(model->k_proj_weight[i]);
            if (model->v_proj_weight && model->v_proj_weight[i]) free(model->v_proj_weight[i]);
            if (model->o_proj_weight && model->o_proj_weight[i]) free(model->o_proj_weight[i]);
            if (model->gate_proj_weight && model->gate_proj_weight[i]) free(model->gate_proj_weight[i]);
            if (model->up_proj_weight && model->up_proj_weight[i]) free(model->up_proj_weight[i]);
            if (model->down_proj_weight && model->down_proj_weight[i]) free(model->down_proj_weight[i]);
        }
        if (model->q_proj_weight) free(model->q_proj_weight);
        if (model->k_proj_weight) free(model->k_proj_weight);
        if (model->v_proj_weight) free(model->v_proj_weight);
        if (model->o_proj_weight) free(model->o_proj_weight);
        if (model->gate_proj_weight) free(model->gate_proj_weight);
        if (model->up_proj_weight) free(model->up_proj_weight);
        if (model->down_proj_weight) free(model->down_proj_weight);
    }
    
    for (int i = 0; i < model->num_readers; i++) {
        safetensors_close(model->readers[i]);
    }
    free(model->readers);
    
    free(model);
}

int model_generate(Qwen3Model* model, const int* input_tokens, size_t input_len,
                   float temperature, int top_k, float top_p, int max_length,
                   int** output_tokens, size_t* output_len) {
    int vocab_size = model->config.vocab_size;
    int hidden_size = model->config.hidden_size;
    int eos_token_id = model->config.eos_token_id;
    bool is_awq = (model->config.quant_method == QUANT_AWQ);
    
    printf("Starting generation (AWQ: %s)...\n", is_awq ? "yes" : "no");
    
    int* all_ids = (int*)malloc((input_len + max_length) * sizeof(int));
    memcpy(all_ids, input_tokens, input_len * sizeof(int));
    int total_len = input_len;
    
    for (int gen_idx = 0; gen_idx < max_length; gen_idx++) {
        float* hidden_states = forward(model, all_ids, total_len);
        
        float* logits = (float*)malloc(vocab_size * sizeof(float));
        
        if (is_awq && model->lm_head_awq && model->lm_head_awq->qweight) {
            awq_matmul(hidden_states + (total_len - 1) * hidden_size,
                       model->lm_head_awq->qweight, model->lm_head_awq->scales_fp16,
                       model->lm_head_awq->qzeros, logits,
                       hidden_size, vocab_size, model->config.quant_group_size);
        } else if (model->embed_is_bf16 && model->embed_tokens_bf16) {
            for (int v = 0; v < vocab_size; v++) {
                logits[v] = 0.0f;
                for (int h = 0; h < hidden_size; h++) {
                    uint16_t bf16_val = model->embed_tokens_bf16[v * hidden_size + h];
                    logits[v] += hidden_states[(total_len - 1) * hidden_size + h] * bf16_to_f32(bf16_val);
                }
            }
        } else if (model->embed_tokens) {
            for (int v = 0; v < vocab_size; v++) {
                logits[v] = 0.0f;
                for (int h = 0; h < hidden_size; h++) {
                    logits[v] += hidden_states[(total_len - 1) * hidden_size + h] * model->embed_tokens[v * hidden_size + h];
                }
            }
        } else {
            memset(logits, 0, vocab_size * sizeof(float));
        }
        
        // Debug: print logits statistics
        float max_logit = logits[0];
        float min_logit = logits[0];
        int max_idx = 0;
        for (int v = 0; v < vocab_size; v++) {
            if (logits[v] > max_logit) { max_logit = logits[v]; max_idx = v; }
            if (logits[v] < min_logit) min_logit = logits[v];
        }
        fprintf(stderr, "DEBUG: logits min=%.4f, max=%.4f, max_idx=%d\n", min_logit, max_logit, max_idx);
        
        int next_token = sample_top_k(logits, vocab_size, top_k, temperature);
        
        free(logits);
        free(hidden_states);
        
        all_ids[total_len++] = next_token;
        
        printf("Generated token %d: %d\n", gen_idx + 1, next_token);
        
        if (next_token == eos_token_id) {
            break;
        }
    }
    
    *output_tokens = (int*)malloc(total_len * sizeof(int));
    memcpy(*output_tokens, all_ids, total_len * sizeof(int));
    *output_len = total_len;
    
    free(all_ids);
    
    return 0;
}

void model_init_cache(Qwen3Model* model, int max_seq_len) {
    if (model->kv_cache) {
        kv_cache_free(model->kv_cache);
    }
    model->kv_cache = kv_cache_create(
        model->config.num_hidden_layers,
        model->config.num_attention_heads,
        model->config.num_key_value_heads,
        model->config.head_dim,
        max_seq_len
    );
    model->kv_cache_len = 0;
    printf("Initialized KV cache: max_seq_len=%d, layers=%d\n", 
           max_seq_len, model->config.num_hidden_layers);
}

void model_free_cache(Qwen3Model* model) {
    if (model->kv_cache) {
        kv_cache_free(model->kv_cache);
        model->kv_cache = NULL;
    }
    model->kv_cache_len = 0;
}

static float* forward_single_token(Qwen3Model* model, int token_id, int pos) {
    int hidden_size = model->config.hidden_size;
    int num_layers = model->config.num_hidden_layers;
    float eps = model->config.rms_norm_eps;
    bool is_awq = (model->config.quant_method == QUANT_AWQ);
    
    float* hidden = (float*)malloc(hidden_size * sizeof(float));
    
    if (token_id >= 0 && token_id < model->config.vocab_size) {
        if (model->embed_is_bf16 && model->embed_tokens_bf16) {
            for (int h = 0; h < hidden_size; h++) {
                uint16_t bf16_val = model->embed_tokens_bf16[token_id * hidden_size + h];
                hidden[h] = bf16_to_f32(bf16_val);
            }
        } else if (model->embed_tokens) {
            memcpy(hidden, model->embed_tokens + token_id * hidden_size, hidden_size * sizeof(float));
        } else {
            memset(hidden, 0, hidden_size * sizeof(float));
        }
    } else {
        memset(hidden, 0, hidden_size * sizeof(float));
    }
    
    float* layer_input = (float*)malloc(hidden_size * sizeof(float));
    float* attn_output = (float*)malloc(hidden_size * sizeof(float));
    float* mlp_output = (float*)malloc(hidden_size * sizeof(float));
    float* norm_hidden = (float*)malloc(hidden_size * sizeof(float));
    
    for (int layer = 0; layer < num_layers; layer++) {
        memcpy(layer_input, hidden, hidden_size * sizeof(float));
        
        rms_norm(norm_hidden, hidden, model->input_layernorm_weights + layer * hidden_size, hidden_size, eps);
        
        if (is_awq) {
            awq_attention_single(model, norm_hidden, attn_output, pos, layer);
        } else {
            attention_single(model, norm_hidden, attn_output, pos, layer);
        }
        
        for (int i = 0; i < hidden_size; i++) {
            hidden[i] = layer_input[i] + attn_output[i];
        }
        
        memcpy(layer_input, hidden, hidden_size * sizeof(float));
        
        rms_norm(norm_hidden, hidden, model->post_attention_layernorm_weights + layer * hidden_size, hidden_size, eps);
        
        if (is_awq) {
            awq_mlp_single(model, norm_hidden, mlp_output, layer);
        } else {
            mlp_single(model, norm_hidden, mlp_output, layer);
        }
        
        for (int i = 0; i < hidden_size; i++) {
            hidden[i] = layer_input[i] + mlp_output[i];
        }
    }
    
    free(layer_input);
    free(attn_output);
    free(mlp_output);
    free(norm_hidden);
    
    return hidden;
}

int model_generate_with_cache(Qwen3Model* model, const int* input_tokens, size_t input_len,
                               float temperature, int top_k, float top_p, int max_length,
                               int** output_tokens, size_t* output_len) {
    int vocab_size = model->config.vocab_size;
    int hidden_size = model->config.hidden_size;
    int eos_token_id = model->config.eos_token_id;
    bool is_awq = (model->config.quant_method == QUANT_AWQ);
    
    if (!model->kv_cache) {
        model_init_cache(model, input_len + max_length);
    }
    
    printf("Starting incremental generation (AWQ: %s, KV Cache: enabled)...\n", is_awq ? "yes" : "no");
    
    int* all_ids = (int*)malloc((input_len + max_length) * sizeof(int));
    memcpy(all_ids, input_tokens, input_len * sizeof(int));
    int total_len = input_len;
    
    float* hidden = NULL;
    for (size_t i = 0; i < input_len; i++) {
        hidden = forward_single_token(model, input_tokens[i], model->kv_cache_len);
        model->kv_cache_len++;
    }
    
    printf("Prefill complete: %zu tokens\n", input_len);
    
    for (int gen_idx = 0; gen_idx < max_length; gen_idx++) {
        float* logits = (float*)malloc(vocab_size * sizeof(float));
        
        if (is_awq && model->lm_head_awq && model->lm_head_awq->qweight) {
            awq_matmul(hidden,
                       model->lm_head_awq->qweight, model->lm_head_awq->scales_fp16,
                       model->lm_head_awq->qzeros, logits,
                       hidden_size, vocab_size, model->config.quant_group_size);
        } else if (model->embed_is_bf16 && model->embed_tokens_bf16) {
            for (int v = 0; v < vocab_size; v++) {
                logits[v] = 0.0f;
                for (int h = 0; h < hidden_size; h++) {
                    uint16_t bf16_val = model->embed_tokens_bf16[v * hidden_size + h];
                    logits[v] += hidden[h] * bf16_to_f32(bf16_val);
                }
            }
        } else if (model->embed_tokens) {
            for (int v = 0; v < vocab_size; v++) {
                logits[v] = 0.0f;
                for (int h = 0; h < hidden_size; h++) {
                    logits[v] += hidden[h] * model->embed_tokens[v * hidden_size + h];
                }
            }
        } else {
            memset(logits, 0, vocab_size * sizeof(float));
        }
        
        int next_token = sample_top_k(logits, vocab_size, top_k, temperature);
        
        free(logits);
        if (hidden) free(hidden);
        
        all_ids[total_len++] = next_token;
        
        printf("Generated token %d: %d\n", gen_idx + 1, next_token);
        
        if (next_token == eos_token_id) {
            break;
        }
        
        hidden = forward_single_token(model, next_token, model->kv_cache_len);
        model->kv_cache_len++;
    }
    
    if (hidden) free(hidden);
    
    *output_tokens = (int*)malloc(total_len * sizeof(int));
    memcpy(*output_tokens, all_ids, total_len * sizeof(int));
    *output_len = total_len;
    
    free(all_ids);
    
    return 0;
}
