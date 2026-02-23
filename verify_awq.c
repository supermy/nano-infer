#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "safetensors.h"
#include "config.h"

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
        if (mant == 0) return sign ? -0.0f : 0.0f;
        return (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
    } else if (exp == 31) {
        return sign ? -INFINITY : INFINITY;
    }
    return (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15.0f);
}

static void rms_norm(float* output, const float* input, const float* weight, int hidden_size, float eps) {
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

static void awq_matmul(
    const float* input,
    const uint32_t* qw_int32,
    const uint16_t* scales_fp16,
    const uint32_t* qz_int32,
    float* output,
    int in_features,
    int out_features,
    int group_size
) {
    int oc_int32 = out_features / 8;
    
    memset(output, 0, out_features * sizeof(float));
    
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

int main() {
    const char* model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3-4B-AWQ/model.safetensors";
    
    SafetensorsReader* reader = safetensors_open(model_path);
    if (!reader) {
        printf("Failed to open\n");
        return 1;
    }
    
    void* ptr;
    size_t size;
    
    // Load embed_tokens
    safetensors_get_tensor_ptr(reader, "model.embed_tokens.weight", &ptr, &size);
    uint16_t* embed_bf16 = (uint16_t*)ptr;
    
    // Load layernorm
    safetensors_get_tensor_ptr(reader, "model.layers.0.input_layernorm.weight", &ptr, &size);
    uint16_t* ln_bf16 = (uint16_t*)ptr;
    
    // Load q_proj AWQ weights
    safetensors_get_tensor_ptr(reader, "model.layers.0.self_attn.q_proj.qweight", &ptr, &size);
    uint32_t* qw = (uint32_t*)ptr;
    
    safetensors_get_tensor_ptr(reader, "model.layers.0.self_attn.q_proj.scales", &ptr, &size);
    uint16_t* scales = (uint16_t*)ptr;
    
    safetensors_get_tensor_ptr(reader, "model.layers.0.self_attn.q_proj.qzeros", &ptr, &size);
    uint32_t* qz = (uint32_t*)ptr;
    
    printf("=== Testing AWQ Inference Step by Step ===\n\n");
    
    int hidden_size = 2560;
    int out_features = 4096;
    int group_size = 128;
    
    // Convert layernorm to FP32
    float* ln_weight = (float*)malloc(hidden_size * sizeof(float));
    for (int i = 0; i < hidden_size; i++) {
        ln_weight[i] = bf16_to_f32(ln_bf16[i]);
    }
    
    // Get embedding for token 9906 (Hello)
    float* hidden = (float*)malloc(hidden_size * sizeof(float));
    for (int i = 0; i < hidden_size; i++) {
        hidden[i] = bf16_to_f32(embed_bf16[9906 * hidden_size + i]);
    }
    
    printf("1. Embedding for token 9906 (Hello):\n");
    printf("   First 10 values: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", hidden[i]);
    }
    printf("\n");
    
    // Apply RMS norm
    float* normed = (float*)malloc(hidden_size * sizeof(float));
    rms_norm(normed, hidden, ln_weight, hidden_size, 1e-6f);
    
    printf("\n2. After RMS norm:\n");
    printf("   First 10 values: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", normed[i]);
    }
    printf("\n");
    
    // Apply AWQ matmul
    float* q_output = (float*)malloc(out_features * sizeof(float));
    awq_matmul(normed, qw, scales, qz, q_output, hidden_size, out_features, group_size);
    
    printf("\n3. After q_proj (AWQ):\n");
    printf("   First 10 values: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", q_output[i]);
    }
    printf("\n");
    
    // Check for NaN or Inf
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < out_features; i++) {
        if (isnan(q_output[i])) nan_count++;
        if (isinf(q_output[i])) inf_count++;
    }
    printf("   NaN count: %d, Inf count: %d\n", nan_count, inf_count);
    
    // Check statistics
    float min_val = 1e10, max_val = -1e10, sum = 0;
    for (int i = 0; i < out_features; i++) {
        if (q_output[i] < min_val) min_val = q_output[i];
        if (q_output[i] > max_val) max_val = q_output[i];
        sum += q_output[i];
    }
    printf("   Min: %f, Max: %f, Mean: %f\n", min_val, max_val, sum / out_features);
    
    free(ln_weight);
    free(hidden);
    free(normed);
    free(q_output);
    
    safetensors_close(reader);
    return 0;
}
