#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "safetensors.h"

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

int main() {
    const char* model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3-4B-AWQ/model.safetensors";
    
    SafetensorsReader* reader = safetensors_open(model_path);
    if (!reader) {
        printf("Failed to open\n");
        return 1;
    }
    
    void* ptr;
    size_t size;
    
    // q_proj: qweight [2560, 512], scales [20, 4096], qzeros [20, 512]
    safetensors_get_tensor_ptr(reader, "model.layers.0.self_attn.q_proj.qweight", &ptr, &size);
    uint32_t* qw = (uint32_t*)ptr;
    
    safetensors_get_tensor_ptr(reader, "model.layers.0.self_attn.q_proj.scales", &ptr, &size);
    uint16_t* scales = (uint16_t*)ptr;
    
    safetensors_get_tensor_ptr(reader, "model.layers.0.self_attn.q_proj.qzeros", &ptr, &size);
    uint32_t* qz = (uint32_t*)ptr;
    
    printf("=== AWQ Dequantization Verification ===\n\n");
    
    int in_features = 2560;
    int out_features = 4096;
    int group_size = 128;
    int oc_int32 = out_features / 8;  // 512
    
    // AWQ formula: W = (qweight - qzeros) * scales
    // where qweight and qzeros are packed 4-bit values
    
    // Test: dequantize first column (output 0) for first 10 rows
    printf("Dequantizing output 0 for first 10 inputs:\n");
    int j = 0;  // output index
    int j_int = j / 8;  // 0
    int k = j % 8;      // 0
    
    for (int i = 0; i < 10; i++) {
        int g = i / group_size;  // all in group 0
        
        uint32_t w_packed = qw[i * oc_int32 + j_int];
        uint32_t z_packed = qz[g * oc_int32 + j_int];
        float scale = fp16_to_f32(scales[g * out_features + j]);
        
        uint8_t w_val = (w_packed >> (k * 4)) & 0x0F;
        uint8_t z_val = (z_packed >> (k * 4)) & 0x0F;
        
        float dequant = ((float)w_val - (float)z_val) * scale;
        printf("  i=%d: w_packed=0x%08x, w_val=%u, z_val=%u, scale=%f, dequant=%f\n",
               i, w_packed, w_val, z_val, scale, dequant);
    }
    
    // Check scales distribution
    printf("\nScale statistics for group 0:\n");
    float min_s = 1e10, max_s = 0, sum_s = 0;
    for (int jj = 0; jj < out_features; jj++) {
        float s = fp16_to_f32(scales[jj]);
        if (s < min_s) min_s = s;
        if (s > max_s) max_s = s;
        sum_s += s;
    }
    printf("  min=%f, max=%f, mean=%f\n", min_s, max_s, sum_s / out_features);
    
    // Check qzeros distribution
    printf("\nQzeros distribution for group 0:\n");
    int z_counts[16] = {0};
    for (int j_int = 0; j_int < oc_int32; j_int++) {
        uint32_t z_packed = qz[j_int];
        for (int kk = 0; kk < 8; kk++) {
            uint8_t z_val = (z_packed >> (kk * 4)) & 0x0F;
            z_counts[z_val]++;
        }
    }
    printf("  Zero point distribution:\n");
    for (int zi = 0; zi < 16; zi++) {
        if (z_counts[zi] > 0) {
            printf("    z=%d: count=%d\n", zi, z_counts[zi]);
        }
    }
    
    safetensors_close(reader);
    return 0;
}
