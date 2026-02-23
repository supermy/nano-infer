#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "safetensors.h"

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

int main() {
    const char* model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3-4B-AWQ/model.safetensors";
    
    SafetensorsReader* reader = safetensors_open(model_path);
    if (!reader) {
        printf("Failed to open\n");
        return 1;
    }
    
    void* ptr;
    size_t size;
    
    // Get qweight
    safetensors_get_tensor_ptr(reader, "model.layers.0.self_attn.q_proj.qweight", &ptr, &size);
    uint32_t* qw = (uint32_t*)ptr;
    
    // Get scales
    safetensors_get_tensor_ptr(reader, "model.layers.0.self_attn.q_proj.scales", &ptr, &size);
    uint16_t* scales_fp16 = (uint16_t*)ptr;
    
    // Get qzeros
    safetensors_get_tensor_ptr(reader, "model.layers.0.self_attn.q_proj.qzeros", &ptr, &size);
    uint32_t* qz = (uint32_t*)ptr;
    
    printf("=== AWQ Weight Analysis ===\n\n");
    
    printf("qweight first 5 INT32 values:\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d] 0x%08x\n", i, qw[i]);
    }
    
    printf("\nqzeros first 5 INT32 values:\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d] 0x%08x\n", i, qz[i]);
    }
    
    printf("\nscales first 10 FP16 values:\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] 0x%04x -> %f\n", i, scales_fp16[i], fp16_to_f32(scales_fp16[i]));
    }
    
    // Test unpacking
    printf("\n=== Testing unpacking ===\n");
    uint32_t w_packed = qw[0];
    uint32_t z_packed = qz[0];
    float scale = fp16_to_f32(scales_fp16[0]);
    
    printf("First packed weight: 0x%08x\n", w_packed);
    printf("First packed zero: 0x%08x\n", z_packed);
    printf("First scale: %f\n", scale);
    
    printf("\nUnpacking 8 weights from first INT32 (AWQ format):\n");
    for (int k = 0; k < 8; k++) {
        uint8_t w_val = (w_packed >> (k * 4)) & 0x0F;
        uint8_t z_val = (z_packed >> (k * 4)) & 0x0F;
        float dequant = ((float)w_val - (float)z_val) * scale;
        printf("  k=%d: w=%u, z=%u, dequant=%f\n", k, w_val, z_val, dequant);
    }
    
    safetensors_close(reader);
    return 0;
}
