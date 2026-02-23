#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "safetensors.h"

static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = (uint32_t)bf16 << 16;
    float f32;
    memcpy(&f32, &f32_bits, sizeof(float));
    return f32;
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
    
    safetensors_get_tensor_ptr(reader, "model.embed_tokens.weight", &ptr, &size);
    uint16_t* embed_bf16 = (uint16_t*)ptr;
    
    printf("embed_tokens shape: [151936, 2560]\n");
    printf("Token 0 (first 10 values):\n");
    for (int i = 0; i < 10; i++) {
        printf("  embed[0][%d] = 0x%04x -> %f\n", i, embed_bf16[i], bf16_to_f32(embed_bf16[i]));
    }
    
    printf("\nToken 9906 (Hello, first 10 values):\n");
    int token_id = 9906;
    for (int i = 0; i < 10; i++) {
        int idx = token_id * 2560 + i;
        printf("  embed[9906][%d] = 0x%04x -> %f\n", i, embed_bf16[idx], bf16_to_f32(embed_bf16[idx]));
    }
    
    printf("\nToken 19 (first 10 values):\n");
    token_id = 19;
    for (int i = 0; i < 10; i++) {
        int idx = token_id * 2560 + i;
        printf("  embed[19][%d] = 0x%04x -> %f\n", i, embed_bf16[idx], bf16_to_f32(embed_bf16[idx]));
    }
    
    safetensors_close(reader);
    return 0;
}
