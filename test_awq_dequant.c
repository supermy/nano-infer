#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static float fp16_to_f32(uint16_t fp16) {
    uint32_t sign = (fp16 >> 15) & 1;
    uint32_t exponent = (fp16 >> 10) & 0x1F;
    uint32_t mantissa = fp16 & 0x3FF;
    
    if (exponent == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        }
        exponent = 1;
        while (!(mantissa & 0x400)) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x3FF;
        exponent += 112;
    } else if (exponent == 31) {
        return sign ? -INFINITY : INFINITY;
    } else {
        exponent += 112;
    }
    
    uint32_t f32 = (sign << 31) | (exponent << 23) | (mantissa << 13);
    return *(float*)&f32;
}

int main() {
    // Test AWQ dequantization
    // Simulate a simple case: in_features=4, out_features=8, group_size=2
    
    int in_features = 4;
    int out_features = 8;
    int group_size = 2;
    
    // Input vector
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // Simulated qweight (4-bit packed into int32)
    // Each int32 contains 8 4-bit weights
    // qweight shape: [in_features, out_features/8] = [4, 1]
    uint32_t qweight[] = {
        0x76543210,  // weights for input 0 -> outputs 0-7
        0xFEDCBA98,  // weights for input 1 -> outputs 0-7
        0x01234567,  // weights for input 2 -> outputs 0-7
        0x89ABCDEF   // weights for input 3 -> outputs 0-7
    };
    
    // Simulated scales (FP16)
    // scales shape: [num_groups, out_features] = [2, 8]
    uint16_t scales[] = {
        0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00,  // group 0: all 1.0
        0x4000, 0x4000, 0x4000, 0x4000, 0x4000, 0x4000, 0x4000, 0x4000   // group 1: all 2.0
    };
    
    // Simulated qzeros (4-bit packed into int32)
    // qzeros shape: [num_groups, out_features/8] = [2, 1]
    uint32_t qzeros[] = {
        0x88888888,  // group 0: all zeros are 8
        0x88888888   // group 1: all zeros are 8
    };
    
    float output[8] = {0};
    
    // AWQ matmul
    int oc_int32 = out_features / 8;
    
    for (int i = 0; i < in_features; i++) {
        int g = i / group_size;
        
        uint32_t w_packed = qweight[i];
        uint32_t z_packed = qzeros[g];
        
        for (int k = 0; k < 8; k++) {
            int j = k;
            uint8_t w_val = (w_packed >> (k * 4)) & 0x0F;
            uint8_t z_val = (z_packed >> (k * 4)) & 0x0F;
            
            float scale = fp16_to_f32(scales[g * out_features + j]);
            float w = ((float)w_val - (float)z_val) * scale;
            output[j] += input[i] * w;
            
            printf("i=%d, j=%d: w_val=%d, z_val=%d, scale=%.4f, w=%.4f, contrib=%.4f\n",
                   i, j, w_val, z_val, scale, w, input[i] * w);
        }
    }
    
    printf("\nOutput:\n");
    for (int j = 0; j < out_features; j++) {
        printf("  output[%d] = %.4f\n", j, output[j]);
    }
    
    return 0;
}
