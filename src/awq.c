#include <stdint.h>
#include <string.h>
#include <stdio.h>

void awq_dequantize(
    const uint8_t* qweight,    
    const float* scales,       
    const uint8_t* qzeros,     
    float* out,                
    int num_groups,            
    int group_size,            
    int out_features           
) {
    int num_bits = 4;
    int values_per_byte = 8 / num_bits;
    
    for (int g = 0; g < num_groups; g++) {
        const uint8_t* qw_group = qweight + g * (group_size * out_features / 2);
        const float* scale_row = scales + g * out_features;
        const uint8_t* qz_group = qzeros + g * out_features;
        
        for (int i = 0; i < group_size; i += 2) {
            int in_idx = g * group_size + i;
            
            for (int j = 0; j < out_features; j++) {
                uint8_t byte = qw_group[(i / 2) * out_features + j];
                
                uint8_t lo_nibble = byte & 0x0F;
                uint8_t hi_nibble = (byte >> 4) & 0x0F;
                
                float lo_val = ((float)lo_nibble - (float)qz_group[j]) * scale_row[j];
                float hi_val = ((float)hi_nibble - (float)qz_group[j]) * scale_row[j];
                
                out[in_idx * out_features + j] = lo_val;
                if (i + 1 < group_size) {
                    out[(in_idx + 1) * out_features + j] = hi_val;
                }
            }
        }
    }
}

void awq_dequantize_row(
    const uint8_t* qweight,    
    const float* scales,       
    const uint8_t* qzeros,     
    float* out,                
    int in_features,           
    int out_features           
) {
    int group_size = 128;
    int num_groups = (in_features + group_size - 1) / group_size;
    
    int num_bits = 4;
    int values_per_byte = 8 / num_bits;
    
    memset(out, 0, in_features * out_features * sizeof(float));
    
    for (int g = 0; g < num_groups; g++) {
        int g_start = g * group_size;
        int g_end = (g + 1) * group_size;
        if (g_end > in_features) g_end = in_features;
        int actual_group_size = g_end - g_start;
        
        const uint8_t* qw_group = qweight + (g * group_size / 2) * out_features;
        const float* scale_row = scales + g * out_features;
        const uint8_t* qz_group = qzeros + g * out_features;
        
        for (int i = 0; i < actual_group_size; i += 2) {
            int out_idx = g_start + i;
            int qw_idx = (i / 2) * out_features;
            
            for (int j = 0; j < out_features; j++) {
                uint8_t byte = qw_group[qw_idx + j];
                
                uint8_t lo_nibble = byte & 0x0F;
                uint8_t hi_nibble = (byte >> 4) & 0x0F;
                
                float lo_val = ((float)lo_nibble - (float)qz_group[j]) * scale_row[j];
                out[out_idx * out_features + j] = lo_val;
                
                if (i + 1 < actual_group_size) {
                    out[(out_idx + 1) * out_features + j] = 
                        ((float)hi_nibble - (float)qz_group[j]) * scale_row[j];
                }
            }
        }
    }
}
