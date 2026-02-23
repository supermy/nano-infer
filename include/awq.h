#ifndef AWQ_H
#define AWQ_H

#include <stddef.h>
#include <stdint.h>

void awq_dequantize(
    const uint8_t* qweight,    
    const float* scales,       
    const uint8_t* qzeros,     
    float* out,                
    int num_groups,            
    int group_size,            
    int out_features           
);

void awq_dequantize_row(
    const uint8_t* qweight,    
    const float* scales,       
    const uint8_t* qzeros,     
    float* out,                
    int in_features,           
    int out_features           
);

#endif
