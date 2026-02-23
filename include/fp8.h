#ifndef FP8_H
#define FP8_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef enum {
    FP8_E4M3 = 0,
    FP8_E5M2 = 1,
} FP8Format;

typedef struct {
    uint8_t* qweight;
    float* scales;
    int in_features;
    int out_features;
    FP8Format format;
    bool rowwise_scale;
} FP8Weight;

typedef struct {
    uint8_t* weights;
    float* scale;
    float* scale_inv;
    size_t num_elements;
    FP8Format format;
} FP8Tensor;

uint8_t fp32_to_fp8_e4m3(float val);
uint8_t fp32_to_fp8_e5m2(float val);
float fp8_e4m3_to_fp32(uint8_t val);
float fp8_e5m2_to_fp32(uint8_t val);

void fp8_quantize_row(const float* input, uint8_t* output, float* scale,
                      int n, FP8Format format);
void fp8_dequantize_row(const uint8_t* input, float* output, const float* scale,
                        int n, FP8Format format);

FP8Tensor* fp8_tensor_create(const float* data, size_t num_elements, FP8Format format);
void fp8_tensor_free(FP8Tensor* tensor);
float* fp8_tensor_to_fp32(const FP8Tensor* tensor);

FP8Weight* fp8_weight_create(const float* weight, int in_features, int out_features,
                              FP8Format format, bool rowwise_scale);
void fp8_weight_free(FP8Weight* weight);

void fp8_matmul(const float* input, const FP8Weight* weight, float* output,
                int batch_size, int in_features, int out_features);

void fp8_matmul_with_dequant(const float* input, const uint8_t* qweight,
                              const float* scales, float* output,
                              int in_features, int out_features,
                              FP8Format format, bool rowwise_scale);

size_t fp8_memory_size(size_t num_elements, bool with_scale);

#endif
