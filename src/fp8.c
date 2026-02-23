#include "fp8.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

static inline uint8_t fp32_to_fp8_e4m3_impl(float val) {
    if (val == 0.0f) return 0;
    if (val != val) return 0x80;
    if (val > 448.0f) return 0x7E;
    if (val < -448.0f) return 0xFE;
    
    uint32_t bits;
    memcpy(&bits, &val, sizeof(float));
    
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;
    
    int fp8_exp = exp + 7;
    uint8_t fp8_mant = (mant >> 20) & 0x07;
    
    if (fp8_exp <= 0) {
        return (uint8_t)(sign << 7);
    }
    
    if (fp8_exp >= 15) {
        fp8_exp = 14;
        fp8_mant = 0x07;
    }
    
    return (uint8_t)((sign << 7) | ((fp8_exp & 0x0F) << 3) | fp8_mant);
}

static inline uint8_t fp32_to_fp8_e5m2_impl(float val) {
    if (val == 0.0f) return 0;
    if (val != val) return 0x80;
    if (val > 57344.0f) return 0x7B;
    if (val < -57344.0f) return 0xFB;
    
    uint32_t bits;
    memcpy(&bits, &val, sizeof(float));
    
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;
    
    int fp8_exp = exp + 15;
    uint8_t fp8_mant = (mant >> 21) & 0x03;
    
    if (fp8_exp <= 0) {
        return (uint8_t)(sign << 7);
    }
    
    if (fp8_exp >= 31) {
        fp8_exp = 30;
        fp8_mant = 0x03;
    }
    
    return (uint8_t)((sign << 7) | ((fp8_exp & 0x1F) << 2) | fp8_mant);
}

static inline float fp8_e4m3_to_fp32_impl(uint8_t val) {
    if (val == 0) return 0.0f;
    
    uint32_t sign = (val >> 7) & 1;
    uint32_t exp = (val >> 3) & 0x0F;
    uint32_t mant = val & 0x07;
    
    if (exp == 0 && mant == 0) {
        return sign ? -0.0f : 0.0f;
    }
    
    int32_t fp32_exp = exp - 7 + 127;
    uint32_t fp32_mant = mant << 20;
    
    uint32_t fp32_bits = (sign << 31) | ((fp32_exp & 0xFF) << 23) | fp32_mant;
    float result;
    memcpy(&result, &fp32_bits, sizeof(float));
    
    return result;
}

static inline float fp8_e5m2_to_fp32_impl(uint8_t val) {
    if (val == 0) return 0.0f;
    
    uint32_t sign = (val >> 7) & 1;
    uint32_t exp = (val >> 2) & 0x1F;
    uint32_t mant = val & 0x03;
    
    if (exp == 0 && mant == 0) {
        return sign ? -0.0f : 0.0f;
    }
    
    if (exp == 31) {
        return sign ? -INFINITY : INFINITY;
    }
    
    int32_t fp32_exp = exp - 15 + 127;
    uint32_t fp32_mant = mant << 21;
    
    uint32_t fp32_bits = (sign << 31) | ((fp32_exp & 0xFF) << 23) | fp32_mant;
    float result;
    memcpy(&result, &fp32_bits, sizeof(float));
    
    return result;
}

uint8_t fp32_to_fp8_e4m3(float val) {
    return fp32_to_fp8_e4m3_impl(val);
}

uint8_t fp32_to_fp8_e5m2(float val) {
    return fp32_to_fp8_e5m2_impl(val);
}

float fp8_e4m3_to_fp32(uint8_t val) {
    return fp8_e4m3_to_fp32_impl(val);
}

float fp8_e5m2_to_fp32(uint8_t val) {
    return fp8_e5m2_to_fp32_impl(val);
}

void fp8_quantize_row(const float* input, uint8_t* output, float* scale,
                       int n, FP8Format format) {
    if (!input || !output || !scale || n <= 0) return;
    
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(input[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    
    float max_fp8 = (format == FP8_E4M3) ? 448.0f : 57344.0f;
    *scale = max_abs / max_fp8;
    if (*scale < 1e-10f) *scale = 1.0f;
    
    float inv_scale = 1.0f / *scale;
    
    for (int i = 0; i < n; i++) {
        float scaled = input[i] * inv_scale;
        if (format == FP8_E4M3) {
            output[i] = fp32_to_fp8_e4m3_impl(scaled);
        } else {
            output[i] = fp32_to_fp8_e5m2_impl(scaled);
        }
    }
}

void fp8_dequantize_row(const uint8_t* input, float* output, const float* scale,
                         int n, FP8Format format) {
    if (!input || !output || !scale || n <= 0) return;
    
    for (int i = 0; i < n; i++) {
        float val;
        if (format == FP8_E4M3) {
            val = fp8_e4m3_to_fp32_impl(input[i]);
        } else {
            val = fp8_e5m2_to_fp32_impl(input[i]);
        }
        output[i] = val * (*scale);
    }
}

FP8Tensor* fp8_tensor_create(const float* data, size_t num_elements, FP8Format format) {
    if (!data || num_elements == 0) return NULL;
    
    FP8Tensor* tensor = (FP8Tensor*)calloc(1, sizeof(FP8Tensor));
    if (!tensor) return NULL;
    
    tensor->weights = (uint8_t*)malloc(num_elements * sizeof(uint8_t));
    tensor->scale = (float*)malloc(sizeof(float));
    tensor->num_elements = num_elements;
    tensor->format = format;
    
    if (!tensor->weights || !tensor->scale) {
        free(tensor->weights);
        free(tensor->scale);
        free(tensor);
        return NULL;
    }
    
    fp8_quantize_row(data, tensor->weights, tensor->scale, num_elements, format);
    
    tensor->scale_inv = (float*)malloc(sizeof(float));
    if (tensor->scale_inv) {
        *tensor->scale_inv = 1.0f / *tensor->scale;
    }
    
    return tensor;
}

void fp8_tensor_free(FP8Tensor* tensor) {
    if (!tensor) return;
    free(tensor->weights);
    free(tensor->scale);
    free(tensor->scale_inv);
    free(tensor);
}

float* fp8_tensor_to_fp32(const FP8Tensor* tensor) {
    if (!tensor) return NULL;
    
    float* output = (float*)malloc(tensor->num_elements * sizeof(float));
    if (!output) return NULL;
    
    fp8_dequantize_row(tensor->weights, output, tensor->scale, tensor->num_elements, tensor->format);
    
    return output;
}

FP8Weight* fp8_weight_create(const float* weight, int in_features, int out_features,
                              FP8Format format, bool rowwise_scale) {
    if (!weight || in_features <= 0 || out_features <= 0) return NULL;
    
    FP8Weight* fp8_w = (FP8Weight*)calloc(1, sizeof(FP8Weight));
    if (!fp8_w) return NULL;
    
    fp8_w->in_features = in_features;
    fp8_w->out_features = out_features;
    fp8_w->format = format;
    fp8_w->rowwise_scale = rowwise_scale;
    
    size_t num_weights = (size_t)in_features * out_features;
    fp8_w->qweight = (uint8_t*)malloc(num_weights * sizeof(uint8_t));
    
    if (!fp8_w->qweight) {
        free(fp8_w);
        return NULL;
    }
    
    if (rowwise_scale) {
        fp8_w->scales = (float*)malloc(out_features * sizeof(float));
        if (!fp8_w->scales) {
            free(fp8_w->qweight);
            free(fp8_w);
            return NULL;
        }
        
        for (int j = 0; j < out_features; j++) {
            float max_abs = 0.0f;
            for (int i = 0; i < in_features; i++) {
                float abs_val = fabsf(weight[j * in_features + i]);
                if (abs_val > max_abs) max_abs = abs_val;
            }
            
            float max_fp8 = (format == FP8_E4M3) ? 448.0f : 57344.0f;
            fp8_w->scales[j] = max_abs / max_fp8;
            if (fp8_w->scales[j] < 1e-10f) fp8_w->scales[j] = 1.0f;
            
            float inv_scale = 1.0f / fp8_w->scales[j];
            for (int i = 0; i < in_features; i++) {
                float scaled = weight[j * in_features + i] * inv_scale;
                if (format == FP8_E4M3) {
                    fp8_w->qweight[j * in_features + i] = fp32_to_fp8_e4m3_impl(scaled);
                } else {
                    fp8_w->qweight[j * in_features + i] = fp32_to_fp8_e5m2_impl(scaled);
                }
            }
        }
    } else {
        fp8_w->scales = (float*)malloc(sizeof(float));
        if (!fp8_w->scales) {
            free(fp8_w->qweight);
            free(fp8_w);
            return NULL;
        }
        
        float max_abs = 0.0f;
        for (size_t i = 0; i < num_weights; i++) {
            float abs_val = fabsf(weight[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        
        float max_fp8 = (format == FP8_E4M3) ? 448.0f : 57344.0f;
        fp8_w->scales[0] = max_abs / max_fp8;
        if (fp8_w->scales[0] < 1e-10f) fp8_w->scales[0] = 1.0f;
        
        fp8_quantize_row(weight, fp8_w->qweight, fp8_w->scales, num_weights, format);
    }
    
    return fp8_w;
}

void fp8_weight_free(FP8Weight* weight) {
    if (!weight) return;
    free(weight->qweight);
    free(weight->scales);
    free(weight);
}

void fp8_matmul(const float* input, const FP8Weight* weight, float* output,
                int batch_size, int in_features, int out_features) {
    if (!input || !weight || !output) return;
    
    memset(output, 0, batch_size * out_features * sizeof(float));
    
    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < out_features; j++) {
            float sum = 0.0f;
            float scale = weight->rowwise_scale ? weight->scales[j] : weight->scales[0];
            
            for (int i = 0; i < in_features; i++) {
                uint8_t qval = weight->qweight[j * in_features + i];
                float w;
                if (weight->format == FP8_E4M3) {
                    w = fp8_e4m3_to_fp32_impl(qval) * scale;
                } else {
                    w = fp8_e5m2_to_fp32_impl(qval) * scale;
                }
                sum += input[b * in_features + i] * w;
            }
            output[b * out_features + j] = sum;
        }
    }
}

void fp8_matmul_with_dequant(const float* input, const uint8_t* qweight,
                              const float* scales, float* output,
                              int in_features, int out_features,
                              FP8Format format, bool rowwise_scale) {
    if (!input || !qweight || !scales || !output) return;
    
    memset(output, 0, out_features * sizeof(float));
    
    for (int j = 0; j < out_features; j++) {
        float sum = 0.0f;
        float scale = rowwise_scale ? scales[j] : scales[0];
        
        for (int i = 0; i < in_features; i++) {
            uint8_t qval = qweight[j * in_features + i];
            float w;
            if (format == FP8_E4M3) {
                w = fp8_e4m3_to_fp32_impl(qval) * scale;
            } else {
                w = fp8_e5m2_to_fp32_impl(qval) * scale;
            }
            sum += input[i] * w;
        }
        output[j] = sum;
    }
}

size_t fp8_memory_size(size_t num_elements, bool with_scale) {
    size_t size = num_elements * sizeof(uint8_t);
    if (with_scale) {
        size += sizeof(float);
    }
    return size;
}
