#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>

#define QWEN3_VERSION "0.1.0"

typedef enum {
    DTYPE_FLOAT32 = 0,
    DTYPE_FLOAT16 = 1,
    DTYPE_BFLOAT16 = 2,
    DTYPE_INT8 = 3,
    DTYPE_UINT8 = 4,
    DTYPE_INT32 = 5,
    DTYPE_INT64 = 6,
} DataType;

typedef struct {
    DataType dtype;
    size_t shape[4];
    int ndim;
    size_t offset;
    size_t size_bytes;
    size_t n_elements;
} TensorMeta;

typedef struct {
    void* data;
    TensorMeta meta;
} Tensor;

typedef struct {
    float* data;
    size_t size;
} Buffer;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#endif
