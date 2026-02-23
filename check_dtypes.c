#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "safetensors.h"

int main() {
    const char* model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3-4B-AWQ/model.safetensors";
    
    SafetensorsReader* reader = safetensors_open(model_path);
    if (!reader) {
        printf("Failed to open\n");
        return 1;
    }
    
    printf("=== Checking embed_tokens dtype ===\n");
    for (size_t i = 0; i < reader->num_tensors; i++) {
        const char* name = reader->tensor_names[i];
        if (strstr(name, "embed_tokens")) {
            TensorMeta* meta = &reader->tensor_metas[i];
            printf("%s: dtype=%zu, shape=[%zu,%zu], size=%zu\n",
                   name, meta->dtype, meta->shape[0], meta->shape[1], meta->size_bytes);
        }
    }
    
    printf("\n=== Checking layernorm dtype ===\n");
    for (size_t i = 0; i < reader->num_tensors; i++) {
        const char* name = reader->tensor_names[i];
        if (strstr(name, "layernorm")) {
            TensorMeta* meta = &reader->tensor_metas[i];
            printf("%s: dtype=%zu, shape=[%zu,%zu], size=%zu\n",
                   name, meta->dtype, meta->shape[0], meta->shape[1], meta->size_bytes);
        }
    }
    
    printf("\n=== Checking norm dtype ===\n");
    for (size_t i = 0; i < reader->num_tensors; i++) {
        const char* name = reader->tensor_names[i];
        if (strstr(name, "norm.weigh")) {
            TensorMeta* meta = &reader->tensor_metas[i];
            printf("%s: dtype=%zu, shape=[%zu,%zu], size=%zu\n",
                   name, meta->dtype, meta->shape[0], meta->shape[1], meta->size_bytes);
        }
    }
    
    safetensors_close(reader);
    return 0;
}
