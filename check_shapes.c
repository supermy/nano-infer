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
    
    printf("=== AWQ Tensor Shapes ===\n\n");
    
    const char* tensors[] = {
        "model.layers.0.self_attn.q_proj.qweight",
        "model.layers.0.self_attn.q_proj.scales",
        "model.layers.0.self_attn.q_proj.qzeros",
        "model.layers.0.mlp.gate_proj.qweight",
        "model.layers.0.mlp.gate_proj.scales",
        "model.layers.0.mlp.gate_proj.qzeros",
        "model.embed_tokens.weight",
        "model.norm.weight"
    };
    
    for (int i = 0; i < 8; i++) {
        for (size_t j = 0; j < reader->num_tensors; j++) {
            if (strcmp(reader->tensor_names[j], tensors[i]) == 0) {
                TensorMeta* meta = &reader->tensor_metas[j];
                printf("%s:\n", tensors[i]);
                printf("  dtype=%zu, shape=[", meta->dtype);
                for (int k = 0; k < meta->ndim; k++) {
                    printf("%zu", meta->shape[k]);
                    if (k < meta->ndim - 1) printf(", ");
                }
                printf("], n_elements=%zu, size_bytes=%zu\n\n", meta->n_elements, meta->size_bytes);
                break;
            }
        }
    }
    
    safetensors_close(reader);
    return 0;
}
