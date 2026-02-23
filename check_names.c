#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "safetensors.h"

int main() {
    const char* model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3-4B-AWQ/model.safetensors";
    
    SafetensorsReader* reader = safetensors_open(model_path);
    if (!reader) {
        printf("Failed to open\n");
        return 1;
    }
    
    printf("Looking for embed_tokens:\n");
    for (size_t i = 0; i < reader->num_tensors; i++) {
        if (strstr(reader->tensor_names[i], "embed")) {
            printf("  [%zu] %s\n", i, reader->tensor_names[i]);
        }
    }
    
    printf("\nLooking for norm:\n");
    for (size_t i = 0; i < reader->num_tensors; i++) {
        if (strstr(reader->tensor_names[i], "norm")) {
            printf("  [%zu] %s\n", i, reader->tensor_names[i]);
        }
    }
    
    safetensors_close(reader);
    return 0;
}
