#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include "common.h"
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct {
    char* data;
    size_t size;
    int fd;
    bool is_mapped;
} MappedFile;

typedef struct {
    char** tensor_names;
    TensorMeta* tensor_metas;
    size_t num_tensors;
    MappedFile* mapped_file;
    size_t data_offset;
} SafetensorsReader;

SafetensorsReader* safetensors_open(const char* file_path);
void safetensors_close(SafetensorsReader* reader);
Tensor* safetensors_read_tensor(SafetensorsReader* reader, const char* name);
bool safetensors_load_tensor_data(SafetensorsReader* reader, Tensor* tensor);
bool safetensors_get_tensor_ptr(SafetensorsReader* reader, const char* name, void** ptr_out, size_t* size_out);
void tensor_free(Tensor* tensor);

#endif
