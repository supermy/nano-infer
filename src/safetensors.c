#include "safetensors.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static char* trim(char* str) {
    while (isspace((unsigned char)*str)) str++;
    if (*str == 0) return str;
    
    char* end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return str;
}

static int parse_shape_from_json(const char* json, size_t* shape, int* ndim) {
    char* shape_start = strstr(json, "\"shape\":");
    if (!shape_start) return -1;
    
    shape_start = strchr(shape_start, '[');
    if (!shape_start) return -1;
    
    char* shape_end = strchr(shape_start, ']');
    if (!shape_end) return -1;
    
    char* ptr = shape_start + 1;
    int count = 0;
    while (ptr < shape_end) {
        while (ptr < shape_end && (*ptr == ' ' || *ptr == ',')) ptr++;
        if (ptr >= shape_end) break;
        shape[count++] = (size_t)strtol(ptr, &ptr, 10);
    }
    
    *ndim = count;
    return 0;
}

static int parse_data_offsets(const char* json, size_t* start, size_t* end) {
    char* offsets_start = strstr(json, "\"data_offsets\":");
    if (!offsets_start) return -1;
    
    char* bracket1 = strchr(offsets_start, '[');
    if (!bracket1) return -1;
    
    char* bracket2 = strchr(bracket1, ']');
    if (!bracket2) return -1;
    
    char* comma = strchr(bracket1, ',');
    if (!comma || comma > bracket2) return -1;
    
    *start = strtoull(bracket1 + 1, NULL, 10);
    *end = strtoull(comma + 1, NULL, 10);
    
    return 0;
}

static int parse_dtype_from_json(const char* json, DataType* dtype) {
    char* dtype_start = strstr(json, "\"dtype\":");
    if (!dtype_start) return -1;
    
    char* quote1 = strchr(dtype_start, '"');
    if (!quote1) return -1;
    
    char* quote2 = strchr(quote1 + 1, '"');
    if (!quote2) return -1;
    
    size_t len = quote2 - quote1 - 1;
    char dtype_str[32] = {0};
    strncpy(dtype_str, quote1 + 1, len < 31 ? len : 31);
    
    if (strcmp(dtype_str, "float32") == 0) *dtype = DTYPE_FLOAT32;
    else if (strcmp(dtype_str, "float16") == 0) *dtype = DTYPE_FLOAT16;
    else if (strcmp(dtype_str, "bfloat16") == 0) *dtype = DTYPE_BFLOAT16;
    else if (strcmp(dtype_str, "int8") == 0) *dtype = DTYPE_INT8;
    else if (strcmp(dtype_str, "uint8") == 0) *dtype = DTYPE_UINT8;
    else if (strcmp(dtype_str, "int32") == 0) *dtype = DTYPE_INT32;
    else if (strcmp(dtype_str, "int64") == 0) *dtype = DTYPE_INT64;
    else *dtype = DTYPE_BFLOAT16;
    
    return 0;
}

static size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return 4;
        case DTYPE_FLOAT16: return 2;
        case DTYPE_BFLOAT16: return 2;
        case DTYPE_INT8: return 1;
        case DTYPE_UINT8: return 1;
        case DTYPE_INT32: return 4;
        case DTYPE_INT64: return 8;
        default: return 2;
    }
}

static MappedFile* mmap_file(const char* file_path) {
    MappedFile* mf = (MappedFile*)calloc(1, sizeof(MappedFile));
    if (!mf) return NULL;
    
    mf->fd = open(file_path, O_RDONLY);
    if (mf->fd < 0) {
        fprintf(stderr, "Failed to open file for mmap: %s\n", file_path);
        free(mf);
        return NULL;
    }
    
    struct stat st;
    if (fstat(mf->fd, &st) < 0) {
        fprintf(stderr, "Failed to get file size: %s\n", file_path);
        close(mf->fd);
        free(mf);
        return NULL;
    }
    
    mf->size = st.st_size;
    
    mf->data = mmap(NULL, mf->size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
    if (mf->data == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap file: %s\n", file_path);
        close(mf->fd);
        free(mf);
        return NULL;
    }
    
    madvise(mf->data, mf->size, MADV_WILLNEED);
    
    mf->is_mapped = true;
    
    return mf;
}

static void munmap_file(MappedFile* mf) {
    if (!mf) return;
    
    if (mf->data && mf->data != MAP_FAILED) {
        munmap(mf->data, mf->size);
    }
    
    if (mf->fd >= 0) {
        close(mf->fd);
    }
    
    free(mf);
}

SafetensorsReader* safetensors_open(const char* file_path) {
    MappedFile* mf = mmap_file(file_path);
    if (!mf) {
        fprintf(stderr, "Failed to mmap safetensors file: %s\n", file_path);
        return NULL;
    }
    
    if (mf->size < 8) {
        fprintf(stderr, "File too small: %s\n", file_path);
        munmap_file(mf);
        return NULL;
    }
    
    char* data = mf->data;
    
    size_t header_size = 0;
    for (int i = 7; i >= 0; i--) {
        header_size = (header_size << 8) | (unsigned char)data[i];
    }
    
    if (header_size == 0 || header_size > mf->size - 8) {
        fprintf(stderr, "Invalid header size: %zu (file size: %zu)\n", header_size, mf->size);
        munmap_file(mf);
        return NULL;
    }
    
    char* header_json = data + 8;
    
    size_t data_offset = 8 + header_size;
    
    int num_tensors = 0;
    char* ptr = header_json;
    while ((ptr = strstr(ptr, "\":{\"dtype\":")) != NULL) {
        num_tensors++;
        ptr++;
    }
    
    SafetensorsReader* reader = (SafetensorsReader*)calloc(1, sizeof(SafetensorsReader));
    if (!reader) {
        munmap_file(mf);
        return NULL;
    }
    
    reader->tensor_names = (char**)calloc(num_tensors, sizeof(char*));
    reader->tensor_metas = (TensorMeta*)calloc(num_tensors, sizeof(TensorMeta));
    reader->num_tensors = num_tensors;
    reader->mapped_file = mf;
    reader->data_offset = data_offset;
    
    char* tensor_start = header_json;
    int idx = 0;
    
    while (idx < num_tensors && (tensor_start = strstr(tensor_start, "\":{\"dtype\":")) != NULL) {
        char* tensor_end = strstr(tensor_start + 1, "\":{\"dtype\":");
        if (!tensor_end) tensor_end = header_json + header_size;
        
        // tensor_start points to the opening quote of ":{"dtype":
        // So name_end is tensor_start (the quote before the colon)
        char* name_end = tensor_start;
        
        // Find the opening quote of the tensor name
        char* name_start = name_end - 1;
        while (name_start > header_json && *name_start != '"') name_start--;
        
        if (name_start >= header_json && name_end > name_start) {
            name_start++;  // Skip the opening quote
            size_t name_len = name_end - name_start;
            reader->tensor_names[idx] = (char*)malloc(name_len + 1);
            strncpy(reader->tensor_names[idx], name_start, name_len);
            reader->tensor_names[idx][name_len] = '\0';
            
            parse_dtype_from_json(tensor_start, &reader->tensor_metas[idx].dtype);
            parse_shape_from_json(tensor_start, reader->tensor_metas[idx].shape, &reader->tensor_metas[idx].ndim);
            
            reader->tensor_metas[idx].n_elements = 1;
            for (int i = 0; i < reader->tensor_metas[idx].ndim; i++) {
                reader->tensor_metas[idx].n_elements *= reader->tensor_metas[idx].shape[i];
            }
            
            size_t data_start, data_end;
            if (parse_data_offsets(tensor_start, &data_start, &data_end) == 0) {
                reader->tensor_metas[idx].offset = data_offset + data_start;
                reader->tensor_metas[idx].size_bytes = data_end - data_start;
            } else {
                reader->tensor_metas[idx].size_bytes = reader->tensor_metas[idx].n_elements * get_dtype_size(reader->tensor_metas[idx].dtype);
                reader->tensor_metas[idx].offset = data_offset;
                for (int j = 0; j < idx; j++) {
                    reader->tensor_metas[idx].offset += reader->tensor_metas[j].size_bytes;
                }
            }
            
            idx++;
        }
        
        tensor_start = tensor_end;
    }
    
    return reader;
}

void safetensors_close(SafetensorsReader* reader) {
    if (!reader) return;
    
    if (reader->mapped_file) {
        munmap_file(reader->mapped_file);
    }
    
    for (size_t i = 0; i < reader->num_tensors; i++) {
        free(reader->tensor_names[i]);
    }
    free(reader->tensor_names);
    free(reader->tensor_metas);
    free(reader);
}

static int find_tensor_index(SafetensorsReader* reader, const char* name) {
    for (size_t i = 0; i < reader->num_tensors; i++) {
        if (strcmp(reader->tensor_names[i], name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

Tensor* safetensors_read_tensor(SafetensorsReader* reader, const char* name) {
    int idx = find_tensor_index(reader, name);
    if (idx < 0) {
        fprintf(stderr, "Tensor not found: %s\n", name);
        return NULL;
    }
    
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;
    
    tensor->meta = reader->tensor_metas[idx];
    tensor->data = NULL;
    
    return tensor;
}

bool safetensors_load_tensor_data(SafetensorsReader* reader, Tensor* tensor) {
    if (!reader || !tensor) return false;
    
    if (!reader->mapped_file || !reader->mapped_file->data) {
        fprintf(stderr, "No mmap data available\n");
        return false;
    }
    
    size_t offset = tensor->meta.offset;
    size_t size = tensor->meta.size_bytes;
    
    if (offset + size > reader->mapped_file->size) {
        fprintf(stderr, "Tensor data out of bounds\n");
        return false;
    }
    
    tensor->data = reader->mapped_file->data + offset;
    
    return true;
}

bool safetensors_get_tensor_ptr(SafetensorsReader* reader, const char* name, void** ptr_out, size_t* size_out) {
    if (!reader || !name || !ptr_out) return false;
    
    int idx = find_tensor_index(reader, name);
    if (idx < 0) {
        fprintf(stderr, "Tensor not found: %s\n", name);
        return false;
    }
    
    TensorMeta* meta = &reader->tensor_metas[idx];
    
    if (meta->offset + meta->size_bytes > reader->mapped_file->size) {
        fprintf(stderr, "Tensor data out of bounds: %s\n", name);
        return false;
    }
    
    *ptr_out = reader->mapped_file->data + meta->offset;
    if (size_out) {
        *size_out = meta->size_bytes;
    }
    
    return true;
}

void tensor_free(Tensor* tensor) {
    if (!tensor) return;
    if (tensor->data) {
    }
    free(tensor);
}
