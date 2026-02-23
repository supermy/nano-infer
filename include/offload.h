#ifndef OFFLOAD_H
#define OFFLOAD_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef enum {
    DEVICE_CPU = 0,
    DEVICE_DISK = 1,
} DeviceType;

typedef struct {
    int layer_idx;
    DeviceType device;
    bool is_loaded;
    char* file_path;
    size_t memory_size;
    void* data;
} LayerOffloadInfo;

typedef struct {
    LayerOffloadInfo* layers;
    int num_layers;
    size_t total_cpu_memory;
    size_t max_cpu_memory;
    size_t total_disk_memory;
    char* cache_dir;
    bool auto_offload;
    float memory_threshold;
} OffloadManager;

typedef struct {
    int* cpu_layers;
    int num_cpu_layers;
    int* disk_layers;
    int num_disk_layers;
    size_t estimated_memory;
} OffloadConfig;

OffloadManager* offload_manager_create(int num_layers, size_t max_cpu_memory, const char* cache_dir);
void offload_manager_free(OffloadManager* manager);

int offload_manager_configure(OffloadManager* manager, const OffloadConfig* config);
int offload_layer_to_disk(OffloadManager* manager, int layer_idx, const void* data, size_t size);
int offload_layer_to_cpu(OffloadManager* manager, int layer_idx);
void* offload_manager_get_layer(OffloadManager* manager, int layer_idx, bool load_if_needed);

OffloadConfig* offload_config_create_default(int num_layers, size_t total_memory, size_t layer_memory);
void offload_config_free(OffloadConfig* config);
OffloadConfig* offload_config_parse(const char* config_str);

size_t get_available_memory(void);
size_t get_process_memory_usage(void);

#endif
