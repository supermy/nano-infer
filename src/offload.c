#include "offload.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <fstream>
#include <sstream>
#endif

OffloadManager* offload_manager_create(int num_layers, size_t max_cpu_memory, const char* cache_dir) {
    OffloadManager* manager = (OffloadManager*)calloc(1, sizeof(OffloadManager));
    if (!manager) return NULL;
    
    manager->layers = (LayerOffloadInfo*)calloc(num_layers, sizeof(LayerOffloadInfo));
    if (!manager->layers) {
        free(manager);
        return NULL;
    }
    
    manager->num_layers = num_layers;
    manager->max_cpu_memory = max_cpu_memory;
    manager->total_cpu_memory = 0;
    manager->total_disk_memory = 0;
    manager->auto_offload = true;
    manager->memory_threshold = 0.9f;
    
    if (cache_dir) {
        manager->cache_dir = strdup(cache_dir);
        mkdir(cache_dir, 0755);
    } else {
        manager->cache_dir = strdup("/tmp/nano_infer_cache");
        mkdir(manager->cache_dir, 0755);
    }
    
    for (int i = 0; i < num_layers; i++) {
        manager->layers[i].layer_idx = i;
        manager->layers[i].device = DEVICE_CPU;
        manager->layers[i].is_loaded = false;
        manager->layers[i].file_path = NULL;
        manager->layers[i].memory_size = 0;
        manager->layers[i].data = NULL;
    }
    
    return manager;
}

void offload_manager_free(OffloadManager* manager) {
    if (!manager) return;
    
    for (int i = 0; i < manager->num_layers; i++) {
        if (manager->layers[i].file_path) {
            unlink(manager->layers[i].file_path);
            free(manager->layers[i].file_path);
        }
        if (manager->layers[i].data) {
            free(manager->layers[i].data);
        }
    }
    free(manager->layers);
    free(manager->cache_dir);
    free(manager);
}

int offload_manager_configure(OffloadManager* manager, const OffloadConfig* config) {
    if (!manager || !config) return -1;
    
    for (int i = 0; i < config->num_cpu_layers && i < manager->num_layers; i++) {
        int layer_idx = config->cpu_layers[i];
        if (layer_idx >= 0 && layer_idx < manager->num_layers) {
            manager->layers[layer_idx].device = DEVICE_CPU;
        }
    }
    
    for (int i = 0; i < config->num_disk_layers && i < manager->num_layers; i++) {
        int layer_idx = config->disk_layers[i];
        if (layer_idx >= 0 && layer_idx < manager->num_layers) {
            manager->layers[layer_idx].device = DEVICE_DISK;
        }
    }
    
    return 0;
}

int offload_layer_to_disk(OffloadManager* manager, int layer_idx, const void* data, size_t size) {
    if (!manager || layer_idx < 0 || layer_idx >= manager->num_layers) return -1;
    if (!data || size == 0) return -1;
    
    LayerOffloadInfo* layer = &manager->layers[layer_idx];
    
    if (!layer->file_path) {
        char path[512];
        snprintf(path, sizeof(path), "%s/layer_%d.bin", manager->cache_dir, layer_idx);
        layer->file_path = strdup(path);
    }
    
    FILE* fp = fopen(layer->file_path, "wb");
    if (!fp) return -1;
    
    size_t written = fwrite(data, 1, size, fp);
    fclose(fp);
    
    if (written != size) return -1;
    
    layer->device = DEVICE_DISK;
    layer->memory_size = size;
    layer->is_loaded = false;
    
    if (layer->data) {
        free(layer->data);
        layer->data = NULL;
        manager->total_cpu_memory -= size;
    }
    
    manager->total_disk_memory += size;
    
    return 0;
}

int offload_layer_to_cpu(OffloadManager* manager, int layer_idx) {
    if (!manager || layer_idx < 0 || layer_idx >= manager->num_layers) return -1;
    
    LayerOffloadInfo* layer = &manager->layers[layer_idx];
    
    if (layer->device != DEVICE_DISK || !layer->file_path) return -1;
    
    FILE* fp = fopen(layer->file_path, "rb");
    if (!fp) return -1;
    
    layer->data = malloc(layer->memory_size);
    if (!layer->data) {
        fclose(fp);
        return -1;
    }
    
    size_t read_size = fread(layer->data, 1, layer->memory_size, fp);
    fclose(fp);
    
    if (read_size != layer->memory_size) {
        free(layer->data);
        layer->data = NULL;
        return -1;
    }
    
    layer->device = DEVICE_CPU;
    layer->is_loaded = true;
    manager->total_cpu_memory += layer->memory_size;
    manager->total_disk_memory -= layer->memory_size;
    
    return 0;
}

void* offload_manager_get_layer(OffloadManager* manager, int layer_idx, bool load_if_needed) {
    if (!manager || layer_idx < 0 || layer_idx >= manager->num_layers) return NULL;
    
    LayerOffloadInfo* layer = &manager->layers[layer_idx];
    
    if (layer->device == DEVICE_CPU && layer->is_loaded) {
        return layer->data;
    }
    
    if (layer->device == DEVICE_DISK && load_if_needed) {
        if (offload_layer_to_cpu(manager, layer_idx) == 0) {
            return layer->data;
        }
    }
    
    return NULL;
}

OffloadConfig* offload_config_create_default(int num_layers, size_t total_memory, size_t layer_memory) {
    OffloadConfig* config = (OffloadConfig*)calloc(1, sizeof(OffloadConfig));
    if (!config) return NULL;
    
    int max_cpu_layers = (int)(total_memory / layer_memory);
    if (max_cpu_layers > num_layers) max_cpu_layers = num_layers;
    
    config->cpu_layers = (int*)malloc(max_cpu_layers * sizeof(int));
    config->disk_layers = (int*)malloc((num_layers - max_cpu_layers) * sizeof(int));
    
    if (!config->cpu_layers || !config->disk_layers) {
        free(config->cpu_layers);
        free(config->disk_layers);
        free(config);
        return NULL;
    }
    
    config->num_cpu_layers = max_cpu_layers;
    config->num_disk_layers = num_layers - max_cpu_layers;
    
    for (int i = 0; i < max_cpu_layers; i++) {
        config->cpu_layers[i] = i;
    }
    for (int i = max_cpu_layers; i < num_layers; i++) {
        config->disk_layers[i - max_cpu_layers] = i;
    }
    
    config->estimated_memory = max_cpu_layers * layer_memory;
    
    return config;
}

void offload_config_free(OffloadConfig* config) {
    if (!config) return;
    free(config->cpu_layers);
    free(config->disk_layers);
    free(config);
}

OffloadConfig* offload_config_parse(const char* config_str) {
    if (!config_str) return NULL;
    
    OffloadConfig* config = (OffloadConfig*)calloc(1, sizeof(OffloadConfig));
    if (!config) return NULL;
    
    int cpu_layers[256];
    int disk_layers[256];
    int cpu_count = 0;
    int disk_count = 0;
    
    const char* p = config_str;
    char mode = 'c';
    
    while (*p) {
        if (*p == 'c' || *p == 'C') {
            mode = 'c';
            p++;
        } else if (*p == 'd' || *p == 'D') {
            mode = 'd';
            p++;
        } else if (*p >= '0' && *p <= '9') {
            int val = 0;
            while (*p >= '0' && *p <= '9') {
                val = val * 10 + (*p - '0');
                p++;
            }
            if (mode == 'c' && cpu_count < 256) {
                cpu_layers[cpu_count++] = val;
            } else if (mode == 'd' && disk_count < 256) {
                disk_layers[disk_count++] = val;
            }
        } else {
            p++;
        }
    }
    
    config->cpu_layers = (int*)malloc(cpu_count * sizeof(int));
    config->disk_layers = (int*)malloc(disk_count * sizeof(int));
    
    if (!config->cpu_layers || !config->disk_layers) {
        free(config->cpu_layers);
        free(config->disk_layers);
        free(config);
        return NULL;
    }
    
    memcpy(config->cpu_layers, cpu_layers, cpu_count * sizeof(int));
    memcpy(config->disk_layers, disk_layers, disk_count * sizeof(int));
    config->num_cpu_layers = cpu_count;
    config->num_disk_layers = disk_count;
    
    return config;
}

size_t get_available_memory(void) {
#ifdef __APPLE__
    mach_port_t host = mach_host_self();
    vm_size_t page_size;
    host_page_size(host, &page_size);
    
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        return (size_t)vm_stats.free_count * page_size;
    }
    return 0;
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram * info.mem_unit;
    }
    return 0;
#else
    return 0;
#endif
}

size_t get_process_memory_usage(void) {
#ifdef __APPLE__
    struct task_basic_info info;
    mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
    
    if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
#elif defined(__linux__)
    FILE* fp = fopen("/proc/self/status", "r");
    if (!fp) return 0;
    
    char line[256];
    size_t rss = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "VmRSS: %zu kB", &rss);
            rss *= 1024;
            break;
        }
    }
    fclose(fp);
    return rss;
#else
    return 0;
#endif
}
