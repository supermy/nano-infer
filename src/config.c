#include "config.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* read_file_content(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", path);
        return NULL;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* content = (char*)malloc(size + 1);
    if (!content) {
        fclose(f);
        return NULL;
    }
    
    fread(content, 1, size, f);
    content[size] = '\0';
    fclose(f);
    
    *out_size = size;
    return content;
}

static int parse_dtype(const char* dtype_str) {
    if (strcmp(dtype_str, "float32") == 0) return DTYPE_FLOAT32;
    if (strcmp(dtype_str, "float16") == 0) return DTYPE_FLOAT16;
    if (strcmp(dtype_str, "bfloat16") == 0) return DTYPE_BFLOAT16;
    if (strcmp(dtype_str, "int8") == 0) return DTYPE_INT8;
    if (strcmp(dtype_str, "uint8") == 0) return DTYPE_UINT8;
    if (strcmp(dtype_str, "int32") == 0) return DTYPE_INT32;
    if (strcmp(dtype_str, "int64") == 0) return DTYPE_INT64;
    return DTYPE_BFLOAT16;
}

static char* json_get_string(const char* json, const char* key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    char* pos = strstr(json, pattern);
    if (!pos) return NULL;
    
    pos = strchr(pos, '"');
    if (!pos) return NULL;
    pos++;
    
    char* end = strchr(pos, '"');
    if (!end) return NULL;
    
    size_t len = end - pos;
    char* result = (char*)malloc(len + 1);
    strncpy(result, pos, len);
    result[len] = '\0';
    return result;
}

static int json_get_int(const char* json, const char* key, int default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    char* pos = strstr(json, pattern);
    if (!pos) return default_val;
    
    pos = strchr(pos, ':');
    if (!pos) return default_val;
    pos++;
    
    while (*pos == ' ') pos++;
    
    return (int)strtol(pos, NULL, 10);
}

static float json_get_float(const char* json, const char* key, float default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    char* pos = strstr(json, pattern);
    if (!pos) return default_val;
    
    pos = strchr(pos, ':');
    if (!pos) return default_val;
    pos++;
    
    while (*pos == ' ') pos++;
    
    return (float)strtod(pos, NULL);
}

static bool json_get_bool(const char* json, const char* key, bool default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    char* pos = strstr(json, pattern);
    if (!pos) return default_val;
    
    pos = strchr(pos, ':');
    if (!pos) return default_val;
    pos++;
    
    while (*pos == ' ') pos++;
    
    if (strncmp(pos, "true", 4) == 0) return true;
    if (strncmp(pos, "false", 5) == 0) return false;
    return default_val;
}

int config_load_from_json(ModelConfig* config, const char* json_path) {
    size_t size;
    char* content = read_file_content(json_path, &size);
    if (!content) {
        return -1;
    }
    
    memset(config, 0, sizeof(ModelConfig));
    
    char* arch = json_get_string(content, "architectures");
    if (arch) {
        char* p = strchr(arch, '"');
        char* p2 = strchr(arch, ',');
        if (p2) *p2 = '\0';
        if (p) {
            char* start = p + 1;
            char* end = strchr(start, '"');
            if (end) *end = '\0';
            strncpy(config->architecture, start, sizeof(config->architecture) - 1);
        }
        free(arch);
    }
    
    config->attention_bias = json_get_bool(content, "attention_bias", false);
    config->attention_dropout = json_get_float(content, "attention_dropout", 0.0f);
    config->bos_token_id = json_get_int(content, "bos_token_id", 151643);
    config->eos_token_id = json_get_int(content, "eos_token_id", 151645);
    config->head_dim = json_get_int(content, "head_dim", 128);
    
    char* hidden_act = json_get_string(content, "hidden_act");
    if (hidden_act) {
        char* p = strchr(hidden_act, '"');
        char* p2 = p ? strchr(p + 1, '"') : NULL;
        if (p && p2) {
            size_t len = p2 - p - 1;
            if (len > 31) len = 31;
            strncpy(config->hidden_act, p + 1, len);
            config->hidden_act[len] = '\0';
        }
        free(hidden_act);
    }
    
    config->hidden_size = json_get_int(content, "hidden_size", 2560);
    config->initializer_range = json_get_float(content, "initializer_range", 0.02f);
    config->intermediate_size = json_get_int(content, "intermediate_size", 9728);
    config->max_position_embeddings = json_get_int(content, "max_position_embeddings", 40960);
    config->max_window_layers = json_get_int(content, "max_window_layers", 36);
    
    char* model_type = json_get_string(content, "model_type");
    if (model_type) {
        char* p = strchr(model_type, '"');
        char* p2 = p ? strchr(p + 1, '"') : NULL;
        if (p && p2) {
            size_t len = p2 - p - 1;
            if (len > 31) len = 31;
            strncpy(config->model_type, p + 1, len);
            config->model_type[len] = '\0';
        }
        free(model_type);
    }
    
    config->num_attention_heads = json_get_int(content, "num_attention_heads", 32);
    config->num_hidden_layers = json_get_int(content, "num_hidden_layers", 36);
    config->num_key_value_heads = json_get_int(content, "num_key_value_heads", 8);
    config->rms_norm_eps = json_get_float(content, "rms_norm_eps", 1e-6f);
    config->rope_theta = json_get_float(content, "rope_theta", 1000000.0f);
    config->tie_word_embeddings = json_get_bool(content, "tie_word_embeddings", true);
    
    char* torch_dtype = json_get_string(content, "torch_dtype");
    if (torch_dtype) {
        config->torch_dtype = parse_dtype(torch_dtype);
        free(torch_dtype);
    } else {
        config->torch_dtype = DTYPE_BFLOAT16;
    }
    
    config->use_cache = json_get_bool(content, "use_cache", true);
    config->use_sliding_window = json_get_bool(content, "use_sliding_window", false);
    config->vocab_size = json_get_int(content, "vocab_size", 151936);
    
    config->quant_method = QUANT_NONE;
    config->quant_bits = 0;
    config->quant_group_size = 128;
    config->quant_zero_point = true;
    
    char* quant_method = json_get_string(content, "quant_method");
    if (!quant_method) {
        const char* qpos = strstr(content, "\"quant_method\":");
        if (qpos) {
            const char* colon = qpos + strlen("\"quant_method\":");
            while (*colon && (*colon == ' ' || *colon == '\n' || *colon == '\t')) colon++;
            const char* start = colon;
            while (*start && *start != '"') start++;
            start++;
            const char* end = start;
            while (*end && *end != '"') end++;
            if (end > start) {
                quant_method = strndup(start, end - start);
            }
        }
    }
    
    if (quant_method) {
        if (strcmp(quant_method, "awq") == 0) {
            config->quant_method = QUANT_AWQ;
            const char* bits_pos = strstr(content, "\"bits\":");
            if (bits_pos) {
                config->quant_bits = atoi(bits_pos + strlen("\"bits\":"));
            } else {
                config->quant_bits = 4;
            }
            const char* group_pos = strstr(content, "\"group_size\":");
            if (group_pos) {
                config->quant_group_size = atoi(group_pos + strlen("\"group_size\":"));
            }
        } else if (strcmp(quant_method, "fp8") == 0) {
            config->quant_method = QUANT_FP8;
        }
        free(quant_method);
    }
    
    config->generation_max_length = 2048;
    config->generation_min_length = 0;
    config->temperature = 0.7f;
    config->top_p = 0.9f;
    config->top_k = 20;
    config->do_sample = true;
    config->pad_token_id = 151643;
    config->eos_token_id2 = 151645;
    
    free(content);
    return 0;
}

void config_print(const ModelConfig* config) {
    printf("=== Qwen3 Model Configuration ===\n");
    printf("Architecture: %s\n", config->architecture);
    printf("Model Type: %s\n", config->model_type);
    printf("Hidden Size: %d\n", config->hidden_size);
    printf("Num Attention Heads: %d\n", config->num_attention_heads);
    printf("Num KV Heads: %d\n", config->num_key_value_heads);
    printf("Num Hidden Layers: %d\n", config->num_hidden_layers);
    printf("Intermediate Size: %d\n", config->intermediate_size);
    printf("Vocab Size: %d\n", config->vocab_size);
    printf("Head Dim: %d\n", config->head_dim);
    printf("Max Position Embeddings: %d\n", config->max_position_embeddings);
    printf("RMS Norm Eps: %.6f\n", config->rms_norm_eps);
    printf("RoPE Theta: %.1f\n", config->rope_theta);
    printf("Hidden Act: %s\n", config->hidden_act);
    printf("EOS Token ID: %d\n", config->eos_token_id);
    if (config->quant_method == QUANT_AWQ) {
        printf("Quantization: AWQ %d-bit (group_size=%d)\n", 
               config->quant_bits, config->quant_group_size);
    } else if (config->quant_method == QUANT_FP8) {
        printf("Quantization: FP8\n");
    }
    printf("==================================\n");
}
