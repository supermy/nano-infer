#ifndef MODEL_H
#define MODEL_H

#include "config.h"
#include "safetensors.h"
#include "common.h"

typedef struct {
    uint8_t* qweight;
    uint16_t* scales_fp16;
    uint8_t* qzeros;
    int in_features;
    int out_features;
} AWQWeight;

typedef struct {
    ModelConfig config;
    
    float* embed_tokens;
    
    float** q_proj_weight;
    float** k_proj_weight;
    float** v_proj_weight;
    float** o_proj_weight;
    
    float** gate_proj_weight;
    float** up_proj_weight;
    float** down_proj_weight;
    
    AWQWeight* q_proj_awq;
    AWQWeight* k_proj_awq;
    AWQWeight* v_proj_awq;
    AWQWeight* o_proj_awq;
    AWQWeight* gate_proj_awq;
    AWQWeight* up_proj_awq;
    AWQWeight* down_proj_awq;
    AWQWeight* lm_head_awq;
    
    float* input_layernorm_weights;
    float* post_attention_layernorm_weights;
    
    float* q_norm_weights;
    float* k_norm_weights;
    
    float* final_layernorm_weight;
    float* lm_head_weight;
    
    int num_layers_loaded;
    int num_layers_total;
    
    SafetensorsReader** readers;
    int num_readers;
} Qwen3Model;

Qwen3Model* model_load(const char* model_dir, const ModelConfig* config);
void model_free(Qwen3Model* model);

int model_generate(Qwen3Model* model, const int* input_tokens, size_t input_len,
                   float temperature, int top_k, float top_p, int max_length,
                   int** output_tokens, size_t* output_len);

typedef struct {
    float* data;
    int dim;
    int batch_size;
} HiddenState;

typedef struct {
    float* q;
    float* k;
    float* v;
    int q_len;
    int kv_len;
} KVCache;

KVCache* kv_cache_create(int num_layers, int num_heads, int num_kv_heads, int head_dim, int max_seq_len);
void kv_cache_free(KVCache* cache);
void kv_cache_update(KVCache* cache, int layer_idx, int pos, const float* k, const float* v);

#endif
