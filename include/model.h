#ifndef MODEL_H
#define MODEL_H

#include "config.h"
#include "safetensors.h"
#include "common.h"
#include "kv_cache.h"

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
    uint16_t* embed_tokens_bf16;
    bool embed_is_bf16;
    
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
    
    KVCache* kv_cache;
    int kv_cache_len;
} Qwen3Model;

Qwen3Model* model_load(const char* model_dir, const ModelConfig* config);
void model_free(Qwen3Model* model);

int model_generate(Qwen3Model* model, const int* input_tokens, size_t input_len,
                   float temperature, int top_k, float top_p, int max_length,
                   int** output_tokens, size_t* output_len);

int model_generate_with_cache(Qwen3Model* model, const int* input_tokens, size_t input_len,
                               float temperature, int top_k, float top_p, int max_length,
                               int** output_tokens, size_t* output_len);

void model_init_cache(Qwen3Model* model, int max_seq_len);
void model_free_cache(Qwen3Model* model);

typedef struct {
    float* data;
    int dim;
    int batch_size;
} HiddenState;

#endif
