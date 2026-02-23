#ifndef CONFIG_H
#define CONFIG_H

#include "common.h"

typedef enum {
    QUANT_NONE = 0,
    QUANT_AWQ,
    QUANT_FP8
} QuantMethod;

typedef struct {
    char architecture[64];
    bool attention_bias;
    float attention_dropout;
    int bos_token_id;
    int eos_token_id;
    int head_dim;
    char hidden_act[32];
    int hidden_size;
    float initializer_range;
    int intermediate_size;
    int max_position_embeddings;
    int max_window_layers;
    char model_type[32];
    int num_attention_heads;
    int num_hidden_layers;
    int num_key_value_heads;
    float rms_norm_eps;
    float rope_scaling;
    float rope_theta;
    int sliding_window;
    bool tie_word_embeddings;
    DataType torch_dtype;
    bool use_cache;
    bool use_sliding_window;
    int vocab_size;
    
    int generation_max_length;
    int generation_min_length;
    float temperature;
    float top_p;
    int top_k;
    bool do_sample;
    int pad_token_id;
    int eos_token_id2;
    
    QuantMethod quant_method;
    int quant_bits;
    int quant_group_size;
    bool quant_zero_point;
} ModelConfig;

int config_load_from_json(ModelConfig* config, const char* json_path);
void config_print(const ModelConfig* config);

#endif
