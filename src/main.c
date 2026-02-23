#include "common.h"
#include "config.h"
#include "tokenizer.h"
#include "model.h"
#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_usage(const char* prog) {
    printf("Qwen3-4B Inference Engine v%s\n\n", QWEN3_VERSION);
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model <path>     Model directory path (default: ../models/Qwen/Qwen3-4B/)\n");
    printf("  -p, --prompt <text>    Input prompt for generation\n");
    printf("  -t, --temperature <f>  Sampling temperature (default: 0.7)\n");
    printf("  -k, --top-k <n>        Top-k sampling (default: 20)\n");
    printf("  -P, --top-p <f>        Top-p nucleus sampling (default: 0.9)\n");
    printf("  -l, --max-length <n>   Maximum generation length (default: 512)\n");
    printf("  -s, --seed <n>         Random seed\n");
    printf("  -i, --interactive      Interactive chat mode\n");
    printf("  -v, --verbose         Verbose output\n");
    printf("  -h, --help             Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s -p \"Hello, how are you?\"\n", prog);
    printf("  %s -m /path/to/model -p \"Tell me a story\"\n", prog);
    printf("  %s -i\n", prog);
}

static char* default_model_path = "../models/Qwen/Qwen3-4B/";

typedef struct {
    char* model_path;
    char* prompt;
    float temperature;
    int top_k;
    float top_p;
    int max_length;
    unsigned int seed;
    bool interactive;
    bool verbose;
    bool has_prompt;
} Options;

static Options parse_args(int argc, char** argv) {
    Options opts = {
        .model_path = default_model_path,
        .prompt = NULL,
        .temperature = 0.7f,
        .top_k = 20,
        .top_p = 0.9f,
        .max_length = 512,
        .seed = 0,
        .interactive = false,
        .verbose = false,
        .has_prompt = false
    };
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) opts.model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (i + 1 < argc) {
                opts.prompt = argv[++i];
                opts.has_prompt = true;
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temperature") == 0) {
            if (i + 1 < argc) opts.temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--top-k") == 0) {
            if (i + 1 < argc) opts.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-P") == 0 || strcmp(argv[i], "--top-p") == 0) {
            if (i + 1 < argc) opts.top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--max-length") == 0) {
            if (i + 1 < argc) opts.max_length = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) opts.seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            opts.interactive = true;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            opts.verbose = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }
    
    if (opts.seed == 0) {
        opts.seed = (unsigned int)time(NULL);
    }
    
    return opts;
}

static char* read_stdin_line(void) {
    char buffer[4096];
    if (fgets(buffer, sizeof(buffer), stdin)) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
        char* result = (char*)malloc(len + 1);
        strcpy(result, buffer);
        return result;
    }
    return NULL;
}

static int generate(Qwen3Model* model, Tokenizer* tokenizer, const char* prompt,
                    float temperature, int top_k, float top_p, int max_length) {
    int* tokens = NULL;
    size_t token_len = 0;
    
    if (tokenizer_encode(tokenizer, prompt, &tokens, &token_len) != 0) {
        fprintf(stderr, "Failed to encode prompt\n");
        return -1;
    }
    
    if (token_len == 0) {
        fprintf(stderr, "Empty token sequence\n");
        return -1;
    }
    
    printf("[Input tokens: %zu]\n", token_len);
    printf("Prompt: %s\n\n", prompt);
    printf("Generating...\n\n");
    
    int* output_tokens = NULL;
    size_t output_len = 0;
    
    int result = model_generate(model, tokens, token_len, temperature, top_k, top_p, max_length, &output_tokens, &output_len);
    
    if (result != 0) {
        fprintf(stderr, "Generation failed\n");
        tokenizer_free_tokens(tokens);
        return -1;
    }
    
    if (output_tokens && output_len > 0) {
        int* generated_only = output_tokens + token_len;
        size_t generated_len = output_len - token_len;
        
        if (generated_len > 0) {
            char* decoded = tokenizer_decode(tokenizer, generated_only, generated_len);
            if (decoded) {
                printf("%s", decoded);
                fflush(stdout);
                free(decoded);
            }
        }
        free(output_tokens);
    }
    
    tokenizer_free_tokens(tokens);
    
    printf("\n\n[Generation complete]\n");
    
    return 0;
}

int main(int argc, char** argv) {
    Options opts = parse_args(argc, argv);
    
    printf("=== Qwen3-4B Inference Engine ===\n");
    printf("Version: %s\n", QWEN3_VERSION);
    printf("Model: %s\n", opts.model_path);
    
    if (opts.verbose) {
        printf("\nConfiguration:\n");
        printf("  Temperature: %.2f\n", opts.temperature);
        printf("  Top-K: %d\n", opts.top_k);
        printf("  Top-P: %.2f\n", opts.top_p);
        printf("  Max Length: %d\n", opts.max_length);
        printf("  Seed: %u\n", opts.seed);
    }
    
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", opts.model_path);
    
    ModelConfig config;
    if (config_load_from_json(&config, config_path) != 0) {
        fprintf(stderr, "Failed to load model config from %s\n", config_path);
        return 1;
    }
    
    if (opts.verbose) {
        config_print(&config);
    }
    
    Tokenizer* tokenizer = tokenizer_load(opts.model_path);
    if (!tokenizer) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }
    printf("Tokenizer loaded successfully\n");
    
    Qwen3Model* model = model_load(opts.model_path, &config);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        tokenizer_free(tokenizer);
        return 1;
    }
    printf("Model loaded successfully\n");
    
    if (opts.interactive) {
        printf("\n=== Interactive Mode ===\n");
        printf("Type your messages below (Ctrl+C to exit):\n\n");
        
        while (1) {
            printf("\n> ");
            fflush(stdout);
            
            char* input = read_stdin_line();
            if (!input) break;
            
            if (strlen(input) == 0) {
                free(input);
                continue;
            }
            
            if (strcmp(input, "/quit") == 0 || strcmp(input, "/exit") == 0) {
                free(input);
                break;
            }
            
            generate(model, tokenizer, input, opts.temperature, opts.top_k, opts.top_p, opts.max_length);
            
            free(input);
        }
    } else if (opts.has_prompt) {
        generate(model, tokenizer, opts.prompt, opts.temperature, opts.top_k, opts.top_p, opts.max_length);
    } else {
        print_usage(argv[0]);
    }
    
    model_free(model);
    tokenizer_free(tokenizer);
    
    printf("\nCleaned up resources.\n");
    
    return 0;
}
