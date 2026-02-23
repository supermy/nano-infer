#include "tokenizer.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_TEXT 256
#define INITIAL_VOCAB_SIZE 200000

static char* read_file(const char* path, size_t* size_out) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* data = (char*)malloc(size + 1);
    if (!data) {
        fclose(f);
        return NULL;
    }
    
    size_t read_size = fread(data, 1, size, f);
    data[read_size] = '\0';
    fclose(f);
    
    if (size_out) *size_out = read_size;
    return data;
}

static void parse_vocab(Tokenizer* tok, const char* vocab_start, const char* vocab_end) {
    tok->vocab = (char**)malloc(INITIAL_VOCAB_SIZE * sizeof(char*));
    tok->vocab_ids = (int*)malloc(INITIAL_VOCAB_SIZE * sizeof(int));
    tok->vocab_size = 0;
    
    const char* ptr = vocab_start;
    
    while (ptr < vocab_end && tok->vocab_size < INITIAL_VOCAB_SIZE) {
        while (ptr < vocab_end && *ptr != '"') ptr++;
        if (ptr >= vocab_end) break;
        
        ptr++;
        const char* key_start = ptr;
        
        int in_escape = 0;
        while (ptr < vocab_end) {
            if (in_escape) {
                in_escape = 0;
                ptr++;
            } else if (*ptr == '\\') {
                in_escape = 1;
                ptr++;
            } else if (*ptr == '"') {
                break;
            } else {
                ptr++;
            }
        }
        if (ptr >= vocab_end) break;
        
        size_t key_len = ptr - key_start;
        char* token = (char*)malloc(key_len + 1);
        if (!token) break;
        
        size_t j = 0;
        for (size_t i = 0; i < key_len && j < MAX_TOKEN_TEXT - 1; i++) {
            if (key_start[i] == '\\' && i + 1 < key_len) {
                i++;
                switch (key_start[i]) {
                    case 'n': token[j++] = '\n'; break;
                    case 't': token[j++] = '\t'; break;
                    case 'r': token[j++] = '\r'; break;
                    case '\\': token[j++] = '\\'; break;
                    case '"': token[j++] = '"'; break;
                    case 'x':
                        if (i + 2 < key_len) {
                            char hex[3] = {key_start[i+1], key_start[i+2], 0};
                            token[j++] = (char)strtol(hex, NULL, 16);
                            i += 2;
                        }
                        break;
                    default:
                        if (key_start[i] >= '0' && key_start[i] <= '9') {
                            if (i + 4 < key_len && key_start[i+1] >= '0' && key_start[i+1] <= '9' &&
                                key_start[i+2] >= '0' && key_start[i+2] <= '9' &&
                                key_start[i+3] >= '0' && key_start[i+3] <= '9') {
                                char uni[5] = {key_start[i], key_start[i+1], key_start[i+2], key_start[i+3], 0};
                                int cp = (int)strtol(uni, NULL, 10);
                                if (cp < 0x80) {
                                    token[j++] = (char)cp;
                                } else if (cp < 0x800) {
                                    token[j++] = (char)(0xC0 | (cp >> 6));
                                    token[j++] = (char)(0x80 | (cp & 0x3F));
                                } else {
                                    token[j++] = (char)(0xE0 | (cp >> 12));
                                    token[j++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                                    token[j++] = (char)(0x80 | (cp & 0x3F));
                                }
                                i += 4;
                            } else {
                                token[j++] = key_start[i];
                            }
                        } else {
                            token[j++] = key_start[i];
                        }
                        break;
                }
            } else {
                token[j++] = key_start[i];
            }
        }
        token[j] = '\0';
        
        ptr++;
        
        while (ptr < vocab_end && (*ptr == ' ' || *ptr == '\n' || *ptr == '\t' || *ptr == '\r')) ptr++;
        
        if (ptr >= vocab_end || *ptr != ':') {
            free(token);
            break;
        }
        ptr++;
        
        while (ptr < vocab_end && (*ptr == ' ' || *ptr == '\n' || *ptr == '\t' || *ptr == '\r')) ptr++;
        
        int id = 0;
        while (ptr < vocab_end && *ptr >= '0' && *ptr <= '9') {
            id = id * 10 + (*ptr - '0');
            ptr++;
        }
        
        tok->vocab[tok->vocab_size] = token;
        tok->vocab_ids[tok->vocab_size] = id;
        tok->vocab_size++;
        
        while (ptr < vocab_end && *ptr != ',' && *ptr != '}') ptr++;
        if (ptr >= vocab_end || *ptr == '}') break;
        ptr++;
    }
}

static int find_token_id(Tokenizer* tok, const char* text) {
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->vocab[i] && strcmp(tok->vocab[i], text) == 0) {
            return tok->vocab_ids[i];
        }
    }
    return -1;
}

static const char* find_token_text(Tokenizer* tok, int id) {
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->vocab_ids[i] == id) {
            return tok->vocab[i];
        }
    }
    return NULL;
}

Tokenizer* tokenizer_load(const char* tokenizer_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/tokenizer.json", tokenizer_dir);
    
    size_t size;
    char* data = read_file(path, &size);
    if (!data) {
        fprintf(stderr, "Failed to read tokenizer.json\n");
        return NULL;
    }
    
    Tokenizer* tok = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    if (!tok) {
        free(data);
        return NULL;
    }
    
    const char* model_key = "\"model\":";
    const char* model_start = strstr(data, model_key);
    if (!model_start) {
        const char* vocab_key = "\"vocab\":";
        model_start = strstr(data, vocab_key);
    } else {
        const char* vocab_key = "\"vocab\":";
        model_start = strstr(model_start, vocab_key);
    }
    
    if (!model_start) {
        fprintf(stderr, "Cannot find vocab in tokenizer.json\n");
        free(data);
        free(tok);
        return NULL;
    }
    
    const char* vocab_start = model_start + strlen("\"vocab\":");
    while (*vocab_start && (*vocab_start == ' ' || *vocab_start == '\n' || *vocab_start == '\t')) {
        vocab_start++;
    }
    
    if (*vocab_start != '{') {
        fprintf(stderr, "Invalid vocab format\n");
        free(data);
        free(tok);
        return NULL;
    }
    
    vocab_start++;
    
    int brace_count = 1;
    int in_string = 0;
    int in_escape = 0;
    const char* vocab_end = vocab_start;
    while (*vocab_end && brace_count > 0) {
        if (in_escape) {
            in_escape = 0;
        } else if (*vocab_end == '\\') {
            in_escape = 1;
        } else if (*vocab_end == '"') {
            in_string = !in_string;
        } else if (!in_string) {
            if (*vocab_end == '{') brace_count++;
            else if (*vocab_end == '}') brace_count--;
        }
        vocab_end++;
    }
    
    parse_vocab(tok, vocab_start, vocab_end);
    
    tok->bos_token_id = find_token_id(tok, "<|endoftext|>");
    tok->eos_token_id = find_token_id(tok, "<|endoftext|>");
    tok->pad_token_id = find_token_id(tok, "<|endoftext|>");
    
    if (tok->bos_token_id < 0) tok->bos_token_id = 151643;
    if (tok->eos_token_id < 0) tok->eos_token_id = 151643;
    if (tok->pad_token_id < 0) tok->pad_token_id = 151643;
    
    printf("Tokenizer loaded: %d vocab entries\n", tok->vocab_size);
    
    free(data);
    return tok;
}

void tokenizer_free(Tokenizer* tokenizer) {
    if (!tokenizer) return;
    
    if (tokenizer->vocab) {
        for (int i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->vocab[i]);
        }
        free(tokenizer->vocab);
    }
    free(tokenizer->vocab_ids);
    free(tokenizer->chat_template);
    free(tokenizer);
}

int tokenizer_encode(Tokenizer* tokenizer, const char* text, int** tokens_out, size_t* len_out) {
    if (!tokenizer || !text || !tokens_out || !len_out) return -1;
    
    size_t text_len = strlen(text);
    int* tokens = (int*)malloc((text_len + 100) * sizeof(int));
    if (!tokens) return -1;
    
    size_t n_tokens = 0;
    size_t pos = 0;
    
    while (pos < text_len) {
        int best_id = -1;
        int best_len = 0;
        
        for (int i = 0; i < tokenizer->vocab_size; i++) {
            const char* token_text = tokenizer->vocab[i];
            if (!token_text) continue;
            
            size_t token_len = strlen(token_text);
            if (token_len == 0 || token_len > text_len - pos) continue;
            
            if (strncmp(text + pos, token_text, token_len) == 0) {
                if (token_len > (size_t)best_len) {
                    best_len = (int)token_len;
                    best_id = tokenizer->vocab_ids[i];
                }
            }
        }
        
        if (best_id >= 0 && best_len > 0) {
            tokens[n_tokens++] = best_id;
            pos += best_len;
        } else {
            unsigned char c = (unsigned char)text[pos];
            if (c < 128) {
                char single[2] = {text[pos], '\0'};
                int id = find_token_id(tokenizer, single);
                if (id >= 0) {
                    tokens[n_tokens++] = id;
                } else {
                    tokens[n_tokens++] = (int)c;
                }
                pos++;
            } else {
                int char_len = 1;
                if ((c & 0xE0) == 0xC0) char_len = 2;
                else if ((c & 0xF0) == 0xE0) char_len = 3;
                else if ((c & 0xF8) == 0xF0) char_len = 4;
                
                char utf8_char[5] = {0};
                for (int i = 0; i < char_len && pos + i < text_len; i++) {
                    utf8_char[i] = text[pos + i];
                }
                
                int id = find_token_id(tokenizer, utf8_char);
                if (id >= 0) {
                    tokens[n_tokens++] = id;
                } else {
                    for (int i = 0; i < char_len; i++) {
                        tokens[n_tokens++] = (unsigned char)text[pos + i];
                    }
                }
                pos += char_len;
            }
        }
    }
    
    *tokens_out = tokens;
    *len_out = n_tokens;
    return 0;
}

char* tokenizer_decode(Tokenizer* tokenizer, const int* tokens, size_t len) {
    if (!tokenizer || !tokens || len == 0) return NULL;
    
    size_t buf_size = len * 64 + 1;
    char* result = (char*)malloc(buf_size);
    if (!result) return NULL;
    
    size_t pos = 0;
    for (size_t i = 0; i < len; i++) {
        const char* text = find_token_text(tokenizer, tokens[i]);
        if (text) {
            size_t text_len = strlen(text);
            if (pos + text_len < buf_size) {
                strcpy(result + pos, text);
                pos += text_len;
            }
        } else {
            if (tokens[i] >= 32 && tokens[i] < 127) {
                if (pos + 1 < buf_size) {
                    result[pos++] = (char)tokens[i];
                }
            } else if (tokens[i] < 256) {
                pos += snprintf(result + pos, buf_size - pos, "[%d]", tokens[i]);
            } else {
                pos += snprintf(result + pos, buf_size - pos, "[%d]", tokens[i]);
            }
        }
    }
    
    result[pos] = '\0';
    return result;
}

void tokenizer_free_tokens(int* tokens) {
    free(tokens);
}
