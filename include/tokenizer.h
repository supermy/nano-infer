#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "common.h"
#include <stdbool.h>

#define MAX_TOKEN_LENGTH 64
#define SPECIAL_TOKEN_START 151643

typedef struct {
    int id;
    char text[MAX_TOKEN_LENGTH];
    bool is_special;
} Token;

typedef struct {
    char** vocab;
    int* vocab_ids;
    int vocab_size;
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
    int eos_token_id2;
    char* chat_template;
    char* merges;
    size_t merges_size;
} Tokenizer;

Tokenizer* tokenizer_load(const char* tokenizer_dir);
void tokenizer_free(Tokenizer* tokenizer);
int tokenizer_encode(Tokenizer* tokenizer, const char* text, int** tokens_out, size_t* len_out);
char* tokenizer_decode(Tokenizer* tokenizer, const int* tokens, size_t len);
void tokenizer_free_tokens(int* tokens);

#endif
