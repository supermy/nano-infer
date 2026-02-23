#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"

int main() {
    const char* model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3-4B-AWQ";
    
    Tokenizer* tok = tokenizer_load(model_path);
    if (!tok) {
        printf("Failed to load tokenizer\n");
        return 1;
    }
    
    printf("=== Token Decoding Test ===\n\n");
    
    // Generated tokens
    int tokens[] = {89935, 3834, 82048, 40971, 1841};
    int n_tokens = 5;
    
    for (int i = 0; i < n_tokens; i++) {
        char* decoded = tokenizer_decode(tok, &tokens[i], 1);
        printf("Token %d: %d -> '%s'\n", i+1, tokens[i], decoded ? decoded : "(null)");
        if (decoded) free(decoded);
    }
    
    // Try decoding all together
    printf("\nDecoding all tokens together:\n");
    char* result = tokenizer_decode(tok, tokens, n_tokens);
    printf("Result: '%s'\n", result ? result : "(null)");
    if (result) free(result);
    
    // Test encoding "Hello"
    printf("\n=== Token Encoding Test ===\n");
    const char* test_str = "Hello";
    int* encoded = NULL;
    size_t n_encoded = 0;
    tokenizer_encode(tok, test_str, &encoded, &n_encoded);
    printf("Encoding '%s': %zu tokens\n", test_str, n_encoded);
    for (size_t i = 0; i < n_encoded; i++) {
        printf("  token[%zu] = %d\n", i, encoded[i]);
    }
    if (encoded) tokenizer_free_tokens(encoded);
    
    // Check vocab size
    printf("\nVocab size: %d\n", tok->vocab_size);
    
    // Check if specific tokens exist
    printf("\nChecking first 20 tokens:\n");
    for (int t = 0; t < tok->vocab_size && t < 20; t++) {
        if (tok->vocab[t]) {
            printf("  token[%d] = '%s'\n", t, tok->vocab[t]);
        }
    }
    
    tokenizer_free(tok);
    return 0;
}
