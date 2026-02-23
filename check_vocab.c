#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    const char* path = "/Users/moyong/project/ai/models/Qwen/Qwen3-4B-AWQ/tokenizer.json";
    
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Failed to open\n");
        return 1;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* json = (char*)malloc(size + 1);
    fread(json, 1, size, f);
    json[size] = '\0';
    fclose(f);
    
    printf("File size: %ld bytes\n", size);
    
    // Find vocab
    const char* vocab = strstr(json, "\"vocab\":");
    if (vocab) {
        printf("Found 'vocab' at position: %ld\n", vocab - json);
        
        // Find the opening brace
        const char* brace = strchr(vocab, '{');
        if (brace) {
            printf("Found opening brace at position: %ld\n", brace - json);
            
            // Count entries
            int count = 0;
            const char* ptr = brace + 1;
            while (*ptr) {
                const char* quote = strchr(ptr, '"');
                if (!quote) break;
                
                // Check if this is a key (followed by colon)
                const char* colon = strchr(quote, ':');
                if (!colon) break;
                
                // Check if there's another quote before the colon
                const char* next_quote = strchr(quote + 1, '"');
                if (!next_quote || next_quote > colon) {
                    ptr = colon + 1;
                    continue;
                }
                
                count++;
                
                // Move to next entry
                ptr = colon + 1;
                while (*ptr && *ptr != ',' && *ptr != '}') ptr++;
                if (*ptr == '}') break;
                if (*ptr) ptr++;
            }
            
            printf("Counted %d entries in vocab\n", count);
        }
    }
    
    // Check for "Hello" token
    const char* hello = strstr(json, "\"Hello\":");
    if (hello) {
        printf("Found 'Hello' token at position: %ld\n", hello - json);
        
        // Get the ID
        const char* colon = strchr(hello, ':');
        if (colon) {
            colon++;
            while (*colon == ' ') colon++;
            int id = atoi(colon);
            printf("Hello token ID: %d\n", id);
        }
    }
    
    free(json);
    return 0;
}
