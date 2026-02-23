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
        while (ptr < vocab_end && *ptr != '"') ptr++;
        if (ptr >= vocab_end) break;
        
        size_t key_len = ptr - key_start;
        char* token = (char*)malloc(key_len + 1);
        if (!token) break;
        
        size_t j = 0;
        for (size_t i = 0; i < key_len && j < MAX_TOKEN_TEXT - 1; i++) {
            if (key_start[i] == '\\' && i + 1 < key_len) {
                if (key_start[i+1] == 'n') {
                    token[j++] = '\n';
                    i++;
                } else if (key_start[i+1] == 't') {
                    token[j++] = '\t';
                    i++;
                } else if (key_start[i+1] == 'r') {
                    token[j++] = '\r';
                    i++;
                } else if (key_start[i+1] == '\\') {
                    token[j++] = '\\';
                    i++;
                } else if (key_start[i+1] == '"') {
                    token[j++] = '"';
                    i++;
                } else {
                    token[j++] = key_start[i];
                }
            } else {
                token[j++] = key_start[i];
            }
        }
        token[j] = '\0';
        
        ptr = strchr(ptr, ':');
        if (!ptr) break;
        ptr++;
        
        while (ptr < vocab_end && (*ptr == ' ' || *ptr == '\n' || *ptr == '\t')) ptr++;
        
        int id = atoi(ptr);
        
        tok->vocab[tok->vocab_size] = token;
        tok->vocab_ids[tok->vocab_size] = id;
        tok->vocab_size++;
        
        while (ptr < vocab_end && (*ptr != ',' && *ptr != '}')) ptr++;
        if (*ptr == '}') break;
        ptr++;
    }
}
