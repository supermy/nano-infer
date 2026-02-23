#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    const char* model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3-4B-AWQ/model.safetensors";
    
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        printf("Failed to open file\n");
        return 1;
    }
    
    size_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    
    uint8_t* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    // Read header size (little-endian 8 bytes)
    size_t header_size = 0;
    for (int i = 7; i >= 0; i--) {
        header_size = (header_size << 8) | data[i];
    }
    
    printf("Header size: %zu\n", header_size);
    
    char* header_json = (char*)data + 8;
    
    // Find first tensor entry
    char* first_tensor = strstr(header_json, "\":{\"dtype\":");
    if (first_tensor) {
        printf("\nFirst tensor context (100 chars before and 50 after):\n");
        int start = first_tensor - header_json - 100;
        if (start < 0) start = 0;
        printf("...%.*s...\n", 150, header_json + start);
    }
    
    munmap(data, file_size);
    close(fd);
    return 0;
}
