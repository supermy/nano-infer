# Direct Makefile build

CC = gcc
CFLAGS = -std=c11 -O3 -Wall -Wextra -Iinclude
LDFLAGS = -lm

# Architecture and SIMD detection
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_M),arm64)
    # Apple Silicon
    CFLAGS += -mcpu=apple-m1
    SIMD_FLAGS = -DHAS_NEON=1
else ifeq ($(UNAME_M),aarch64)
    # ARM64 Linux
    CFLAGS += -march=armv8-a
    SIMD_FLAGS = -DHAS_NEON=1
else ifeq ($(UNAME_S),Darwin)
    # macOS x86 (Rosetta)
    CFLAGS += -march=native
    SIMD_FLAGS = -mavx -mavx2 -mfma -DHAS_AVX=1 -DHAS_AVX2=1
else
    # x86 Linux
    CFLAGS += -march=native
    SIMD_FLAGS = -mavx -mavx2 -mfma -DHAS_AVX=1 -DHAS_AVX2=1
endif

CFLAGS += $(SIMD_FLAGS)

SRC = src/config.c src/safetensors.c src/tokenizer.c src/awq.c src/model.c \
      src/kv_cache.c src/offload.c src/simd.c src/fp8.c src/main.c
OBJ = $(SRC:.c=.o)
TARGET = qwen3-infer

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
