# Direct Makefile build

CC = gcc
CFLAGS = -std=c11 -O3 -Wall -Wextra -Iinclude
LDFLAGS = -lm

SRC = src/config.c src/safetensors.c src/tokenizer.c src/awq.c src/model.c src/main.c
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
