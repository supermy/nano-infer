cmake_minimum_required(VERSION 3.16)
project(qwen3_infer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(ENABLE_OPENMP "Enable OpenMP for multi-threading" ON)
option(ENABLE_BF16 "Enable bfloat16 support" ON)
option(ENABLE_SIMD "Enable SIMD optimizations" ON)
option(ENABLE_FP8 "Enable FP8 quantization support" ON)

if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    endif()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCES
    src/config.c
    src/safetensors.c
    src/tokenizer.c
    src/awq.c
    src/model.c
    src/kv_cache.c
    src/offload.c
    src/simd.c
    src/fp8.c
    src/main.c
)

add_executable(qwen3-infer ${SOURCES})

if(OpenMP_C_FOUND)
    target_link_libraries(qwen3-infer OpenMP::OpenMP_C)
endif()

target_compile_options(qwen3-infer PRIVATE -Wall -Wextra -O3)

if(ENABLE_SIMD)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        # ARM64 (Apple Silicon or Linux ARM)
        if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
            target_compile_options(qwen3-infer PRIVATE -mcpu=apple-m1)
        else()
            target_compile_options(qwen3-infer PRIVATE -march=armv8-a)
        endif()
        target_compile_definitions(qwen3-infer PRIVATE HAS_NEON=1)
    else()
        # x86_64
        target_compile_options(qwen3-infer PRIVATE -march=native -mavx -mavx2 -mfma)
        target_compile_definitions(qwen3-infer PRIVATE HAS_AVX=1 HAS_AVX2=1)
    endif()
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
            target_compile_options(qwen3-infer PRIVATE -mcpu=apple-m1)
        else()
            target_compile_options(qwen3-infer PRIVATE -march=armv8-a)
        endif()
    else()
        target_compile_options(qwen3-infer PRIVATE -march=native)
    endif()
endif()

if(ENABLE_FP8)
    target_compile_definitions(qwen3-infer PRIVATE ENABLE_FP8=1)
endif()

message(STATUS "Qwen3 Inference Engine Build Configuration")
message(STATUS "  OpenMP: ${OpenMP_C_FOUND}")
message(STATUS "  SIMD: ${ENABLE_SIMD}")
message(STATUS "  FP8: ${ENABLE_FP8}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
