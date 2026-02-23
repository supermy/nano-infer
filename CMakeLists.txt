cmake_minimum_required(VERSION 3.16)
project(qwen3_infer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(ENABLE_OPENMP "Enable OpenMP for multi-threading" ON)
option(ENABLE_BF16 "Enable bfloat16 support" ON)

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
    src/model.c
    src/main.c
)

add_executable(qwen3-infer ${SOURCES})

if(OpenMP_C_FOUND)
    target_link_libraries(qwen3-infer OpenMP::OpenMP_C)
endif()

target_compile_options(qwen3-infer PRIVATE -Wall -Wextra -O3)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_compile_options(qwen3-infer PRIVATE -march=armv8.5-a -mtune=apple-m1)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_compile_options(qwen3-infer PRIVATE -march=native)
endif()

message(STATUS "Qwen3 Inference Engine Build Configuration")
message(STATUS "  OpenMP: ${OpenMP_C_FOUND}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
