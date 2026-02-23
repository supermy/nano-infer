#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    float* k_cache;
    float* v_cache;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    int current_len;
    int* token_ids;
} KVCache;

typedef struct {
    uint64_t hash;
    int* token_ids;
    int token_len;
    float** k_caches;
    float** v_caches;
    int num_layers;
    int ref_count;
    size_t memory_size;
} PrefixCacheEntry;

typedef struct {
    PrefixCacheEntry** entries;
    int capacity;
    int size;
    size_t max_memory;
    size_t current_memory;
    uint64_t (*hash_func)(const int*, int);
} PrefixCache;

typedef struct KVPage {
    float* k_data;
    float* v_data;
    int page_id;
    int block_size;
    int num_layers;
    int num_kv_heads;
    int head_dim;
    bool in_use;
    int ref_count;
    struct KVPage* next;
} KVPage;

typedef struct {
    KVPage** pages;
    int num_pages;
    int block_size;
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int* free_list;
    int free_count;
    size_t total_memory;
} PageMemoryPool;

typedef struct {
    int* page_ids;
    int num_pages;
    int max_pages;
    int slot_mapping;
} BlockTable;

typedef struct {
    PageMemoryPool* pool;
    BlockTable** block_tables;
    int num_sequences;
    int max_sequences;
    int* seq_lens;
} PagedKVCache;

typedef struct RadixNode {
    int token_id;
    int depth;
    int ref_count;
    float** k_caches;
    float** v_caches;
    struct RadixNode** children;
    int num_children;
    int children_capacity;
    struct RadixNode* parent;
} RadixNode;

typedef struct {
    RadixNode* root;
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int block_size;
    size_t total_memory;
    size_t max_memory;
} RadixCache;

KVCache* kv_cache_create(int num_layers, int num_heads, int num_kv_heads, int head_dim, int max_seq_len);
void kv_cache_free(KVCache* cache);
void kv_cache_clear(KVCache* cache);
int kv_cache_append(KVCache* cache, int layer_idx, int pos, const float* k, const float* v);
int kv_cache_get(const KVCache* cache, int layer_idx, int pos, float* k_out, float* v_out);
int kv_cache_update_seq(KVCache* cache, const int* new_tokens, int num_tokens);

PrefixCache* prefix_cache_create(int capacity, size_t max_memory);
void prefix_cache_free(PrefixCache* cache);
PrefixCacheEntry* prefix_cache_lookup(PrefixCache* cache, const int* token_ids, int len);
PrefixCacheEntry* prefix_cache_insert(PrefixCache* cache, const int* token_ids, int len,
                                       float** k_caches, float** v_caches, int num_layers);
void prefix_cache_release(PrefixCache* cache, PrefixCacheEntry* entry);
void prefix_cache_evict_lru(PrefixCache* cache);

PageMemoryPool* page_pool_create(int num_pages, int block_size, int num_layers, int num_kv_heads, int head_dim);
void page_pool_free(PageMemoryPool* pool);
KVPage* page_pool_alloc(PageMemoryPool* pool);
void page_pool_free_page(PageMemoryPool* pool, KVPage* page);

PagedKVCache* paged_kv_cache_create(int num_pages, int block_size, int num_layers,
                                     int num_kv_heads, int head_dim, int max_sequences);
void paged_kv_cache_free(PagedKVCache* cache);
int paged_kv_cache_allocate_seq(PagedKVCache* cache, int seq_id);
void paged_kv_cache_free_seq(PagedKVCache* cache, int seq_id);
int paged_kv_cache_append(PagedKVCache* cache, int seq_id, int layer_idx, int pos,
                          const float* k, const float* v);
int paged_kv_cache_get(const PagedKVCache* cache, int seq_id, int layer_idx, int pos,
                       float* k_out, float* v_out);

RadixCache* radix_cache_create(int num_layers, int num_kv_heads, int head_dim, int block_size, size_t max_memory);
void radix_cache_free(RadixCache* cache);
RadixNode* radix_cache_find_or_create(RadixCache* cache, const int* token_ids, int len);
int radix_cache_match_prefix(RadixCache* cache, const int* token_ids, int len, int* matched_len);
void radix_cache_release_path(RadixCache* cache, const int* token_ids, int len);

uint64_t compute_token_hash(const int* tokens, int len);

#endif
