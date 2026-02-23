#include "kv_cache.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static uint64_t fnv1a_hash(const int* tokens, int len) {
    uint64_t hash = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        hash ^= (uint64_t)tokens[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t compute_token_hash(const int* tokens, int len) {
    return fnv1a_hash(tokens, len);
}

KVCache* kv_cache_create(int num_layers, int num_heads, int num_kv_heads, int head_dim, int max_seq_len) {
    KVCache* cache = (KVCache*)calloc(1, sizeof(KVCache));
    if (!cache) return NULL;
    
    cache->num_layers = num_layers;
    cache->num_heads = num_heads;
    cache->num_kv_heads = num_kv_heads;
    cache->head_dim = head_dim;
    cache->max_seq_len = max_seq_len;
    cache->current_len = 0;
    
    size_t kv_size = (size_t)num_layers * max_seq_len * num_kv_heads * head_dim * sizeof(float);
    cache->k_cache = (float*)calloc(kv_size, sizeof(float));
    cache->v_cache = (float*)calloc(kv_size, sizeof(float));
    
    if (!cache->k_cache || !cache->v_cache) {
        free(cache->k_cache);
        free(cache->v_cache);
        free(cache);
        return NULL;
    }
    
    cache->token_ids = (int*)calloc(max_seq_len, sizeof(int));
    if (!cache->token_ids) {
        free(cache->k_cache);
        free(cache->v_cache);
        free(cache);
        return NULL;
    }
    
    return cache;
}

void kv_cache_free(KVCache* cache) {
    if (!cache) return;
    free(cache->k_cache);
    free(cache->v_cache);
    free(cache->token_ids);
    free(cache);
}

void kv_cache_clear(KVCache* cache) {
    if (!cache) return;
    cache->current_len = 0;
    memset(cache->k_cache, 0, (size_t)cache->num_layers * cache->max_seq_len * 
           cache->num_kv_heads * cache->head_dim * sizeof(float));
    memset(cache->v_cache, 0, (size_t)cache->num_layers * cache->max_seq_len * 
           cache->num_kv_heads * cache->head_dim * sizeof(float));
}

int kv_cache_append(KVCache* cache, int layer_idx, int pos, const float* k, const float* v) {
    if (!cache || layer_idx < 0 || layer_idx >= cache->num_layers) return -1;
    if (pos < 0 || pos >= cache->max_seq_len) return -1;
    
    size_t offset = (size_t)layer_idx * cache->max_seq_len * cache->num_kv_heads * cache->head_dim +
                    (size_t)pos * cache->num_kv_heads * cache->head_dim;
    
    memcpy(cache->k_cache + offset, k, cache->num_kv_heads * cache->head_dim * sizeof(float));
    memcpy(cache->v_cache + offset, v, cache->num_kv_heads * cache->head_dim * sizeof(float));
    
    if (pos >= cache->current_len) {
        cache->current_len = pos + 1;
    }
    
    return 0;
}

int kv_cache_get(const KVCache* cache, int layer_idx, int pos, float* k_out, float* v_out) {
    if (!cache || layer_idx < 0 || layer_idx >= cache->num_layers) return -1;
    if (pos < 0 || pos >= cache->current_len) return -1;
    
    size_t offset = (size_t)layer_idx * cache->max_seq_len * cache->num_kv_heads * cache->head_dim +
                    (size_t)pos * cache->num_kv_heads * cache->head_dim;
    
    memcpy(k_out, cache->k_cache + offset, cache->num_kv_heads * cache->head_dim * sizeof(float));
    memcpy(v_out, cache->v_cache + offset, cache->num_kv_heads * cache->head_dim * sizeof(float));
    
    return 0;
}

int kv_cache_update_seq(KVCache* cache, const int* new_tokens, int num_tokens) {
    if (!cache || num_tokens < 0) return -1;
    if (cache->current_len + num_tokens > cache->max_seq_len) return -1;
    
    memcpy(cache->token_ids + cache->current_len, new_tokens, num_tokens * sizeof(int));
    cache->current_len += num_tokens;
    
    return 0;
}

PrefixCache* prefix_cache_create(int capacity, size_t max_memory) {
    PrefixCache* cache = (PrefixCache*)calloc(1, sizeof(PrefixCache));
    if (!cache) return NULL;
    
    cache->entries = (PrefixCacheEntry**)calloc(capacity, sizeof(PrefixCacheEntry*));
    if (!cache->entries) {
        free(cache);
        return NULL;
    }
    
    cache->capacity = capacity;
    cache->size = 0;
    cache->max_memory = max_memory;
    cache->current_memory = 0;
    cache->hash_func = fnv1a_hash;
    
    return cache;
}

void prefix_cache_free(PrefixCache* cache) {
    if (!cache) return;
    
    for (int i = 0; i < cache->size; i++) {
        PrefixCacheEntry* entry = cache->entries[i];
        if (entry) {
            free(entry->token_ids);
            for (int l = 0; l < entry->num_layers; l++) {
                free(entry->k_caches[l]);
                free(entry->v_caches[l]);
            }
            free(entry->k_caches);
            free(entry->v_caches);
            free(entry);
        }
    }
    free(cache->entries);
    free(cache);
}

PrefixCacheEntry* prefix_cache_lookup(PrefixCache* cache, const int* token_ids, int len) {
    if (!cache || !token_ids || len <= 0) return NULL;
    
    uint64_t hash = cache->hash_func(token_ids, len);
    
    for (int i = 0; i < cache->size; i++) {
        PrefixCacheEntry* entry = cache->entries[i];
        if (entry && entry->hash == hash && entry->token_len == len) {
            if (memcmp(entry->token_ids, token_ids, len * sizeof(int)) == 0) {
                entry->ref_count++;
                return entry;
            }
        }
    }
    
    return NULL;
}

PrefixCacheEntry* prefix_cache_insert(PrefixCache* cache, const int* token_ids, int len,
                                       float** k_caches, float** v_caches, int num_layers) {
    if (!cache || !token_ids || len <= 0 || !k_caches || !v_caches) return NULL;
    if (cache->size >= cache->capacity) return NULL;
    
    size_t entry_memory = (size_t)num_layers * 2 * len * sizeof(float);
    while (cache->current_memory + entry_memory > cache->max_memory && cache->size > 0) {
        prefix_cache_evict_lru(cache);
    }
    
    PrefixCacheEntry* entry = (PrefixCacheEntry*)calloc(1, sizeof(PrefixCacheEntry));
    if (!entry) return NULL;
    
    entry->hash = cache->hash_func(token_ids, len);
    entry->token_len = len;
    entry->token_ids = (int*)malloc(len * sizeof(int));
    memcpy(entry->token_ids, token_ids, len * sizeof(int));
    
    entry->k_caches = (float**)malloc(num_layers * sizeof(float*));
    entry->v_caches = (float**)malloc(num_layers * sizeof(float*));
    entry->num_layers = num_layers;
    
    for (int l = 0; l < num_layers; l++) {
        entry->k_caches[l] = (float*)malloc(len * sizeof(float));
        entry->v_caches[l] = (float*)malloc(len * sizeof(float));
        memcpy(entry->k_caches[l], k_caches[l], len * sizeof(float));
        memcpy(entry->v_caches[l], v_caches[l], len * sizeof(float));
    }
    
    entry->ref_count = 1;
    entry->memory_size = entry_memory;
    
    cache->entries[cache->size++] = entry;
    cache->current_memory += entry_memory;
    
    return entry;
}

void prefix_cache_release(PrefixCache* cache, PrefixCacheEntry* entry) {
    if (!cache || !entry) return;
    entry->ref_count--;
    if (entry->ref_count <= 0) {
        prefix_cache_evict_lru(cache);
    }
}

void prefix_cache_evict_lru(PrefixCache* cache) {
    if (!cache || cache->size == 0) return;
    
    int lru_idx = 0;
    int min_ref = cache->entries[0]->ref_count;
    
    for (int i = 1; i < cache->size; i++) {
        if (cache->entries[i]->ref_count < min_ref) {
            min_ref = cache->entries[i]->ref_count;
            lru_idx = i;
        }
    }
    
    PrefixCacheEntry* entry = cache->entries[lru_idx];
    cache->current_memory -= entry->memory_size;
    
    free(entry->token_ids);
    for (int l = 0; l < entry->num_layers; l++) {
        free(entry->k_caches[l]);
        free(entry->v_caches[l]);
    }
    free(entry->k_caches);
    free(entry->v_caches);
    free(entry);
    
    for (int i = lru_idx; i < cache->size - 1; i++) {
        cache->entries[i] = cache->entries[i + 1];
    }
    cache->size--;
}

PageMemoryPool* page_pool_create(int num_pages, int block_size, int num_layers,
                                  int num_kv_heads, int head_dim) {
    PageMemoryPool* pool = (PageMemoryPool*)calloc(1, sizeof(PageMemoryPool));
    if (!pool) return NULL;
    
    pool->num_pages = num_pages;
    pool->block_size = block_size;
    pool->num_layers = num_layers;
    pool->num_kv_heads = num_kv_heads;
    pool->head_dim = head_dim;
    
    pool->pages = (KVPage**)calloc(num_pages, sizeof(KVPage*));
    pool->free_list = (int*)malloc(num_pages * sizeof(int));
    
    if (!pool->pages || !pool->free_list) {
        free(pool->pages);
        free(pool->free_list);
        free(pool);
        return NULL;
    }
    
    size_t page_data_size = (size_t)num_layers * block_size * num_kv_heads * head_dim * sizeof(float);
    pool->total_memory = page_data_size * num_pages;
    
    for (int i = 0; i < num_pages; i++) {
        KVPage* page = (KVPage*)calloc(1, sizeof(KVPage));
        page->k_data = (float*)calloc(page_data_size, sizeof(float));
        page->v_data = (float*)calloc(page_data_size, sizeof(float));
        page->page_id = i;
        page->block_size = block_size;
        page->num_layers = num_layers;
        page->num_kv_heads = num_kv_heads;
        page->head_dim = head_dim;
        page->in_use = false;
        page->ref_count = 0;
        
        pool->pages[i] = page;
        pool->free_list[i] = i;
    }
    
    pool->free_count = num_pages;
    
    return pool;
}

void page_pool_free(PageMemoryPool* pool) {
    if (!pool) return;
    
    for (int i = 0; i < pool->num_pages; i++) {
        if (pool->pages[i]) {
            free(pool->pages[i]->k_data);
            free(pool->pages[i]->v_data);
            free(pool->pages[i]);
        }
    }
    free(pool->pages);
    free(pool->free_list);
    free(pool);
}

KVPage* page_pool_alloc(PageMemoryPool* pool) {
    if (!pool || pool->free_count == 0) return NULL;
    
    int page_id = pool->free_list[--pool->free_count];
    KVPage* page = pool->pages[page_id];
    page->in_use = true;
    page->ref_count = 1;
    
    return page;
}

void page_pool_free_page(PageMemoryPool* pool, KVPage* page) {
    if (!pool || !page || !page->in_use) return;
    
    page->ref_count--;
    if (page->ref_count <= 0) {
        page->in_use = false;
        pool->free_list[pool->free_count++] = page->page_id;
    }
}

PagedKVCache* paged_kv_cache_create(int num_pages, int block_size, int num_layers,
                                     int num_kv_heads, int head_dim, int max_sequences) {
    PagedKVCache* cache = (PagedKVCache*)calloc(1, sizeof(PagedKVCache));
    if (!cache) return NULL;
    
    cache->pool = page_pool_create(num_pages, block_size, num_layers, num_kv_heads, head_dim);
    if (!cache->pool) {
        free(cache);
        return NULL;
    }
    
    cache->block_tables = (BlockTable**)calloc(max_sequences, sizeof(BlockTable*));
    cache->seq_lens = (int*)calloc(max_sequences, sizeof(int));
    cache->max_sequences = max_sequences;
    cache->num_sequences = 0;
    
    if (!cache->block_tables || !cache->seq_lens) {
        page_pool_free(cache->pool);
        free(cache->block_tables);
        free(cache->seq_lens);
        free(cache);
        return NULL;
    }
    
    return cache;
}

void paged_kv_cache_free(PagedKVCache* cache) {
    if (!cache) return;
    
    for (int i = 0; i < cache->max_sequences; i++) {
        if (cache->block_tables[i]) {
            free(cache->block_tables[i]->page_ids);
            free(cache->block_tables[i]);
        }
    }
    page_pool_free(cache->pool);
    free(cache->block_tables);
    free(cache->seq_lens);
    free(cache);
}

int paged_kv_cache_allocate_seq(PagedKVCache* cache, int seq_id) {
    if (!cache || seq_id < 0 || seq_id >= cache->max_sequences) return -1;
    
    if (cache->block_tables[seq_id]) {
        return 0;
    }
    
    BlockTable* table = (BlockTable*)calloc(1, sizeof(BlockTable));
    table->page_ids = (int*)malloc(cache->pool->num_pages * sizeof(int));
    table->num_pages = 0;
    table->max_pages = cache->pool->num_pages;
    
    cache->block_tables[seq_id] = table;
    cache->seq_lens[seq_id] = 0;
    cache->num_sequences++;
    
    return 0;
}

void paged_kv_cache_free_seq(PagedKVCache* cache, int seq_id) {
    if (!cache || seq_id < 0 || seq_id >= cache->max_sequences) return;
    if (!cache->block_tables[seq_id]) return;
    
    BlockTable* table = cache->block_tables[seq_id];
    for (int i = 0; i < table->num_pages; i++) {
        KVPage* page = cache->pool->pages[table->page_ids[i]];
        page_pool_free_page(cache->pool, page);
    }
    
    free(table->page_ids);
    free(table);
    cache->block_tables[seq_id] = NULL;
    cache->seq_lens[seq_id] = 0;
    cache->num_sequences--;
}

int paged_kv_cache_append(PagedKVCache* cache, int seq_id, int layer_idx, int pos,
                          const float* k, const float* v) {
    if (!cache || seq_id < 0 || seq_id >= cache->max_sequences) return -1;
    
    BlockTable* table = cache->block_tables[seq_id];
    if (!table) return -1;
    
    int block_size = cache->pool->block_size;
    int page_idx = pos / block_size;
    int offset_in_page = pos % block_size;
    
    while (page_idx >= table->num_pages) {
        KVPage* page = page_pool_alloc(cache->pool);
        if (!page) return -1;
        table->page_ids[table->num_pages++] = page->page_id;
    }
    
    KVPage* page = cache->pool->pages[table->page_ids[page_idx]];
    size_t data_offset = (size_t)layer_idx * block_size * cache->pool->num_kv_heads * cache->pool->head_dim +
                         (size_t)offset_in_page * cache->pool->num_kv_heads * cache->pool->head_dim;
    
    memcpy(page->k_data + data_offset, k, cache->pool->num_kv_heads * cache->pool->head_dim * sizeof(float));
    memcpy(page->v_data + data_offset, v, cache->pool->num_kv_heads * cache->pool->head_dim * sizeof(float));
    
    cache->seq_lens[seq_id] = pos + 1;
    
    return 0;
}

int paged_kv_cache_get(const PagedKVCache* cache, int seq_id, int layer_idx, int pos,
                       float* k_out, float* v_out) {
    if (!cache || seq_id < 0 || seq_id >= cache->max_sequences) return -1;
    
    BlockTable* table = cache->block_tables[seq_id];
    if (!table) return -1;
    
    if (pos >= cache->seq_lens[seq_id]) return -1;
    
    int block_size = cache->pool->block_size;
    int page_idx = pos / block_size;
    int offset_in_page = pos % block_size;
    
    if (page_idx >= table->num_pages) return -1;
    
    KVPage* page = cache->pool->pages[table->page_ids[page_idx]];
    size_t data_offset = (size_t)layer_idx * block_size * cache->pool->num_kv_heads * cache->pool->head_dim +
                         (size_t)offset_in_page * cache->pool->num_kv_heads * cache->pool->head_dim;
    
    memcpy(k_out, page->k_data + data_offset, cache->pool->num_kv_heads * cache->pool->head_dim * sizeof(float));
    memcpy(v_out, page->v_data + data_offset, cache->pool->num_kv_heads * cache->pool->head_dim * sizeof(float));
    
    return 0;
}

static RadixNode* radix_node_create(int token_id, int depth, int num_layers,
                                     int num_kv_heads, int head_dim) {
    RadixNode* node = (RadixNode*)calloc(1, sizeof(RadixNode));
    if (!node) return NULL;
    
    node->token_id = token_id;
    node->depth = depth;
    node->ref_count = 0;
    node->num_children = 0;
    node->children_capacity = 16;
    node->children = (RadixNode**)calloc(node->children_capacity, sizeof(RadixNode*));
    node->parent = NULL;
    
    node->k_caches = (float**)calloc(num_layers, sizeof(float*));
    node->v_caches = (float**)calloc(num_layers, sizeof(float*));
    
    return node;
}

static void radix_node_free(RadixNode* node, int num_layers) {
    if (!node) return;
    
    for (int i = 0; i < node->num_children; i++) {
        radix_node_free(node->children[i], num_layers);
    }
    free(node->children);
    
    for (int l = 0; l < num_layers; l++) {
        free(node->k_caches[l]);
        free(node->v_caches[l]);
    }
    free(node->k_caches);
    free(node->v_caches);
    free(node);
}

RadixCache* radix_cache_create(int num_layers, int num_kv_heads, int head_dim,
                                int block_size, size_t max_memory) {
    RadixCache* cache = (RadixCache*)calloc(1, sizeof(RadixCache));
    if (!cache) return NULL;
    
    cache->root = radix_node_create(-1, 0, num_layers, num_kv_heads, head_dim);
    cache->num_layers = num_layers;
    cache->num_kv_heads = num_kv_heads;
    cache->head_dim = head_dim;
    cache->block_size = block_size;
    cache->total_memory = 0;
    cache->max_memory = max_memory;
    
    return cache;
}

void radix_cache_free(RadixCache* cache) {
    if (!cache) return;
    radix_node_free(cache->root, cache->num_layers);
    free(cache);
}

static RadixNode* radix_node_find_child(RadixNode* node, int token_id) {
    for (int i = 0; i < node->num_children; i++) {
        if (node->children[i]->token_id == token_id) {
            return node->children[i];
        }
    }
    return NULL;
}

static RadixNode* radix_node_add_child(RadixNode* parent, int token_id,
                                        int num_layers, int num_kv_heads, int head_dim) {
    if (parent->num_children >= parent->children_capacity) {
        parent->children_capacity *= 2;
        parent->children = (RadixNode**)realloc(parent->children,
                                                 parent->children_capacity * sizeof(RadixNode*));
    }
    
    RadixNode* child = radix_node_create(token_id, parent->depth + 1, num_layers, num_kv_heads, head_dim);
    child->parent = parent;
    parent->children[parent->num_children++] = child;
    
    return child;
}

RadixNode* radix_cache_find_or_create(RadixCache* cache, const int* token_ids, int len) {
    if (!cache || !token_ids || len <= 0) return NULL;
    
    RadixNode* current = cache->root;
    
    for (int i = 0; i < len; i++) {
        RadixNode* child = radix_node_find_child(current, token_ids[i]);
        if (!child) {
            child = radix_node_add_child(current, token_ids[i], cache->num_layers,
                                         cache->num_kv_heads, cache->head_dim);
            if (!child) return NULL;
        }
        current = child;
        current->ref_count++;
    }
    
    return current;
}

int radix_cache_match_prefix(RadixCache* cache, const int* token_ids, int len, int* matched_len) {
    if (!cache || !token_ids || len <= 0 || !matched_len) return -1;
    
    RadixNode* current = cache->root;
    *matched_len = 0;
    
    for (int i = 0; i < len; i++) {
        RadixNode* child = radix_node_find_child(current, token_ids[i]);
        if (!child) {
            break;
        }
        current = child;
        (*matched_len)++;
    }
    
    return 0;
}

void radix_cache_release_path(RadixCache* cache, const int* token_ids, int len) {
    if (!cache || !token_ids || len <= 0) return;
    
    RadixNode* current = cache->root;
    
    for (int i = 0; i < len; i++) {
        RadixNode* child = radix_node_find_child(current, token_ids[i]);
        if (!child) break;
        child->ref_count--;
        current = child;
    }
}
