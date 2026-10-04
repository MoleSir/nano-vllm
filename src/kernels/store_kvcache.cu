#include <cuda_runtime.h>

__global__ void store_kvcache_kernel(
    const float* k_ptr, const float* v_ptr, float* k_cache_ptr, float* v_cache_ptr, const int* slot_mapping_ptr,
    int num_tokens, int hidden_size
) {
    const int token_id = blockIdx.x;
    const int tid = threadIdx.x; 

    // 找到 k/v 的 token 起始地址
    const float* k = k_ptr + token_id * hidden_size;
    const float* v = v_ptr + token_id * hidden_size;

    __shared__ int slot;
    if (tid == 0) {
        slot = slot_mapping_ptr[token_id]; 
    }  
    __syncthreads();

    // 找到 k_cache/v_cache 的 token 起始地址
    float* k_cache = k_cache_ptr + slot * hidden_size;
    float* v_cache = v_cache_ptr + slot * hidden_size; 

    // grid-stride：hidden_size 可能大于 blockDim.x（上限 1024）
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        k_cache[i] = k[i];
        v_cache[i] = v[i];
    }
}

/// @param k (num_tokens, num_heads, head_size)
/// @param v (num_tokens, num_heads, head_size)
/// @param k_cache (num_blocks, block_size, num_heads, head_size)
/// @param v_cache (num_blocks, block_size, num_heads, head_size)
/// @param slot_mapping (num_tokens)
/// @param num_tokens 
/// @param hidden_size num_heads * head_size
extern "C" void store_kvcache(
    const float* k, const float* v, float* k_cache, float* v_cache, const int* slot_mapping,
    int num_tokens, int hidden_size
) {
    // 划分为 num_tokens 个 block，每个 block 负责拷贝一个
    const int threads = hidden_size < 1024 ? hidden_size : 1024;
    store_kvcache_kernel<<<num_tokens, threads>>>(k, v, k_cache, v_cache, slot_mapping, num_tokens, hidden_size);
} 
