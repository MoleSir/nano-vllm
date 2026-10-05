#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void store_kvcache_kernel(
    const T* k_ptr, const T* v_ptr, T* k_cache_ptr, T* v_cache_ptr, const int* slot_mapping_ptr,
    int num_tokens, int hidden_size
) {
    const int token_id = blockIdx.x;
    const int tid = threadIdx.x; 

    // 找到 k/v 的 token 起始地址
    const T* k = k_ptr + token_id * hidden_size;
    const T* v = v_ptr + token_id * hidden_size;

    __shared__ int slot;
    if (tid == 0) {
        slot = slot_mapping_ptr[token_id]; 
    }  
    __syncthreads();

    // 找到 k_cache/v_cache 的 token 起始地址
    T* k_cache = k_cache_ptr + slot * hidden_size;
    T* v_cache = v_cache_ptr + slot * hidden_size; 

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
template <typename T>
static void launch_store_kvcache(
    const T* k, const T* v, T* k_cache, T* v_cache, const int* slot_mapping,
    int num_tokens, int hidden_size
) {
    // 划分为 num_tokens 个 block，每个 block 负责拷贝一个 token
    const int threads = hidden_size < 1024 ? hidden_size : 1024;
    store_kvcache_kernel<T><<<num_tokens, threads>>>(k, v, k_cache, v_cache, slot_mapping, num_tokens, hidden_size);
}

extern "C" void store_kvcache_f32(
    const float* k, const float* v, float* k_cache, float* v_cache, const int* slot_mapping,
    int num_tokens, int hidden_size
) {
    launch_store_kvcache<float>(k, v, k_cache, v_cache, slot_mapping, num_tokens, hidden_size);
}

extern "C" void store_kvcache_f16(
    const __half* k, const __half* v, __half* k_cache, __half* v_cache, const int* slot_mapping,
    int num_tokens, int hidden_size
) {
    launch_store_kvcache<__half>(k, v, k_cache, v_cache, slot_mapping, num_tokens, hidden_size);
}

extern "C" void store_kvcache_bf16(
    const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, const int* slot_mapping,
    int num_tokens, int hidden_size
) {
    launch_store_kvcache<__nv_bfloat16>(k, v, k_cache, v_cache, slot_mapping, num_tokens, hidden_size);
}
