// Paged attention kernels (prefill + decode).
//
// This file is the *scaffolding*: argument validation, launch-geometry and the
// C ABI are done. The `__global__` kernel bodies are intentionally left as
// `// TODO(user): implement`. See `paged_attn_common.cuh` for shared helpers
// and `../kernels/README.md` for the exact semantics/shapes.
//
// Layouts (all contiguous):
//   q   (total_q,   q_heads,  head_size)       f32
//   o   (total_q,   q_heads,  head_size)       f32
//   k/v contiguous: (total_kv, kv_heads, head_size)
//   k/v paged:      (num_blocks, block_size, kv_heads, head_size)
//   block_tables:   (num_seqs, max_blocks)     i32   (-1 padding, never read)
//   cu_seqlens_q/k: (num_seqs + 1,)            i32
//   context_lens:   (num_seqs,)                i32
//
// Return codes: see `paged::Status` in the header. Host-side getter:
//   const char* paged_attn_last_error();

#include "paged_attn_common.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

thread_local char g_last_error[256] = {0};

void clear_last_error() { g_last_error[0] = '\0'; }

void set_last_error(const char* msg) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", msg);
}

// Sets MaxDynamicSharedMemorySize when needed; returns false on failure.
template <typename Kernel>
bool ensure_smem(Kernel kernel, size_t smem) {
    if (smem <= 48 * 1024) {
        return true;
    }
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)
    );
    if (err != cudaSuccess) {
        set_last_error(cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool check_shape_common(int num_seqs, int q_heads, int kv_heads, int head_size) {
    if (num_seqs <= 0 || q_heads <= 0 || kv_heads <= 0) {
        set_last_error("paged_attn: num_seqs/q_heads/kv_heads must be > 0");
        return false;
    }
    if (q_heads % kv_heads != 0) {
        set_last_error("paged_attn: q_heads must be a multiple of kv_heads (GQA)");
        return false;
    }
    if (head_size != 32 && head_size != 64 && head_size != 128) {
        set_last_error("paged_attn: head_size must be 32/64/128");
        return false;
    }
    return true;
}

// ===========================================================================
// Prefill: varlen attention over packed sequences, optionally reading a paged
// prefix cache.
// ===========================================================================

template <const int H>
__global__ void flash_attn_varlen_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_k,
    const int* __restrict__ block_tables,  // may be nullptr (no prefix cache)
    int num_seqs,
    int max_seqlen_q,
    int max_seqlen_k,
    int q_heads,
    int kv_heads,
    int block_size,
    int max_blocks,
    int k_paged,
    float scale
) {
    // TODO(user): implement.
    //
    // Suggested grid = (num_seqs, q_heads, ceil(max_seqlen_q / Br)),
    //          block = (Br), dynamic smem = 2 * Bc * H * sizeof(float).
    //
    // Per block:
    //   s = blockIdx.x; q_head = blockIdx.y; tile = blockIdx.z; row = tile*Br + tid;
    //   q_start = cu_seqlens_q[s]; q_len = cu_seqlens_q[s+1] - q_start;
    //   k_start = cu_seqlens_k[s]; k_len = cu_seqlens_k[s+1] - k_start;
    //   start_pos = k_len - q_len;                 // right-aligned causal
    //   q_global  = start_pos + row;
    //   kv_head   = q_head / (q_heads / kv_heads);
    //
    // Loop KV tiles (j = jt*Bc ..):
    //   if (jt * Bc > start_pos + tile*Br + (Br-1)) break;
    //   load K/V tile; for row_g = jt*Bc + r < k_len:
    //       if (k_paged) idx = block_slot(block_tables, s, max_blocks, row_g, block_size)
    //                    else idx = k_start + row_g            (contiguous)
    //       k = k_pool_or_contig + idx * kv_heads*H + kv_head*H
    //   score = dot(q_row, k_row) * scale  if (row_g < k_len && q_global >= row_g) else -inf
    //   online softmax (see README section 2).
    //
    // Write o at ((q_start + row) * q_heads + q_head) * H.
    (void)q; (void)k; (void)v; (void)o;
    (void)cu_seqlens_q; (void)cu_seqlens_k; (void)block_tables;
    (void)num_seqs; (void)max_seqlen_q; (void)max_seqlen_k;
    (void)q_heads; (void)kv_heads; (void)block_size; (void)max_blocks;
    (void)k_paged; (void)scale;
}

template <const int H>
int launch_varlen(
    const float* q, const float* k, const float* v, float* o,
    const int* cu_seqlens_q, const int* cu_seqlens_k, const int* block_tables,
    int num_seqs, int max_seqlen_q, int max_seqlen_k,
    int q_heads, int kv_heads, int block_size, int max_blocks, int k_paged,
    float scale, cudaStream_t stream
) {
    constexpr int Bc = paged::kBlockN;
    constexpr int Br = paged::kBlockM;
    const size_t smem = 2 * Bc * H * sizeof(float);

    if (!ensure_smem(flash_attn_varlen_kernel<H>, smem)) {
        return paged::ERR_CUDA;
    }

    dim3 grid(num_seqs, q_heads, (max_seqlen_q + Br - 1) / Br);
    dim3 block(Br);
    flash_attn_varlen_kernel<H><<<grid, block, smem, stream>>>(
        q, k, v, o, cu_seqlens_q, cu_seqlens_k, block_tables,
        num_seqs, max_seqlen_q, max_seqlen_k, q_heads, kv_heads,
        block_size, max_blocks, k_paged, scale
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_error(cudaGetErrorString(err));
        return paged::ERR_CUDA;
    }
    return paged::OK;
}

// ===========================================================================
// Decode: one query token per sequence, reading the whole paged KV cache.
// ===========================================================================

template <const int H>
__global__ void flash_attn_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ o,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int num_seqs,
    int q_heads,
    int kv_heads,
    int block_size,
    int max_blocks,
    float scale
) {
    // TODO(user): implement.
    //
    // Suggested grid = (num_seqs, q_heads), block = (32)  -> one warp per head.
    //   lane splits head_size (H / 32 elements each).
    //   L  = context_lens[s];  kv_head = q_head / (q_heads / kv_heads);
    //   loop j in [0, L):
    //       idx = block_slot(block_tables, s, max_blocks, j, block_size);
    //       k = k_cache + idx * kv_heads*H + kv_head*H;
    //       v = v_cache + idx * kv_heads*H + kv_head*H;
    //       dot = warp_reduce_sum(sum_lane q_l * k_l);  score = dot * scale;
    //       online softmax update acc/l/m.
    //   o[s, q_head, :] = acc / l.
    (void)q; (void)k_cache; (void)v_cache; (void)o;
    (void)block_tables; (void)context_lens;
    (void)num_seqs; (void)q_heads; (void)kv_heads;
    (void)block_size; (void)max_blocks; (void)scale;
}

template <const int H>
int launch_decode(
    const float* q, const float* k_cache, const float* v_cache, float* o,
    const int* block_tables, const int* context_lens,
    int num_seqs, int q_heads, int kv_heads, int block_size, int max_blocks,
    float scale, cudaStream_t stream
) {
    dim3 grid(num_seqs);
    dim3 block(q_heads * 32);
    flash_attn_decode_kernel<H><<<grid, block, 0, stream>>>(
        q, k_cache, v_cache, o, block_tables, context_lens,
        num_seqs, q_heads, kv_heads, block_size, max_blocks, scale
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_error(cudaGetErrorString(err));
        return paged::ERR_CUDA;
    }
    return paged::OK;
}

}  // namespace

extern "C" const char* paged_attn_last_error(void) noexcept {
    return g_last_error;
}

extern "C" int flash_attn_varlen_f32(
    const float* q, const float* k, const float* v, float* o,
    const int* cu_seqlens_q, const int* cu_seqlens_k, const int* block_tables,
    int num_seqs, int max_seqlen_q, int max_seqlen_k,
    int q_heads, int kv_heads, int head_size,
    int block_size, int max_blocks, int k_paged,
    float scale, void* stream
) noexcept {
    clear_last_error();
    cudaGetLastError();

    if (q == nullptr || o == nullptr || cu_seqlens_q == nullptr || cu_seqlens_k == nullptr) {
        set_last_error("flash_attn_varlen: q/o/cu_seqlens must not be null");
        return paged::ERR_NULL_PTR;
    }
    if ((k_paged && (k == nullptr || v == nullptr || block_tables == nullptr)) ||
        (!k_paged && (k == nullptr || v == nullptr))) {
        set_last_error("flash_attn_varlen: k/v (and block_tables when paged) must not be null");
        return paged::ERR_NULL_PTR;
    }
    if (!check_shape_common(num_seqs, q_heads, kv_heads, head_size)) {
        return paged::ERR_INVALID_SHAPE;
    }
    if (max_seqlen_q <= 0 || max_seqlen_k <= 0 || block_size <= 0) {
        set_last_error("flash_attn_varlen: max_seqlen_q/k and block_size must be > 0");
        return paged::ERR_INVALID_SHAPE;
    }

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (head_size) {
        case 32:
            return launch_varlen<32>(q, k, v, o, cu_seqlens_q, cu_seqlens_k, block_tables,
                                     num_seqs, max_seqlen_q, max_seqlen_k, q_heads, kv_heads,
                                     block_size, max_blocks, k_paged, scale, s);
        case 64:
            return launch_varlen<64>(q, k, v, o, cu_seqlens_q, cu_seqlens_k, block_tables,
                                     num_seqs, max_seqlen_q, max_seqlen_k, q_heads, kv_heads,
                                     block_size, max_blocks, k_paged, scale, s);
        case 128:
            return launch_varlen<128>(q, k, v, o, cu_seqlens_q, cu_seqlens_k, block_tables,
                                      num_seqs, max_seqlen_q, max_seqlen_k, q_heads, kv_heads,
                                      block_size, max_blocks, k_paged, scale, s);
        default:
            return paged::ERR_INVALID_HEAD_SIZE;
    }
}

extern "C" int flash_attn_with_kvcache_f32(
    const float* q, const float* k_cache, const float* v_cache, float* o,
    const int* block_tables, const int* context_lens,
    int num_seqs, int q_heads, int kv_heads, int head_size,
    int block_size, int max_blocks,
    float scale, void* stream
) noexcept {
    clear_last_error();
    cudaGetLastError();

    if (q == nullptr || k_cache == nullptr || v_cache == nullptr || o == nullptr ||
        block_tables == nullptr || context_lens == nullptr) {
        set_last_error("flash_attn_with_kvcache: null pointer argument");
        return paged::ERR_NULL_PTR;
    }
    if (!check_shape_common(num_seqs, q_heads, kv_heads, head_size)) {
        return paged::ERR_INVALID_SHAPE;
    }
    if (block_size <= 0) {
        set_last_error("flash_attn_with_kvcache: block_size must be > 0");
        return paged::ERR_INVALID_SHAPE;
    }

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (head_size) {
        case 32:
            return launch_decode<32>(q, k_cache, v_cache, o, block_tables, context_lens,
                                     num_seqs, q_heads, kv_heads, block_size, max_blocks, scale, s);
        case 64:
            return launch_decode<64>(q, k_cache, v_cache, o, block_tables, context_lens,
                                     num_seqs, q_heads, kv_heads, block_size, max_blocks, scale, s);
        case 128:
            return launch_decode<128>(q, k_cache, v_cache, o, block_tables, context_lens,
                                      num_seqs, q_heads, kv_heads, block_size, max_blocks, scale, s);
        default:
            return paged::ERR_INVALID_HEAD_SIZE;
    }
}
