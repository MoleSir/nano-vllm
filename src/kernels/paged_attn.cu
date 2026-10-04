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
    constexpr int Bc = paged::kBlockN;
    constexpr int Br = paged::kBlockM;

    const int tid = threadIdx.x;
    const int s = blockIdx.x;
    const int q_head = blockIdx.y;
    const int q_tile = blockIdx.z;

    const int q_start = cu_seqlens_q[s];
    const int q_len = cu_seqlens_q[s + 1] - q_start;
    const int kv_start = cu_seqlens_k[s];
    const int k_len = cu_seqlens_k[s + 1] - kv_start;
    const int start_pos = k_len - q_len;  // right-aligned causal

    const int row = q_tile * Br + tid;
    const int q_global = start_pos + row;

    const int group_size = q_heads / kv_heads;
    const int kv_head = q_head / group_size;
    const int D = kv_heads * H;

    const bool row_valid = (row < q_len);
    const long q_off = (long)(q_start + row) * q_heads * H + (long)q_head * H;

    extern __shared__ float sram[];
    float* k_block = sram;              // (Bc, H)
    float* v_block = sram + (Bc * H);   // (Bc, H)

    float qi[H];
    float oi[H];
    if (row_valid) {
#pragma unroll
        for (int x = 0; x < H; ++x) qi[x] = q[q_off + x];
    }
#pragma unroll
    for (int x = 0; x < H; ++x) oi[x] = 0.0f;

    float m_prev = -CUDART_INF_F;
    float l_prev = 0.0f;

    const int Tc = (k_len + Bc - 1) / Bc;
    const int q_global_max = start_pos + q_tile * Br + (Br - 1);

    for (int jt = 0; jt < Tc; ++jt) {
        // All remaining KV tiles are invisible to every row in this q tile.
        if (jt * Bc > q_global_max) break;

        // Cooperative load of K/V tile (Bc, H); out-of-range rows are zeroed.
        for (int i = tid; i < Bc * H; i += Br) {
            const int r = i / H;
            const int col = i % H;
            const int row_g = jt * Bc + r;
            if (row_g < k_len) {
                const float* kbase;
                const float* vbase;
                if (k_paged) {
                    const int slot = paged::block_slot(block_tables, s, max_blocks, row_g, block_size);
                    kbase = k + (long)slot * D + (long)kv_head * H;
                    vbase = v + (long)slot * D + (long)kv_head * H;
                } else {
                    const long off = (long)(kv_start + row_g) * D + (long)kv_head * H;
                    kbase = k + off;
                    vbase = v + off;
                }
                k_block[i] = kbase[col];
                v_block[i] = vbase[col];
            } else {
                k_block[i] = 0.0f;
                v_block[i] = 0.0f;
            }
        }
        __syncthreads();

        if (row_valid) {
            float m_curr = -CUDART_INF_F;
            float sc[Bc];

            // 1. S = q·Kᵀ * scale, with causal + range mask.
            for (int y = 0; y < Bc; ++y) {
                const int kv_idx = jt * Bc + y;
                const bool visible = (kv_idx < k_len) && (q_global >= kv_idx);
                float score = -CUDART_INF_F;
                if (visible) {
                    float sum = 0.0f;
#pragma unroll
                    for (int x = 0; x < H; ++x) sum += qi[x] * k_block[y * H + x];
                    score = sum * scale;
                }
                sc[y] = score;
                m_curr = fmaxf(m_curr, score);
            }

            // 2. p = exp(s - m_curr), running sum.
            float l_curr = 0.0f;
            for (int y = 0; y < Bc; ++y) {
                const float p = (sc[y] == -CUDART_INF_F) ? 0.0f : __expf(sc[y] - m_curr);
                sc[y] = p;
                l_curr += p;
            }

            // 3. Online rescale + accumulate P·V.
            if (m_curr != -CUDART_INF_F) {
                const float m_new = fmaxf(m_prev, m_curr);
                const float alpha = __expf(m_prev - m_new);
                const float beta = __expf(m_curr - m_new);
                const float l_new = alpha * l_prev + beta * l_curr;
                for (int x = 0; x < H; ++x) {
                    float pv = 0.0f;
                    for (int y = 0; y < Bc; ++y) pv += sc[y] * v_block[y * H + x];
                    oi[x] = (alpha * l_prev * oi[x] + beta * pv) / l_new;
                }
                m_prev = m_new;
                l_prev = l_new;
            }
        }

        __syncthreads();
    }

    if (row_valid) {
        const bool write = (l_prev > 0.0f);
#pragma unroll
        for (int x = 0; x < H; ++x) o[q_off + x] = write ? oi[x] : 0.0f;
    }
    (void)num_seqs; (void)max_seqlen_q; (void)max_seqlen_k;
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
    constexpr int ELEMS = H / 32;  // head-dim elements handled per lane

    const int s = blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= q_heads) return;
    const int q_head = warp;

    const int group_size = q_heads / kv_heads;
    const int kv_head = q_head / group_size;
    const int D = kv_heads * H;
    const int L = context_lens[s];

    const long q_off = (long)s * q_heads * H + (long)q_head * H;

    float qi[ELEMS];
    float acc[ELEMS];
#pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
        qi[e] = q[q_off + lane + e * 32];
        acc[e] = 0.0f;
    }

    float m = -CUDART_INF_F;
    float l = 0.0f;

    for (int j = 0; j < L; ++j) {
        const int slot = paged::block_slot(block_tables, s, max_blocks, j, block_size);
        const float* kbase = k_cache + (long)slot * D + (long)kv_head * H;
        const float* vbase = v_cache + (long)slot * D + (long)kv_head * H;

        float partial = 0.0f;
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) partial += qi[e] * kbase[lane + e * 32];
        const float dot = paged::warp_reduce_sum(partial);
        const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;

        const float m_new = fmaxf(m, score);
        const float alpha = __expf(m - m_new);
        const float p = __expf(score - m_new);
        l = l * alpha + p;
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) acc[e] = acc[e] * alpha + p * vbase[lane + e * 32];
        m = m_new;
    }

    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll
    for (int e = 0; e < ELEMS; ++e) o[q_off + lane + e * 32] = acc[e] * inv;

    (void)num_seqs;
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
