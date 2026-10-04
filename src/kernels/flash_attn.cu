#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// Flash Attention 前向（causal mask + 可选 attention mask，支持 GQA）
// 的无异常 / C ABI 版本，供 Rust FFI 调用。
//
// 与 v2_mask.cu 的区别：不使用任何异常，所有错误通过返回码表达。
//
// 布局约定（全部要求连续内存）：
//   q, o : (batch, q_seq_len,  q_num_heads,  head_size)   f32 或 f16
//   k, v : (batch, kv_seq_len, kv_num_heads, head_size)   f32 或 f16
//   attn_mask: (batch, kv_seq_len)，1=有效 / 0=padding，可为 nullptr
//              Q 的第 i 个 token 全局位置为 start_pos + i，据此索引 mask；K 用索引 j
//   head_size 仅支持 32 / 64 / 128
//   q_num_heads 必须是 kv_num_heads 的整数倍（GQA）
//
// causal：Q 的第 i 个 token（全局位置 start_pos + i）只能看到 K 的索引 j <= start_pos + i。
//
// 错误码（返回 int，与 Rust 侧对应）：
//   0  FLASH_ATTN_OK
//   1  FLASH_ATTN_ERR_NULL_PTR          q/k/v/o 为空
//   2  FLASH_ATTN_ERR_INVALID_SHAPE     batch/seq/heads 非正，或 q_num_heads 不是 kv_num_heads 的整数倍
//   3  FLASH_ATTN_ERR_INVALID_HEAD_SIZE head_size 不是 32/64/128
//   4  FLASH_ATTN_ERR_INVALID_START_POS start_pos < 0
//   5  FLASH_ATTN_ERR_CUDA              cudaFuncSetAttribute / kernel launch 失败
//
// C ABI（Rust 侧手写声明即可）：
//   int flash_attn_f32(const float* q, const float* k, const float* v, float* o,
//                      int batch, int q_seq_len, int kv_seq_len,
//                      int q_num_heads, int kv_num_heads, int head_size,
//                      int start_pos, const uint8_t* attn_mask, void* stream) noexcept;
//   int flash_attn_f16(const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* o,
//                      ... 同上 ...) noexcept;
//   const char* flash_attn_last_error(void) noexcept;   // thread_local，出错原因
//
// stream 传 NULL 表示默认流。
// ---------------------------------------------------------------------------

enum flash_attn_status {
    FLASH_ATTN_OK = 0,
    FLASH_ATTN_ERR_NULL_PTR = 1,
    FLASH_ATTN_ERR_INVALID_SHAPE = 2,
    FLASH_ATTN_ERR_INVALID_HEAD_SIZE = 3,
    FLASH_ATTN_ERR_INVALID_START_POS = 4,
    FLASH_ATTN_ERR_CUDA = 5,
};

namespace {

thread_local char g_last_error[256] = {0};

void clear_last_error() { g_last_error[0] = '\0'; }

void set_last_error(const char* msg) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", msg);
}

constexpr int kBlockN = 64;  // Bc
constexpr int kBlockM = 64;  // Br

}  // namespace

extern "C" const char* flash_attn_last_error(void) noexcept {
    return g_last_error;
}

// ---------------------------------------------------------------------------
// kernel（与 v2_mask.cu 完全一致）
// ---------------------------------------------------------------------------
template <typename T, const int Bc, const int Br, const int H>
__global__ void flash_attn_kernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, T* __restrict__ o,
    const int q_seq_len, const int kv_seq_len, const int q_num_heads, const int kv_num_heads,
    const float d_scale, const int Tc, const uint8_t* __restrict__ attn_mask, const int start_pos
) {
    const int group_size = q_num_heads / kv_num_heads;

    const int tid         = threadIdx.x;
    const int batch_idx   = blockIdx.x;
    const int q_head_idx  = blockIdx.y;
    const int q_tile_idx  = blockIdx.z;
    const int kv_head_idx = q_head_idx / group_size;

    // shared memory：k/v 各一块 (Bc, H)
    extern __shared__ float sram[];
    float* k_block = sram;              // (Bc * H)
    float* v_block = sram + (Bc * H);   // (Bc * H)

    // 布局 (batch, seq_len, num_heads, H)
    const int q_batch_stride  = q_seq_len  * q_num_heads  * H;
    const int kv_batch_stride = kv_seq_len * kv_num_heads * H;
    const int q_head_stride   = H;
    const int kv_head_stride  = H;

    const T* q_ptr = q + batch_idx * q_batch_stride + q_head_idx * q_head_stride;
    const T* k_ptr = k + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    const T* v_ptr = v + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    T* o_ptr       = o + batch_idx * q_batch_stride + q_head_idx * q_head_stride;

    // 行（seq）方向的步长
    const int q_stride  = q_num_heads * H;
    const int kv_stride = kv_num_heads * H;

    const int q_row_idx = q_tile_idx * Br + tid;
    const int q_global_idx = start_pos + q_row_idx;

    const uint8_t* mask_row = (attn_mask != nullptr) ? (attn_mask + batch_idx * kv_seq_len) : nullptr;

    // 该行 Q 是否有效：行内 + 全局位置在 mask 范围内 + mask 为 1（无效行输出 0）
    bool q_valid = (q_row_idx < q_seq_len);
    if (q_valid && mask_row != nullptr) {
        q_valid = (q_global_idx < kv_seq_len) && (mask_row[q_global_idx] != 0);
    }

    float qi[H];
    float oi[H];

    // 加载当前 thread 负责的那一行 Q（带越界保护）
    if (q_row_idx < q_seq_len) {
        #pragma unroll
        for (int x = 0; x < H; ++x) {
            qi[x] = static_cast<float>(q_ptr[q_row_idx * q_stride + x]);
        }
    }
    #pragma unroll
    for (int x = 0; x < H; ++x) {
        oi[x] = 0.0f;
    }

    float m_prev = -CUDART_INF_F;
    float l_prev = 0.0f;

    for (int j = 0; j < Tc; ++j) {
        // causal：本 block 内 Q 的最大全局位置
        const int q_global_max = start_pos + q_tile_idx * Br + (Br - 1);
        if (j * Bc > q_global_max) {
            break;  // 之后所有 K tile 对全部行都不可见
        }

        // 加载 K/V 块 (Bc, H)：Br 个线程非对称加载
        for (int i = tid; i < Bc * H; i += Br) {
            const int row = i / H;
            const int col = i % H;
            const int row_g = j * Bc + row;
            if (row_g < kv_seq_len) {
                k_block[i] = static_cast<float>(k_ptr[row_g * kv_stride + col]);
                v_block[i] = static_cast<float>(v_ptr[row_g * kv_stride + col]);
            } else {
                k_block[i] = 0.0f;
                v_block[i] = 0.0f;
            }
        }
        __syncthreads();

        if (q_row_idx < q_seq_len) {
            float m_curr = -CUDART_INF_F;
            float s[Bc];

            // 1. S = q·Kᵀ，同时求当前 tile 的行最大值
            for (int y = 0; y < Bc; ++y) {
                const int kv_idx = j * Bc + y;

                bool visible = (kv_idx < kv_seq_len);
                if (visible && mask_row != nullptr) {
                    visible = (mask_row[kv_idx] != 0);
                }
                // causal：只能看到 kv_idx <= start_pos + q_row_idx
                if (visible) {
                    visible = (q_global_idx >= kv_idx);
                }

                float score = -CUDART_INF_F;
                if (visible) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int x = 0; x < H; ++x) {
                        sum += qi[x] * k_block[y * H + x];
                    }
                    score = sum * d_scale;
                }
                s[y] = score;
                m_curr = fmaxf(m_curr, score);
            }

            // 2. p = exp(s - m_curr)，并累加归一化分母
            float l_curr = 0.0f;
            for (int y = 0; y < Bc; ++y) {
                float p = (s[y] == -CUDART_INF_F) ? 0.0f : __expf(s[y] - m_curr);
                s[y] = p;
                l_curr += p;
            }

            // 整块都被屏蔽（m_curr 仍为 -inf）时跳过，避免 exp(-inf - -inf) = NaN
            if (m_curr != -CUDART_INF_F) {
                const float m_new = fmaxf(m_prev, m_curr);
                const float alpha = __expf(m_prev - m_new);
                const float beta  = __expf(m_curr - m_new);
                const float l_new = alpha * l_prev + beta * l_curr;

                // 3. oi = (alpha*l_prev*oi + beta*(P@V)) / l_new
                for (int x = 0; x < H; ++x) {
                    float pv = 0.0f;
                    for (int y = 0; y < Bc; ++y) {
                        pv += s[y] * v_block[y * H + x];
                    }
                    oi[x] = (alpha * l_prev * oi[x] + beta * pv) / l_new;
                }

                m_prev = m_new;
                l_prev = l_new;
            }
        }

        __syncthreads();
    }

    // 写回输出：无效 Q 行或整行全被屏蔽（l_prev == 0）时写 0
    if (q_row_idx < q_seq_len) {
        const bool write_valid = q_valid && (l_prev > 0.0f);
        #pragma unroll
        for (int x = 0; x < H; ++x) {
            o_ptr[q_row_idx * q_stride + x] = write_valid ? static_cast<T>(oi[x]) : static_cast<T>(0);
        }
    }
}

// ---------------------------------------------------------------------------
// launch（模板，Bc/Br 固定，H 由分派决定），返回错误码
// ---------------------------------------------------------------------------
template <typename T, const int Bc, const int Br, const int H>
int launch_flash_attn(
    const T* q, const T* k, const T* v, T* o,
    int batch, int q_seq_len, int kv_seq_len, int q_num_heads, int kv_num_heads,
    int start_pos, const uint8_t* attn_mask, cudaStream_t stream
) {
    const int Tr = (q_seq_len  + Br - 1) / Br;
    const int Tc = (kv_seq_len + Bc - 1) / Bc;
    const float d_scale = 1.0f / sqrtf(static_cast<float>(H));

    // dynamic shared memory：2 * Bc * H * sizeof(float)，H=128/Bc=64 时 64KB > 48KB 需显式开启
    const size_t smem = 2 * Bc * H * sizeof(float);
    if (smem > 48 * 1024) {
        cudaError_t err = cudaFuncSetAttribute(
            flash_attn_kernel<T, Bc, Br, H>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)
        );
        if (err != cudaSuccess) {
            set_last_error(cudaGetErrorString(err));
            return FLASH_ATTN_ERR_CUDA;
        }
    }

    dim3 grid(batch, q_num_heads, Tr);
    dim3 block(Br);

    flash_attn_kernel<T, Bc, Br, H><<<grid, block, smem, stream>>>(
        q, k, v, o,
        q_seq_len, kv_seq_len, q_num_heads, kv_num_heads,
        d_scale, Tc, attn_mask, start_pos
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_error(cudaGetErrorString(err));
        return FLASH_ATTN_ERR_CUDA;
    }
    return FLASH_ATTN_OK;
}

// ---------------------------------------------------------------------------
// 校验 + 分派（模板），返回错误码
// ---------------------------------------------------------------------------
template <typename T>
int flash_attn_impl(
    const T* q, const T* k, const T* v, T* o,
    int batch, int q_seq_len, int kv_seq_len, int q_num_heads, int kv_num_heads, int head_size,
    int start_pos, const uint8_t* attn_mask, cudaStream_t stream
) {
    if (q == nullptr || k == nullptr || v == nullptr || o == nullptr) {
        set_last_error("flash_attn: q/k/v/o 不能为空");
        return FLASH_ATTN_ERR_NULL_PTR;
    }
    if (batch <= 0 || q_seq_len <= 0 || kv_seq_len <= 0 || q_num_heads <= 0 || kv_num_heads <= 0) {
        set_last_error("flash_attn: 形状参数必须为正");
        return FLASH_ATTN_ERR_INVALID_SHAPE;
    }
    if (q_num_heads % kv_num_heads != 0) {
        set_last_error("flash_attn: q_num_heads 必须是 kv_num_heads 的整数倍（GQA）");
        return FLASH_ATTN_ERR_INVALID_SHAPE;
    }
    if (start_pos < 0) {
        set_last_error("flash_attn: start_pos 不能为负");
        return FLASH_ATTN_ERR_INVALID_START_POS;
    }

    switch (head_size) {
        case 32:
            return launch_flash_attn<T, kBlockN, kBlockM, 32>(q, k, v, o, batch, q_seq_len, kv_seq_len, q_num_heads, kv_num_heads, start_pos, attn_mask, stream);
        case 64:
            return launch_flash_attn<T, kBlockN, kBlockM, 64>(q, k, v, o, batch, q_seq_len, kv_seq_len, q_num_heads, kv_num_heads, start_pos, attn_mask, stream);
        case 128:
            return launch_flash_attn<T, kBlockN, kBlockM, 128>(q, k, v, o, batch, q_seq_len, kv_seq_len, q_num_heads, kv_num_heads, start_pos, attn_mask, stream);
        default:
            set_last_error("flash_attn: 不支持的 head_size（仅支持 32/64/128）");
            return FLASH_ATTN_ERR_INVALID_HEAD_SIZE;
    }
}

extern "C" int flash_attn_f32(
    const float* q, const float* k, const float* v, float* o,
    int batch, int q_seq_len, int kv_seq_len, int q_num_heads, int kv_num_heads, int head_size,
    int start_pos, const uint8_t* attn_mask, void* stream
) noexcept {
    clear_last_error();
    cudaGetLastError();  // 清掉陈旧错误，避免误判本次 launch
    return flash_attn_impl<float>(q, k, v, o, batch, q_seq_len, kv_seq_len, q_num_heads, kv_num_heads,
                                  head_size, start_pos, attn_mask, reinterpret_cast<cudaStream_t>(stream));
}

