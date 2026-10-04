#pragma once

// Shared helpers for the paged attention kernels.
// Header-only: constants, status codes and small device helpers.
// Keep it free of any global variables so it can be included from multiple
// translation units without duplicate-symbol linker errors.

#include <cuda_runtime.h>
#include <math_constants.h>

namespace paged {

constexpr int kBlockN = 64;  // Bc: KV tile along the sequence
constexpr int kBlockM = 64;  // Br: query tile along the sequence

enum Status {
    OK = 0,
    ERR_NULL_PTR = 1,
    ERR_INVALID_SHAPE = 2,
    ERR_INVALID_HEAD_SIZE = 3,
    ERR_CUDA = 4,
};

// Logical KV position `j` of sequence `s` -> physical slot in the pooled cache.
__device__ __forceinline__ int block_slot(
    const int* __restrict__ block_tables,
    int s,
    int max_blocks,
    int j,
    int block_size
) {
    const int logical_block = j / block_size;
    const int offset = j % block_size;
    return block_tables[s * max_blocks + logical_block] * block_size + offset;
}

// Warp-wide sum reduction (32 lanes).
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

}  // namespace paged
