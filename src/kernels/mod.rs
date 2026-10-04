//! CUDA kernels for paged attention.
//!
//! All of these are intentionally `todo!()` for now; the engine and model are
//! wired against these signatures so the kernels can be dropped in later.

use luma::{Cuda, Float, Int, Tensor};

use crate::components::Context;

/// Scatter newly computed K/V into the paged cache at `slot_mapping`.
///
/// - `k`, `v`: `(num_tokens, num_kv_heads, head_dim)`
/// - `k_cache`, `v_cache`: `(num_blocks, block_size, num_kv_heads, head_dim)`
/// - `slot_mapping`: `(num_tokens,)`
pub fn store_kvcache(
    _k: &Tensor<Cuda, Float>,
    _v: &Tensor<Cuda, Float>,
    _k_cache: &Tensor<Cuda, Float>,
    _v_cache: &Tensor<Cuda, Float>,
    _slot_mapping: &Tensor<Cuda, Int>,
) -> anyhow::Result<()> {
    todo!("cuda kernel: store_kvcache")
}

/// Varlen prefill attention (maps to `flash_attn_varlen_func` with an optional
/// `block_table` for prefix cache).
pub fn flash_attn_varlen(
    _q: &Tensor<Cuda, Float>,
    _k: &Tensor<Cuda, Float>,
    _v: &Tensor<Cuda, Float>,
    _ctx: &Context,
    _scale: f64,
) -> anyhow::Result<Tensor<Cuda, Float>> {
    todo!("cuda kernel: flash_attn_varlen")
}

/// Decode attention reading the paged KV cache (maps to `flash_attn_with_kvcache`).
pub fn flash_attn_with_kvcache(
    _q: &Tensor<Cuda, Float>,
    _k_cache: &Tensor<Cuda, Float>,
    _v_cache: &Tensor<Cuda, Float>,
    _ctx: &Context,
    _scale: f64,
) -> anyhow::Result<Tensor<Cuda, Float>> {
    todo!("cuda kernel: flash_attn_with_kvcache")
}
