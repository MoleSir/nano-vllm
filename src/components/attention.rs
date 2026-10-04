use luma::{Cuda, Float, Tensor};

use crate::components::Context;
use crate::kernels;

/// Paged attention component, mirroring `nanovllm/layers/attention.py`.
///
/// It owns the per-layer K/V cache views into the runner's big paged cache.
/// Kernels are not implemented yet (see [`crate::kernels`]).
pub struct Attention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub scale: f64,

    /// `(num_blocks, block_size, num_kv_heads, head_dim)` view.
    pub k_cache: Tensor<Cuda, Float>,
    pub v_cache: Tensor<Cuda, Float>,
}

impl Attention {
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        num_kv_heads: usize,
        scale: f64,
        k_cache: Tensor<Cuda, Float>,
        v_cache: Tensor<Cuda, Float>,
    ) -> Self {
        Self { num_heads, head_dim, num_kv_heads, scale, k_cache, v_cache }
    }

    /// `q/k/v` arrive with shape `(num_tokens, heads * head_dim)`.
    pub fn forward(
        &self,
        q: &Tensor<Cuda, Float>,
        k: &Tensor<Cuda, Float>,
        v: &Tensor<Cuda, Float>,
        ctx: &Context,
    ) -> anyhow::Result<Tensor<Cuda, Float>> {
        let num_tokens = q.dims()[0];
        let q = q.reshape((num_tokens, self.num_heads, self.head_dim))?;
        let k = k.reshape((num_tokens, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((num_tokens, self.num_kv_heads, self.head_dim))?;

        kernels::store_kvcache(&k, &v, &self.k_cache, &self.v_cache, &ctx.slot_mapping)?;

        if ctx.is_prefill {
            // With a cached prefix we read the whole paged pool, otherwise the
            // freshly computed contiguous K/V.
            if ctx.block_tables.is_some() {
                kernels::flash_attn_varlen(&q, &self.k_cache, &self.v_cache, ctx, self.scale)
            } else {
                kernels::flash_attn_varlen(&q, &k, &v, ctx, self.scale)
            }
        } else {
            kernels::flash_attn_with_kvcache(&q, &self.k_cache, &self.v_cache, ctx, self.scale)
        }
    }
}
