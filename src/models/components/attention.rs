use luma::{Cuda, Float, Tensor};
use crate::engine::Phase;
use crate::models::Context;
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
        
        /*
            - k/v: (num_tokens, num_kv_heads, head_size)
            - k/v_cache: (num_block, block_size, num_kv_heads, head_size)
            - slot_mapping: (num_tokens,)
            k/v 是刚计算出来的值，需要填充到 k/v_cache 里面。slot_mapping[i] 是 k/v[i] 在 k/v_cache 的位置
        */
        kernels::store_kvcache(&k, &v, &self.k_cache, &self.v_cache, &ctx.slot_mapping)?;

        // 最后都输出：(num_tokens, num_q_heads, head_size) -> (num_tokens, num_q_heads, head_size) 都 q 的 shape 一致
        match ctx.phase {
            // 完成了 prefill 阶段，并且根据 block_tables 判断是否有前缀缓存
            Phase::Prefill => if ctx.block_tables.is_some() {
                // 有前缀缓存，直接传入 kv cache，
                kernels::flash_attn_varlen(&q, &self.k_cache, &self.v_cache, ctx, self.scale)
            } else {
                // 没有前缀缓存，不需要用到 kv cache
                kernels::flash_attn_varlen(&q, &k, &v, ctx, self.scale)
            }
            // Decode 阶段
            Phase::Decode => kernels::flash_attn_with_kvcache(&q, &self.k_cache, &self.v_cache, ctx, self.scale),
        }
    }
}
