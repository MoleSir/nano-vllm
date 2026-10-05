use anyhow::Context as _;
use luma::{Cuda, Float, Tensor};
use luma_flash_attn::{flash_attn_varlen_func, flash_attn_with_kvcache};

use crate::engine::Phase;
use crate::kernels;
use crate::models::Context;

/// Paged attention component, mirroring `nanovllm/layers/attention.py`.
///
/// It owns the per-layer K/V cache views into the runner's big paged cache and
/// runs the two flash-attention entry points from `luma-flash-attn`:
/// `flash_attn_varlen_func` for prefill and `flash_attn_with_kvcache` for
/// decode.
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
            k/v 是刚计算出来的值，需要填充到 k/v_cache 里面。slot_mapping[i] 是 k/v[i] 在 k/v_cache 的位置。
        */
        kernels::store_kvcache(&k, &v, &self.k_cache, &self.v_cache, &ctx.slot_mapping)?;

        let scale = Some(self.scale);
        let block_tables = ctx.block_tables.as_ref().context("attention missing block_tables")?;

        // 最后都输出 (num_tokens, num_q_heads, head_size)。
        let out = match ctx.phase {
            // Prefill：变长打包，统一从分页 cache 读取（新 K/V 已写入）。
            Phase::Prefill => {
                let cu_seqlens_q = ctx.cu_seqlens_q.as_ref().context("prefill missing cu_seqlens_q")?;
                let cu_seqlens_k = ctx.cu_seqlens_k.as_ref().context("prefill missing cu_seqlens_k")?;
                flash_attn_varlen_func(
                    &q,
                    &self.k_cache,
                    &self.v_cache,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    ctx.max_seqlen_q,
                    ctx.max_seqlen_k,
                    scale,
                    true,
                    Some(block_tables),
                )?
            }
            // Decode：每序列一个 token。
            Phase::Decode => {
                let context_lens = ctx.context_lens.as_ref().context("decode missing context_lens")?;
                let q = q.reshape((num_tokens, 1, self.num_heads, self.head_dim))?;
                flash_attn_with_kvcache(&q, &self.k_cache, &self.v_cache, context_lens, block_tables, scale)?
            }
        };

        Ok(out.reshape((num_tokens, self.num_heads * self.head_dim))?)
    }
}
