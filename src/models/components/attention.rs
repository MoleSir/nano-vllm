use anyhow::Context;
use luma::tensor::IntTensor;
use luma::{Cuda, Float, Tensor};
use luma_flash_attn::{flash_attn_varlen_func, flash_attn_with_kvcache};
use crate::kernels;

pub struct PrefillState {
    /// `(num_seqs + 1,)` — cumulative query lengths (prefill).
    pub cu_seqlens_q: IntTensor<Cuda>,
    /// `(num_seqs + 1,)` — cumulative key lengths (prefill).
    pub cu_seqlens_k: IntTensor<Cuda>,

    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
}

pub struct DecodeState {
    /// `(num_seqs,)` — total KV length per sequence (used by decode reads).
    pub context_lens: IntTensor<Cuda>,
}

pub enum AttnState {
    Prefill(PrefillState),
    Decode(DecodeState),
}

pub struct AttnContext {
    /// `(num_seqs, max_blocks)` — logical block -> physical block page table.
    /// Present during prefill only when a cached prefix exists.
    pub block_tables: IntTensor<Cuda>,

    /// `(total_tokens,)` — physical slot for each newly computed token, used by
    /// the `store_kvcache` write kernel.
    pub slot_mapping: IntTensor<Cuda>,

    pub state: AttnState,
}

#[derive(Clone)]
pub struct Attention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub scale: f64,
    /// `(num_blocks, block_size, num_kv_heads, head_dim)` view.
    pub k_cache: Option<Tensor<Cuda, Float>>,
    pub v_cache: Option<Tensor<Cuda, Float>>,
}

impl Attention {
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        num_kv_heads: usize,
        scale: f64,
    ) -> Self {
        Self { num_heads, head_dim, num_kv_heads, scale, k_cache: None, v_cache: None }
    }

    /// `q/k/v` arrive with shape `(num_tokens, heads * head_dim)`.
    pub fn forward(
        &self,
        q: &Tensor<Cuda, Float>,
        k: &Tensor<Cuda, Float>,
        v: &Tensor<Cuda, Float>,
        ctx: &AttnContext,
    ) -> anyhow::Result<Tensor<Cuda, Float>> {
        let num_tokens = q.dims()[0];
        let q = q.reshape((num_tokens, self.num_heads, self.head_dim))?;
        let k = k.reshape((num_tokens, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((num_tokens, self.num_kv_heads, self.head_dim))?;

        let k_cache = self.k_cache.as_ref().context("k cache not register")?;
        let v_cache = self.v_cache.as_ref().context("k cache not register")?;

        /*
            - k/v: (num_tokens, num_kv_heads, head_size)
            - k/v_cache: (num_block, block_size, num_kv_heads, head_size)
            - slot_mapping: (num_tokens,)
            k/v 是刚计算出来的值，需要填充到 k/v_cache 里面。slot_mapping[i] 是 k/v[i] 在 k/v_cache 的位置。
        */
        kernels::store_kvcache(&k, &v, k_cache, v_cache, &ctx.slot_mapping)?;

        let scale = Some(self.scale);
        let block_tables = &ctx.block_tables;

        // 最后都输出 (num_tokens, num_q_heads, head_size)。
        let out = match &ctx.state {
            AttnState::Prefill(state) => {
                flash_attn_varlen_func(
                    &q,
                    k_cache,
                    v_cache,
                    &state.cu_seqlens_q,
                    &state.cu_seqlens_k,
                    state.max_seqlen_q,
                    state.max_seqlen_k,
                    scale,
                    true,
                    Some(block_tables),
                )?
            }
            AttnState::Decode(state) => {
                let q = q.reshape((num_tokens, 1, self.num_heads, self.head_dim))?;
                flash_attn_with_kvcache(&q, k_cache, v_cache, &state.context_lens, block_tables, scale)?
            }
        };

        Ok(out.reshape((num_tokens, self.num_heads * self.head_dim))?)
    }
}
