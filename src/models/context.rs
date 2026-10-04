use luma::{Cuda, Int, Tensor};

use crate::engine::Phase;

/// Per-step paged-attention metadata, built fresh by the model runner and passed
/// into the model's paged forward (no global state).
///
/// All shapes are flattened over the scheduled sequences of this step.
pub struct Context {
    /// Whether this batch is a prefill (varlen) or a decode (one token / seq).
    pub phase: Phase,

    /// `(total_tokens,)` — physical slot for each newly computed token, used by
    /// the `store_kvcache` write kernel.
    pub slot_mapping: Tensor<Cuda, Int>,

    /// `(num_seqs, max_blocks)` — logical block -> physical block page table.
    /// Present during prefill only when a cached prefix exists.
    pub block_tables: Option<Tensor<Cuda, Int>>,

    /// `(num_seqs,)` — total KV length per sequence (used by decode reads).
    pub context_lens: Option<Tensor<Cuda, Int>>,

    /// `(num_seqs + 1,)` — cumulative query lengths (prefill).
    pub cu_seqlens_q: Option<Tensor<Cuda, Int>>,

    /// `(num_seqs + 1,)` — cumulative key lengths (prefill).
    pub cu_seqlens_k: Option<Tensor<Cuda, Int>>,

    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
}
