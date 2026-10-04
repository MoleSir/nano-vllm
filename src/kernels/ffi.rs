//! Raw C ABI declarations for the hand-written CUDA kernels.
//!
//! Implementations live in `src/kernels/*.cu` (compiled by `build.rs`).

#![allow(non_camel_case_types)]
#![allow(clippy::too_many_arguments)]

use core::ffi::{c_char, c_void};

/// Status codes shared by the paged attention kernels (`paged::Status`).
pub const PAGED_ATTN_OK: i32 = 0;

unsafe extern "C" {
    /// Scatter newly computed K/V into the paged cache.
    pub fn store_kvcache(
        k: *const f32,
        v: *const f32,
        k_cache: *mut f32,
        v_cache: *mut f32,
        slot_mapping: *const i32,
        num_tokens: i32,
        hidden_size: i32,
    );

    /// Varlen prefill attention (optional paged prefix cache).
    pub fn flash_attn_varlen_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        o: *mut f32,
        cu_seqlens_q: *const i32,
        cu_seqlens_k: *const i32,
        block_tables: *const i32,
        num_seqs: i32,
        max_seqlen_q: i32,
        max_seqlen_k: i32,
        q_heads: i32,
        kv_heads: i32,
        head_size: i32,
        block_size: i32,
        max_blocks: i32,
        k_paged: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;

    /// Decode attention reading the paged KV cache.
    pub fn flash_attn_with_kvcache_f32(
        q: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        o: *mut f32,
        block_tables: *const i32,
        context_lens: *const i32,
        num_seqs: i32,
        q_heads: i32,
        kv_heads: i32,
        head_size: i32,
        block_size: i32,
        max_blocks: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;

    /// Thread-local error string for the paged attention kernels.
    pub fn paged_attn_last_error() -> *const c_char;
}
