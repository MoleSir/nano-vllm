//! Raw C ABI declarations for the hand-written CUDA kernels.
//!
//! Implementations live in `src/kernels/*.cu` (compiled by `build.rs`). The
//! half-precision variants take `void*` because the C types `__half` /
//! `__nv_bfloat16` are layout-compatible with 16-bit buffers.

#![allow(non_camel_case_types)]
#![allow(clippy::too_many_arguments)]

use core::ffi::c_void;

unsafe extern "C" {
    /// Scatter newly computed f32 K/V into the paged cache.
    pub fn store_kvcache_f32(
        k: *const f32,
        v: *const f32,
        k_cache: *mut f32,
        v_cache: *mut f32,
        slot_mapping: *const i32,
        num_tokens: i32,
        hidden_size: i32,
    );

    /// Scatter newly computed f16 K/V into the paged cache.
    pub fn store_kvcache_f16(
        k: *const c_void,
        v: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        slot_mapping: *const i32,
        num_tokens: i32,
        hidden_size: i32,
    );

    /// Scatter newly computed bf16 K/V into the paged cache.
    pub fn store_kvcache_bf16(
        k: *const c_void,
        v: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        slot_mapping: *const i32,
        num_tokens: i32,
        hidden_size: i32,
    );
}
