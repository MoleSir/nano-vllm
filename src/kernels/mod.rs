//! CUDA kernels for paged attention.
//!
//! Only the KV-cache scatter write is hand-written (`.cu` compiled by
//! `build.rs`); flash attention itself comes from the `luma-flash-attn` crate.
//!
//! NOTE: luma tensors share their storage across views and carry a
//! `Layout::start_offset()`. Every pointer below adds that offset, which is
//! required for the per-layer `k_cache`/`v_cache` views built by the runner.

mod ffi;
use core::ffi::c_void;

use luma::{Cuda, Float, FloatDType, Int, Tensor};

/// Bind an f32 tensor's raw pointer (with view offset) plus the guard keeping
/// its storage alive.
macro_rules! f32_ptr {
    ($guard:ident, $ptr:ident, $t:expr, $stream:expr) => {
        let $guard = $t.storage_read()?;
        let $ptr: *const f32 = match &$guard.slice {
            ::luma::CudaFloatSlice::F32(s) => {
                let (p, _sync) = ::cudarc::driver::DevicePtr::device_ptr(s, $stream);
                (p as *const f32).wrapping_add($t.layout().start_offset())
            }
            _ => anyhow::bail!("store_kvcache expects f32 storage"),
        };
    };
}

/// Same as [`f32_ptr!`] for f16 tensors (offset counted in 16-bit elements).
macro_rules! f16_ptr {
    ($guard:ident, $ptr:ident, $t:expr, $stream:expr) => {
        let $guard = $t.storage_read()?;
        let $ptr: *const u16 = match &$guard.slice {
            ::luma::CudaFloatSlice::F16(s) => {
                let (p, _sync) = ::cudarc::driver::DevicePtr::device_ptr(s, $stream);
                (p as *const u16).wrapping_add($t.layout().start_offset())
            }
            _ => anyhow::bail!("store_kvcache expects f16 storage"),
        };
    };
}

/// Same as [`f32_ptr!`] for bf16 tensors (offset counted in 16-bit elements).
macro_rules! bf16_ptr {
    ($guard:ident, $ptr:ident, $t:expr, $stream:expr) => {
        let $guard = $t.storage_read()?;
        let $ptr: *const u16 = match &$guard.slice {
            ::luma::CudaFloatSlice::BF16(s) => {
                let (p, _sync) = ::cudarc::driver::DevicePtr::device_ptr(s, $stream);
                (p as *const u16).wrapping_add($t.layout().start_offset())
            }
            _ => anyhow::bail!("store_kvcache expects bf16 storage"),
        };
    };
}

/// Same as [`f32_ptr!`] for i32 tensors.
macro_rules! i32_ptr {
    ($guard:ident, $ptr:ident, $t:expr, $stream:expr) => {
        let $guard = $t.storage_read()?;
        let $ptr: *const i32 = match &$guard.slice {
            ::luma::CudaIntSlice::I32(s) => {
                let (p, _sync) = ::cudarc::driver::DevicePtr::device_ptr(s, $stream);
                (p as *const i32).wrapping_add($t.layout().start_offset())
            }
            _ => anyhow::bail!("store_kvcache expects i32 storage"),
        };
    };
}

/// Scatter newly computed K/V into the paged cache at `slot_mapping`.
///
/// - `k`, `v`: `(num_tokens, num_kv_heads, head_dim)`
/// - `k_cache`, `v_cache`: `(num_blocks, block_size, num_kv_heads, head_dim)` views
/// - `slot_mapping`: `(num_tokens,)`
///
/// All four float tensors must share a dtype (f32/f16/bf16).
pub fn store_kvcache(
    k: &Tensor<Cuda, Float>,
    v: &Tensor<Cuda, Float>,
    k_cache: &Tensor<Cuda, Float>,
    v_cache: &Tensor<Cuda, Float>,
    slot_mapping: &Tensor<Cuda, Int>,
) -> anyhow::Result<()> {
    let dev = k.device();
    let stream = dev.stream();

    let num_tokens = k.dims()[0];
    let hidden_size: usize = k.dims()[1..].iter().product();

    let dtype = k.dtype();
    if v.dtype() != dtype || k_cache.dtype() != dtype || v_cache.dtype() != dtype {
        anyhow::bail!("store_kvcache: k/v/k_cache/v_cache must share one dtype, got {dtype:?}");
    }

    i32_ptr!(g_sm, p_sm, slot_mapping, &stream);

    match dtype {
        FloatDType::F32 => {
            f32_ptr!(g_k, p_k, k, &stream);
            f32_ptr!(g_v, p_v, v, &stream);
            f32_ptr!(g_kc, p_kc, k_cache, &stream);
            f32_ptr!(g_vc, p_vc, v_cache, &stream);
            unsafe {
                ffi::store_kvcache_f32(
                    p_k,
                    p_v,
                    p_kc as *mut f32,
                    p_vc as *mut f32,
                    p_sm,
                    num_tokens as i32,
                    hidden_size as i32,
                );
            }
        }
        FloatDType::F16 => {
            f16_ptr!(g_k, p_k, k, &stream);
            f16_ptr!(g_v, p_v, v, &stream);
            f16_ptr!(g_kc, p_kc, k_cache, &stream);
            f16_ptr!(g_vc, p_vc, v_cache, &stream);
            unsafe {
                ffi::store_kvcache_f16(
                    p_k as *const c_void,
                    p_v as *const c_void,
                    p_kc as *mut c_void,
                    p_vc as *mut c_void,
                    p_sm,
                    num_tokens as i32,
                    hidden_size as i32,
                );
            }
        }
        FloatDType::BF16 => {
            bf16_ptr!(g_k, p_k, k, &stream);
            bf16_ptr!(g_v, p_v, v, &stream);
            bf16_ptr!(g_kc, p_kc, k_cache, &stream);
            bf16_ptr!(g_vc, p_vc, v_cache, &stream);
            unsafe {
                ffi::store_kvcache_bf16(
                    p_k as *const c_void,
                    p_v as *const c_void,
                    p_kc as *mut c_void,
                    p_vc as *mut c_void,
                    p_sm,
                    num_tokens as i32,
                    hidden_size as i32,
                );
            }
        }
        other => anyhow::bail!("store_kvcache: unsupported dtype {other:?} (expected f32/f16/bf16)"),
    }
    Ok(())
}
