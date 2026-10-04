//! CUDA kernels for paged attention.
//!
//! The `.cu` sources are compiled by `build.rs`; this module extracts raw device
//! pointers from luma tensors and calls the C ABI in [`ffi`].
//!
//! NOTE: luma tensors share their storage across views and carry a
//! `Layout::start_offset()`. Every pointer below adds that offset, which is
//! required for the per-layer `k_cache`/`v_cache` views built by the runner.

mod ffi;
use std::ffi::c_void;
use luma::{Cuda, Float, FloatDType, Tensor};
use crate::{models::Context, engine::BLOCK_SIZE};

/// Bind an f32 tensor's raw pointer (with view offset) plus the guard keeping
/// its storage alive. `_sync` is a single-stream ordering marker and may drop.
macro_rules! f32_ptr {
    ($guard:ident, $ptr:ident, $t:expr, $stream:expr) => {
        let $guard = $t.storage_read()?;
        let $ptr: *const f32 = match &$guard.slice {
            ::luma::CudaFloatSlice::F32(s) => {
                let (p, _sync) = ::cudarc::driver::DevicePtr::device_ptr(s, $stream);
                (p as *const f32).wrapping_add($t.layout().start_offset())
            }
            ::luma::CudaFloatSlice::F64(_) => anyhow::bail!("paged kernel expects f32 storage"),
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
            _ => anyhow::bail!("paged kernel expects i32 storage"),
        };
    };
}

fn check_status(code: i32) -> anyhow::Result<()> {
    if code == ffi::PAGED_ATTN_OK {
        return Ok(());
    }
    let msg = unsafe {
        let p = ffi::paged_attn_last_error();
        if p.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    anyhow::bail!("paged attention kernel failed (code {code}): {msg}")
}

/// Scatter newly computed K/V into the paged cache at `slot_mapping`.
///
/// - `k`, `v`: `(num_tokens, num_kv_heads, head_dim)`
/// - `k_cache`, `v_cache`: `(num_blocks, block_size, num_kv_heads, head_dim)` views
/// - `slot_mapping`: `(num_tokens,)`
pub fn store_kvcache(
    k: &Tensor<Cuda, Float>,
    v: &Tensor<Cuda, Float>,
    k_cache: &Tensor<Cuda, Float>,
    v_cache: &Tensor<Cuda, Float>,
    slot_mapping: &Tensor<Cuda, luma::Int>,
) -> anyhow::Result<()> {
    let dev = k.device();
    let stream = dev.stream();

    let num_tokens = k.dims()[0];
    let hidden_size: usize = k.dims()[1..].iter().product();

    f32_ptr!(g_k, p_k, k, &stream);
    f32_ptr!(g_v, p_v, v, &stream);
    f32_ptr!(g_kc, p_kc, k_cache, &stream);
    f32_ptr!(g_vc, p_vc, v_cache, &stream);
    i32_ptr!(g_sm, p_sm, slot_mapping, &stream);

    unsafe {
        ffi::store_kvcache(
            p_k,
            p_v,
            p_kc as *mut f32,
            p_vc as *mut f32,
            p_sm,
            num_tokens as i32,
            hidden_size as i32,
        );
    }
    Ok(())
}

/// Varlen prefill attention (maps to `flash_attn_varlen_func` with an optional
/// `block_table` for prefix cache).
///
/// `q`: `(total_q, q_heads, head_dim)`.
/// `k`/`v`: contiguous `(total_kv, kv_heads, head_dim)` when there is no prefix
/// cache, otherwise the paged pool `(num_blocks, block_size, kv_heads, head_dim)`.
pub fn flash_attn_varlen(
    q: &Tensor<Cuda, Float>,
    k: &Tensor<Cuda, Float>,
    v: &Tensor<Cuda, Float>,
    ctx: &Context,
    scale: f64,
) -> anyhow::Result<Tensor<Cuda, Float>> {
    let dev = q.device();
    let stream = dev.stream();

    let total_q = q.dims()[0];
    let q_heads = q.dims()[1];
    let head_dim = q.dims()[2];

    let k_paged = k.dims().len() == 4;
    let kv_heads = if k_paged { k.dims()[2] } else { k.dims()[1] };

    let cu_q = ctx
        .cu_seqlens_q
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("flash_attn_varlen: ctx.cu_seqlens_q is missing"))?;
    let cu_k = ctx
        .cu_seqlens_k
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("flash_attn_varlen: ctx.cu_seqlens_k is missing"))?;
    let num_seqs = cu_q.dims()[0] - 1;

    let out = Tensor::<Cuda, Float>::zeros((total_q, q_heads, head_dim), (dev, FloatDType::F32))?;

    f32_ptr!(g_q, p_q, q, &stream);
    f32_ptr!(g_k, p_k, k, &stream);
    f32_ptr!(g_v, p_v, v, &stream);
    f32_ptr!(g_o, p_o, &out, &stream);
    i32_ptr!(g_cq, p_cq, cu_q, &stream);
    i32_ptr!(g_ck, p_ck, cu_k, &stream);

    let (g_bt, p_bt, max_blocks) = match &ctx.block_tables {
        Some(bt) => {
            i32_ptr!(g, p, bt, &stream);
            (Some(g), p, bt.dims()[1] as i32)
        }
        None => (None, std::ptr::null(), 0),
    };

    let code = unsafe {
        ffi::flash_attn_varlen_f32(
            p_q,
            p_k,
            p_v,
            p_o as *mut f32,
            p_cq,
            p_ck,
            p_bt,
            num_seqs as i32,
            ctx.max_seqlen_q as i32,
            ctx.max_seqlen_k as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            BLOCK_SIZE as i32,
            max_blocks,
            k_paged as i32,
            scale as f32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop(g_bt);
    drop(g_o);
    check_status(code)?;
    Ok(out)
}

/// Decode attention reading the paged KV cache (maps to `flash_attn_with_kvcache`).
///
/// `q`: `(num_seqs, q_heads, head_dim)` (one token per sequence).
pub fn flash_attn_with_kvcache(
    q: &Tensor<Cuda, Float>,
    k_cache: &Tensor<Cuda, Float>,
    v_cache: &Tensor<Cuda, Float>,
    ctx: &Context,
    scale: f64,
) -> anyhow::Result<Tensor<Cuda, Float>> {
    let dev = q.device();
    let stream = dev.stream();

    let num_seqs = q.dims()[0];
    let q_heads = q.dims()[1];
    let head_dim = q.dims()[2];
    let kv_heads = k_cache.dims()[2];

    let block_tables = ctx
        .block_tables
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("flash_attn_with_kvcache: ctx.block_tables is missing"))?;
    let context_lens = ctx
        .context_lens
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("flash_attn_with_kvcache: ctx.context_lens is missing"))?;
    let max_blocks = block_tables.dims()[1] as i32;

    let out = Tensor::<Cuda, Float>::zeros((num_seqs, q_heads, head_dim), (dev, FloatDType::F32))?;

    f32_ptr!(g_q, p_q, q, &stream);
    f32_ptr!(g_k, p_k, k_cache, &stream);
    f32_ptr!(g_v, p_v, v_cache, &stream);
    f32_ptr!(g_o, p_o, &out, &stream);
    i32_ptr!(g_bt, p_bt, block_tables, &stream);
    i32_ptr!(g_cl, p_cl, context_lens, &stream);

    let code = unsafe {
        ffi::flash_attn_with_kvcache_f32(
            p_q,
            p_k,
            p_v,
            p_o as *mut f32,
            p_bt,
            p_cl,
            num_seqs as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            BLOCK_SIZE as i32,
            max_blocks,
            scale as f32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop(g_o);
    check_status(code)?;
    Ok(out)
}
