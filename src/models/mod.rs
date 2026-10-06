pub mod components;
pub mod qwen2;

mod sample;
pub use sample::*;

use std::path::Path;

use luma::{Cuda, FloatDType, Tensor, tensor::IntTensor};

use crate::models::components::AttnContext;

/// 分页推理的模型接口（CUDA）。
///
/// 对象安全，供 `Box<dyn ForCausalLM>` 运行时分发。构造函数不在 trait 里，
/// 由 [`load_model`] 按 `config.json` 的 `model_type` 决定具体类型。
pub trait ForCausalLM {
    fn forward(
        &self,
        input_ids: &IntTensor<Cuda>,
        positions: &IntTensor<Cuda>,
        ctx: &AttnContext,
    ) -> anyhow::Result<Tensor<Cuda>>;

    /// 把 runner 分配的大 KV cache 的每层视图注入模型。
    ///
    /// `kv_cache`: `(2, num_layers, num_blocks, block_size, num_kv_heads, head_size)`
    fn register_kvcache(&mut self, kv_cache: &Tensor<Cuda>) -> anyhow::Result<()>;

    fn num_hidden_layers(&self) -> usize;
    fn num_key_value_heads(&self) -> usize;
    fn head_size(&self) -> usize;
    fn device(&self) -> &Cuda;
}

/// 读取模型目录里的 `config.json`，按 `model_type` 构造对应模型。
pub fn load_model(dir: &Path, dtype: FloatDType, device: &Cuda) -> anyhow::Result<Box<dyn ForCausalLM>> {
    let config_path = dir.join("config.json");
    let raw = std::fs::read_to_string(&config_path).map_err(|e| anyhow::anyhow!("read {}: {e}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("parse {}: {e}", config_path.display()))?;
    let model_type = json.get("model_type").and_then(|v| v.as_str()).unwrap_or("");

    match model_type {
        "qwen2" => Ok(Box::new(qwen2::Qwen2ForCausalLM::<Cuda>::from_pretrained(dir, dtype, device)?)),
        other => anyhow::bail!("unsupported model_type `{other}` in {}", config_path.display()),
    }
}
