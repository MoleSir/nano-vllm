use derive_builder::Builder;
use serde::{Deserialize, Serialize};

/// Qwen3 的 HuggingFace `config.json`。
///
/// 与 Qwen2 的关键区别：`head_dim` 是显式字段，**不等于** `hidden_size /
/// num_attention_heads`（例如 0.6B：hidden 1024 / 16 heads = 64，但 head_dim = 128）。
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(pattern = "owned", build_fn(validate = "Self::validate"))]
#[serde(default)]
pub struct Qwen3Config {
    #[builder(default = "151643")]
    pub bos_token_id: i64,

    #[builder(default = "151645")]
    pub eos_token_id: i64,

    #[builder(default = "\"silu\".to_string()")]
    pub hidden_act: String,

    #[builder(default = "1024")]
    pub hidden_size: usize,

    #[builder(default = "3072")]
    pub intermediate_size: usize,

    #[builder(default = "40960")]
    pub max_position_embeddings: usize,

    #[builder(default = "16")]
    pub num_attention_heads: usize,

    #[builder(default = "28")]
    pub num_hidden_layers: usize,

    #[builder(default = "8")]
    pub num_key_value_heads: usize,

    /// 每个 attention head 的维度（Qwen3 显式配置）。
    #[builder(default = "128")]
    pub head_dim: usize,

    #[builder(default = "151936")]
    pub vocab_size: usize,

    #[builder(default = "1e-6")]
    pub rms_norm_eps: f64,

    #[builder(default = "1_000_000.0")]
    pub rope_theta: f64,

    #[builder(default = "false")]
    pub attention_bias: bool,

    #[builder(default = "true")]
    pub tie_word_embeddings: bool,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Qwen3ConfigBuilder::default().build().expect("Qwen3Config defaults are valid")
    }
}

impl Qwen3Config {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn head_size(&self) -> usize {
        self.head_dim
    }
}

impl Qwen3ConfigBuilder {
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}
