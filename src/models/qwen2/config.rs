use derive_builder::Builder;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    pub attention_factor: f64,
    pub r#type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(pattern = "owned", build_fn(validate = "Self::validate"))]
#[serde(default)]
pub struct Qwen2Config {
    #[builder(default = "1")]
    pub bos_token_id: i64,

    #[builder(default = "2")]
    pub eos_token_id: i64,

    #[builder(default = "\"silu\".to_string()")]
    pub hidden_act: String,

    #[builder(default = "512")]
    pub hidden_size: usize,

    #[builder(default)]
    pub intermediate_size: Option<usize>,

    #[builder(default = "32768")]
    pub max_position_embeddings: usize,

    #[builder(default = "8")]
    pub num_attention_heads: usize,

    #[builder(default = "8")]
    pub num_hidden_layers: usize,

    #[builder(default = "2")]
    pub num_key_value_heads: usize,

    #[builder(default = "6400")]
    pub vocab_size: usize,

    #[builder(default = "1e-6")]
    pub rms_norm_eps: f64,

    #[builder(default = "1_000_000.0")]
    pub rope_theta: f64,

    #[builder(default = "false")]
    pub inference_rope_scaling: bool,

    #[builder(default)]
    pub rope_scaling: Option<RopeScaling>,

    #[builder(default = "true")]
    pub flash_attn: bool,

    #[builder(default = "true")]
    pub attention_bias: bool,

    #[builder(default = "true")]
    pub tie_word_embeddings: bool,

    // ===== MoE =====
    #[builder(default = "false")]
    pub use_moe: bool,

    #[builder(default = "2")]
    pub num_experts_per_tok: usize,

    #[builder(default = "4")]
    pub n_routed_experts: usize,

    #[builder(default = "1")]
    pub n_shared_experts: usize,

    #[builder(default = "\"softmax\".to_string()")]
    pub scoring_func: String,

    #[builder(default = "0.01")]
    pub aux_loss_alpha: f64,

    #[builder(default = "true")]
    pub seq_aux: bool,

    #[builder(default = "true")]
    pub norm_topk_prob: bool,
}

impl Default for Qwen2Config {
    fn default() -> Self {
        Qwen2ConfigBuilder::default().build().expect("Qwen2Config defaults are valid")
    }
}

impl Qwen2Config {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn small() -> Self {
        Qwen2ConfigBuilder::default()
            .vocab_size(6400)
            .rope_theta(1e6)
            .num_hidden_layers(8)
            .hidden_size(512)
            .num_key_value_heads(2)
            .num_attention_heads(8)
            .build()
            .expect("build small")
    }
}

impl Qwen2ConfigBuilder {
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Qwen2Config {
    pub fn finalize(mut self) -> Self {
        if self.inference_rope_scaling {
            self.rope_scaling = Some(RopeScaling {
                beta_fast: 32.0,
                beta_slow: 1.0,
                factor: 16.0,
                original_max_position_embeddings: 2048,
                attention_factor: 1.0,
                r#type: "yarn".to_string(),
            });
        }
        self
    }

    pub fn intermediate_size(&self) -> usize {
        match self.intermediate_size {
            Some(intermediate_size) => intermediate_size,
            None => {
                let intermediate_size = (self.hidden_size as f64 * std::f64::consts::PI / 64.0).ceil() as usize;
                64 * intermediate_size
            }
        }
    }
}