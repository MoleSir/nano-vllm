use derive_builder::Builder;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(pattern = "owned")]
#[serde(default)]
pub struct Config {
    #[builder(default = "String::new()")]
    pub model: String,

    #[builder(default = 32768)]
    pub max_batched_tokens: usize,

    #[builder(default = 512)]
    pub max_num_seqs: usize,

    #[builder(default = 4096)]
    pub max_model_len: usize,

    #[builder(default = 0.9)]
    pub gpu_memory_utilization: f64,

    #[builder(default = false)]
    pub enforce_eager: bool,

    #[builder(default = 0)]
    pub eos_token_id: u32,

    #[builder(default = 256)]
    pub kvcache_block_size: usize,

    #[builder(default = 0)]
    pub num_kvcache_blocks: usize,
}

impl Default for Config {
    fn default() -> Self {
        ConfigBuilder::default().build().expect("Qwen2Config defaults are valid")
    }
}