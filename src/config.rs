use luma::FloatDType;

/// 引擎的**用户输入**配置。
///
/// 只包含调用方可调的外部参数。运行时派生值不放这里：
/// `eos_token_id` 由 `Scheduler` 持有，`num_kvcache_blocks` 由 `ModelRunner`
/// 根据显存和模型维度算出并持有。
#[derive(Debug, Clone)]
pub struct Config {
    pub max_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub gpu_memory_utilization: f64,

    /// 模型 / KV cache 的计算 dtype。`load_model` 从这里取。
    /// checkpoint 与目标 dtype 不一致时加载会自动 cast。
    pub dtype: FloatDType,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_batched_tokens: 32768,
            max_num_seqs: 512,
            gpu_memory_utilization: 0.9,
            dtype: FloatDType::F32,
        }
    }
}
