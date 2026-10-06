mod block;
mod runner;
mod scheduler;
mod sequence;
pub use block::*;
pub use runner::*;
pub use scheduler::*;
pub use sequence::*;

use std::path::Path;

use luma::Cuda;
use crate::{
    config::Config,
    models::{SamplingParams, load_model},
    tokenizer::{EncodeOptions, Tokenizer},
};

/// 每个 KV cache 物理块容纳的 token 数。
///
/// 这是**全引擎唯一的块大小来源**（编译期常量）：`Sequence` 的块划分、
/// `BlockManager` 的分配、`ModelRunner` 的 slot 计算以及 paged kernel 都基于它。
/// 不再通过 `Config` 配置；要修改请直接改这里并重新编译。
pub const BLOCK_SIZE: usize = 256;

/// 本步批次的阶段：prefill（变长打包）或 decode（每序列一个 token）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Prefill,
    Decode,
}

pub struct LLMEngine {
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub model_runner: ModelRunner,
    pub scheduler: Scheduler,
}

impl LLMEngine {
    pub fn new(model_dir: impl AsRef<Path>, config: Config) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer = Tokenizer::from_pretrained(model_dir)?;

        let dev = Cuda::new(0)?;
        let model = load_model(model_dir, config.dtype, &dev)?;
        let model_runner = ModelRunner::new(&config, model)?;
        // runner 根据显存/模型维度算出块数；scheduler 据此建 BlockManager。
        let scheduler = Scheduler::new(&config, tokenizer.eos_token_id(), model_runner.num_kvcache_blocks);

        Ok(Self { config, tokenizer, model_runner, scheduler })
    }

    pub fn add_request(&mut self, prompt: &str, sampling_params: SamplingParams) -> anyhow::Result<()> {
        let encoding = self.tokenizer.encode(prompt, EncodeOptions::default())?;
        self.scheduler.enqueue(Sequence::new(encoding.get_ids().to_vec(), sampling_params));
        Ok(())
    }

    pub fn add_token_request(&mut self, token_ids: Vec<u32>, sampling_params: SamplingParams) {
        self.scheduler.enqueue(Sequence::new(token_ids, sampling_params));
    }

    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }

    /// One engine step: schedule -> forward -> register blocks -> append tokens.
    ///
    /// Returns `(seq_id, completion_token_ids)` for sequences finished this step.
    pub fn step(&mut self) -> anyhow::Result<Vec<(usize, Vec<u32>)>> {
        // 1. 要求 scheduler 调度
        let mut batch = self.scheduler.schedule()?;
        // 2. 要求模型 forward， 每个 seq 得到一个 u32 作为下一个 token
        let token_ids = self.model_runner.forward_batch(&batch.seqs, batch.phase)?;
        // 3. cache 刚计算的 tokens
        self.scheduler.register_computed(&mut batch);
        // 4. 得到一组 id + 生成的 token id（如果有 seq 以及结束，返回 id + 所有生成的 token）
        Ok(self.scheduler.append_sampled(batch, &token_ids))
    }

    /// Run until all requests finish; returns `(seq_id, completion_token_ids)` sorted by id.
    pub fn generate(&mut self) -> anyhow::Result<Vec<(usize, Vec<u32>)>> {
        let mut outputs = Vec::new();
        while !self.is_finished() {
            outputs.extend(self.step()?);
        }
        outputs.sort_by_key(|(id, _)| *id);
        Ok(outputs)
    }
}
