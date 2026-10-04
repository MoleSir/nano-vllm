mod block;
mod runner;
mod scheduler;
mod sequence;
pub use block::*;
pub use runner::*;
pub use scheduler::*;
pub use sequence::*;
pub use sequence::SamplingParams;

use luma::Cuda;
use crate::{
    config::Config,
    engine::{ModelRunner, Scheduler, Sequence, SequenceStatus},
    tokenizer::{EncodeOptions, Tokenizer},
};

pub const BLOCK_SIZE: usize = 256;

pub struct LLMEngine {
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub model_runner: ModelRunner,
    pub scheduler: Scheduler,
}

impl LLMEngine {
    pub fn new(mut config: Config) -> anyhow::Result<Self> {
        anyhow::ensure!(config.kvcache_block_size % 256 == 0, "kvcache_block_size 必须是 256 的倍数");

        let tokenizer = Tokenizer::from_file(&config.model)?;
        config.eos = tokenizer.eos_token_id();

        let dev = Cuda::new(0)?;
        let model_runner = ModelRunner::new(config.clone(), dev)?;
        // `allocate_kv_cache` fills in the number of blocks on the runner's
        // config; propagate it so the scheduler's block manager is sized correctly.
        config.num_kvcache_blocks = model_runner.config.num_kvcache_blocks;
        let scheduler = Scheduler::new(&config);

        Ok(Self { config, tokenizer, model_runner, scheduler })
    }

    pub fn add_request(&mut self, prompt: &str, sampling_params: SamplingParams) -> anyhow::Result<()> {
        let encoding = self.tokenizer.encode(prompt, EncodeOptions::default())?;
        self.scheduler.add_seq(Sequence::new(encoding.get_ids().to_vec(), sampling_params));
        Ok(())
    }

    pub fn add_request_tokens(&mut self, token_ids: Vec<u32>, sampling_params: SamplingParams) {
        self.scheduler.add_seq(Sequence::new(token_ids, sampling_params));
    }

    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }

    /// One engine step: schedule -> forward -> register blocks -> append tokens.
    ///
    /// Returns `(seq_id, completion_token_ids)` for sequences finished this step.
    pub fn step(&mut self) -> anyhow::Result<Vec<(usize, Vec<u32>)>> {
        let (seqs, is_prefill) = self.scheduler.schedule()?;

        let token_ids = {
            let guards: Vec<_> = seqs.iter().map(|seq| seq.read().expect("seq read")).collect();
            let refs: Vec<&Sequence> = guards.iter().map(|guard| &**guard).collect();
            self.model_runner.run(&refs, is_prefill)?
        };

        self.scheduler.post_forward(&seqs, is_prefill);
        self.scheduler.post_process(&seqs, &token_ids);

        let outputs = seqs
            .iter()
            .filter_map(|seq| {
                let guard = seq.read().expect("seq read");
                (guard.status == SequenceStatus::Finished)
                    .then(|| (guard.id, guard.token_ids[guard.num_prompt_tokens..].to_vec()))
            })
            .collect();

        Ok(outputs)
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
