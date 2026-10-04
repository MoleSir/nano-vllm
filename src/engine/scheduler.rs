use std::{
    collections::VecDeque,
    sync::{Arc, RwLock},
};

use crate::{
    config::Config,
    engine::{BlockManager, Sequence, SequenceStatus},
};

pub struct Scheduler {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub eos: u32,
    pub block_manager: BlockManager,
    pub waiting: VecDeque<Arc<RwLock<Sequence>>>,
    pub running: VecDeque<Arc<RwLock<Sequence>>>,
    pub num_finished: usize,
    pub num_tokens: usize,
}

impl Scheduler {
    pub fn new(config: &Config) -> Self {
        Self {
            max_num_seqs: config.max_num_seqs,
            max_num_batched_tokens: config.max_num_batched_tokens,
            eos: config.eos,
            block_manager: BlockManager::new(config.num_kvcache_blocks),
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            num_finished: 0,
            num_tokens: 0,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    pub fn add_seq(&mut self, seq: Sequence) {
        self.waiting.push_back(Arc::new(RwLock::new(seq)));
    }

    /// Schedule the next batch. The second return value is `true` for a prefill
    /// (varlen) batch and `false` for a decode batch.
    pub fn schedule(&mut self) -> anyhow::Result<(Vec<Arc<RwLock<Sequence>>>, bool)> {
        let mut scheduled = Vec::new();
        let mut num_seqs = 0usize;
        let mut num_batched_tokens = 0usize;

        // ---- prefill: drain from waiting until a budget / block limit is hit ----
        while !self.waiting.is_empty() && num_seqs < self.max_num_seqs {
            let seq = self.waiting.front().cloned().expect("empty waiting");
            {
                let mut guard = seq.write().expect("seq write");
                if num_batched_tokens + guard.len() > self.max_num_batched_tokens
                    || !self.block_manager.can_allocate(&guard)
                {
                    break;
                }
                self.block_manager.allocate(&mut guard)?;
                num_batched_tokens += guard.len() - guard.num_cached_tokens;
                guard.status = SequenceStatus::Running;
            } // drop the lock before touching the queues

            self.waiting.pop_front();
            self.running.push_back(seq.clone());
            scheduled.push(seq);
            num_seqs += 1;
        }

        if !scheduled.is_empty() {
            return Ok((scheduled, true));
        }

        // ---- decode: round-robin, preempting from the back until there is room ----
        while !self.running.is_empty() && num_seqs < self.max_num_seqs {
            let seq = self.running.pop_front().expect("empty running");

            let mut can_schedule = true;
            while !self.block_manager.can_append(&*seq.read().expect("seq read")) {
                if let Some(victim) = self.running.pop_back() {
                    self.preempt(victim);
                } else {
                    self.preempt(seq.clone());
                    can_schedule = false;
                    break;
                }
            }
            if !can_schedule {
                continue;
            }

            {
                let mut guard = seq.write().expect("seq write");
                self.block_manager.append_tokens(&mut guard)?;
            }
            num_seqs += 1;
            scheduled.push(seq);
        }

        // Round-robin: scheduled sequences go back to the front.
        let mut running: VecDeque<_> = scheduled.iter().cloned().collect();
        running.extend(self.running.drain(..));
        self.running = running;

        Ok((scheduled, false))
    }

    /// Register the blocks computed by the model forward into the prefix cache.
    pub fn post_forward(&mut self, seqs: &[Arc<RwLock<Sequence>>], is_prefill: bool) {
        for seq in seqs {
            let mut guard = seq.write().expect("seq write");
            let num_computed = if is_prefill { guard.len() - guard.num_cached_tokens } else { 1 };
            self.block_manager.on_computed(&mut guard, num_computed);
        }
    }

    pub fn preempt(&mut self, seq: Arc<RwLock<Sequence>>) {
        {
            let mut guard = seq.write().expect("seq write");
            guard.status = SequenceStatus::Waiting;
            self.block_manager.deallocate(&mut guard);
        }
        self.waiting.push_front(seq);
    }

    /// Append the sampled token of each scheduled sequence and retire finished ones.
    pub fn post_process(&mut self, seqs: &[Arc<RwLock<Sequence>>], token_ids: &[u32]) {
        self.num_tokens += token_ids.len();
        for (seq, token_id) in seqs.iter().zip(token_ids.iter()) {
            let finished = {
                let mut guard = seq.write().expect("seq write");
                guard.append_token(*token_id);
                (!guard.sampling_params.ignore_eos && *token_id == self.eos)
                    || guard.num_completion_tokens() >= guard.sampling_params.max_tokens
            };

            if finished {
                {
                    let mut guard = seq.write().expect("seq write");
                    guard.status = SequenceStatus::Finished;
                    self.block_manager.deallocate(&mut guard);
                }
                self.running.retain(|running| !Arc::ptr_eq(running, seq));
                self.num_finished += 1;
            }
        }
    }
}
