use std::collections::VecDeque;

use crate::{
    config::Config,
    engine::{BlockManager, Phase, Sequence, SequenceState},
};

/// 一步调度出来的批次：本步要 forward 的序列 + 阶段。
pub struct Batch {
    pub seqs: Vec<Sequence>,
    pub phase: Phase,
}

pub struct Scheduler {
    pub max_num_seqs: usize,
    pub max_batched_tokens: usize,
    pub eos_token_id: u32,
    pub block_manager: BlockManager,
    /// 尚未 prefill（或被抢占）的请求
    pub waiting: VecDeque<Sequence>,
    /// 已经 prefill、正在逐 token 生成的请求
    pub running: VecDeque<Sequence>,
    pub num_finished: usize,
    pub num_tokens: usize,
}

impl Scheduler {
    pub fn new(config: &Config) -> Self {
        Self {
            max_num_seqs: config.max_num_seqs,
            max_batched_tokens: config.max_batched_tokens,
            eos_token_id: config.eos_token_id,
            block_manager: BlockManager::new(config.num_kvcache_blocks),
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            num_finished: 0,
            num_tokens: 0,
        }
    }

    /// 无 waiting 也 无 running
    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    /// 插入一个全新请求
    pub fn enqueue(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    /// 调度下一步：优先 prefill（清空 waiting），否则对 running 做一轮 decode。
    pub fn schedule(&mut self) -> anyhow::Result<Batch> {
        let mut seqs = Vec::new();
        let mut num_batched_tokens = 0usize;

        // prefill：从 waiting 里把能装下的请求都排进来
        while seqs.len() < self.max_num_seqs {
            // 取出最早的 waiting seq
            let Some(seq) = self.waiting.front() else { break };
            // 需要满足：token 不超过上限 && block 系统有空闲
            if num_batched_tokens + seq.len() > self.max_batched_tokens || !self.block_manager.can_reserve_prefill(seq) {
                break;
            }
            let mut seq = self.waiting.pop_front().expect("non-empty waiting");
            // 让 block_manager 给这个 seq 填充 block，有些可以复用，剩下的申请好 Block 
            self.block_manager.reserve_prefill(&mut seq)?;
            num_batched_tokens += seq.uncached_len();
            seqs.push(seq);
        }

        // 有限调度 prefill，如果有 prefill 就先处理
        if !seqs.is_empty() {
            return Ok(Batch { seqs, phase: Phase::Prefill });
        }

        // decode：轮转 running，空间不够就抢占队尾 
        while !self.running.is_empty() && seqs.len() < self.max_num_seqs {
            // 取出最早 decode 的 seq
            let mut seq = self.running.pop_front().expect("non-empty running");

            // 先抢占队尾，尽力为它腾出跨块所需的空闲块
            while !self.block_manager.can_extend_for_token(&seq) {
                // 空间不够，取出最后一个 decode 的 seq 进行释放，再次尝试，直到没有 decode 可以释放了
                let Some(victim) = self.running.pop_back() else { break };
                self.preempt(victim);
            }
            
            // 退出这个 while 有可能：1. 空间足够了；2. 释放了全部的 running 都不够用
            if self.block_manager.can_extend_for_token(&seq) {
                // 空间足够，申请申请一下
                self.block_manager.extend_for_token(&mut seq)?;
                seqs.push(seq);
            } else {
                // 连自己都腾不出来：退回 waiting，等下次再 prefill
                self.preempt(seq);
            }
        }

        Ok(Batch { seqs, phase: Phase::Decode })
    }

    /// 抢占：释放它的块、退回 waiting 队首（下次需要重新 prefill）。
    pub fn preempt(&mut self, mut seq: Sequence) {
        seq.state = SequenceState::Prefill;
        self.block_manager.release(&mut seq);
        self.waiting.push_front(seq);
    }

    /// forward 算完后，把写满的块登记进前缀缓存；prefill 的序列同时进入 Decode。
    pub fn register_computed(&mut self, batch: &mut Batch) {
        for seq in &mut batch.seqs {
            // 判断当前阶段，如果完成的是一个 prefill，计算 q 的情况比较负责，可能有多个 token 会计算
            // decode 阶段就值计算一个
            let num_computed = match batch.phase {
                Phase::Prefill => seq.uncached_len(),
                Phase::Decode => 1,
            };
            // 给这个 seq 设置又计算了 num_computed 个 token
            //调用完 register_computed，给 seq 的 cached_len 增加 num_computed，对齐了 kvcache 的状态
            self.block_manager.register_computed(seq, num_computed);
            // 如果刚才是 Prefill，下一个阶段就是 Decode
            if batch.phase == Phase::Prefill {
                seq.state = SequenceState::Decode;
            }
        }
    }

    /// 追加采样出的 token、退休结束的序列，并把存活的序列放回调度队列。
    /// 返回本步结束的 `(seq_id, completion_token_ids)`。
    pub fn append_sampled(&mut self, batch: Batch, token_ids: &[u32]) -> Vec<(usize, Vec<u32>)> {
        self.num_tokens += token_ids.len();

        let mut finished = Vec::new();
        let mut survivors: VecDeque<Sequence> = VecDeque::new();
        // 遍历每个 seq 以及新增加的 token
        for (mut seq, &token_id) in batch.seqs.into_iter().zip(token_ids) {
            // 插入 token
            seq.push_token(token_id);
            let done = (!seq.sampling_params.ignore_eos && token_id == self.eos_token_id)
                || seq.completion_len() >= seq.sampling_params.max_new_tokens;

            if done {
                seq.state = SequenceState::Finished;
                self.block_manager.release(&mut seq);
                self.num_finished += 1;
                // 保存 prompt len 后生成的 token
                finished.push((seq.id, seq.token_ids[seq.prompt_len..].to_vec()));
            } else {
                survivors.push_back(seq);
            }
        }

        match batch.phase {
            // 新 prefilled 的放队尾
            Phase::Prefill => self.running.extend(survivors),
            // round-robin：本步调度的放队首，其余保持相对顺序
            Phase::Decode => {
                let mut running = survivors;
                running.extend(self.running.drain(..));
                self.running = running;
            }
        }

        finished
    }
}
