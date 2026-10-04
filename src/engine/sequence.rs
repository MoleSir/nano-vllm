use std::sync::atomic::{AtomicUsize, Ordering};

use super::BLOCK_SIZE;
use crate::models::SamplingParams;

/// 一条序列在其生命周期中的阶段（决定它下一步需要 prefill 还是 decode）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    /// prompt（或前缀）尚未进入 KV，下一次被调度时需要一次 prefill。
    /// 新请求、以及被抢占回退的请求都处于这个状态。
    Prefill,
    /// prompt 已算完，正在逐 token 生成。
    Decode,
    /// 生成结束。
    Finished,
}

/// 表示一个完整请求
pub struct Sequence {
    pub id: usize,
    pub state: SequenceState,
    /// 当前的所有 token ids，请求刚到就是 prompt，后面逐个 token 生成
    pub token_ids: Vec<u32>,
    /// prompt 长度（生成起点）
    pub prompt_len: usize,
    /// 已经写入 KV 的 token 数（`block_table` 覆盖的前缀长度）
    pub cached_len: usize,
    /// "页表"，当前 token ids 按照 Block 划分，block_table[i] 表示第 i 个 token ids 块的 kv cache 保存到的是哪个 Block 对象
    /// 可以理解为 seq 是一个进程，j = block_table[i] 中 i 是虚拟地址，逻辑连续，j 是物理地址
    pub block_table: Vec<usize>,
    pub sampling_params: SamplingParams,
}

impl Sequence {
    pub fn new(token_ids: impl Into<Vec<u32>>, sampling_params: SamplingParams) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let token_ids = token_ids.into();
        let prompt_len = token_ids.len();
        Self {
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
            state: SequenceState::Prefill,
            token_ids,
            prompt_len,
            cached_len: 0,
            block_table: vec![],
            sampling_params,
        }
    }

    /// self.token_ids.len() 
    #[inline]
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// 已生成（非 prompt）的 token 数
    #[inline]
    pub fn completion_len(&self) -> usize {
        self.token_ids.len() - self.prompt_len
    }

    /// 还有多少 token 没有写进 KV：self.token_ids.len() - self.cached_len
    #[inline]
    pub fn uncached_len(&self) -> usize {
        self.token_ids.len() - self.cached_len
    }

    /// 当前长度至少需要多少个 block
    pub fn needed_blocks(&self) -> usize {
        (self.token_ids.len() + BLOCK_SIZE - 1) / BLOCK_SIZE
    }

    /// 计算过 kv cache 的 block：self.cached_len / BLOCK_SIZE
    pub fn cached_blocks(&self) -> usize {
        self.cached_len / BLOCK_SIZE
    }

    pub fn last_token(&self) -> u32 {
        self.token_ids.last().cloned().expect("empty token ids")
    }

    /// 取出第 i 个逻辑 block 包含的 token
    pub fn token_block(&self, index: usize) -> &[u32] {
        let begin = index * BLOCK_SIZE;
        // 最后一块可能没写满,夹一下,Rust 的切片不像 Python 会自动截断
        let end = ((index + 1) * BLOCK_SIZE).min(self.token_ids.len());
        &self.token_ids[begin..end]
    }

    /// 最后一个逻辑 block 的 token（可能没写满）
    pub fn last_block_tokens(&self) -> &[u32] {
        let blocks = self.needed_blocks();
        let begin = (blocks - 1) * BLOCK_SIZE;
        &self.token_ids[begin..]
    }

    pub fn push_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
    }
}
