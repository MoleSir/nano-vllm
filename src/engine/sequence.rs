use std::{ops::{Index, IndexMut}, sync::atomic::{AtomicUsize, Ordering}};

use super::BLOCK_SIZE;

pub struct SamplingParams {
    pub temperature: f64,
    pub max_tokens: usize,
    pub ignore_eos: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
}

/// 表示一个完整请求
pub struct Sequence {
    pub id: usize,
    pub status: SequenceStatus,
    /// 当前的所有 token ids，请求刚到就是 prompt，后面逐个 token 生成
    pub token_ids: Vec<u32>,
    pub num_prompt_tokens: usize,
    pub num_cached_tokens: usize,
    /// "页表"，当前 token ids 按照 Block 划分，block_table[i] 表示第 i 个 token ids 块的 kv cache 保存到的是哪个 Block 对象
    /// 可以理解为 seq 是一个进程，j = block_table[i] 中 i 是虚拟地址，逻辑连续，j 是物理地址
    pub block_table: Vec<usize>,
    pub sampling_params: SamplingParams,
}

impl Sequence {
    pub fn new(token_ids: impl Into<Vec<u32>>, sampling_params: SamplingParams) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let token_ids = token_ids.into();
        let num_prompt_tokens = token_ids.len();
        Self {
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
            status: SequenceStatus::Waiting,
            token_ids: token_ids,
            // 由 BlockManager 维护:已经算完(或复用了缓存)的 token 数
            num_cached_tokens: 0,
            num_prompt_tokens,
            block_table: vec![],
            sampling_params,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    #[inline]
    pub fn num_completion_tokens(&self) -> usize {
        self.token_ids.len() - self.num_prompt_tokens
    }

    /// 保存下 token_ids 这么多 token 最少要多少 block
    pub fn num_blocks(&self) -> usize {
        (self.token_ids.len() + BLOCK_SIZE - 1) / BLOCK_SIZE
    }

    pub fn num_cached_blocks(&self) -> usize {
        self.num_cached_tokens / BLOCK_SIZE
    }

    pub fn last_token(&self) -> u32 {
        self.token_ids.last().cloned().expect("empty token ids")
    }

    /// 取出第 i 个 block 包含的 token
    pub fn block(&self, index: usize) -> &[u32] {
        let block_begin = index * BLOCK_SIZE;
        // 最后一块可能没写满,夹一下,Rust 的切片不像 Python 会自动截断
        let block_end = ((index + 1) * BLOCK_SIZE).min(self.token_ids.len());
        &self.token_ids[block_begin..block_end]
    }

    pub fn last_block(&self) -> &[u32] {
        let num_blocks = self.num_blocks();
        let block_begin = (num_blocks - 1) * BLOCK_SIZE;
        &self.token_ids[block_begin..]
    }

    pub fn append_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
    }
}

impl Index<usize> for Sequence {
    type Output = u32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.token_ids[index]
    }
}

impl IndexMut<usize> for Sequence {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.token_ids[index]
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            max_tokens: 64,
            ignore_eos: false,
        }
    }
}

impl SamplingParams {
    pub fn new() -> Self {
        Self::default()
    }
}

