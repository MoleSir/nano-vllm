use std::{collections::{HashMap, HashSet, VecDeque}, fmt::Display};
use super::{BLOCK_SIZE, Sequence};
use crate::error::{OutOfBlocks, Result};

pub type BlockId = usize;
pub type Hash = u64;

pub struct Block {
    pub id: BlockId,
    pub ref_count: usize,
    pub hash: Option<Hash>,
    pub token_ids: Vec<u32>,
}

pub struct BlockManager {
    pub blocks: Vec<Block>,
    pub hash_to_block_id: HashMap<Hash, BlockId>,
    pub free_block_ids: VecDeque<BlockId>,
    pub used_block_ids: HashSet<BlockId>,
}

impl BlockManager {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            blocks: (0..num_blocks).map(|id| Block::new(id) ).collect(),
            hash_to_block_id: HashMap::new(),
            free_block_ids: (0..num_blocks).collect(),
            used_block_ids: HashSet::new(),
        }
    }

    /// 新请求到达，给 seq 分配 block_table，返回这次复用了多少 token
    /// 空间不足就返回失败
    pub fn allocate(&mut self, seq: &mut Sequence) -> Result<usize> {
        let plan = self.probe(seq)?;
        let num_cached_tokens = plan.num_cached_tokens();
        self.commit(seq, plan);
        Ok(num_cached_tokens)
    }

    /// 释放这个 seq 中的 block
    pub fn deallocate(&mut self, seq: &mut Sequence) {
        for &block_id in seq.block_table.iter().rev() {
            self.blocks[block_id].ref_count -= 1;
            if self.blocks[block_id].ref_count == 0 {
                self.deallocate_block(block_id);
            }
        }
        seq.num_cached_tokens = 0;
        seq.block_table.clear();
    }

    /// 每次 prefill/decode 得到一个 append token 后调用
    /// token 跨过块边界就补一个新块。
    /// 如果这一步腾不出块，调用方该抢占/换出这个序列 —— 同样不改任何状态。
    pub fn append_tokens(&mut self, seq: &mut Sequence) -> Result<()> {
        // 给 seq push token 后，看下需要的 block 数量和当前 block table 的数量
        let needed = seq.num_blocks() - seq.block_table.len();
        if needed == 0 {
            return Ok(());
        }
        // 需要 needed 个 block，空间不足
        if self.free_block_ids.len() < needed {
            return Err(OutOfBlocks { needed, free: self.free_block_ids.len() }.into());
        }
        // 对每个需要补充的 block，申请一个空块，然后插入 table
        // 同样的，只是申请到了空间，新快都没有被 hash，他们的 hash 都是空，需要调用 on_computed
        for _ in 0..needed {
            seq.block_table.push(self.allocate_block());
        }
        Ok(())
    }

    /// forward 算完之后调用，把这次新写满的块登记进缓存目录，别的序列才能复用。
    /// allocate 后，还没有 refill，所以通过 allocate_block 得到的这些 block 只是有了空间，但没有被 hash，无法共享
    pub fn on_computed(&mut self, seq: &mut Sequence, num_computed: usize) {
        let start = seq.num_cached_tokens / BLOCK_SIZE;
        let end = (seq.num_cached_tokens + num_computed) / BLOCK_SIZE;
        seq.num_cached_tokens += num_computed;
        if start == end {
            return;
        }

        let mut h = if start > 0 { self.blocks[seq.block_table[start - 1]].hash } else { None };
        debug_assert!(start == 0 || h.is_some(), "前一个块要先登记,前缀链才接得上");

        for i in start..end {
            let block_id = seq.block_table[i];
            let token_ids = seq.block(i);
            let cur_h = compute_hash(token_ids, h);
            h = Some(cur_h);
            self.blocks[block_id].update(cur_h, token_ids.to_vec());
            self.hash_to_block_id.insert(cur_h, block_id);
        }
    }
}

impl BlockManager {
    /// 为一个新到来的请求（只有 prompt 部分）申请这些 token 需要的 Block
    /// 前缀判断下是否可以缓存，如果空间不够会报错
    fn probe(&self, seq: &Sequence) -> Result<Plan> {
        let num_blocks = seq.num_blocks();
        let mut cached = vec![];
        let mut h = None;
        let mut num_cached_free_blocks = 0;

        for i in 0..num_blocks - 1 {
            // 计算这块的 hash（之前的 hash + token ids）
            let token_ids = seq.block(i);
            let cur_h = compute_hash(token_ids, h);
            h = Some(cur_h);
            
            // 检查是否可以复用，条件：当前 hash 值出现在 hash_to_block_id 中，并且 token id 完全一致（主要防止 hash 冲突）
            // 只要有一块不满足，之后所有的都不可以复用，直接 break
            let Some(&block_id) = self.hash_to_block_id.get(&cur_h) else { break; };
            if token_ids != self.blocks[block_id].token_ids {
                break;
            }

            // hash 的块可能是空闲的
            // hash 和 used 是分开的两个系统，例如一个不完整的 block 一直 used 但不可以被 hash 记录来共享
            // 一个 block 释放只是移除 used，不会删除 hash 和 block 的映射
            // 例如 system prompt，在前一条完成释放后，system prompt 的 hash 还在，直接复用
            if !self.used_block_ids.contains(&block_id) {
                num_cached_free_blocks += 1;
            }
            
            // 不需要去申请的 block，这些有两种情况
            // 1. 已经释放了，但 hash 和 kv cache 还在
            // 2. 没有释放，还在跟其他人共享呢
            cached.push(block_id);
        }

        // cached 中是不需要申请的，总块 - 不需要申请的 = 需要申请的
        let num_fresh_blocks = num_blocks - cached.len();
        // 需要申请的 + 不需要申请但当前 free 的，这些都需要从 free block 中取
        let num_free_blocks_needed = num_cached_free_blocks + num_fresh_blocks;
        if self.free_block_ids.len() < num_free_blocks_needed {
            return Err(OutOfBlocks { needed: num_free_blocks_needed, free: self.free_block_ids.len() }.into());
        }
        Ok(Plan { cached, num_fresh_blocks })
    }

    /// 把 plan 变成 seq 的 block_table。plan 是刚探测出来的
    fn commit(&mut self, seq: &mut Sequence, plan: Plan) {
        let Plan { cached, num_fresh_blocks } = plan;
        let num_cached_blocks = cached.len();
        let mut block_table = Vec::with_capacity(seq.num_blocks());
        
        // cached 表示这个前缀部分可以被缓存。但分为两种情况：
        // 1. 原来使用的 block 被释放了，但 hash 和 kvcache 还保存
        // 2. 正在使用这个 block，直接和其他 seq 共享
        for block_id in cached {
            if self.used_block_ids.contains(&block_id) {
                // 还被别人引用:共享,加一次引用
                self.blocks[block_id].ref_count += 1;
            } else {
                // 已经空闲但内容还在:从缓存里捡回来。
                self.blocks[block_id].ref_count = 1;
                self.free_block_ids.retain(|&id| id != block_id);   // 冷路径,O(n)
                self.used_block_ids.insert(block_id);
            }
            // 到这里两种情况都处理好了，进入 used，没有新申请 block
            block_table.push(block_id);
        }

        // 剩下的必须分配
        for _ in 0..num_fresh_blocks {
            block_table.push(self.allocate_block());
        }

        // 记录下 cached 的 token 数量，并且更新 block table
        seq.num_cached_tokens = num_cached_blocks * BLOCK_SIZE;
        seq.block_table = block_table;
    }
    
    /// 从 free_block_ids 取出最早的空闲 block
    /// 插入到 used_block_ids
    /// 注意不会填充 block 的 hash 也不会插入到 hash to block 
    fn allocate_block(&mut self) -> BlockId {
        let block_id = self.free_block_ids.pop_front().expect("empty free list");
        let block = &mut self.blocks[block_id];
        // 如果这个空闲块之前有计算过 hash，需要删除这个 hash 记录
        if let Some(hash) = block.hash {
            if self.hash_to_block_id.get(&hash) == Some(&block_id) {
                self.hash_to_block_id.remove(&hash);
            }
        }
        block.reset();
        self.used_block_ids.insert(block_id);
        block_id
    }

    /// 将 block_id 这个 Block 从 used 移除，插入 free
    fn deallocate_block(&mut self, block_id: BlockId) {
        self.used_block_ids.remove(&block_id);
        self.free_block_ids.push_back(block_id);
    }
}

impl Block {
    pub fn new(id: BlockId) -> Self {
        Self {
            id, ref_count: 0, hash: None, token_ids: vec![],
        }
    }

    pub fn update(&mut self, hash: u64, token_ids: Vec<u32>) {
        self.hash = Some(hash);
        self.token_ids = token_ids;
    }

    pub fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = None;
        self.token_ids = vec![];
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} {} {:?}", self.id, self.ref_count, self.hash)
    }
}

#[derive(Debug)]
struct Plan {
    cached: Vec<BlockId>,
    num_fresh_blocks: usize,
}

impl Plan {
    fn num_cached_tokens(&self) -> usize {
        self.cached.len() * BLOCK_SIZE
    }
}

use xxhash_rust::xxh64::Xxh64;

pub fn compute_hash(token_ids: &[u32], prefix: Option<u64>) -> u64 {
    let mut h = Xxh64::new(0);
    if let Some(prefix) = prefix {
        h.update(&prefix.to_le_bytes());
    }
    h.update(bytemuck::cast_slice(token_ids));
    h.digest()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::SamplingParams;
    use crate::error::Error;

    fn tokens(range: std::ops::Range<u32>) -> Vec<u32> {
        range.collect()
    }

    /// 模拟一次「分配 + 前向算完」:分配 block_table,并把算完的块登记进缓存
    fn prefill(bm: &mut BlockManager, token_ids: Vec<u32>) -> Sequence {
        let mut seq = Sequence::new(token_ids, SamplingParams::default());
        bm.allocate(&mut seq).expect("空闲块不够");
        let num_computed = seq.len() - seq.num_cached_tokens;
        bm.on_computed(&mut seq, num_computed);
        seq
    }

    #[test]
    fn partial_tail_block_is_never_shared() {
        let mut bm = BlockManager::new(16);
        let seq = prefill(&mut bm, tokens(0..300)); // 1 个满块 + 44 token 的尾块

        assert_eq!(seq.block_table.len(), 2);
        assert_eq!(bm.hash_to_block_id.len(), 1, "只有写满的块能进 hash 表");
        assert_eq!(bm.blocks[seq.block_table[1]].ref_count, 1, "尾块必须独占");
    }

    #[test]
    fn cached_prefix_block_is_shared() {
        let mut bm = BlockManager::new(16);
        let a = prefill(&mut bm, tokens(0..300));

        // 白盒看一眼探测结果(allocate 内部就是它)
        let mut b = Sequence::new(tokens(0..300), SamplingParams::default());
        let plan = bm.probe(&b).expect("空闲块不够");
        assert_eq!(plan.cached.len(), 1, "只有写满的第 0 块能命中");
        assert_eq!(plan.num_fresh_blocks, 1, "尾块要新分配");

        let num_cached_tokens = bm.allocate(&mut b).expect("空闲块不够");
        assert_eq!(num_cached_tokens, BLOCK_SIZE);
        assert_eq!(b.block_table[0], a.block_table[0]);
        assert_eq!(bm.blocks[a.block_table[0]].ref_count, 2);
        assert_ne!(b.block_table[1], a.block_table[1], "尾块不共享");
    }

    #[test]
    fn free_but_cached_block_is_reused() {
        let mut bm = BlockManager::new(8);
        let mut a = prefill(&mut bm, tokens(0..600)); // 2 个满块 + 88 token 的尾块
        let a_blocks = a.block_table.clone();
        bm.deallocate(&mut a);                        // A 结束:块回空闲队列,内容还在

        // 命中的两块都是「空闲但带缓存」的状态 —— 它们也要占空闲队列的名额
        let mut b = Sequence::new(tokens(0..600), SamplingParams::default());
        let num_cached_tokens = bm.allocate(&mut b).expect("空闲块不够");

        assert_eq!(num_cached_tokens, 2 * BLOCK_SIZE);
        assert_eq!(b.block_table.len(), 3, "block_table 长度必须等于总块数");
        assert_eq!(&b.block_table[..2], &a_blocks[..2], "前两块直接复用");
        assert_eq!(bm.free_block_ids.len(), 5, "3 个块都从空闲队列里拿");
    }

    #[test]
    fn reusing_a_free_block_evicts_its_cache_entry() {
        let mut bm = BlockManager::new(2);
        let mut a = prefill(&mut bm, tokens(0..300)); // 块 0 写满并登记
        assert_eq!(bm.hash_to_block_id.len(), 1);
        bm.deallocate(&mut a);                        // 空闲,但缓存还在

        // 一个毫不相干的序列把块拿走 —— 内容即将被覆盖,映射必须作废
        let mut b = Sequence::new(tokens(1000..1300), SamplingParams::default());
        bm.allocate(&mut b).expect("空闲块不够");

        assert!(bm.hash_to_block_id.is_empty(), "被覆盖的块不能再留着缓存映射");
        assert!(bm.blocks.iter().all(|block| block.hash.is_none()));
    }

    #[test]
    fn cached_free_blocks_count_toward_the_budget() {
        let mut bm = BlockManager::new(4);
        let mut a = prefill(&mut bm, tokens(0..600)); // 3 块全用掉
        bm.deallocate(&mut a);                        // 都回到空闲队列,前两块带着缓存

        let mut c = Sequence::new(tokens(1000..1300), SamplingParams::default()); // 占 2 块
        bm.allocate(&mut c).expect("空闲块不够");

        let mut b = Sequence::new(tokens(0..600), SamplingParams::default());
        // 命中 2 块(空闲但带缓存,要摘回来)+ 1 个新块 = 3 > 剩下的 2 块
        let Error::OutOfBlocks(e) = bm.allocate(&mut b).unwrap_err();
        assert_eq!((e.needed, e.free), (3, 2));
        assert!(b.block_table.is_empty(), "失败不能留下半截 block_table");
    }

    #[test]
    fn hash_chain_prevents_false_sharing() {
        let mut bm = BlockManager::new(32);
        let p0 = tokens(0..256);
        let q = tokens(500..756);
        let x = tokens(1000..1256);
        let y = tokens(2000..2256);
        let z = tokens(3000..3256);

        prefill(&mut bm, [p0.clone(), x].concat());                   // 登记 p0 这一块
        let c = prefill(&mut bm, [p0.clone(), y.clone(), z.clone()].concat());
        let e = prefill(&mut bm, [q, y.clone(), z.clone()].concat()); // 不同前缀,同样的 y

        let mut b = Sequence::new([p0, y, z].concat(), SamplingParams::default());
        let num_cached_tokens = bm.allocate(&mut b).expect("空闲块不够");
        assert_eq!(num_cached_tokens, 2 * BLOCK_SIZE);

        assert_eq!(b.block_table[1], c.block_table[1], "同前缀的 y 块才能复用");
        assert_ne!(b.block_table[1], e.block_table[1], "不同前缀的 y 块不能复用");
    }

    #[test]
    fn failed_allocation_leaves_state_untouched() {
        let mut bm = BlockManager::new(2);
        let mut a = Sequence::new(tokens(0..300), SamplingParams::default());
        bm.allocate(&mut a).expect("应该能装下"); // 2 个块全用掉

        let mut b = Sequence::new(tokens(1000..2000), SamplingParams::default()); // 要 4 块
        let free_before = bm.free_block_ids.clone();
        let Error::OutOfBlocks(e) = bm.allocate(&mut b).unwrap_err();
        assert_eq!((e.needed, e.free), (4, 0));

        assert_eq!(bm.free_block_ids, free_before, "失败不能动空闲队列");
        assert!(b.block_table.is_empty(), "失败不能留下半截 block_table");
    }

    #[test]
    fn append_tokens_grows_block_table_at_boundary() {
        let mut bm = BlockManager::new(4);
        let mut seq = prefill(&mut bm, tokens(0..255)); // 1 块,还没写满
        assert_eq!(seq.block_table.len(), 1);

        seq.append_token(9999);                         // 256 个 token:还是 1 块
        bm.append_tokens(&mut seq).expect("不该失败");
        assert_eq!(seq.block_table.len(), 1);

        seq.append_token(9999);                         // 257 个 token:要第 2 块了
        bm.append_tokens(&mut seq).expect("不该失败");
        assert_eq!(seq.block_table.len(), 2);
        assert_eq!(bm.blocks[seq.block_table[1]].ref_count, 1, "新尾块独占");
    }

    #[test]
    fn append_tokens_fails_cleanly_when_out_of_blocks() {
        let mut bm = BlockManager::new(1);
        let mut seq = prefill(&mut bm, tokens(0..256)); // 唯一的块,正好写满
        let table_before = seq.block_table.clone();

        seq.append_token(1);                            // 跨块,但没有空闲块了
        assert!(bm.append_tokens(&mut seq).is_err());
        assert_eq!(seq.block_table, table_before, "失败不能动 block_table");
    }
}
