use std::collections::{HashMap, VecDeque};
use super::{BLOCK_SIZE, Sequence};
use crate::error::{OutOfBlocks, Result};
pub type BlockId = usize;
pub type Hash = u64;

/// 空闲块（引用计数为 0）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreeBlock {
    /// 干净空块
    Free { id: BlockId },
    /// 内容仍有效，可被相同前缀复用
    FreeCached { id: BlockId, hash: Hash },
}

impl FreeBlock {
    pub fn id(&self) -> BlockId {
        match self {
            FreeBlock::Free { id } | FreeBlock::FreeCached { id, .. } => *id,
        }
    }
}

/// 在用块（引用计数 >= 1）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveBlock {
    /// refs = 1，内容尚未登记，不可共享
    Exclusive,
    /// refs >= 1，内容已登记，可被同前缀请求共享
    Cached { refs: usize, hash: Hash },
}

/// 内容索引里的一条：hash 对应的物理块 + 用于冲突校验的 token。
#[derive(Clone, Debug)]
struct CachedEntry {
    id: BlockId,
    token_ids: Box<[u32]>,
}

/// `probe` 的命中结果：直接告诉 `commit` 该怎么落地，避免再次回查状态。
#[derive(Clone, Copy, Debug)]
enum CachedHit {
    /// 命中一个在用块：`refs += 1`
    Share(BlockId),
    /// 命中一个空闲缓存块：从 free 池摘回，变成独占在用
    Reclaim { id: BlockId, hash: Hash },
}

struct Plan {
    cached: Vec<CachedHit>,
    num_fresh_blocks: usize,
}

impl Plan {
    fn num_cached_tokens(&self) -> usize {
        self.cached.len() * BLOCK_SIZE
    }
}

pub struct BlockManager {
    /// 物理块表：`None` = 空闲（其状态保存在 free 池里），`Some` = 在用
    blocks: Vec<Option<LiveBlock>>,
    /// 空闲池：元素只可能是 `Free` / `FreeCached`，按回收顺序（FIFO 驱逐）
    free: VecDeque<FreeBlock>,
    /// 内容索引：hash -> (块, token_ids)，同时包含在用与空闲两类已登记块
    hash_index: HashMap<Hash, CachedEntry>,
}

impl BlockManager {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            // 初始化所有 Block 都是 None 空闲
            blocks: (0..num_blocks).map(|_| None).collect(),
            // 初始化所有的 free 
            free: (0..num_blocks).map(|id| FreeBlock::Free { id }).collect(),
            hash_index: HashMap::new(),
        }
    }

    /// Whether `seq` (a fresh prefill request) can get all its blocks.
    pub fn can_allocate(&self, seq: &Sequence) -> bool {
        self.free.len() >= seq.num_blocks()
    }

    /// Whether the next decode token can be appended (a new block is needed
    /// only when the sequence crosses a block boundary).
    pub fn can_append(&self, seq: &Sequence) -> bool {
        let remid = seq.len() % BLOCK_SIZE;
        if remid == 1 {
            self.free.len() >= 1
        } else {
            true
        }
    }

    /// 新请求到达，给 seq 分配 block_table，返回这次复用了多少 token。
    /// 空间不足就返回失败，且不改动任何状态（probe 无副作用）。
    pub fn allocate(&mut self, seq: &mut Sequence) -> Result<usize> {
        let plan = self.probe(seq)?;
        let num_cached_tokens = plan.num_cached_tokens();
        self.commit(seq, plan);
        Ok(num_cached_tokens)
    }

    /// 释放这个 seq 中的 block
    pub fn deallocate(&mut self, seq: &mut Sequence) {
        // 释放 seq 的所有 block，遍历 seq 的 block table
        for &id in seq.block_table.iter().rev() {
            // 取出这个 block，判断其状态（暂时将 id 位置设置为空闲）
            match self.blocks[id].take() {
                // block 独占，释放很简单，直接插入 free 队列，由于之前独占，也不可以共享，直接完全 Free
                Some(LiveBlock::Exclusive) => {
                    self.free.push_back(FreeBlock::Free { id });
                }
                // block 正在被贡献，同时引用计数大于 1，释放很简单，这个位置不可以被释放，修改 refs 即可
                Some(LiveBlock::Cached { refs, hash }) if refs > 1 => {
                    self.blocks[id] = Some(LiveBlock::Cached { refs: refs - 1, hash });
                }
                // block 正在被贡献，同时引用计数 == 1，需要被释放
                Some(LiveBlock::Cached { hash, .. }) => {
                    // 由于这个 block 是 hash 的，可以转为 FreeCached 用于之后的复用
                    self.free.push_back(FreeBlock::FreeCached { id, hash });
                }
                None => unreachable!("releasing an already-free block {id}"),
            }
        }
        seq.num_cached_tokens = 0;
        seq.block_table.clear();
    }

    /// 每次 prefill/decode 得到一个 append token 后调用。
    /// token 跨过块边界就补一个新块。空间不足时返回失败且不改状态。
    pub fn append_tokens(&mut self, seq: &mut Sequence) -> Result<()> {
        let needed = seq.num_blocks() - seq.block_table.len();
        if needed == 0 {
            return Ok(());
        }
        if self.free.len() < needed {
            return Err(OutOfBlocks { needed, free: self.free.len() }.into());
        }
        for _ in 0..needed {
            seq.block_table.push(self.take_free());
        }
        Ok(())
    }

    /// forward 算完之后调用，把这次新写满的块登记进缓存目录，别的序列才能复用。
    pub fn on_computed(&mut self, seq: &mut Sequence, num_computed: usize) {
        let start = seq.num_cached_tokens / BLOCK_SIZE;
        let end = (seq.num_cached_tokens + num_computed) / BLOCK_SIZE;
        seq.num_cached_tokens += num_computed;
        if start == end {
            return;
        }

        // 前缀链：第 start 个块的 hash 接在它前一个块上（若存在）。
        let mut h = if start > 0 {
            match &self.blocks[seq.block_table[start - 1]] {
                Some(LiveBlock::Cached { hash, .. }) => Some(*hash),
                _ => unreachable!("previous block must be registered before its successor"),
            }
        } else {
            None
        };

        for i in start..end {
            let id = seq.block_table[i];
            let token_ids: Box<[u32]> = seq.block(i).to_vec().into_boxed_slice();
            let cur_h = compute_hash(&token_ids, h);
            h = Some(cur_h);
            debug_assert!(self.blocks[id].is_some(), "登记一个空闲块 {id}");
            self.blocks[id] = Some(LiveBlock::Cached { refs: 1, hash: cur_h });
            self.hash_index.insert(cur_h, CachedEntry { id, token_ids });
        }
    }
}

impl BlockManager {
    /// 为一个新到来的请求（只有 prompt 部分）探测前缀复用方案。
    /// 纯读，不产生任何副作用；失败时调用方保证状态不变。
    fn probe(&self, seq: &Sequence) -> Result<Plan> {
        let num_blocks = seq.num_blocks();
        let mut cached = Vec::new();
        let mut h = None;
        let mut num_cached_free_blocks = 0usize;

        // 只复用写满的块（最后一块往往是半块，不共享）。
        for i in 0..num_blocks.saturating_sub(1) {
            // 为当前 块 计算 hash（包含前缀）
            let token_ids = seq.block(i);
            let cur_h = compute_hash(token_ids, h);
            h = Some(cur_h);

            // 命中条件：hash 在索引里，且 token id 完全一致
            let Some(entry) = self.hash_index.get(&cur_h) else { break };
            if entry.token_ids.as_ref() != token_ids {
                break;
            }

            // 通过 hash 得到缓存命中的 block id，我们希望使用 entry.id 这个 Block
            // 有两种情况：1. 复用之前的 block；2. 需要一个新 block（有 hash 的）
            match &self.blocks[entry.id] {
                // 复用已经存在的
                Some(LiveBlock::Cached { .. }) => cached.push(CachedHit::Share(entry.id)),
                // 之前的代码会保证不可能出现独占的 block 会被尝试共享
                Some(_) => unreachable!("hash_index points at a non-cached live block"),
                // 复用之前存在的（Hash 还没删除）
                None => {
                    cached.push(CachedHit::Reclaim { id: entry.id, hash: cur_h });
                    num_cached_free_blocks += 1;
                }
            }
        }

        // num_fresh_blocks：不可以被缓存的 block 数量
        let num_fresh_blocks = num_blocks - cached.len();
        // 需要从 free 池取出的块 = 命中的空闲缓存块 + 全新块
        let num_free_blocks_needed = num_cached_free_blocks + num_fresh_blocks;
        if self.free.len() < num_free_blocks_needed {
            return Err(OutOfBlocks { needed: num_free_blocks_needed, free: self.free.len() }.into());
        }
        Ok(Plan { cached, num_fresh_blocks })
    }

    /// 把 plan 变成 seq 的 block_table
    fn commit(&mut self, seq: &mut Sequence, plan: Plan) {
        let num_cached_blocks = plan.cached.len();
        let num_fresh_blocks = plan.num_fresh_blocks;
        let mut block_table = Vec::with_capacity(num_cached_blocks + num_fresh_blocks);

        // 先插入 cached 的 block，还是两种情况：复用 shared / 从 free 取出之前 hash 过还没删除的
        for hit in plan.cached {
            match hit {
                // 复用 shared block，只需增加引用计数
                CachedHit::Share(id) => {
                    match &mut self.blocks[id] {
                        Some(LiveBlock::Cached { refs, .. }) => *refs += 1,
                        // 之前的代码会保证不可能出现独占的 block 会被尝试共享
                        _ => unreachable!("Share hit on a non-cached block {id}"),
                    }
                    block_table.push(id);
                }
                // 从 free 中取出这个复用 hash 
                CachedHit::Reclaim { id, hash } => {
                    // 从空闲池摘回来继续用，内容不变。
                    self.free.retain(|fb| fb.id() != id);
                    // id 位置从 free 回到共享状态
                    self.blocks[id] = Some(LiveBlock::Cached { refs: 1, hash });
                    block_table.push(id);
                }
            }
        }

        // 剩下 block 直接去申请
        for _ in 0..num_fresh_blocks {
            block_table.push(self.take_free());
        }

        seq.num_cached_tokens = num_cached_blocks * BLOCK_SIZE;
        seq.block_table = block_table;
    }

    /// 从 free 池取出最早的空闲块并置为在用（未登记）。
    /// 若取到的是带缓存的空闲块，说明内容即将被覆盖，先作废它的索引条目。
    fn take_free(&mut self) -> BlockId {
        // 取出最早 free 的 block
        let free = self.free.pop_front().expect("empty free list");
        // free 有两种状态：1. 完全 free；2. 已经没有使用了，但 hash 还在（用于前缀缓存命中）
        let id = match free {
            // 完全 free 的 block
            FreeBlock::Free { id } => id,
            // hash 信息还在，要申请这个 block，需要删除之前的 hash 信息
            FreeBlock::FreeCached { id, hash } => {
                debug_assert_eq!(self.hash_index.get(&hash).map(|e| e.id), Some(id));
                self.hash_index.remove(&hash);
                id
            }
        };
        // 刚申请的 block 只能独占，还没填满 token id 
        self.blocks[id] = Some(LiveBlock::Exclusive);
        id
    }

    /// 测试用：校验 free 池 / blocks / hash_index 三者一致。
    #[cfg(test)]
    fn check_invariants(&self) {
        use std::collections::HashSet;
        let mut in_free = HashSet::new();
        for fb in &self.free {
            let id = fb.id();
            assert!(in_free.insert(id), "free 池出现重复 id {id}");
            assert!(self.blocks[id].is_none(), "free 池里的 {id} 却是 live");
        }
        for (id, block) in self.blocks.iter().enumerate() {
            if block.is_some() {
                assert!(!in_free.contains(&id), "live 块 {id} 同时出现在 free 池");
            }
        }
        for (h, entry) in &self.hash_index {
            assert_eq!(entry.token_ids.len(), BLOCK_SIZE, "登记的块必须写满");
            match &self.blocks[entry.id] {
                Some(LiveBlock::Cached { hash, .. }) => assert_eq!(hash, h),
                Some(_) => panic!("hash_index 指向非 Cached 的在用块 {}", entry.id),
                None => assert!(
                    self.free
                        .iter()
                        .any(|fb| matches!(fb, FreeBlock::FreeCached { id, hash } if *id == entry.id && hash == h)),
                    "hash_index 指向不存在的空闲缓存块 {}",
                    entry.id
                ),
            }
        }
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
        assert_eq!(bm.hash_index.len(), 1, "只有写满的块能进 hash 表");
        assert!(
            matches!(bm.blocks[seq.block_table[1]], Some(LiveBlock::Exclusive)),
            "尾块必须独占"
        );
        bm.check_invariants();
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
        assert!(
            matches!(bm.blocks[a.block_table[0]], Some(LiveBlock::Cached { refs: 2, .. })),
            "被共享的块 refs 应该 +1"
        );
        assert_ne!(b.block_table[1], a.block_table[1], "尾块不共享");
        bm.check_invariants();
    }

    #[test]
    fn free_but_cached_block_is_reused() {
        let mut bm = BlockManager::new(8);
        let mut a = prefill(&mut bm, tokens(0..600)); // 2 个满块 + 88 token 的尾块
        let a_blocks = a.block_table.clone();
        bm.deallocate(&mut a); // A 结束:块回空闲队列,内容还在

        // 命中的两块都是「空闲但带缓存」的状态 —— 它们也要占空闲队列的名额
        let mut b = Sequence::new(tokens(0..600), SamplingParams::default());
        let num_cached_tokens = bm.allocate(&mut b).expect("空闲块不够");

        assert_eq!(num_cached_tokens, 2 * BLOCK_SIZE);
        assert_eq!(b.block_table.len(), 3, "block_table 长度必须等于总块数");
        assert_eq!(&b.block_table[..2], &a_blocks[..2], "前两块直接复用");
        assert_eq!(bm.free.len(), 5, "3 个块都从空闲队列里拿");
        bm.check_invariants();
    }

    #[test]
    fn reusing_a_free_block_evicts_its_cache_entry() {
        let mut bm = BlockManager::new(2);
        let mut a = prefill(&mut bm, tokens(0..300)); // 块 0 写满并登记
        assert_eq!(bm.hash_index.len(), 1);
        bm.deallocate(&mut a); // 空闲,但缓存还在

        // 一个毫不相干的序列把块拿走 —— 内容即将被覆盖,映射必须作废
        let mut b = Sequence::new(tokens(1000..1300), SamplingParams::default());
        bm.allocate(&mut b).expect("空闲块不够");

        assert!(bm.hash_index.is_empty(), "被覆盖的块不能再留着缓存映射");
        assert!(
            bm.free.iter().all(|fb| matches!(fb, FreeBlock::Free { .. })),
            "不应再有带缓存的空闲块"
        );
        assert!(
            bm.blocks.iter().all(|b| !matches!(b, Some(LiveBlock::Cached { .. }))),
            "不应再有已登记的在用块"
        );
        bm.check_invariants();
    }

    #[test]
    fn cached_free_blocks_count_toward_the_budget() {
        let mut bm = BlockManager::new(4);
        let mut a = prefill(&mut bm, tokens(0..600)); // 3 块全用掉
        bm.deallocate(&mut a); // 都回到空闲队列,前两块带着缓存

        let mut c = Sequence::new(tokens(1000..1300), SamplingParams::default()); // 占 2 块
        bm.allocate(&mut c).expect("空闲块不够");

        let mut b = Sequence::new(tokens(0..600), SamplingParams::default());
        // 命中 2 块(空闲但带缓存,要摘回来)+ 1 个新块 = 3 > 剩下的 2 块
        let Error::OutOfBlocks(e) = bm.allocate(&mut b).unwrap_err();
        assert_eq!((e.needed, e.free), (3, 2));
        assert!(b.block_table.is_empty(), "失败不能留下半截 block_table");
        bm.check_invariants();
    }

    #[test]
    fn hash_chain_prevents_false_sharing() {
        let mut bm = BlockManager::new(32);
        let p0 = tokens(0..256);
        let q = tokens(500..756);
        let x = tokens(1000..1256);
        let y = tokens(2000..2256);
        let z = tokens(3000..3256);

        prefill(&mut bm, [p0.clone(), x].concat()); // 登记 p0 这一块
        let c = prefill(&mut bm, [p0.clone(), y.clone(), z.clone()].concat());
        let e = prefill(&mut bm, [q, y.clone(), z.clone()].concat()); // 不同前缀,同样的 y

        let mut b = Sequence::new([p0, y, z].concat(), SamplingParams::default());
        let num_cached_tokens = bm.allocate(&mut b).expect("空闲块不够");
        assert_eq!(num_cached_tokens, 2 * BLOCK_SIZE);

        assert_eq!(b.block_table[1], c.block_table[1], "同前缀的 y 块才能复用");
        assert_ne!(b.block_table[1], e.block_table[1], "不同前缀的 y 块不能复用");
        bm.check_invariants();
    }

    #[test]
    fn failed_allocation_leaves_state_untouched() {
        let mut bm = BlockManager::new(2);
        let mut a = Sequence::new(tokens(0..300), SamplingParams::default());
        bm.allocate(&mut a).expect("应该能装下"); // 2 个块全用掉

        let mut b = Sequence::new(tokens(1000..2000), SamplingParams::default()); // 要 4 块
        let free_before = bm.free.clone();
        let Error::OutOfBlocks(e) = bm.allocate(&mut b).unwrap_err();
        assert_eq!((e.needed, e.free), (4, 0));

        assert_eq!(bm.free, free_before, "失败不能动空闲队列");
        assert!(b.block_table.is_empty(), "失败不能留下半截 block_table");
        bm.check_invariants();
    }

    #[test]
    fn append_tokens_grows_block_table_at_boundary() {
        let mut bm = BlockManager::new(4);
        let mut seq = prefill(&mut bm, tokens(0..255)); // 1 块,还没写满
        assert_eq!(seq.block_table.len(), 1);

        seq.append_token(9999); // 256 个 token:还是 1 块
        bm.append_tokens(&mut seq).expect("不该失败");
        assert_eq!(seq.block_table.len(), 1);

        seq.append_token(9999); // 257 个 token:要第 2 块了
        bm.append_tokens(&mut seq).expect("不该失败");
        assert_eq!(seq.block_table.len(), 2);
        assert!(
            matches!(bm.blocks[seq.block_table[1]], Some(LiveBlock::Exclusive)),
            "新尾块独占"
        );
        bm.check_invariants();
    }

    #[test]
    fn append_tokens_fails_cleanly_when_out_of_blocks() {
        let mut bm = BlockManager::new(1);
        let mut seq = prefill(&mut bm, tokens(0..256)); // 唯一的块,正好写满
        let table_before = seq.block_table.clone();

        seq.append_token(1); // 跨块,但没有空闲块了
        assert!(bm.append_tokens(&mut seq).is_err());
        assert_eq!(seq.block_table, table_before, "失败不能动 block_table");
        bm.check_invariants();
    }

    // ---- 状态迁移 -----------------------------------------------------------

    #[test]
    fn take_free_from_empty_and_cached() {
        let mut bm = BlockManager::new(2);
        let mut a = prefill(&mut bm, tokens(0..300)); // 块0 登记,块1 尾块
        let cached_id = a.block_table[0];
        bm.deallocate(&mut a); // free = [块1(Free), 块0(FreeCached)]

        assert!(bm.free.iter().any(|fb| matches!(fb, FreeBlock::FreeCached { id, .. } if *id == cached_id)));
        assert!(bm.free.iter().any(|fb| matches!(fb, FreeBlock::Free { .. })));

        // 取走纯空块：不影响 hash_index
        let first = bm.take_free();
        assert!(matches!(bm.blocks[first], Some(LiveBlock::Exclusive)));
        assert_eq!(bm.hash_index.len(), 1);

        // 取走缓存空闲块：必须摘掉它的 hash_index 条目
        let second = bm.take_free();
        assert_eq!(second, cached_id);
        assert!(bm.hash_index.is_empty());
        assert!(matches!(bm.blocks[second], Some(LiveBlock::Exclusive)));
        bm.check_invariants();
    }

    #[test]
    fn share_is_only_on_cached_live_blocks() {
        let mut bm = BlockManager::new(8);
        let a = prefill(&mut bm, tokens(0..300));
        let mut b = Sequence::new(tokens(0..300), SamplingParams::default());
        bm.allocate(&mut b).expect("空闲块不够");
        assert!(matches!(bm.blocks[a.block_table[0]], Some(LiveBlock::Cached { refs: 2, .. })));
        assert!(matches!(bm.blocks[a.block_table[1]], Some(LiveBlock::Exclusive)));
        bm.check_invariants();
    }

    #[test]
    fn release_steps_refs_then_frees_cached() {
        let mut bm = BlockManager::new(8);
        let mut a = prefill(&mut bm, tokens(0..300));
        let mut b = Sequence::new(tokens(0..300), SamplingParams::default());
        bm.allocate(&mut b).expect("空闲块不够");
        let shared_id = a.block_table[0];
        assert!(matches!(bm.blocks[shared_id], Some(LiveBlock::Cached { refs: 2, .. })));

        bm.deallocate(&mut b); // refs 2 -> 1,仍在使用
        assert!(matches!(bm.blocks[shared_id], Some(LiveBlock::Cached { refs: 1, .. })));

        bm.deallocate(&mut a); // refs 1 -> 0,变成空闲缓存块
        assert!(bm.blocks[shared_id].is_none());
        assert!(bm.free.iter().any(|fb| matches!(fb, FreeBlock::FreeCached { id, .. } if *id == shared_id)));
        bm.check_invariants();
    }
}
