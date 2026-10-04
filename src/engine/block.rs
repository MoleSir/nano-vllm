use std::collections::{HashMap, VecDeque};

use super::{BLOCK_SIZE, Sequence};
use crate::error::{OutOfBlocks, Result};

pub type BlockId = usize;
pub type Hash = u64;

// ===========================================================================
//  状态模型
//
//  一个物理块（BlockId，对应一段 KV cache 显存）始终处于两个池之一：
//
//    - free 池：无引用。元素类型 `FreeBlock` 只会是 `Free` / `FreeCached`，
//      **不可能是在用块**——这点由类型保证，取出时不必再判断 `Live`。
//    - live 池（`slots` 中的 `Some`）：被 >=1 个序列引用。
//
//  「归属（refs）」与「内容缓存（hash）」是两个正交维度：
//    Free       : refs=0, 未登记
//    FreeCached : refs=0, 已登记（可被同前缀复用）
//    Exclusive  : refs=1, 未登记（独占，不可共享）
//    Cached     : refs>=1, 已登记（可被同前缀共享）
//
//  未登记的块一定独占（refs=1）：只有内容登记进 `hash_index` 后，才可能被
//  另一个序列通过前缀命中而 `refs += 1`。
// ===========================================================================

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
    /// refs = 1，内容尚未登记 —— 不可共享
    Exclusive,
    /// refs >= 1，内容已登记 —— 可被同前缀请求共享
    Cached { refs: usize, hash: Hash },
}

/// 内容索引里的一条：hash 对应的物理块 + 用于冲突校验的 token。
#[derive(Clone, Debug)]
struct CachedEntry {
    id: BlockId,
    token_ids: Box<[u32]>,
}

/// `plan_prefix` 的命中结果：直接告诉 `apply_plan` 该怎么落地，避免再次回查状态。
#[derive(Clone, Copy, Debug)]
enum PrefixHit {
    /// 命中一个在用块：`refs += 1`
    Share(BlockId),
    /// 命中一个空闲缓存块：从 free 池摘回，变成独占在用
    Reuse { id: BlockId, hash: Hash },
}

struct AllocationPlan {
    hits: Vec<PrefixHit>,
    fresh_blocks: usize,
}

impl AllocationPlan {
    fn reused_tokens(&self) -> usize {
        self.hits.len() * BLOCK_SIZE
    }
}

pub struct BlockManager {
    /// 物理块表：`None` = 空闲（其状态保存在 free 池里），`Some` = 在用
    slots: Vec<Option<LiveBlock>>,
    /// 空闲池：元素只可能是 `Free` / `FreeCached`，按回收顺序（FIFO 驱逐）
    free_blocks: VecDeque<FreeBlock>,
    /// 内容索引：hash -> (块, token_ids)，同时包含在用与空闲两类已登记块
    hash_index: HashMap<Hash, CachedEntry>,
}

impl BlockManager {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            slots: (0..num_blocks).map(|_| None).collect(),
            free_blocks: (0..num_blocks).map(|id| FreeBlock::Free { id }).collect(),
            hash_index: HashMap::new(),
        }
    }

    /// 一条【待 prefill】的 seq（prompt 尚未进入 KV）能否装下它需要的全部块。
    pub fn can_reserve_prefill(&self, seq: &Sequence) -> bool {
        self.free_blocks.len() >= seq.needed_blocks()
    }

    /// decode 追加一个 token 后是否需要新块、以及现在是否拿得出。
    pub fn can_extend_for_token(&self, seq: &Sequence) -> bool {
        // 调用这个函数的时候，之前插入过 token，修改了 seq.len()，但这个 token 还没有申请 block。
        // 所以是判断 remid == 1。
        // 后续的逻辑是：extend_for_token 申请一个 block，但这个 block 还没有写任何 kvcache
        // 因为就这个 token 是新得到的，kvcache 都还没计算，直到下次 attention 就会算出这个 token 的向量，然后写入到 全局 kvcache
        let remid = seq.len() % BLOCK_SIZE;
        if remid == 1 {
            self.free_blocks.len() >= 1
        } else {
            true
        }
    }

    /// 为一条【待 prefill】的 seq 预留全部所需块，并尽力复用公共前缀的缓存。
    /// 前置条件：`seq.cached_len == 0` 且 `block_table` 为空。
    /// 空间不足时返回失败，且不改动任何状态。
    pub fn reserve_prefill(&mut self, seq: &mut Sequence) -> Result<()> {
        debug_assert_eq!(seq.cached_len, 0, "reserve_prefill 只用于尚未 prefill 的序列");
        let plan = self.plan_prefix(seq)?;
        self.apply_plan(seq, plan);
        Ok(())
    }

    /// 释放这个 seq 中的全部块（结束或抢占时调用）。
    pub fn release(&mut self, seq: &mut Sequence) {
        for &id in seq.block_table.iter().rev() {
            match self.slots[id].take() {
                Some(LiveBlock::Exclusive) => {
                    self.free_blocks.push_back(FreeBlock::Free { id });
                }
                Some(LiveBlock::Cached { refs, hash }) if refs > 1 => {
                    self.slots[id] = Some(LiveBlock::Cached { refs: refs - 1, hash });
                }
                Some(LiveBlock::Cached { hash, .. }) => {
                    self.free_blocks.push_back(FreeBlock::FreeCached { id, hash });
                }
                None => unreachable!("releasing an already-free block {id}"),
            }
        }
        seq.cached_len = 0;
        seq.block_table.clear();
    }

    /// 【decode】追加一个 token 后，若跨过块边界则补一个新块。
    /// 前置条件：`block_table` 已经覆盖到 `len - 1`。
    pub fn extend_for_token(&mut self, seq: &mut Sequence) -> Result<()> {
        let needed = seq.needed_blocks() - seq.block_table.len();
        if needed == 0 {
            return Ok(());
        }
        if self.free_blocks.len() < needed {
            return Err(OutOfBlocks { needed, free: self.free_blocks.len() }.into());
        }
        for _ in 0..needed {
            seq.block_table.push(self.take_free_block());
        }
        Ok(())
    }

    /// forward 算完之后调用，把这次新写满的块登记进缓存目录，别的序列才能复用。
    pub fn register_computed(&mut self, seq: &mut Sequence, num_computed: usize) {
        // 找到还没有 cached 的 block，之前的 block 都 cached 过了
        let start = seq.cached_len / BLOCK_SIZE;
        // cached_len 后增加 num_computed 个 token
        let end = (seq.cached_len + num_computed) / BLOCK_SIZE;
        // 总 cached 数量增加
        seq.cached_len += num_computed;
        // 如果增加的 cache 数量不会满出一个 token，啥也不用干
        if start == end {
            return;
        }

        // 前缀链：第 start 个块的 hash 接在它前一个块上（若存在）。
        let mut h = if start > 0 {
            match &self.slots[seq.block_table[start - 1]] {
                Some(LiveBlock::Cached { hash, .. }) => Some(*hash),
                _ => unreachable!("previous block must be registered before its successor"),
            }
        } else {
            None
        };

        // 对这些 cache 但还没 hash 的 block
        for i in start..end {
            // 取出这个 block id
            let id = seq.block_table[i];
            // 这个 block 已经满了，可以 hash
            let token_ids: Box<[u32]> = seq.token_block(i).to_vec().into_boxed_slice();
            let cur_h = compute_hash(&token_ids, h);
            h = Some(cur_h);
            debug_assert!(self.slots[id].is_some(), "登记一个空闲块 {id}");
            // 更新，从 Exclusive -> Cached
            self.slots[id] = Some(LiveBlock::Cached { refs: 1, hash: cur_h });
            // 增加 hash 
            self.hash_index.insert(cur_h, CachedEntry { id, token_ids });
        }
    }

    /// 为一个新到来的请求（只有 prompt 部分）探测前缀复用方案。
    /// 纯读，不产生任何副作用；失败时调用方保证状态不变。
    fn plan_prefix(&self, seq: &Sequence) -> Result<AllocationPlan> {
        let needed = seq.needed_blocks();
        let mut hits = Vec::new();
        let mut h = None;
        let mut num_cached_free_blocks = 0usize;

        // 只复用写满的块（最后一块往往是半块，不共享）。
        for i in 0..needed.saturating_sub(1) {
            let token_ids = seq.token_block(i);
            let cur_h = compute_hash(token_ids, h);
            h = Some(cur_h);

            // 命中条件：hash 在索引里，且 token id 逐一对上（防 hash 冲突）。
            let Some(entry) = self.hash_index.get(&cur_h) else { break };
            if entry.token_ids.as_ref() != token_ids {
                break;
            }

            match &self.slots[entry.id] {
                Some(LiveBlock::Cached { .. }) => hits.push(PrefixHit::Share(entry.id)),
                Some(_) => unreachable!("hash_index points at a non-cached live block"),
                None => {
                    hits.push(PrefixHit::Reuse { id: entry.id, hash: cur_h });
                    num_cached_free_blocks += 1;
                }
            }
        }

        let fresh_blocks = needed - hits.len();
        // 需要从 free 池取出的块 = 命中的空闲缓存块 + 全新块
        let num_free_blocks_needed = num_cached_free_blocks + fresh_blocks;
        if self.free_blocks.len() < num_free_blocks_needed {
            return Err(OutOfBlocks { needed: num_free_blocks_needed, free: self.free_blocks.len() }.into());
        }
        Ok(AllocationPlan { hits, fresh_blocks })
    }

    /// 把 plan 变成 seq 的 block_table，并把复用的前缀长度写进 `seq.cached_len`。
    fn apply_plan(&mut self, seq: &mut Sequence, plan: AllocationPlan) {
        let reused_len = plan.reused_tokens();
        let fresh_blocks = plan.fresh_blocks;
        let mut block_table = Vec::with_capacity(plan.hits.len() + fresh_blocks);

        for hit in plan.hits {
            match hit {
                PrefixHit::Share(id) => {
                    match &mut self.slots[id] {
                        Some(LiveBlock::Cached { refs, .. }) => *refs += 1,
                        _ => unreachable!("Share hit on a non-cached block {id}"),
                    }
                    block_table.push(id);
                }
                PrefixHit::Reuse { id, hash } => {
                    // 从空闲池摘回来继续用，内容不变。
                    self.free_blocks.retain(|fb| fb.id() != id);
                    self.slots[id] = Some(LiveBlock::Cached { refs: 1, hash });
                    block_table.push(id);
                }
            }
        }

        for _ in 0..fresh_blocks {
            block_table.push(self.take_free_block());
        }

        seq.cached_len = reused_len;
        seq.block_table = block_table;
    }

    /// 从 free 池取出最早的空闲块并置为在用（未登记）。
    /// 若取到的是带缓存的空闲块，说明内容即将被覆盖，先作废它的索引条目。
    fn take_free_block(&mut self) -> BlockId {
        let free = self.free_blocks.pop_front().expect("empty free list");
        let id = match free {
            FreeBlock::Free { id } => id,
            FreeBlock::FreeCached { id, hash } => {
                debug_assert_eq!(self.hash_index.get(&hash).map(|e| e.id), Some(id));
                self.hash_index.remove(&hash);
                id
            }
        };
        self.slots[id] = Some(LiveBlock::Exclusive);
        id
    }

    /// 测试用：校验 free 池 / slots / hash_index 三者一致。
    #[cfg(test)]
    fn check_invariants(&self) {
        use std::collections::HashSet;
        let mut in_free = HashSet::new();
        for fb in &self.free_blocks {
            let id = fb.id();
            assert!(in_free.insert(id), "free 池出现重复 id {id}");
            assert!(self.slots[id].is_none(), "free 池里的 {id} 却是 live");
        }
        for (id, block) in self.slots.iter().enumerate() {
            if block.is_some() {
                assert!(!in_free.contains(&id), "live 块 {id} 同时出现在 free 池");
            }
        }
        for (h, entry) in &self.hash_index {
            assert_eq!(entry.token_ids.len(), BLOCK_SIZE, "登记的块必须写满");
            match &self.slots[entry.id] {
                Some(LiveBlock::Cached { hash, .. }) => assert_eq!(hash, h),
                Some(_) => panic!("hash_index 指向非 Cached 的在用块 {}", entry.id),
                None => assert!(
                    self.free_blocks
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
        bm.reserve_prefill(&mut seq).expect("空闲块不够");
        let num_computed = seq.uncached_len();
        bm.register_computed(&mut seq, num_computed);
        seq
    }

    #[test]
    fn partial_tail_block_is_never_shared() {
        let mut bm = BlockManager::new(16);
        let seq = prefill(&mut bm, tokens(0..300)); // 1 个满块 + 44 token 的尾块

        assert_eq!(seq.block_table.len(), 2);
        assert_eq!(bm.hash_index.len(), 1, "只有写满的块能进 hash 表");
        assert!(
            matches!(bm.slots[seq.block_table[1]], Some(LiveBlock::Exclusive)),
            "尾块必须独占"
        );
        bm.check_invariants();
    }

    #[test]
    fn cached_prefix_block_is_shared() {
        let mut bm = BlockManager::new(16);
        let a = prefill(&mut bm, tokens(0..300));

        // 白盒看一眼探测结果(reserve_prefill 内部就是它)
        let mut b = Sequence::new(tokens(0..300), SamplingParams::default());
        let plan = bm.plan_prefix(&b).expect("空闲块不够");
        assert_eq!(plan.hits.len(), 1, "只有写满的第 0 块能命中");
        assert_eq!(plan.fresh_blocks, 1, "尾块要新分配");

        bm.reserve_prefill(&mut b).expect("空闲块不够");
        assert_eq!(b.cached_len, BLOCK_SIZE);
        assert_eq!(b.block_table[0], a.block_table[0]);
        assert!(
            matches!(bm.slots[a.block_table[0]], Some(LiveBlock::Cached { refs: 2, .. })),
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
        bm.release(&mut a); // A 结束:块回空闲队列,内容还在

        // 命中的两块都是「空闲但带缓存」的状态 —— 它们也要占空闲队列的名额
        let mut b = Sequence::new(tokens(0..600), SamplingParams::default());
        bm.reserve_prefill(&mut b).expect("空闲块不够");

        assert_eq!(b.cached_len, 2 * BLOCK_SIZE);
        assert_eq!(b.block_table.len(), 3, "block_table 长度必须等于总块数");
        assert_eq!(&b.block_table[..2], &a_blocks[..2], "前两块直接复用");
        assert_eq!(bm.free_blocks.len(), 5, "3 个块都从空闲队列里拿");
        bm.check_invariants();
    }

    #[test]
    fn reusing_a_free_block_evicts_its_cache_entry() {
        let mut bm = BlockManager::new(2);
        let mut a = prefill(&mut bm, tokens(0..300)); // 块 0 写满并登记
        assert_eq!(bm.hash_index.len(), 1);
        bm.release(&mut a); // 空闲,但缓存还在

        // 一个毫不相干的序列把块拿走 —— 内容即将被覆盖,映射必须作废
        let mut b = Sequence::new(tokens(1000..1300), SamplingParams::default());
        bm.reserve_prefill(&mut b).expect("空闲块不够");

        assert!(bm.hash_index.is_empty(), "被覆盖的块不能再留着缓存映射");
        assert!(
            bm.free_blocks.iter().all(|fb| matches!(fb, FreeBlock::Free { .. })),
            "不应再有带缓存的空闲块"
        );
        assert!(
            bm.slots.iter().all(|b| !matches!(b, Some(LiveBlock::Cached { .. }))),
            "不应再有已登记的在用块"
        );
        bm.check_invariants();
    }

    #[test]
    fn cached_free_blocks_count_toward_the_budget() {
        let mut bm = BlockManager::new(4);
        let mut a = prefill(&mut bm, tokens(0..600)); // 3 块全用掉
        bm.release(&mut a); // 都回到空闲队列,前两块带着缓存

        let mut c = Sequence::new(tokens(1000..1300), SamplingParams::default()); // 占 2 块
        bm.reserve_prefill(&mut c).expect("空闲块不够");

        let mut b = Sequence::new(tokens(0..600), SamplingParams::default());
        // 命中 2 块(空闲但带缓存,要摘回来)+ 1 个新块 = 3 > 剩下的 2 块
        let Error::OutOfBlocks(e) = bm.reserve_prefill(&mut b).unwrap_err();
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
        bm.reserve_prefill(&mut b).expect("空闲块不够");
        assert_eq!(b.cached_len, 2 * BLOCK_SIZE);

        assert_eq!(b.block_table[1], c.block_table[1], "同前缀的 y 块才能复用");
        assert_ne!(b.block_table[1], e.block_table[1], "不同前缀的 y 块不能复用");
        bm.check_invariants();
    }

    #[test]
    fn failed_allocation_leaves_state_untouched() {
        let mut bm = BlockManager::new(2);
        let mut a = Sequence::new(tokens(0..300), SamplingParams::default());
        bm.reserve_prefill(&mut a).expect("应该能装下"); // 2 个块全用掉

        let mut b = Sequence::new(tokens(1000..2000), SamplingParams::default()); // 要 4 块
        let free_before = bm.free_blocks.clone();
        let Error::OutOfBlocks(e) = bm.reserve_prefill(&mut b).unwrap_err();
        assert_eq!((e.needed, e.free), (4, 0));

        assert_eq!(bm.free_blocks, free_before, "失败不能动空闲队列");
        assert!(b.block_table.is_empty(), "失败不能留下半截 block_table");
        bm.check_invariants();
    }

    #[test]
    fn extend_for_token_grows_block_table_at_boundary() {
        let mut bm = BlockManager::new(4);
        let mut seq = prefill(&mut bm, tokens(0..255)); // 1 块,还没写满
        assert_eq!(seq.block_table.len(), 1);

        seq.push_token(9999); // 256 个 token:还是 1 块
        bm.extend_for_token(&mut seq).expect("不该失败");
        assert_eq!(seq.block_table.len(), 1);

        seq.push_token(9999); // 257 个 token:要第 2 块了
        bm.extend_for_token(&mut seq).expect("不该失败");
        assert_eq!(seq.block_table.len(), 2);
        assert!(
            matches!(bm.slots[seq.block_table[1]], Some(LiveBlock::Exclusive)),
            "新尾块独占"
        );
        bm.check_invariants();
    }

    #[test]
    fn extend_for_token_fails_cleanly_when_out_of_blocks() {
        let mut bm = BlockManager::new(1);
        let mut seq = prefill(&mut bm, tokens(0..256)); // 唯一的块,正好写满
        let table_before = seq.block_table.clone();

        seq.push_token(1); // 跨块,但没有空闲块了
        assert!(bm.extend_for_token(&mut seq).is_err());
        assert_eq!(seq.block_table, table_before, "失败不能动 block_table");
        bm.check_invariants();
    }

    // ---- 状态迁移 -----------------------------------------------------------

    #[test]
    fn take_free_block_from_empty_and_cached() {
        let mut bm = BlockManager::new(2);
        let mut a = prefill(&mut bm, tokens(0..300)); // 块0 登记,块1 尾块
        let cached_id = a.block_table[0];
        bm.release(&mut a); // free = [块1(Free), 块0(FreeCached)]

        assert!(bm.free_blocks.iter().any(|fb| matches!(fb, FreeBlock::FreeCached { id, .. } if *id == cached_id)));
        assert!(bm.free_blocks.iter().any(|fb| matches!(fb, FreeBlock::Free { .. })));

        // 取走纯空块：不影响 hash_index
        let first = bm.take_free_block();
        assert!(matches!(bm.slots[first], Some(LiveBlock::Exclusive)));
        assert_eq!(bm.hash_index.len(), 1);

        // 取走缓存空闲块：必须摘掉它的 hash_index 条目
        let second = bm.take_free_block();
        assert_eq!(second, cached_id);
        assert!(bm.hash_index.is_empty());
        assert!(matches!(bm.slots[second], Some(LiveBlock::Exclusive)));
        bm.check_invariants();
    }

    #[test]
    fn share_is_only_on_cached_live_blocks() {
        let mut bm = BlockManager::new(8);
        let a = prefill(&mut bm, tokens(0..300));
        let mut b = Sequence::new(tokens(0..300), SamplingParams::default());
        bm.reserve_prefill(&mut b).expect("空闲块不够");
        assert!(matches!(bm.slots[a.block_table[0]], Some(LiveBlock::Cached { refs: 2, .. })));
        assert!(matches!(bm.slots[a.block_table[1]], Some(LiveBlock::Exclusive)));
        bm.check_invariants();
    }

    #[test]
    fn release_steps_refs_then_frees_cached() {
        let mut bm = BlockManager::new(8);
        let mut a = prefill(&mut bm, tokens(0..300));
        let mut b = Sequence::new(tokens(0..300), SamplingParams::default());
        bm.reserve_prefill(&mut b).expect("空闲块不够");
        let shared_id = a.block_table[0];
        assert!(matches!(bm.slots[shared_id], Some(LiveBlock::Cached { refs: 2, .. })));

        bm.release(&mut b); // refs 2 -> 1,仍在使用
        assert!(matches!(bm.slots[shared_id], Some(LiveBlock::Cached { refs: 1, .. })));

        bm.release(&mut a); // refs 1 -> 0,变成空闲缓存块
        assert!(bm.slots[shared_id].is_none());
        assert!(bm.free_blocks.iter().any(|fb| matches!(fb, FreeBlock::FreeCached { id, .. } if *id == shared_id)));
        bm.check_invariants();
    }
}
