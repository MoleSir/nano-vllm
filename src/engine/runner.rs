use std::path::Path;

use luma::{Cuda, FloatDType, IndexOp, Int, Tensor, tensor::IntTensor};

use crate::{
    config::Config,
    engine::{BLOCK_SIZE, Phase, Sequence},
    models::{
        components::Attention,
        qwen2::{Qwen2Config, Qwen2ForCausalLM},
        Context,
    },
};

pub struct ModelRunner {
    pub config: Config,
    pub model_config: Qwen2Config,
    pub model: Qwen2ForCausalLM<Cuda>,
    pub kv_cache: Tensor<Cuda>,
    pub attn: Vec<Attention>,
    pub dev: Cuda,
}

impl ModelRunner {
    pub fn new(config: Config, dev: Cuda) -> anyhow::Result<Self> {
        let model_dir = Path::new(&config.model);
        let model_config = Qwen2Config::from_file(model_dir.join("config.json"))?;
        let model = Qwen2ForCausalLM::<Cuda>::from_pretrained(model_dir, &dev)?;
        let kv_cache = Tensor::zeros((0,), (&dev, FloatDType::F32))?;

        let mut runner = Self {
            config,
            model_config,
            model,
            kv_cache,
            attn: Vec::new(),
            dev,
        };
        runner.allocate_kv_cache()?;
        Ok(runner)
    }

    /// 对一个批次做一次 forward，并返回每序列采样出的一个 token。
    pub fn forward_batch(&mut self, seqs: &[Sequence], phase: Phase) -> anyhow::Result<Vec<u32>> {
        // 根据 phase 准备 input_ids/context
        let (input_ids, positions, context) = match phase {
            Phase::Prefill => self.build_prefill(seqs)?,
            Phase::Decode => self.build_decode(seqs)?,
        };

        // input_ids/positions: (num_tokens,) -> (num_seq, vocab_size)
        // 此时 kvcache 中已经被插入了新计算出来的 token，但 seq.len() 还不变
        let logits = self.model.forward_paged(&input_ids, &positions, &context, &self.attn)?;

        // (num_seq, vocab_size)：逐个 seq 用各自的采样参数采一个 token
        let vocab = *logits.dims().last().expect("logits must be non-empty");
        let flat = logits.to_vec()?;
        debug_assert_eq!(flat.len(), seqs.len() * vocab);
        let mut tokens = Vec::with_capacity(seqs.len());
        for (row, seq) in seqs.iter().enumerate() {
            tokens.push(seq.sampling_params.sample(&flat[row * vocab..(row + 1) * vocab]));
        }
        Ok(tokens)
    }

    /// 为一系列需要 prefill 的 seq 准备输入
    pub fn build_prefill(&mut self, seqs: &[Sequence]) -> anyhow::Result<(IntTensor<Cuda>, IntTensor<Cuda>, Context)> {
        let mut input_ids: Vec<i32> = vec![];
        let mut positions: Vec<i32> = vec![];
        let mut cu_seqlens_q = vec![0i32];
        let mut cu_seqlens_k = vec![0i32];
        let mut max_seqlen_q = 0usize;
        let mut max_seqlen_k = 0usize;
        let mut slot_mapping: Vec<i32> = vec![];
        let mut context_lens = None;
        let mut block_tables = None;

        // 对每个 seq
        for seq in seqs {
            /*
                A sequence is split into [cached prefix, uncached suffix].
                取出 seq 中无法命中前缀缓存的部分 token ids
                同时这部分 token ids 在原来 seq 中的逻辑索引
                例如第一个 block 命中缓存
                [432, 43, ..., 31, 89, 322]
                [256, 257, ..., 510, 511]
            */
            // input_ids 就是给模型进行 forward 的 token ids
            input_ids.extend(seq.token_ids[seq.cached_len..].iter().map(|&t| t as i32));
            positions.extend(seq.cached_len as i32..seq.len() as i32);

            // seqlen_q：总长度 - 缓存 token 数量 = 需要处理的 tokens
            let seqlen_q = seq.uncached_len();
            let seqlen_k = seq.len();

            /*
                因为 input_ids 和 positions 是扁平的，多个 seq 都是一维的，需要记录下
                cu_seqlens_x 就是用来做这个的，每次在原来的最后一个位置加上本轮的长度：
                [0, 8, 12, 21] 对应长度：[8, 4, 9] 三个 seq 的长度
            */
            cu_seqlens_q.push(cu_seqlens_q.last().unwrap() + seqlen_q as i32);
            cu_seqlens_k.push(cu_seqlens_k.last().unwrap() + seqlen_k as i32);

            max_seqlen_q = max_seqlen_q.max(seqlen_q);
            max_seqlen_k = max_seqlen_k.max(seqlen_k);

            // 对当前 seq 中，每个没有被 cache 的 block（需要计算 kv），记录下他们的 slot 位置
            for i in seq.cached_blocks()..seq.needed_blocks() {
                // 逻辑 i block，对应物理 block 的号为 seq.block_table[i]
                // seq.block_table[i] * BLOCK_SIZE 是 token 的 slot 号
                let start = seq.block_table[i] * BLOCK_SIZE;
                // 如果是最后一个，end 只有部分，否则沾满整个 block
                let end = if i != seq.needed_blocks() - 1 { start + BLOCK_SIZE } else { start + seq.last_block_tokens().len() };
                // slot mapping 插入这个 block 内部所有 token 的 slot 索引
                slot_mapping.extend(start as i32..end as i32);
            }

            // positions 和 slot_mapping，在每次新加入一个 seq，增加的长度都是没有 cache 的 tokens 数量
            // 前者是逻辑 token 索引，后者是物理索引
            // 这个新增加的长度：seq.len() - seq.cached_len 被保存到了 cu_seqlens_q 中！
        }

        assert_eq!(input_ids.len(), *cu_seqlens_q.last().unwrap() as usize);
        assert_eq!(input_ids.len(), slot_mapping.len());

        // 如果这个 seq 有前缀缓存
        if cu_seqlens_k.last().unwrap() > cu_seqlens_q.last().unwrap() {
            // 记录每个 seq 的长度
            /*
                为什么有前缀和，需要每个 seqs 的长度？怎么感觉和 cu_seqlens_k 重复啊？
                seq_lens[i] = seqlen_k[i+1] - seqlen_k[i]?

                为什么需要 block_table，因为如果没有前缀和，全部 kv 都需要计算，所以利用 input_ids 就可以得到所有 q k v 
                感觉和 slot_mapping 也有点重叠？
            */
            let seq_lens: Vec<i32> = seqs.iter().map(|s| s.len() as i32).collect();
            context_lens = Some(Tensor::<Cuda, Int>::new(seq_lens, &self.dev)?);
            block_tables = Some(self.prepare_block_tables(seqs)?);
        }

        let input_ids = Tensor::<Cuda, Int>::new(input_ids, &self.dev)?;
        let positions = Tensor::<Cuda, Int>::new(positions, &self.dev)?;
        let cu_seqlens_q = Tensor::<Cuda, Int>::new(cu_seqlens_q, &self.dev)?;
        let cu_seqlens_k = Tensor::<Cuda, Int>::new(cu_seqlens_k, &self.dev)?;
        let slot_mapping = Tensor::<Cuda, Int>::new(slot_mapping, &self.dev)?;

        let context = Context {
            phase: Phase::Prefill,
            cu_seqlens_q: Some(cu_seqlens_q),
            cu_seqlens_k: Some(cu_seqlens_k),
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            context_lens,
            block_tables,
        };

        Ok((input_ids, positions, context))
    }

    pub fn build_decode(&mut self, seqs: &[Sequence]) -> anyhow::Result<(IntTensor<Cuda>, IntTensor<Cuda>, Context)> {
        let mut input_ids: Vec<i32> = vec![];
        let mut positions: Vec<i32> = vec![];
        let mut slot_mapping: Vec<i32> = vec![];
        let mut context_lens: Vec<i32> = vec![];

        for seq in seqs {
            // decode 阶段，要处理的 ids 就是 seq 的最后一个 tokens
            input_ids.push(seq.last_token() as i32);
            // 记录这个 token 的逻辑 token 索引
            positions.push(seq.len() as i32);
            context_lens.push(seq.len() as i32);
            // 计算这个 token 的物理 token 索引。做法是找这个 token 的页表（最后一个） * BLOCK_SIZE + 最后一个块内索引
            slot_mapping.push((seq.block_table.last().unwrap() * BLOCK_SIZE + seq.last_block_tokens().len() - 1) as i32);
        }

        let input_ids = Tensor::<Cuda, Int>::new(input_ids, &self.dev)?;
        let positions = Tensor::<Cuda, Int>::new(positions, &self.dev)?;
        let slot_mapping = Tensor::<Cuda, Int>::new(slot_mapping, &self.dev)?;
        let context_lens = Tensor::<Cuda, Int>::new(context_lens, &self.dev)?;
        let block_tables = self.prepare_block_tables(seqs)?;

        let context = Context {
            phase: Phase::Decode,
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            slot_mapping,
            context_lens: Some(context_lens),
            block_tables: Some(block_tables),
        };

        Ok((input_ids, positions, context))
    }

    /// Allocate the single paged KV cache tensor and build the per-layer
    /// Shape: `(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)`.
    pub fn allocate_kv_cache(&mut self) -> anyhow::Result<()> {
        let (free, total) = self.dev.mem_get_info();
        let used = total.saturating_sub(free);
        let budget = (total as f64 * self.config.gpu_memory_utilization) as usize;
        let available = budget.saturating_sub(used);

        let head_size = self.model_config.head_size();
        let block_bytes = 2
            * self.model_config.num_hidden_layers
            * BLOCK_SIZE
            * self.model_config.num_key_value_heads
            * head_size
            * std::mem::size_of::<f32>();
        self.config.num_kvcache_blocks = available / block_bytes.max(1);

        let shape = (
            2,
            self.model_config.num_hidden_layers,
            self.config.num_kvcache_blocks,
            BLOCK_SIZE,
            self.model_config.num_key_value_heads,
            head_size,
        );
        self.kv_cache = Tensor::zeros(shape, (&self.dev, FloatDType::F32))?;

        let scale = 1.0 / (head_size as f64).sqrt();
        self.attn.clear();
        for i in 0..self.model_config.num_hidden_layers {
            let k_cache = self.kv_cache.i((0usize, i))?;
            let v_cache = self.kv_cache.i((1usize, i))?;
            self.attn.push(Attention::new(
                self.model_config.num_attention_heads,
                head_size,
                self.model_config.num_key_value_heads,
                scale,
                k_cache,
                v_cache,
            ));
        }

        Ok(())
    }

    /// 将输入 seqs 的 block table 整理为 2 维的 Tensor（num_seq, max_len），max_len 是所有 seq 中 block 最大的
    fn prepare_block_tables(&mut self, seqs: &[Sequence]) -> anyhow::Result<Tensor<Cuda, Int>> {
        let max_len = seqs.iter().map(|s| s.block_table.len()).max().unwrap_or(0);

        let mut block_tables: Vec<i32> = vec![];
        for seq in seqs {
            block_tables.extend(seq.block_table.iter().map(|&i| i as i32));
            for _ in seq.block_table.len()..max_len {
                block_tables.push(-1);
            }
        }

        let block_tables = Tensor::<Cuda, Int>::new(block_tables, &self.dev)?.reshape((seqs.len(), max_len))?;
        Ok(block_tables)
    }
}
