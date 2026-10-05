use super::Qwen2Config;
use crate::models::components::{Attention, AttnContext, AttnState};
use anyhow::Context;
use std::path::Path;
use luma::nn::{Buffer, Embedding, EmbeddingConfig, Init, Linear, LinearConfig, Module, RMSNorm, RMSNormConfig};
use luma::{
    D, Device, Float, Tensor,
    dtype::{FloatDType, IntDType},
    no_grad,
    tensor::{FloatTensor, IntTensor},
};

#[derive(Module)]
pub struct Qwen2ForCausalLM<Dev: Device> {
    pub model: Qwen2Model<Dev>,
    /// Only present when the LM head is *not* tied to the token embeddings.
    /// `#[derive(Module)]` skips a `None` field entirely, so tied checkpoints
    /// (which have no `lm_head.weight`) load cleanly with `strict = false`.
    pub lm_head: Option<Linear<Dev>>,
    #[module(skip)]
    pub config: Qwen2Config,
}

impl<Dev: Device> Qwen2ForCausalLM<Dev> {
    pub fn new(config: &Qwen2Config, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        let model = Qwen2Model::new(config, dtype, device).context("init Qwen2 model")?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(gpt2_linear(config.hidden_size, config.vocab_size, dtype, device).context("init lm_head")?)
        };
        Ok(Self { model, lm_head, config: config.clone() })
    }

    /// Load a HuggingFace Qwen2 model directory (containing `config.json` and
    /// `model.safetensors`) into `dtype` on `device`.
    ///
    /// The checkpoint is read directly through `luma::io::safetensors`; weights
    /// whose dtype differs from `dtype` are cast first, because
    /// `load_state_dict`'s `copy_` does not convert types.
    pub fn from_pretrained<P: AsRef<Path>>(dir: P, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        let config = Qwen2Config::from_file(dir.join("config.json")).context("load Qwen2 config")?;
        let mut model = Self::new(&config, dtype, device).context("build Qwen2 model")?;

        let path = dir.join("model.safetensors");
        let content = luma::io::safetensors::load_file(&path, device)
            .with_context(|| format!("load safetensors {}", path.display()))?;

        let mut state = std::collections::HashMap::with_capacity(content.tensors.len());
        for (name, tensor) in content.tensors {
            let tensor = match tensor {
                luma::DynTensor::Float(f) if f.dtype() != dtype => {
                    luma::DynTensor::Float(f.cast_float(dtype).with_context(|| format!("cast `{name}` to {dtype:?}"))?)
                }
                other => other,
            };
            state.insert(name, tensor);
        }

        // strict = false: the rope buffers are generated, not stored, and
        // checkpoints may carry extra tensors (e.g. an untied `lm_head.weight`).
        model.load_state_dict(&state, false).context("apply Qwen2 state dict")?;
        Ok(model)
    }

    pub fn forward(
        &self,
        input_ids: &IntTensor<Dev>,
        start_pos: usize,
        cache: &mut Option<Qwen2KvCache<Dev>>,
    ) -> anyhow::Result<Tensor<Dev>> {
        // (batch_size, seq_len) => (batch_size, seq_len, hidden_size)
        let hidden_states = self.model.forward(input_ids, start_pos, cache).context("model forward")?;

        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, vocab_size)
        let logits = match &self.lm_head {
            Some(lm_head) => lm_head.forward(&hidden_states).context("lm head forward")?,
            None => {
                let wte_weight = &self.model.embed_tokens.weight;
                luma::nn::functional::linear(&hidden_states, wte_weight, None).context("tied lm head forward")?
            }
        };

        Ok(logits)
    }
}

impl Qwen2ForCausalLM<luma::Cuda> {
    /// Paged forward used by the engine's model runner.
    ///
    /// - `input_ids`, `positions`: flattened tokens/rope positions for this step.
    /// - `ctx`: paged-attention metadata built by `ModelRunner`.
    /// - `attn`: per-layer paged attention components (own the K/V cache views).
    ///
    /// Returns logits `(num_seqs, vocab)`: for prefill, only each sequence's last
    /// token is projected (matching nano-vllm's `ParallelLMHead`); for decode,
    /// every row is projected.
    pub fn forward_paged(
        &self,
        input_ids: &IntTensor<luma::Cuda>,
        positions: &IntTensor<luma::Cuda>,
        ctx: &AttnContext,
        attn: &[Attention],
    ) -> anyhow::Result<Tensor<luma::Cuda>> {
        // (num_tokens,) -> (num_tokens, hidden_size)
        let hidden_states = self.model.forward_paged(input_ids, positions, ctx, attn).context("model forward_paged")?;
        
        // (num_tokens, hidden_size) -> (num_tokens, hidden_size)

        let hidden_states = if let AttnState::Prefill(state) = &ctx.state {
            // prefill 阶段的结果，取出 cu_seqlen_q
            let cu = state.cu_seqlens_q.to_vec().context("cu_seqlens_q to_vec")?;

            // cu_seqlens_q 两个元素之差是每个 seq 的输入 q 长度，w[l] - 1 表示每个 seq 的最后一个 token 位置
            let indices: Vec<i64> = cu.windows(2).map(|w| w[1] - 1).collect();
            let indices = IntTensor::<luma::Cuda>::from_slice(&indices, indices.len(), (hidden_states.device(), IntDType::I32))
                .context("last-token indices")?;
            
            /*
                hidden_states: (num_tokens, hidden_size)，表示每个 token 输出的特征向量
                indices: (num_seq)，表示每个 seq 最后一个 token 位置
                调用 index_select 就可以从所有 hidden_states 取出最后一个 token 的预测输出，用来给每个 seq 预测下一个 token
            */
            hidden_states.index_select(&indices, 0).context("select last-token hidden states")?
        } else {
            hidden_states
        };

        // (num_seq, hidden_size) -> (num_seq, vocab_size)
        let logits = match &self.lm_head {
            Some(lm_head) => lm_head.forward(&hidden_states).context("lm head forward")?,
            None => {
                let wte_weight = &self.model.embed_tokens.weight;
                luma::nn::functional::linear(&hidden_states, wte_weight, None).context("tied lm head forward")?
            }
        };

        Ok(logits)
    }
}

#[derive(Module)]
pub struct Qwen2Model<Dev: Device> {
    pub embed_tokens: Embedding<Dev>,
    pub layers: Vec<Qwen2Layer<Dev>>,
    pub norm: RMSNorm<Dev>,
    pub rope: Qwen2Rope<Dev>,
}

impl<Dev: Device> Qwen2Model<Dev> {
    pub fn new(config: &Qwen2Config, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        // Token embeddings use the GPT-2 style N(0, 0.02) init: since the LM head
        // is weight-tied, this keeps the initial logits small and the starting
        // loss close to ln(vocab) instead of being wildly over-confident.
        let embed_config = EmbeddingConfig {
            num_embeddings: config.vocab_size,
            embedding_dim: config.hidden_size,
            padding_idx: None,
            weight_init: Init::normal(0.0, 0.02),
        };
        let embed_tokens = Embedding::from_config(&embed_config, (device, dtype)).context("init embed_tokens")?;

        let mut layers = Vec::new();
        for layer_index in 0..config.num_hidden_layers {
            layers.push(Qwen2Layer::new(config, layer_index, dtype, device).with_context(|| format!("init layer {layer_index}"))?);
        }
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, device, dtype).context("init final rms norm")?;
        let rope = Qwen2Rope::new(config, dtype, device).context("init rope")?;

        Ok(Self { embed_tokens, layers, norm, rope })
    }
}

impl<Dev: Device> Qwen2Model<Dev> {
    pub fn forward(
        &self,
        input_ids: &IntTensor<Dev>,
        start_pos: usize,
        cache: &mut Option<Qwen2KvCache<Dev>>,
    ) -> anyhow::Result<Tensor<Dev>> {
        // embedding: (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        let mut hidden_states = self.embed_tokens.forward(input_ids).context("embed_tokens forward")?;
        for (i, layer) in self.layers.iter().enumerate() {
            // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, hidden_size)
            hidden_states = layer.forward(&hidden_states, start_pos, cache, &self.rope).with_context(|| format!("layer {i} forward"))?;
        }

        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, hidden_size)
        hidden_states = self.norm.forward(&hidden_states).context("final rms norm forward")?;

        Ok(hidden_states)
    }
}

#[derive(Module)]
pub struct Qwen2Layer<Dev: Device> {
    pub self_attn: Qwen2Attention<Dev>,
    pub mlp: Qwen2Mlp<Dev>,
    pub input_layernorm: RMSNorm<Dev>,
    pub post_attention_layernorm: RMSNorm<Dev>,
}

impl<Dev: Device> Qwen2Layer<Dev> {
    pub fn new(config: &Qwen2Config, layer_index: usize, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        let self_attn = Qwen2Attention::<Dev>::new(config, layer_index, dtype, device)
            .with_context(|| format!("init layer {layer_index} attention"))?;
        let mlp = Qwen2Mlp::<Dev>::new(config, dtype, device).with_context(|| format!("init layer {layer_index} mlp"))?;
        let input_layernorm = rms_norm(config.hidden_size, config.rms_norm_eps, device, dtype)
            .with_context(|| format!("init layer {layer_index} input layernorm"))?;
        let post_attention_layernorm = rms_norm(config.hidden_size, config.rms_norm_eps, device, dtype)
            .with_context(|| format!("init layer {layer_index} post-attention layernorm"))?;
        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<Dev>,
        start_pos: usize,
        kv_cache: &mut Option<Qwen2KvCache<Dev>>,
        rope: &Qwen2Rope<Dev>,
    ) -> anyhow::Result<Tensor<Dev>> {
        // Self Attention
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states).context("input layernorm forward")?;
        let hidden_states = self.self_attn.forward(&hidden_states, start_pos, kv_cache, rope).context("self attention forward")?;
        let hidden_states = residual + hidden_states;

        // Mlp
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states).context("post-attention layernorm forward")?;
        let hidden_states = self.mlp.forward(&hidden_states).context("mlp forward")?;
        let hidden_states = residual + hidden_states;

        Ok(hidden_states)
    }
}

#[derive(Module)]
pub struct Qwen2Mlp<Dev: Device> {
    pub gate_proj: Linear<Dev>,
    pub up_proj: Linear<Dev>,
    pub down_proj: Linear<Dev>,
}

impl<Dev: Device> Qwen2Mlp<Dev> {
    pub fn new(config: &Qwen2Config, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        let gate_proj = gpt2_linear(config.hidden_size, config.intermediate_size(), dtype, device).context("init mlp gate_proj")?;
        let up_proj = gpt2_linear(config.hidden_size, config.intermediate_size(), dtype, device).context("init mlp up_proj")?;
        let down_proj = gpt2_linear(config.intermediate_size(), config.hidden_size, dtype, device).context("init mlp down_proj")?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor<Dev>) -> anyhow::Result<Tensor<Dev>> {
        let gate = self.gate_proj.forward(x).context("mlp gate_proj forward")?;
        let up = self.up_proj.forward(x).context("mlp up_proj forward")?;
        let hidden = up * gate.silu().context("mlp gate silu")?;
        let out = self.down_proj.forward(&hidden).context("mlp down_proj forward")?;
        Ok(out)
    }
}

#[derive(Module)]
pub struct Qwen2Attention<Dev: Device> {
    pub q_proj: Linear<Dev>,
    pub k_proj: Linear<Dev>,
    pub v_proj: Linear<Dev>,
    pub o_proj: Linear<Dev>,

    #[module(skip)]
    pub layer_index: usize,

    #[module(skip)]
    pub head_size: usize,
    #[module(skip)]
    pub num_attention_heads: usize,
    #[module(skip)]
    pub num_key_value_heads: usize,
    #[module(skip)]
    pub hidden_size: usize,
}

impl<Dev: Device> Qwen2Attention<Dev> {
    pub fn new(config: &Qwen2Config, layer_index: usize, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        assert!(config.hidden_size % config.num_attention_heads == 0);
        assert!(config.num_attention_heads >= config.num_key_value_heads);
        assert!(config.num_attention_heads % config.num_key_value_heads == 0);

        let head_size = config.hidden_size / config.num_attention_heads;
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let hidden_size = config.hidden_size;

        // Qwen2 applies a bias to q/k/v (unlike o_proj and the MLP projections).
        let attention_bias = config.attention_bias;
        let q_proj = gpt2_linear_bias(hidden_size, head_size * num_attention_heads, attention_bias, dtype, device).context("init attention q_proj")?;
        let k_proj = gpt2_linear_bias(hidden_size, head_size * num_key_value_heads, attention_bias, dtype, device).context("init attention k_proj")?;
        let v_proj = gpt2_linear_bias(hidden_size, head_size * num_key_value_heads, attention_bias, dtype, device).context("init attention v_proj")?;
        let o_proj = gpt2_linear(head_size * num_attention_heads, hidden_size, dtype, device).context("init attention o_proj")?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            layer_index,
            head_size,
            num_attention_heads,
            num_key_value_heads,
            hidden_size,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor<Dev>,
        start_pos: usize,
        kv_cache: &mut Option<Qwen2KvCache<Dev>>,
        rope: &Qwen2Rope<Dev>,
    ) -> anyhow::Result<Tensor<Dev>> {
        // 1. forward QKV
        let q = self.q_proj.forward(x).context("attention q_proj forward")?;
        let k = self.k_proj.forward(x).context("attention k_proj forward")?;
        let v = self.v_proj.forward(x).context("attention v_proj forward")?;

        let q = self.split_heads(q, self.num_attention_heads).context("split q heads")?; // (batch, num_attention_heads, seq, head_size)
        let k = self.split_heads(k, self.num_key_value_heads).context("split k heads")?; // (batch, num_key_value_heads, seq, head_size)
        let v = self.split_heads(v, self.num_key_value_heads).context("split v heads")?; // (batch, num_key_value_heads, seq, head_size)

        // 2. Rope
        let q = rope.apply_rotary_emb(&q, start_pos).context("apply rope to q")?; // (batch, num_attention_heads, seq, head_size)
        let k = rope.apply_rotary_emb(&k, start_pos).context("apply rope to k")?; // (batch, num_key_value_heads, seq, head_size)

        // 3. kv cache
        let (k, v) = match kv_cache {
            Some(kv_cache) => {
                let (k, v) = match &kv_cache.kv_caches[self.layer_index] {
                    Some((past_k, past_v)) => {
                        // k/v: (batch, num_key_value_heads, past_seq, head_size)
                        let k = Tensor::cat(&[past_k, &k], 2).context("cat k with past k")?;
                        let v = Tensor::cat(&[past_v, &v], 2).context("cat v with past v")?;
                        (k, v)
                    }
                    None => (k, v),
                };
                // (batch, num_key_value_heads, past_seq+seq, head_size)
                kv_cache.kv_caches[self.layer_index] = Some((k.clone(), v.clone()));
                (k, v)
            }
            None => (k, v),
        };

        // 4. expand kv
        let k = self.repeat_kv(k).context("repeat k heads")?; // (batch, num_attention_heads, tseq, head_size)
        let v = self.repeat_kv(v).context("repeat v heads")?; // (batch, num_attention_heads, tseq, head_size)

        // 5. attn
        let k_t = k.transpose_last().context("transpose k")?;
        let attn_weight = q.matmul(&k_t).context("q @ k^T")?; // (batch, num_attention_heads, seq, tseq)
        let attn_weight = attn_weight / (self.head_size as f64).sqrt();

        // 6. causal mask
        let (_, _, seq, tseq) = attn_weight.dims4().context("attn_weight dims4")?;
        let q_pos = IntTensor::<Dev>::arange(start_pos as i64, (start_pos + seq) as i64, 1, (x.device(), IntDType::U32))
            .context("arange query positions")?; // (seq,)
        let k_pos = IntTensor::<Dev>::arange(0, tseq as i64, 1, (x.device(), IntDType::U32)).context("arange key positions")?; // (tseq,)
        let q_pos = q_pos.unsqueeze(1).context("unsqueeze query positions")?; // (seq, 1)
        let k_pos = k_pos.unsqueeze(0).context("unsqueeze key positions")?; // (1, tseq)
        let mask = q_pos.broadcast_lt(&k_pos).context("build causal mask")?; // (seq, tseq) Bool
        let mask = mask
            .unsqueeze(0)
            .context("unsqueeze mask batch dim")?
            .unsqueeze(0)
            .context("unsqueeze mask head dim")?
            .broadcast_as(attn_weight.shape())
            .context("broadcast causal mask")?; // (batch, heads, seq, tseq)
        let attn_weight = mask.pick_true(f64::NEG_INFINITY, &attn_weight).context("apply causal mask")?;

        // 7. softmax
        let attn_weight = luma::nn::functional::softmax(&attn_weight, D::Minus1).context("attention softmax")?;

        // 8. with v
        let attn_scores = attn_weight.matmul(&v).context("attn_weight @ v")?; // (batch, num_attention_heads, seq, head_size)

        // 9. output
        // (batch, num_attention_heads, seq, head_size) -> (batch, seq, num_attention_heads, head_size) -> (batch, seq, num_attention_heads*head_size)
        let attn_scores = attn_scores
            .transpose(1, 2)
            .context("transpose attn scores")?
            .flatten_from(2)
            .context("flatten attn scores")?
            .contiguous()
            .context("contiguous attn scores")?;
        let out = self.o_proj.forward(&attn_scores).context("attention o_proj forward")?;

        Ok(out)
    }

    fn split_heads(&self, t: Tensor<Dev>, num_heads: usize) -> anyhow::Result<Tensor<Dev>> {
        let (batch, seq, _) = t.dims3().context("split_heads dims3")?;
        let t = t.reshape((batch, seq, num_heads, self.head_size)).context("split_heads reshape")?;
        let t = t.transpose(1, 2).context("split_heads transpose")?.contiguous().context("split_heads contiguous")?;
        Ok(t)
    }

    fn repeat_kv(&self, t: Tensor<Dev>) -> anyhow::Result<Tensor<Dev>> {
        let (batch, _, tseq, head_size) = t.dims4().context("repeat_kv dims4")?;
        let num_repeat = self.num_attention_heads / self.num_key_value_heads;
        let t = t
            .unsqueeze(2)
            .context("repeat_kv unsqueeze")?
            .repeat_dim(2, num_repeat)
            .context("repeat_kv repeat_dim")?
            .reshape((batch, self.num_attention_heads, tseq, head_size))
            .context("repeat_kv reshape")?;
        Ok(t)
    }
}

// ============================================================================================ //
//     Paged forward (CUDA only) — mirrors nanovllm/layers/attention.py
// ============================================================================================ //

impl Qwen2Attention<luma::Cuda> {
    /// `x`: `(num_tokens, hidden)`, `positions`: `(num_tokens,)`.
    pub fn forward_paged(
        &self,
        x: &Tensor<luma::Cuda>,
        positions: &IntTensor<luma::Cuda>,
        ctx: &AttnContext,
        attn: &Attention,
        rope: &Qwen2Rope<luma::Cuda>,
    ) -> anyhow::Result<Tensor<luma::Cuda>> {
        let num_tokens = x.dims()[0];

        // (num_tokens, hidden_size) => (num_tokens, q_num_heads, head_size)
        let q = self.q_proj.forward(x).context("attention q_proj forward")?;
        // (num_tokens, hidden_size) => (num_tokens, kv_num_heads, head_size)
        let k = self.k_proj.forward(x).context("attention k_proj forward")?;
        let v = self.v_proj.forward(x).context("attention v_proj forward")?;

        let q = q.reshape((num_tokens, self.num_attention_heads, self.head_size)).context("q reshape")?;
        let k = k.reshape((num_tokens, self.num_key_value_heads, self.head_size)).context("k reshape")?;
        let v = v.reshape((num_tokens, self.num_key_value_heads, self.head_size)).context("v reshape")?;

        // (num_tokens, q_num_heads, head_size) / (num_tokens, kv_num_heads, head_size)
        let q = rope.apply_rotary_emb_positions(&q, positions).context("rope q")?;
        let k = rope.apply_rotary_emb_positions(&k, positions).context("rope k")?;

        let o = attn.forward(&q, &k, &v, ctx).context("paged attention forward")?;
        let o = o.reshape((num_tokens, self.num_attention_heads * self.head_size)).context("attn flatten")?;
        let out = self.o_proj.forward(&o).context("attention o_proj forward")?;
        Ok(out)
    }
}

impl Qwen2Layer<luma::Cuda> {
    pub fn forward_paged(
        &self,
        hidden_states: &Tensor<luma::Cuda>,
        positions: &IntTensor<luma::Cuda>,
        ctx: &AttnContext,
        attn: &Attention,
        rope: &Qwen2Rope<luma::Cuda>,
    ) -> anyhow::Result<Tensor<luma::Cuda>> {
        // (num_tokens, hidden_size)
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states).context("input layernorm forward")?;
        let hidden_states = self.self_attn.forward_paged(&hidden_states, positions, ctx, attn, rope).context("self attention forward")?;
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states).context("post-attention layernorm forward")?;
        let hidden_states = self.mlp.forward(&hidden_states).context("mlp forward")?;
        let hidden_states = residual + hidden_states;

        Ok(hidden_states)
    }
}

impl Qwen2Model<luma::Cuda> {
    pub fn forward_paged(
        &self,
        input_ids: &IntTensor<luma::Cuda>,
        positions: &IntTensor<luma::Cuda>,
        ctx: &AttnContext,
        attn: &[Attention],
    ) -> anyhow::Result<Tensor<luma::Cuda>> {
        // (num_tokens,) -> (num_tokens, hidden_size)
        let mut hidden_states = self.embed_tokens.forward(input_ids).context("embed_tokens forward")?;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward_paged(&hidden_states, positions, ctx, &attn[i], &self.rope)
                .with_context(|| format!("layer {i} forward_paged"))?;
        }
        let hidden_states = self.norm.forward(&hidden_states).context("final rms norm forward")?;
        Ok(hidden_states)
    }
}

#[derive(Module)]
pub struct Qwen2Rope<Dev: Device> {
    pub sin: Buffer<Dev, Float>,
    pub cos: Buffer<Dev, Float>,
    #[module(skip)]
    pub max_position_embeddings: usize,
}

impl<Dev: Device> Qwen2Rope<Dev> {
    pub fn new(config: &Qwen2Config, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        no_grad!();
        let head_size = config.hidden_size / config.num_attention_heads;
        let freqs = (0..head_size).step_by(2).map(|i| 1.0 / (config.rope_theta.powf(i as f64 / head_size as f64))).collect::<Vec<_>>();
        let freqs = FloatTensor::<Dev>::from_slice(&freqs, (1, freqs.len()), (device, dtype)).context("rope freqs from_slice")?;
        let positions = IntTensor::<Dev>::arange(0, config.max_position_embeddings as i64, 1, (device, IntDType::U32))
            .context("rope positions arange")?;
        // (max_positions, 1)
        let positions = positions.cast(dtype).context("rope positions cast")?.unsqueeze(1).context("rope positions unsqueeze")?;
        // (max_positions, head_size/2)
        let thetas = positions.matmul(&freqs).context("rope positions @ freqs")?;

        // (max_positions, head_size/2)
        let sin = thetas.sin().context("rope sin")?;
        let cos = thetas.cos().context("rope cos")?;

        // (max_positions, head_size)
        let sin = Tensor::cat(&[&sin, &sin], 1).context("rope cat sin")?;
        let cos = Tensor::cat(&[&cos, &cos], 1).context("rope cat cos")?;

        Ok(Self {
            sin: Buffer::<Dev, Float>::new(sin),
            cos: Buffer::<Dev, Float>::new(cos),
            max_position_embeddings: config.max_position_embeddings,
        })
    }

    pub fn apply_rotary_emb(&self, x: &Tensor<Dev>, start_pos: usize) -> anyhow::Result<Tensor<Dev>> {
        let (_batch_size, _n_heads, seq_len, _head_size) = x.dims4().context("apply_rotary_emb dims4")?;
        assert!(start_pos + seq_len <= self.max_position_embeddings);
        let sin = self.sin.narrow(0, start_pos, seq_len).context("rope narrow sin")?; // (seq_len, head_size)
        let cos = self.cos.narrow(0, start_pos, seq_len).context("rope narrow cos")?; // (seq_len, head_size)
        let sin = sin.unsqueeze(0).context("rope sin unsqueeze batch")?.unsqueeze(0).context("rope sin unsqueeze head")?; // (1, 1, seq_len, head_size)
        let cos = cos.unsqueeze(0).context("rope cos unsqueeze batch")?.unsqueeze(0).context("rope cos unsqueeze head")?; // (1, 1, seq_len, head_size)

        // [x1, x2] -> [-x2, x1]
        let x_rotated = self.rotate_half(x).context("rotate_half")?;
        let x_out = x.broadcast_mul(&cos).context("rope x * cos")? + x_rotated.broadcast_mul(&sin).context("rope x_rotated * sin")?;
        Ok(x_out)
    }

    /// Per-token RoPE for the paged path.
    /// `x`: `(num_tokens, num_heads, head_dim)`, `positions`: `(num_tokens,)`.
    pub fn apply_rotary_emb_positions(&self, x: &Tensor<Dev>, positions: &IntTensor<Dev>) -> anyhow::Result<Tensor<Dev>> {
        let (num_tokens, _n_heads, head_size) = x.dims3().context("apply_rotary_emb_positions dims3")?;
        let sin = self.sin.index_select(positions, 0).context("rope gather sin")?; // (num_tokens, head_size)
        let cos = self.cos.index_select(positions, 0).context("rope gather cos")?; // (num_tokens, head_size)
        let sin = sin.reshape((num_tokens, 1, head_size)).context("rope sin reshape")?;
        let cos = cos.reshape((num_tokens, 1, head_size)).context("rope cos reshape")?;

        let x_rotated = self.rotate_half_last(x).context("rotate_half_last")?;
        let x_out = x.broadcast_mul(&cos).context("rope x * cos")? + x_rotated.broadcast_mul(&sin).context("rope x_rotated * sin")?;
        Ok(x_out)
    }

    /// `[x1, x2] -> [-x2, x1]` along the last dim, for any rank.
    fn rotate_half_last(&self, x: &Tensor<Dev>) -> anyhow::Result<Tensor<Dev>> {
        let head_size = *x.dims().last().context("rotate_half_last empty dims")?;
        let half_size = head_size / 2;
        let x1 = x.narrow(D::Minus1, 0, half_size).context("rotate_half_last narrow x1")?;
        let x2 = x.narrow(D::Minus1, half_size, half_size).context("rotate_half_last narrow x2")?;
        let neg_x2 = x2.neg().context("rotate_half_last neg x2")?;
        Ok(Tensor::cat(&[&neg_x2, &x1], D::Minus1).context("rotate_half_last cat")?)
    }

    fn rotate_half(&self, x: &Tensor<Dev>) -> anyhow::Result<Tensor<Dev>> {
        let (_batch_size, _n_heads, _seq_len, head_size) = x.dims4().context("rotate_half dims4")?;
        let half_size = head_size / 2;
        // (batch, n_heads, seq_len, head_size/2)
        let x1 = x.narrow(D::Minus1, 0, half_size).context("rotate_half narrow x1")?;
        let x2 = x.narrow(D::Minus1, half_size, half_size).context("rotate_half narrow x2")?;
        let neg_x2 = x2.neg().context("rotate_half neg x2")?;
        // (batch, n_heads, seq_len, head_size)
        let x_rotated = Tensor::cat(&[&neg_x2, &x1], D::Minus1).context("rotate_half cat")?;
        Ok(x_rotated)
    }
}

pub struct Qwen2KvCache<Dev: Device> {
    kv_caches: Vec<Option<(Tensor<Dev>, Tensor<Dev>)>>,
}

impl<Dev: Device> Qwen2KvCache<Dev> {
    pub fn new(config: &Qwen2Config) -> Self {
        Self { kv_caches: vec![None; config.num_hidden_layers] }
    }
}

/// RMSNorm with the reference Qwen2 eps (1e-6 by default), wired to config.
fn rms_norm<Dev: Device>(normalized_shape: usize, eps: f64, device: &Dev, dtype: FloatDType) -> anyhow::Result<RMSNorm<Dev>> {
    let config = RMSNormConfig { normalized_shape, eps, weight_init: Init::Ones };
    Ok(RMSNorm::from_config(&config, (device, dtype)).context("init rms norm")?)
}

/// GPT-2 / Qwen2 style linear layer: weights `N(0, 0.02)`, no bias.
///
/// The reference Qwen2 initialises every dense layer this way; using the
/// Kaiming default here makes the initial activations and gradients too large
/// and the loss diverges to NaN within a handful of steps.
fn gpt2_linear<Dev: Device>(in_features: usize, out_features: usize, dtype: FloatDType, device: &Dev) -> anyhow::Result<Linear<Dev>> {
    gpt2_linear_bias(in_features, out_features, false, dtype, device)
}

/// GPT-2 / Qwen2 style linear layer, optionally with a zero-initialised bias
/// (Qwen2 attention q/k/v projections carry a bias; o_proj and the MLP do not).
fn gpt2_linear_bias<Dev: Device>(
    in_features: usize,
    out_features: usize,
    bias: bool,
    dtype: FloatDType,
    device: &Dev,
) -> anyhow::Result<Linear<Dev>> {
    let config = LinearConfig { in_features, out_features, bias, weight_init: Init::normal(0.0, 0.02), bias_init: Init::zeros() };
    Ok(Linear::from_config(&config, (device, dtype)).context("init gpt2 linear")?)
}