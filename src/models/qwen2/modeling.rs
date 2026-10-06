use super::Qwen2Config;
use crate::models::ForCausalLM;
use crate::models::components::{Attention, AttnContext, AttnState, Rope};
use anyhow::Context;
use std::path::Path;
use luma::nn::{Embedding, Init, Linear, LinearConfig, Module, RMSNorm, RMSNormConfig};
use luma::{
    Cuda, Device, IndexOp, Tensor,
    dtype::{FloatDType, IntDType},
    tensor::IntTensor,
};

// ===================================================================================== //
//                                    Qwen2ForCausalLM
// ===================================================================================== //

#[derive(Module)]
pub struct Qwen2ForCausalLM<Dev: Device> {
    pub model: Qwen2Model<Dev>,
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
            Some(linear(config.hidden_size, config.vocab_size, false, dtype, device).context("init lm_head")?)
        };
        Ok(Self { model, lm_head, config: config.clone() })
    }
}

impl Qwen2ForCausalLM<Cuda> {
    pub fn from_pretrained<P: AsRef<Path>>(dir: P, dtype: FloatDType, device: &Cuda) -> anyhow::Result<Self> {
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

    /// - `input_ids`, `positions`: flattened tokens/rope positions for this step.
    /// - `ctx`: paged-attention metadata built by `ModelRunner`.
    /// - `attn`: per-layer paged attention components (own the K/V cache views).
    pub fn forward(
        &self,
        input_ids: &IntTensor<Cuda>,
        positions: &IntTensor<Cuda>,
        ctx: &AttnContext,
    ) -> anyhow::Result<Tensor<Cuda>> {
        // (num_tokens,) -> (num_tokens, hidden_size)
        let hidden_states = self.model.forward(input_ids, positions, ctx).context("model forward")?;
        
        // (num_tokens, hidden_size) -> (num_tokens, hidden_size)
        let hidden_states = if let AttnState::Prefill(state) = &ctx.state {
            // prefill 阶段的结果，取出 cu_seqlen_q
            let cu = state.cu_seqlens_q.to_vec().context("cu_seqlens_q to_vec")?;

            // cu_seqlens_q 两个元素之差是每个 seq 的输入 q 长度，w[l] - 1 表示每个 seq 的最后一个 token 位置
            let indices: Vec<i64> = cu.windows(2).map(|w| w[1] - 1).collect();
            let indices = IntTensor::<Cuda>::from_slice(&indices, indices.len(), (hidden_states.device(), IntDType::I32))
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

// ===================================================================================== //
//                                    Qwen2Model
// ===================================================================================== //

#[derive(Module)]
pub struct Qwen2Model<Dev: Device> {
    pub embed_tokens: Embedding<Dev>,
    pub layers: Vec<Qwen2Layer<Dev>>,
    pub norm: RMSNorm<Dev>,
    pub rope: Rope<Dev>,
}

impl<Dev: Device> Qwen2Model<Dev> {
    pub fn new(config: &Qwen2Config, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, (device, dtype)).context("init embed_tokens")?;
        let mut layers = Vec::new();
        for layer_index in 0..config.num_hidden_layers {
            layers.push(Qwen2Layer::new(config, layer_index, dtype, device).with_context(|| format!("init layer {layer_index}"))?);
        }
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, device, dtype).context("init final rms norm")?;
        let rope = Rope::new(config.head_size(), config.max_position_embeddings, config.rope_theta, dtype, device).context("init rope")?;

        Ok(Self { embed_tokens, layers, norm, rope })
    }
}

impl Qwen2Model<Cuda> {
    pub fn forward(
        &self,
        input_ids: &IntTensor<Cuda>,
        positions: &IntTensor<Cuda>,
        ctx: &AttnContext,
    ) -> anyhow::Result<Tensor<Cuda>> {
        // (num_tokens,) -> (num_tokens, hidden_size)
        let mut hidden_states = self.embed_tokens.forward(input_ids).context("embed_tokens forward")?;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, ctx, &self.rope).with_context(|| format!("layer {i} forward"))?;
        }
        let hidden_states = self.norm.forward(&hidden_states).context("final rms norm forward")?;
        Ok(hidden_states)
    }
}

// ===================================================================================== //
//                                    Qwen2Model
// ===================================================================================== //

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
}

impl Qwen2Layer<Cuda> {
    pub fn forward(
        &self,
        hidden_states: &Tensor<Cuda>,
        positions: &IntTensor<Cuda>,
        ctx: &AttnContext,
        rope: &Rope<Cuda>,
    ) -> anyhow::Result<Tensor<Cuda>> {
        // (num_tokens, hidden_size)
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states).context("input layernorm forward")?;
        let hidden_states = self.self_attn.forward(&hidden_states, positions, ctx, rope).context("self attention forward")?;
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states).context("post-attention layernorm forward")?;
        let hidden_states = self.mlp.forward(&hidden_states).context("mlp forward")?;
        let hidden_states = residual + hidden_states;

        Ok(hidden_states)
    }
}

// ===================================================================================== //
//                                    Qwen2Mlp
// ===================================================================================== //

#[derive(Module)]
pub struct Qwen2Mlp<Dev: Device> {
    pub gate_proj: Linear<Dev>,
    pub up_proj: Linear<Dev>,
    pub down_proj: Linear<Dev>,
}

impl<Dev: Device> Qwen2Mlp<Dev> {
    pub fn new(config: &Qwen2Config, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        let gate_proj = linear(config.hidden_size, config.intermediate_size(), false, dtype, device).context("init mlp gate_proj")?;
        let up_proj = linear(config.hidden_size, config.intermediate_size(), false, dtype, device).context("init mlp up_proj")?;
        let down_proj = linear(config.intermediate_size(), config.hidden_size, false, dtype, device).context("init mlp down_proj")?;
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

// ===================================================================================== //
//                                    Qwen2Attention
// ===================================================================================== //

#[derive(Module)]
pub struct Qwen2Attention<Dev: Device> {
    pub q_proj: Linear<Dev>,
    pub k_proj: Linear<Dev>,
    pub v_proj: Linear<Dev>,
    pub o_proj: Linear<Dev>,

    #[module(skip)]
    pub attn: Attention,

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
        let q_proj = linear(hidden_size, head_size * num_attention_heads, attention_bias, dtype, device).context("init attention q_proj")?;
        let k_proj = linear(hidden_size, head_size * num_key_value_heads, attention_bias, dtype, device).context("init attention k_proj")?;
        let v_proj = linear(hidden_size, head_size * num_key_value_heads, attention_bias, dtype, device).context("init attention v_proj")?;
        let o_proj = linear(head_size * num_attention_heads, hidden_size, false, dtype, device).context("init attention o_proj")?;

        let scale = 1.0 / (head_size as f64).sqrt();
        let attn = Attention::new(num_attention_heads, head_size, num_key_value_heads, scale);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            attn,
            layer_index,
            head_size,
            num_attention_heads,
            num_key_value_heads,
            hidden_size,
        })
    }
}

impl Qwen2Attention<Cuda> {
    /// `x`: `(num_tokens, hidden)`, `positions`: `(num_tokens,)`.
    pub fn forward(
        &self,
        x: &Tensor<Cuda>,
        positions: &IntTensor<Cuda>,
        ctx: &AttnContext,
        rope: &Rope<Cuda>,
    ) -> anyhow::Result<Tensor<Cuda>> {
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

        let o = self.attn.forward(&q, &k, &v, ctx).context("paged attention forward")?;
        let o = o.reshape((num_tokens, self.num_attention_heads * self.head_size)).context("attn flatten")?;
        let out = self.o_proj.forward(&o).context("attention o_proj forward")?;
        Ok(out)
    }
}


/// RMSNorm with the reference Qwen2 eps (1e-6 by default), wired to config.
fn rms_norm<Dev: Device>(normalized_shape: usize, eps: f64, device: &Dev, dtype: FloatDType) -> anyhow::Result<RMSNorm<Dev>> {
    let config = RMSNormConfig { normalized_shape, eps, weight_init: Init::Ones };
    Ok(RMSNorm::from_config(&config, (device, dtype)).context("init rms norm")?)
}

fn linear<Dev: Device>(
    in_features: usize,
    out_features: usize,
    bias: bool,
    dtype: FloatDType,
    device: &Dev,
) -> anyhow::Result<Linear<Dev>> {
    let config = LinearConfig { in_features, out_features, bias, weight_init: Init::normal(0.0, 0.02), bias_init: Init::zeros() };
    Ok(Linear::from_config(&config, (device, dtype)).context("init gpt2 linear")?)
}

// ===================================================================================== //
//                                    ForCausalLM
// ===================================================================================== //

impl ForCausalLM for Qwen2ForCausalLM<Cuda> {
    fn forward(
        &self,
        input_ids: &IntTensor<Cuda>,
        positions: &IntTensor<Cuda>,
        ctx: &AttnContext,
    ) -> anyhow::Result<Tensor<Cuda>> {
        Qwen2ForCausalLM::forward(self, input_ids, positions, ctx)
    }

    fn register_kvcache(&mut self, kv_cache: &Tensor<Cuda>) -> anyhow::Result<()> {
        // kv_cache: (2, num_layers, num_blocks, block_size, num_kv_heads, head_size)
        for (layer_index, layer) in self.model.layers.iter_mut().enumerate() {
            let k_cache = kv_cache.i((0usize, layer_index)).context("k_cache view")?;
            let v_cache = kv_cache.i((1usize, layer_index)).context("v_cache view")?;
            layer.self_attn.attn.k_cache = Some(k_cache);
            layer.self_attn.attn.v_cache = Some(v_cache);
        }
        Ok(())
    }

    fn num_hidden_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn num_key_value_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    fn head_size(&self) -> usize {
        self.config.head_size()
    }

    fn device(&self) -> &Cuda {
        self.model.embed_tokens.weight.device()
    }
}