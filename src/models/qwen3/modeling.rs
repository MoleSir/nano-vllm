use super::Qwen3Config;
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
//                                    Qwen3ForCausalLM
// ===================================================================================== //

#[derive(Module)]
pub struct Qwen3ForCausalLM<Dev: Device> {
    pub model: Qwen3Model<Dev>,
    pub lm_head: Option<Linear<Dev>>,
    #[module(skip)]
    pub config: Qwen3Config,
}

impl<Dev: Device> Qwen3ForCausalLM<Dev> {
    pub fn new(config: &Qwen3Config, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        let model = Qwen3Model::new(config, dtype, device).context("init Qwen3 model")?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(linear(config.hidden_size, config.vocab_size, false, dtype, device).context("init lm_head")?)
        };
        Ok(Self { model, lm_head, config: config.clone() })
    }
}

impl Qwen3ForCausalLM<Cuda> {
    pub fn from_pretrained<P: AsRef<Path>>(dir: P, dtype: FloatDType, device: &Cuda) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        let config = Qwen3Config::from_file(dir.join("config.json")).context("load Qwen3 config")?;
        let mut model = Self::new(&config, dtype, device).context("build Qwen3 model")?;

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

        // strict = false: the rope buffers are generated, not stored, and the
        // checkpoint carries an `lm_head.weight` that is tied and thus unused.
        model.load_state_dict(&state, false).context("apply Qwen3 state dict")?;
        Ok(model)
    }

    /// - `input_ids`, `positions`: flattened tokens/rope positions for this step.
    /// - `ctx`: paged-attention metadata built by `ModelRunner`.
    pub fn forward(
        &self,
        input_ids: &IntTensor<Cuda>,
        positions: &IntTensor<Cuda>,
        ctx: &AttnContext,
    ) -> anyhow::Result<Tensor<Cuda>> {
        // (num_tokens,) -> (num_tokens, hidden_size)
        let hidden_states = self.model.forward(input_ids, positions, ctx).context("model forward")?;

        // prefill 只取每个 seq 的最后一个 token 预测下一个 token。
        let hidden_states = if let AttnState::Prefill(state) = &ctx.state {
            let cu = state.cu_seqlens_q.to_vec().context("cu_seqlens_q to_vec")?;
            let indices: Vec<i64> = cu.windows(2).map(|w| w[1] - 1).collect();
            let indices = IntTensor::<Cuda>::from_slice(&indices, indices.len(), (hidden_states.device(), IntDType::I32))
                .context("last-token indices")?;
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
//                                    Qwen3Model
// ===================================================================================== //

#[derive(Module)]
pub struct Qwen3Model<Dev: Device> {
    pub embed_tokens: Embedding<Dev>,
    pub layers: Vec<Qwen3Layer<Dev>>,
    pub norm: RMSNorm<Dev>,
    pub rope: Rope<Dev>,
}

impl<Dev: Device> Qwen3Model<Dev> {
    pub fn new(config: &Qwen3Config, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, (device, dtype)).context("init embed_tokens")?;
        let mut layers = Vec::new();
        for layer_index in 0..config.num_hidden_layers {
            layers.push(Qwen3Layer::new(config, layer_index, dtype, device).with_context(|| format!("init layer {layer_index}"))?);
        }
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, device, dtype).context("init final rms norm")?;
        let rope = Rope::new(config.head_size(), config.max_position_embeddings, config.rope_theta, dtype, device).context("init rope")?;

        Ok(Self { embed_tokens, layers, norm, rope })
    }
}

impl Qwen3Model<Cuda> {
    pub fn forward(
        &self,
        input_ids: &IntTensor<Cuda>,
        positions: &IntTensor<Cuda>,
        ctx: &AttnContext,
    ) -> anyhow::Result<Tensor<Cuda>> {
        let mut hidden_states = self.embed_tokens.forward(input_ids).context("embed_tokens forward")?;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, ctx, &self.rope).with_context(|| format!("layer {i} forward"))?;
        }
        let hidden_states = self.norm.forward(&hidden_states).context("final rms norm forward")?;
        Ok(hidden_states)
    }
}

// ===================================================================================== //
//                                    Qwen3Layer
// ===================================================================================== //

#[derive(Module)]
pub struct Qwen3Layer<Dev: Device> {
    pub self_attn: Qwen3Attention<Dev>,
    pub mlp: Qwen3Mlp<Dev>,
    pub input_layernorm: RMSNorm<Dev>,
    pub post_attention_layernorm: RMSNorm<Dev>,
}

impl<Dev: Device> Qwen3Layer<Dev> {
    pub fn new(config: &Qwen3Config, layer_index: usize, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        let self_attn = Qwen3Attention::<Dev>::new(config, layer_index, dtype, device)
            .with_context(|| format!("init layer {layer_index} attention"))?;
        let mlp = Qwen3Mlp::<Dev>::new(config, dtype, device).with_context(|| format!("init layer {layer_index} mlp"))?;
        let input_layernorm = rms_norm(config.hidden_size, config.rms_norm_eps, device, dtype)
            .with_context(|| format!("init layer {layer_index} input layernorm"))?;
        let post_attention_layernorm = rms_norm(config.hidden_size, config.rms_norm_eps, device, dtype)
            .with_context(|| format!("init layer {layer_index} post-attention layernorm"))?;
        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm })
    }
}

impl Qwen3Layer<Cuda> {
    pub fn forward(
        &self,
        hidden_states: &Tensor<Cuda>,
        positions: &IntTensor<Cuda>,
        ctx: &AttnContext,
        rope: &Rope<Cuda>,
    ) -> anyhow::Result<Tensor<Cuda>> {
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
//                                    Qwen3Mlp
// ===================================================================================== //

#[derive(Module)]
pub struct Qwen3Mlp<Dev: Device> {
    pub gate_proj: Linear<Dev>,
    pub up_proj: Linear<Dev>,
    pub down_proj: Linear<Dev>,
}

impl<Dev: Device> Qwen3Mlp<Dev> {
    pub fn new(config: &Qwen3Config, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        let gate_proj = linear(config.hidden_size, config.intermediate_size, false, dtype, device).context("init mlp gate_proj")?;
        let up_proj = linear(config.hidden_size, config.intermediate_size, false, dtype, device).context("init mlp up_proj")?;
        let down_proj = linear(config.intermediate_size, config.hidden_size, false, dtype, device).context("init mlp down_proj")?;
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
//                                    Qwen3Attention
// ===================================================================================== //

/// Qwen3 attention：在 Qwen2 基础上加了 **per-head QK RMSNorm**（`q_norm`/`k_norm`），
/// 且 `head_size` 来自 `config.head_dim`（不是 `hidden_size / num_heads`）。
#[derive(Module)]
pub struct Qwen3Attention<Dev: Device> {
    pub q_proj: Linear<Dev>,
    pub k_proj: Linear<Dev>,
    pub v_proj: Linear<Dev>,
    pub o_proj: Linear<Dev>,
    pub q_norm: RMSNorm<Dev>,
    pub k_norm: RMSNorm<Dev>,

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

impl<Dev: Device> Qwen3Attention<Dev> {
    pub fn new(config: &Qwen3Config, layer_index: usize, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        assert!(config.num_attention_heads >= config.num_key_value_heads);
        assert!(config.num_attention_heads % config.num_key_value_heads == 0);

        let head_size = config.head_dim;
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let hidden_size = config.hidden_size;

        let attention_bias = config.attention_bias;
        let q_proj = linear(hidden_size, head_size * num_attention_heads, attention_bias, dtype, device).context("init attention q_proj")?;
        let k_proj = linear(hidden_size, head_size * num_key_value_heads, attention_bias, dtype, device).context("init attention k_proj")?;
        let v_proj = linear(hidden_size, head_size * num_key_value_heads, attention_bias, dtype, device).context("init attention v_proj")?;
        let o_proj = linear(head_size * num_attention_heads, hidden_size, false, dtype, device).context("init attention o_proj")?;

        let q_norm = rms_norm(head_size, config.rms_norm_eps, device, dtype).context("init attention q_norm")?;
        let k_norm = rms_norm(head_size, config.rms_norm_eps, device, dtype).context("init attention k_norm")?;

        let scale = 1.0 / (head_size as f64).sqrt();
        let attn = Attention::new(num_attention_heads, head_size, num_key_value_heads, scale);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            attn,
            layer_index,
            head_size,
            num_attention_heads,
            num_key_value_heads,
            hidden_size,
        })
    }
}

impl Qwen3Attention<Cuda> {
    /// `x`: `(num_tokens, hidden)`, `positions`: `(num_tokens,)`.
    pub fn forward(
        &self,
        x: &Tensor<Cuda>,
        positions: &IntTensor<Cuda>,
        ctx: &AttnContext,
        rope: &Rope<Cuda>,
    ) -> anyhow::Result<Tensor<Cuda>> {
        let num_tokens = x.dims()[0];

        let q = self.q_proj.forward(x).context("attention q_proj forward")?;
        let k = self.k_proj.forward(x).context("attention k_proj forward")?;
        let v = self.v_proj.forward(x).context("attention v_proj forward")?;

        // (num_tokens, q_num_heads, head_size) / (num_tokens, kv_num_heads, head_size)
        let q = q.reshape((num_tokens, self.num_attention_heads, self.head_size)).context("q reshape")?;
        let k = k.reshape((num_tokens, self.num_key_value_heads, self.head_size)).context("k reshape")?;
        let v = v.reshape((num_tokens, self.num_key_value_heads, self.head_size)).context("v reshape")?;

        // Qwen3: per-head QK RMSNorm，在 RoPE 之前。
        let q = self.q_norm.forward(&q).context("q_norm forward")?;
        let k = self.k_norm.forward(&k).context("k_norm forward")?;

        let q = rope.apply_rotary_emb_positions(&q, positions).context("rope q")?;
        let k = rope.apply_rotary_emb_positions(&k, positions).context("rope k")?;

        let o = self.attn.forward(&q, &k, &v, ctx).context("paged attention forward")?;
        let o = o.reshape((num_tokens, self.num_attention_heads * self.head_size)).context("attn flatten")?;
        let out = self.o_proj.forward(&o).context("attention o_proj forward")?;
        Ok(out)
    }
}

// ===================================================================================== //
//                                    helpers
// ===================================================================================== //

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
    Ok(Linear::from_config(&config, (device, dtype)).context("init qwen3 linear")?)
}

// ===================================================================================== //
//                                    ForCausalLM
// ===================================================================================== //

impl ForCausalLM for Qwen3ForCausalLM<Cuda> {
    fn forward(
        &self,
        input_ids: &IntTensor<Cuda>,
        positions: &IntTensor<Cuda>,
        ctx: &AttnContext,
    ) -> anyhow::Result<Tensor<Cuda>> {
        Qwen3ForCausalLM::forward(self, input_ids, positions, ctx)
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
