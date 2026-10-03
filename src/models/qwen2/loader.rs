use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use luma::dtype::FloatDType;
use luma::nn::Module;
use luma::tensor::FloatTensor;
use luma::{DType, Device, DynTensor};
use serde_json::Value;

use super::{Qwen2Config, Qwen2ForCausalLM};

/// Read a safetensors checkpoint into an F32 state dict.
///
/// `luma` only supports F32/F64 floats, while the Qwen2.5 checkpoints ship as
/// `BF16`. BF16 is losslessly widened to F32 here (`(u16 as u32) << 16`), so a
/// BF16 file can be loaded directly without an offline conversion step.
pub fn load_state_dict_f32<Dev: Device>(path: &Path, device: &Dev) -> anyhow::Result<HashMap<String, DynTensor<Dev>>> {
    let bytes = std::fs::read(path).with_context(|| format!("read safetensors file {}", path.display()))?;
    if bytes.len() < 8 {
        anyhow::bail!("safetensors file too small: {}", path.display());
    }

    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_len;
    if header_end > bytes.len() {
        anyhow::bail!("invalid safetensors header length in {}", path.display());
    }

    let header: HashMap<String, Value> = serde_json::from_slice(&bytes[8..header_end]).context("parse safetensors header")?;
    let data_start = header_end;

    let mut state = HashMap::with_capacity(header.len());
    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }

        let dtype = info.get("dtype").and_then(Value::as_str).ok_or_else(|| anyhow::anyhow!("tensor `{name}` is missing dtype"))?;
        let shape: Vec<usize> = info
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("tensor `{name}` is missing shape"))?
            .iter()
            .map(|dim| dim.as_u64().unwrap_or_default() as usize)
            .collect();
        let offsets = info
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("tensor `{name}` is missing data_offsets"))?;
        let start = offsets[0].as_u64().unwrap_or_default() as usize;
        let end = offsets[1].as_u64().unwrap_or_default() as usize;
        let raw = &bytes[data_start + start..data_start + end];

        let tensor = match dtype {
            "BF16" => {
                if raw.len() % 2 != 0 {
                    anyhow::bail!("BF16 tensor `{name}` has an odd byte length");
                }
                let mut data = Vec::with_capacity(raw.len() / 2);
                for chunk in raw.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                    data.push(f32::from_bits(bits << 16));
                }
                DynTensor::Float(FloatTensor::<Dev>::from_vec_f32(data, shape, device)?)
            }
            "F32" => DynTensor::from_bytes(raw, DType::F32, shape, device)?,
            "F64" => DynTensor::from_bytes(raw, DType::F64, shape, device)?,
            other => anyhow::bail!("unsupported safetensors dtype `{other}` for tensor `{name}`"),
        };

        state.insert(name, tensor);
    }

    Ok(state)
}

impl<Dev: Device> Qwen2ForCausalLM<Dev> {
    /// Load a HuggingFace Qwen2 model directory (containing `config.json` and
    /// `model.safetensors`) and build it in F32 on `device`.
    pub fn from_pretrained<P: AsRef<Path>>(dir: P, device: &Dev) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        let config = Qwen2Config::from_file(dir.join("config.json")).context("load Qwen2 config")?;
        let mut model = Self::new(&config, FloatDType::F32, device).context("build Qwen2 model")?;

        // strict = false: the rope `sin`/`cos` buffers are generated, not stored,
        // and checkpoints may carry extra tensors (e.g. an untied `lm_head.weight`).
        let state = load_state_dict_f32(&dir.join("model.safetensors"), device).context("load Qwen2 weights")?;
        model.load_state_dict(&state, false).context("apply Qwen2 state dict")?;

        Ok(model)
    }
}
