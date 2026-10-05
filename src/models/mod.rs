pub mod qwen2;
pub mod components;
mod sample;
use std::path::Path;
use luma::{Cuda, FloatDType, Tensor, tensor::IntTensor};
pub use sample::*;

use crate::models::components::AttnContext;

pub trait ForCausalLM : Sized {
    fn from_pretrained<P: AsRef<Path>>(dir: P, dtype: FloatDType, device: &Cuda) -> anyhow::Result<Self>;
    fn forward(&self, input_ids: &IntTensor<Cuda>, positions: &IntTensor<Cuda>, ctx: &AttnContext) -> anyhow::Result<Tensor<Cuda>>;
}