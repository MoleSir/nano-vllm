# Nano vLLM (Rust)

[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的 Rust 版：不依赖 PyTorch / flash_attn，
基于 [luma](https://github.com/MoleSir/luma) 从零实现的深度学习框架 + CUDA kernel。

## 运行

准备一个 HuggingFace Qwen2 模型目录（默认放 `./cache/Qwen2.5-0.5B-Instruct`），然后：

```bash
# F32
cargo run --release --example engine_generate

# BF16（显存约减半，推荐）
NANOVLLM_DTYPE=bf16 cargo run --release --example engine_generate

# 指定其它模型目录（支持 qwen2 / qwen3）
NANOVLLM_MODEL=./cache/Qwen3-0.6B NANOVLLM_DTYPE=bf16 cargo run --release --example engine_generate
```

## 用法

```rust
use luma::dtype::FloatDType;
use nanovllm::config::Config;
use nanovllm::engine::LLMEngine;
use nanovllm::models::SamplingParams;

let config = Config { dtype: FloatDType::BF16, ..Default::default() };
let mut engine = LLMEngine::new("./cache/Qwen2.5-0.5B-Instruct", config)?;

engine.add_request("你好，请用一句话介绍一下你自己。", SamplingParams::greedy())?;
for (id, tokens) in engine.generate()? {
    println!("{id}: {:?}", tokens);
}
```
