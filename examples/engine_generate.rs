use std::time::Instant;

use anyhow::Context;
use nanovllm::config::Config;
use nanovllm::engine::LLMEngine;
use nanovllm::models::SamplingParams;
use nanovllm::tokenizer::{EncodeOptions, Message, Tokenizer};

const MODEL_DIR: &str = "./cache/Qwen2.5-0.5B-Instruct";
const MAX_NEW_TOKENS: usize = 48;

fn main() -> anyhow::Result<()> {
    let tokenizer = Tokenizer::from_file(MODEL_DIR).context("load tokenizer")?;
    let config = Config { model: MODEL_DIR.to_string(), ..Default::default() };
    let mut engine = LLMEngine::new(config)?;
    println!("kv cache blocks: {}", engine.config.num_kvcache_blocks);

    // A >256-token prompt so at least one full KV block is cacheable.
    let long_prompt = "机器学习是人工智能的一个重要分支，它让计算机从数据中自动学习规律。".repeat(24);

    let prompts: Vec<String> = vec![
        "你好，请用一句话介绍一下你自己。".to_string(),
        "1+1 等于几？只回答数字。".to_string(),
        "写一句关于春天的诗。".to_string(),
        "What is the capital of France? Answer in one word.".to_string(),
        "用一句话解释什么是机器学习。".to_string(),
        long_prompt.clone(),
    ];

    // ---- round 1: one batch of concurrent requests ----
    println!("\n================ round 1: {} concurrent requests ================", prompts.len());
    for p in &prompts {
        add(&mut engine, &tokenizer, p, MAX_NEW_TOKENS)?;
    }
    let t0 = Instant::now();
    let outputs = engine.generate()?;
    let dt = t0.elapsed();
    for ((_id, toks), p) in outputs.iter().zip(&prompts) {
        let short = if p.chars().count() > 24 { format!("{}...", p.chars().take(24).collect::<String>()) } else { p.clone() };
        println!("[{} tokens] {} -> {}", token_count(&tokenizer, p)?, short, tokenizer.decode(toks, true)?);
    }
    println!("round 1: {} seqs, {:.2}s", outputs.len(), dt.as_secs_f64());

    // ---- round 2: repeat the long prompt -> prefix cache hit (paged prefill) ----
    println!("\n================ round 2: repeated long prompt (prefix cache) ================");
    add(&mut engine, &tokenizer, &long_prompt, MAX_NEW_TOKENS)?;
    let outputs = engine.generate()?;
    for (_id, toks) in &outputs {
        println!("repeat -> {}", tokenizer.decode(toks, true)?);
    }

    // ---- round 3: mixed lengths ----
    println!("\n================ round 3: mixed lengths ================");
    let mixed: Vec<String> = vec![
        "Hi".to_string(),
        "请把这句话翻译成英文：今天天气很好。".to_string(),
        "请依次列出 1 到 10 的平方，用逗号分隔。".to_string(),
        "Tell me a short joke.".to_string(),
    ];
    for p in &mixed {
        add(&mut engine, &tokenizer, p, MAX_NEW_TOKENS)?;
    }
    let outputs = engine.generate()?;
    for ((_id, toks), p) in outputs.iter().zip(&mixed) {
        println!("{p} -> {}", tokenizer.decode(toks, true)?);
    }

    println!("\ndone");
    Ok(())
}

fn add(engine: &mut LLMEngine, tokenizer: &Tokenizer, user: &str, max_tokens: usize) -> anyhow::Result<()> {
    let messages = vec![Message::user(user)];
    let prompt = tokenizer.apply_chat_template(&messages, true).context("chat template")?;
    let enc = tokenizer
        .encode(&prompt, EncodeOptions { add_special_tokens: false, ..Default::default() })
        .context("encode")?;
    engine.add_token_request(
        enc.get_ids().to_vec(),
        SamplingParams { max_new_tokens: max_tokens, ..SamplingParams::greedy() },
    );
    Ok(())
}

fn token_count(tokenizer: &Tokenizer, user: &str) -> anyhow::Result<usize> {
    let messages = vec![Message::user(user)];
    let prompt = tokenizer.apply_chat_template(&messages, true)?;
    let enc = tokenizer.encode(&prompt, EncodeOptions { add_special_tokens: false, ..Default::default() })?;
    Ok(enc.get_ids().len())
}
