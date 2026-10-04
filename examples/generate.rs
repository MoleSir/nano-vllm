use anyhow::Context;
use luma::dtype::IntDType;
use luma::tensor::IntTensor;
use luma::{Device, Tensor, no_grad};
use nanovllm::models::Sampler;
use nanovllm::models::qwen2::{Qwen2Config, Qwen2ForCausalLM, Qwen2KvCache};
use nanovllm::tokenizer::{EncodeOptions, Message, Tokenizer};

const MODEL_DIR: &str = "./cache/Qwen2.5-0.5B-Instruct";
const MAX_NEW_TOKENS: usize = 128;

fn main() -> anyhow::Result<()> {
    match luma::Cuda::new(0) {
        Ok(device) => {
            println!("device: CUDA:0");
            run(&device)
        }
        Err(error) => {
            eprintln!("CUDA unavailable ({error}); falling back to CPU");
            run(&luma::Cpu::default())
        }
    }
}

fn run<Dev: Device>(device: &Dev) -> anyhow::Result<()> {
    no_grad!();

    let tokenizer = Tokenizer::from_file(MODEL_DIR).context("load tokenizer")?;
    let model = Qwen2ForCausalLM::<Dev>::from_pretrained(MODEL_DIR, device).context("load model")?;
    println!("model loaded: {} layers, hidden {}", model.config.num_hidden_layers, model.config.hidden_size);

    let messages = vec![Message::user("你好，请用一句话介绍一下你自己。")];
    let prompt = tokenizer.apply_chat_template(&messages, true).context("apply chat template")?;
    let encoding = tokenizer
        .encode(&prompt, EncodeOptions { add_special_tokens: false, ..Default::default() })
        .context("encode prompt")?;
    let prompt_ids = encoding.get_ids().to_vec();
    println!("prompt tokens: {}", prompt_ids.len());

    let sampler = Sampler::strict();
    let generated = generate(&model, &model.config, device, &prompt_ids, MAX_NEW_TOKENS, tokenizer.eos_token_id(), &sampler)?;

    let text = tokenizer.decode(&generated, true).context("decode output")?;
    println!("\n=== generated ===\n{text}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate<Dev: Device>(
    model: &Qwen2ForCausalLM<Dev>,
    config: &Qwen2Config,
    device: &Dev,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    eos_token_id: u32,
    sampler: &Sampler,
) -> anyhow::Result<Vec<u32>> {
    let mut cache = Some(Qwen2KvCache::new(config));

    let prompt: Vec<i64> = prompt_ids.iter().map(|&id| id as i64).collect();
    let input = IntTensor::<Dev>::from_slice(&prompt, (1, prompt.len()), (device, IntDType::U32)).context("build prompt input")?;
    let logits = model.forward(&input, 0, &mut cache).context("prefill")?;
    let mut next = sample_last(sampler, &logits)?;

    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut start_pos = prompt_ids.len();
    for _ in 0..max_new_tokens {
        generated.push(next);
        if next == eos_token_id {
            break;
        }

        let input = IntTensor::<Dev>::from_slice(&[next as i64], (1, 1), (device, IntDType::U32)).context("build decode input")?;
        let logits = model.forward(&input, start_pos, &mut cache).context("decode step")?;
        next = sample_last(sampler, &logits)?;
        start_pos += 1;
    }

    Ok(generated)
}

fn sample_last<Dev: Device>(sampler: &Sampler, logits: &Tensor<Dev>) -> anyhow::Result<u32> {
    let (_, seq, _) = logits.dims3().context("logits dims3")?;
    let last = logits.narrow(1, seq - 1, 1).context("narrow last logits")?.contiguous().context("contiguous logits")?;
    let tokens = sampler.sample(&last, &[sampler.temperature()]).context("sample last")?;
    Ok(tokens[0])
}
