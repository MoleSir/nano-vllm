use super::Tokenizer;
use tokenizers::{Encoding, PaddingParams, TruncationParams};

#[derive(Debug, Clone)]
pub struct EncodeOptions {
    pub add_special_tokens: bool,
    pub truncation: bool,
    pub max_length: Option<usize>,
    pub padding: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self { add_special_tokens: true, truncation: false, max_length: None, padding: false }
    }
}

impl Tokenizer {
    pub fn encode(&self, input: &str, options: EncodeOptions) -> anyhow::Result<Encoding> {
        let mut tokenizer = self.tokenizer.lock().unwrap();

        // ===== 保存旧状态 =====
        let saved_trunc = tokenizer.get_truncation().cloned();
        let saved_padding = tokenizer.get_padding().cloned();

        // ===== truncation =====
        if options.truncation {
            let max_length = options
                .max_length
                .unwrap_or(self.config.extra.get("model_max_length").and_then(|v| v.as_u64()).unwrap_or(2048) as usize);

            tokenizer
                .with_truncation(Some(TruncationParams { max_length, ..Default::default() }))
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        }

        // ===== padding =====
        if options.padding {
            let max_length = options
                .max_length
                .unwrap_or(self.config.extra.get("model_max_length").and_then(|v| v.as_u64()).unwrap_or(2048) as usize);

            tokenizer.with_padding(Some(PaddingParams { strategy: tokenizers::PaddingStrategy::Fixed(max_length), ..Default::default() }));
        }

        // ===== encode =====
        let res = tokenizer.encode(input, options.add_special_tokens).map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;

        // ===== 恢复状态 =====
        tokenizer.with_truncation(saved_trunc).map_err(|e| anyhow::anyhow!(e.to_string()))?;

        tokenizer.with_padding(saved_padding);

        Ok(res)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        let tokenizer = self.tokenizer.lock().unwrap();
        tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))
    }
}
