mod chat_template;
mod encode;
mod types;

pub use encode::*;
pub use types::*;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path, sync::Mutex};

pub struct Tokenizer {
    tokenizer: Mutex<tokenizers::Tokenizer>,
    config: TokenizerConfig,

    eos_token_id: u32,
    bos_token_id: u32,
    pad_token_id: u32,
    unk_token_id: u32,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TokenizerConfig {
    pub add_bos_token: bool,
    pub add_eos_token: bool,
    pub add_prefix_space: bool,

    pub added_tokens_decoder: HashMap<String, AddedToken>,

    pub additional_special_tokens: Vec<String>,

    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub pad_token: Option<String>,
    pub unk_token: Option<String>,

    pub clean_up_tokenization_spaces: bool,
    pub legacy: bool,
    pub model_max_length: usize,

    pub sp_model_kwargs: HashMap<String, serde_json::Value>,

    pub spaces_between_special_tokens: bool,

    pub tokenizer_class: String,

    pub chat_template: String,

    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddedToken {
    pub content: String,
    pub lstrip: bool,
    pub normalized: bool,
    pub rstrip: bool,
    pub single_word: bool,
    pub special: bool,
}

impl Tokenizer {
    pub fn from_pretrained<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let path = path.as_ref();

        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer from file error: {}", e))?;

        let config_path = path.join("tokenizer_config.json");
        let config_content = std::fs::read_to_string(config_path)?;
        let config: TokenizerConfig = serde_json::from_str(&config_content)?;

        let eos_token_id = config
            .eos_token
            .as_deref()
            .and_then(|token| tokenizer.token_to_id(token))
            .ok_or_else(|| anyhow::anyhow!("no eos token!"))?;

        // Some Qwen tokenizers leave bos/pad/unk unset (null in the JSON).
        // Fall back gracefully so generation can still run.
        let lookup = |token: Option<&str>| token.and_then(|token| tokenizer.token_to_id(token));
        let bos_token_id = lookup(config.bos_token.as_deref()).or_else(|| tokenizer.token_to_id("<|endoftext|>")).unwrap_or(eos_token_id);
        let pad_token_id = lookup(config.pad_token.as_deref()).unwrap_or(eos_token_id);
        let unk_token_id = lookup(config.unk_token.as_deref()).unwrap_or(eos_token_id);

        Ok(Self { tokenizer: Mutex::new(tokenizer), config, bos_token_id, eos_token_id, pad_token_id, unk_token_id })
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    pub fn eos_token(&self) -> &str {
        self.config.eos_token.as_deref().unwrap_or("")
    }

    pub fn bos_token(&self) -> &str {
        self.config.bos_token.as_deref().unwrap_or("")
    }

    pub fn pad_token(&self) -> &str {
        self.config.pad_token.as_deref().unwrap_or("")
    }

    pub fn unk_token(&self) -> &str {
        self.config.unk_token.as_deref().unwrap_or("")
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::Message;

    use super::Tokenizer;

    #[test]
    fn test_load_file() {
        let tokenizer = Tokenizer::from_pretrained("./assets/tokenizer").unwrap();
        assert_eq!(tokenizer.bos_token_id(), 1);
    }

    #[test]
    fn test_chat_template() {
        let tokenizer = Tokenizer::from_pretrained("./assets/tokenizer").unwrap();
        let messages = vec![
            Message::system("You are a helpful and concise AI assistant.", []),
            Message::user("Hello, who are you?"),
            Message::assistant("I am an AI. How can I help you today?", []),
            Message::user("Can you write a Rust struct for me?"),
        ];

        let prompt_string = tokenizer.apply_chat_template(&messages, false).unwrap();

        println!("{}", prompt_string);
    }
}
