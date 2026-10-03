use anyhow::Context;

use super::{Message, Tokenizer, ToolDefinition};

const DEFAULT_SYSTEM_PROMPT: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

impl Tokenizer {
    /// Render the prompt for a conversation.
    ///
    /// Prefers the model's own Jinja `chat_template` (from `tokenizer_config.json`)
    /// so any HF-compatible model can be used; falls back to the built-in Qwen
    /// formatter when the template is missing or fails to render.
    pub fn apply_chat_template(&self, messages: &[Message], add_generation_prompt: bool) -> anyhow::Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        let template = self.config.chat_template.as_str();
        if !template.is_empty() {
            match self.render_chat_template(template, messages, add_generation_prompt) {
                Ok(rendered) => return Ok(rendered),
                Err(error) => {
                    eprintln!("chat_template render failed ({error:#}); falling back to built-in Qwen template");
                }
            }
        }

        Ok(self.apply_chat_template_fallback(messages, add_generation_prompt))
    }

    fn render_chat_template(&self, template: &str, messages: &[Message], add_generation_prompt: bool) -> anyhow::Result<String> {
        use minijinja::{Environment, Error, ErrorKind};

        let tools: Vec<ToolDefinition> = messages
            .iter()
            .find_map(|message| match message {
                Message::System(system) if !system.tools.is_empty() => Some(system.tools.clone()),
                _ => None,
            })
            .unwrap_or_default();

        let mut env = Environment::new();
        env.add_function("raise_exception", |message: String| -> Result<minijinja::Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        });

        let context = serde_json::json!({
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": self.bos_token(),
            "eos_token": self.eos_token(),
        });

        env.render_str(template, context).context("render chat_template")
    }

    /// Built-in Qwen2/Qwen2.5 chat template, used when no Jinja template is available.
    fn apply_chat_template_fallback(&self, messages: &[Message], add_generation_prompt: bool) -> String {
        let mut prompt = String::new();
        if messages.len() == 0 {
            return "".to_string();
        }

        // --- 1. 处理 System & Tools 头部 ---
        let messages = if let Message::System(msg) = &messages[0] {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(&msg.content);

            if msg.tools.is_empty() {
                prompt.push_str("<|im_end|>\n");
            } else {
                prompt.push_str("\n\n");
                prompt.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
                for tool in &msg.tools {
                    prompt.push_str("\n");
                    prompt.push_str(&serde_json::to_string(tool).unwrap_or_default());
                }
                prompt.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
            }
            &messages[1..]
        } else {
            prompt.push_str(&format!("<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"));
            messages
        };

        // --- 2. 遍历消息 ---
        for (i, msg) in messages.iter().enumerate() {
            match msg {
                Message::System(m) => {
                    prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", m.content));
                }
                Message::User(m) => {
                    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", m.content));
                }
                Message::Assistant(m) => {
                    prompt.push_str("<|im_start|>assistant\n");
                    prompt.push_str(&m.content);

                    for (idx, tc) in m.tool_calls.iter().enumerate() {
                        // Jinja: {%- if (loop.first and content) or (not loop.first) %}{{- '\n' }}{%- endif %}
                        if (idx == 0 && !m.content.is_empty()) || idx > 0 {
                            prompt.push_str("\n");
                        }
                        prompt.push_str("<tool_call>\n{\"name\": \"");
                        prompt.push_str(&tc.function.name);
                        prompt.push_str("\", \"arguments\": ");
                        // Jinja: {%- if tool_call.arguments is string %}...{%- else %}...|tojson{%- endif %}
                        match &tc.function.arguments {
                            serde_json::Value::String(s) => prompt.push_str(s),
                            other => prompt.push_str(&serde_json::to_string(other).unwrap_or_default()),
                        }
                        prompt.push_str("}\n</tool_call>");
                    }
                    prompt.push_str("<|im_end|>\n");
                }
                Message::Tool(m) => {
                    // --- Tool 消息特殊处理：包装在 user 角色中 ---
                    let prev_is_not_tool = i == 0 || !matches!(messages[i - 1], Message::Tool(_));
                    if prev_is_not_tool {
                        prompt.push_str("<|im_start|>user");
                    }

                    prompt.push_str(&format!("\n<tool_response>\n{}\n</tool_response>", m.content));

                    let next_is_not_tool = i + 1 == messages.len() || !matches!(messages[i + 1], Message::Tool(_));
                    if next_is_not_tool {
                        prompt.push_str("<|im_end|>\n");
                    }
                }
            }
        }

        // --- 3. 生成提示符 ---
        if add_generation_prompt {
            prompt.push_str("<|im_start|>assistant\n");
        }

        prompt
    }
}
