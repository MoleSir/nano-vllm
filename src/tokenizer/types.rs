use serde::{Deserialize, Deserializer, Serialize};
use std::fmt::Display;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "role")]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolMessage),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    pub content: String,
    #[serde(default, rename = "tools", deserialize_with = "deserialize_json_string_or_seq")]
    pub tools: Vec<ToolDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    pub content: String,
    #[serde(default, deserialize_with = "deserialize_json_string_or_seq")]
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMessage {
    pub content: String,
}

/// A tool definition as stored in the data: `{"function": {"name", "description", "parameters"}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

/// A tool call as stored in the data: `{"function": {"name", "arguments"}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Deserialize a field that may be either a JSON-encoded string (e.g. `"[...]"`)
/// or a plain sequence, into a `Vec<T>`.
fn deserialize_json_string_or_seq<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrSeq<S> {
        Str(String),
        Seq(Vec<S>),
    }

    match StringOrSeq::<T>::deserialize(deserializer)? {
        StringOrSeq::Str(s) => serde_json::from_str(&s).map_err(serde::de::Error::custom),
        StringOrSeq::Seq(v) => Ok(v),
    }
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self { function: FunctionDefinition { name: name.into(), description: description.into(), parameters } }
    }
}

impl Message {
    pub fn role(&self) -> Role {
        match self {
            Self::Assistant(_) => Role::Assistant,
            Self::User(_) => Role::User,
            Self::Tool(_) => Role::Tool,
            Self::System(_) => Role::System,
        }
    }

    pub fn system(content: impl Into<String>, tools: impl Into<Vec<ToolDefinition>>) -> Self {
        Self::System(SystemMessage { content: content.into(), tools: tools.into() })
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User(UserMessage { content: content.into() })
    }

    pub fn assistant(content: impl Into<String>, tool_calls: impl Into<Vec<ToolCall>>) -> Self {
        Self::Assistant(AssistantMessage { content: content.into(), tool_calls: tool_calls.into() })
    }

    pub fn tool(content: impl Into<String>) -> Self {
        Self::Tool(ToolMessage { content: content.into() })
    }

    pub fn is_system(&self) -> bool {
        matches!(self, Self::System(_))
    }

    pub fn is_user(&self) -> bool {
        matches!(self, Self::User(_))
    }

    pub fn is_assistant(&self) -> bool {
        matches!(self, Self::Assistant(_))
    }

    pub fn is_tool(&self) -> bool {
        matches!(self, Self::Tool(_))
    }
}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

impl Into<Message> for SystemMessage {
    fn into(self) -> Message {
        Message::System(self)
    }
}

impl Into<Message> for UserMessage {
    fn into(self) -> Message {
        Message::User(self)
    }
}

impl Into<Message> for AssistantMessage {
    fn into(self) -> Message {
        Message::Assistant(self)
    }
}

impl Into<Message> for ToolMessage {
    fn into(self) -> Message {
        Message::Tool(self)
    }
}