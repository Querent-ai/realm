/// Chat template support for different model formats
use wasm_chord_core::error::Result;

#[derive(Debug, Clone)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: ChatRole::System, content: content.into() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: ChatRole::User, content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: ChatRole::Assistant, content: content.into() }
    }
}

#[derive(Debug, Clone)]
pub enum ChatTemplate {
    /// ChatML format (used by TinyLlama, Mistral, etc.)
    ///
    /// Format: `<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n`
    ChatML,

    /// Llama 2 format
    ///
    /// Format: `\[INST\] <<SYS>>\n{system}\n<</SYS>>\n\n{user} \[/INST\]`
    Llama2,

    /// Alpaca format
    ///
    /// Format: `### Instruction:\n{instruction}\n\n### Response:\n`
    Alpaca,
}

impl ChatTemplate {
    /// Format a list of chat messages into a prompt string
    pub fn format(&self, messages: &[ChatMessage]) -> Result<String> {
        match self {
            ChatTemplate::ChatML => self.format_chatml(messages),
            ChatTemplate::Llama2 => self.format_llama2(messages),
            ChatTemplate::Alpaca => self.format_alpaca(messages),
        }
    }

    fn format_chatml(&self, messages: &[ChatMessage]) -> Result<String> {
        let mut prompt = String::new();

        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    prompt.push_str("<|system|>\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("</s>\n");
                }
                ChatRole::User => {
                    prompt.push_str("<|user|>\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("</s>\n");
                }
                ChatRole::Assistant => {
                    prompt.push_str("<|assistant|>\n");
                    prompt.push_str(&msg.content);
                    if !msg.content.is_empty() {
                        prompt.push_str("</s>\n");
                    }
                }
            }
        }

        // If last message is not assistant, add assistant prefix
        if let Some(last) = messages.last() {
            if !matches!(last.role, ChatRole::Assistant) {
                prompt.push_str("<|assistant|>\n");
            }
        }

        Ok(prompt)
    }

    fn format_llama2(&self, messages: &[ChatMessage]) -> Result<String> {
        let mut prompt = String::new();
        let mut system_msg = String::new();

        // Extract system message if present
        for msg in messages {
            if matches!(msg.role, ChatRole::System) {
                system_msg = msg.content.clone();
                break;
            }
        }

        // Format conversation
        let mut in_conversation = false;
        for msg in messages {
            match msg.role {
                ChatRole::System => continue, // Already handled
                ChatRole::User => {
                    if !in_conversation && !system_msg.is_empty() {
                        prompt.push_str("[INST] <<SYS>>\n");
                        prompt.push_str(&system_msg);
                        prompt.push_str("\n<</SYS>>\n\n");
                        prompt.push_str(&msg.content);
                        prompt.push_str(" [/INST] ");
                        in_conversation = true;
                    } else {
                        prompt.push_str("[INST] ");
                        prompt.push_str(&msg.content);
                        prompt.push_str(" [/INST] ");
                    }
                }
                ChatRole::Assistant => {
                    prompt.push_str(&msg.content);
                    if !msg.content.is_empty() {
                        prompt.push(' ');
                    }
                }
            }
        }

        Ok(prompt)
    }

    fn format_alpaca(&self, messages: &[ChatMessage]) -> Result<String> {
        let mut prompt = String::new();

        // For Alpaca, combine system + last user message as instruction
        let mut instruction = String::new();

        for msg in messages {
            if matches!(msg.role, ChatRole::System) {
                instruction = msg.content.clone();
            }
        }

        // Get last user message
        for msg in messages.iter().rev() {
            if matches!(msg.role, ChatRole::User) {
                if !instruction.is_empty() {
                    instruction.push_str("\n\n");
                }
                instruction.push_str(&msg.content);
                break;
            }
        }

        if !instruction.is_empty() {
            prompt.push_str("### Instruction:\n");
            prompt.push_str(&instruction);
            prompt.push_str("\n\n### Response:\n");
        }

        Ok(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_format() {
        let messages =
            vec![ChatMessage::system("You are a helpful assistant."), ChatMessage::user("Hello!")];

        let template = ChatTemplate::ChatML;
        let formatted = template.format(&messages).unwrap();

        assert!(formatted.contains("<|system|>"));
        assert!(formatted.contains("<|user|>"));
        assert!(formatted.contains("<|assistant|>"));
    }

    #[test]
    fn test_llama2_format() {
        let messages =
            vec![ChatMessage::system("You are a helpful assistant."), ChatMessage::user("Hello!")];

        let template = ChatTemplate::Llama2;
        let formatted = template.format(&messages).unwrap();

        assert!(formatted.contains("[INST]"));
        assert!(formatted.contains("<<SYS>>"));
        assert!(formatted.contains("[/INST]"));
    }
}
