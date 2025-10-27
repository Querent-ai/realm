//! Streaming inference implementation for real-time text generation
//!
//! This module provides streaming capabilities that allow the model to generate
//! text incrementally, providing real-time output as tokens are generated.

use crate::{GenerationConfig, Model};
use wasm_chord_core::error::Result;
use wasm_chord_core::tokenizer::Tokenizer;

/// Type alias for token callback function
#[allow(dead_code)]
pub type TokenCallback = Box<dyn Fn(&str) + Send + Sync>;

/// Type alias for completion callback function
#[allow(dead_code)]
pub type CompleteCallback = Box<dyn Fn() + Send + Sync>;

/// Streaming inference handler for real-time text generation
pub struct StreamingInference {
    model: Model,
    tokenizer: Tokenizer,
    config: GenerationConfig,
    /// Current sequence of tokens
    tokens: Vec<u32>,
    /// Maximum sequence length to prevent memory issues
    max_sequence_length: usize,
}

impl StreamingInference {
    /// Create a new streaming inference handler
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        config: GenerationConfig,
        max_sequence_length: Option<usize>,
    ) -> Self {
        Self {
            model,
            tokenizer,
            config,
            tokens: Vec::new(),
            max_sequence_length: max_sequence_length.unwrap_or(2048),
        }
    }

    /// Start streaming inference with an initial prompt
    pub fn start_streaming(&mut self, prompt: &str) -> Result<()> {
        // Reset tokens
        self.tokens.clear();

        // Tokenize the prompt
        let prompt_tokens = self.tokenizer.encode(prompt, true)?;
        self.tokens.extend(prompt_tokens);

        println!("🚀 Starting streaming inference with prompt: {:?}", prompt);
        println!("📝 Initial tokens: {:?}", self.tokens);

        Ok(())
    }

    /// Generate the next token and return it
    ///
    /// This method performs a single forward pass to generate one token at a time.
    /// It uses the model's forward method directly for efficient streaming.
    pub fn generate_next_token(&mut self) -> Result<Option<String>> {
        // Check if we've reached the maximum sequence length
        if self.tokens.len() >= self.max_sequence_length {
            return Ok(None);
        }

        // For true streaming, we'd use model.forward() with KV caching
        // For now, use the generate API with max_tokens=1 as a simpler approach

        // Convert current tokens to text
        let current_text = self.tokenizer.decode(&self.tokens, false)?;

        // Create a single-token generation config
        let mut streaming_config = self.config.clone();
        streaming_config.max_tokens = 1;

        // Generate the next token
        let output = self.model.generate(&current_text, &self.tokenizer, &streaming_config)?;

        // Extract the new token by comparing lengths
        if output.len() > current_text.len() {
            let new_text = &output[current_text.len()..];

            // Encode to get the actual token ID
            let new_token_ids = self.tokenizer.encode(new_text, false)?;
            if let Some(&token_id) = new_token_ids.first() {
                self.tokens.push(token_id);
                return Ok(Some(new_text.to_string()));
            }
        }

        // If no new token was generated, stop
        Ok(None)
    }

    /// Generate multiple tokens and return them as a batch
    #[allow(dead_code)]
    pub fn generate_batch(&mut self, batch_size: usize) -> Result<Vec<String>> {
        let mut batch = Vec::new();

        for _ in 0..batch_size {
            if let Some(token_text) = self.generate_next_token()? {
                batch.push(token_text);
            } else {
                break; // Reached end of generation
            }
        }

        Ok(batch)
    }

    /// Get the current generated text
    #[allow(dead_code)]
    pub fn get_current_text(&self) -> Result<String> {
        self.tokenizer.decode(&self.tokens, false)
    }

    /// Get the current token count
    #[allow(dead_code)]
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Check if generation should continue
    pub fn should_continue(&self) -> bool {
        self.tokens.len() < self.max_sequence_length
    }

    /// Reset the streaming state
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.tokens.clear();
    }

    /// Update generation configuration
    #[allow(dead_code)]
    pub fn update_config(&mut self, config: GenerationConfig) {
        self.config = config;
    }
}

/// Streaming inference with callback support
#[allow(dead_code)]
pub struct CallbackStreamingInference {
    streaming: StreamingInference,
    /// Callback function called for each generated token
    on_token: Option<TokenCallback>,
    /// Callback function called when generation completes
    on_complete: Option<CompleteCallback>,
}

impl CallbackStreamingInference {
    /// Create a new callback streaming inference handler
    #[allow(dead_code)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        config: GenerationConfig,
        max_sequence_length: Option<usize>,
    ) -> Self {
        Self {
            streaming: StreamingInference::new(model, tokenizer, config, max_sequence_length),
            on_token: None,
            on_complete: None,
        }
    }

    /// Set the token callback
    #[allow(dead_code)]
    pub fn set_token_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.on_token = Some(Box::new(callback));
    }

    /// Set the completion callback
    #[allow(dead_code)]
    pub fn set_complete_callback<F>(&mut self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.on_complete = Some(Box::new(callback));
    }

    /// Start streaming with callbacks
    #[allow(dead_code)]
    pub fn start_streaming(&mut self, prompt: &str) -> Result<()> {
        self.streaming.start_streaming(prompt)?;

        // Call token callback for initial prompt
        if let Some(ref callback) = self.on_token {
            callback(prompt);
        }

        Ok(())
    }

    /// Generate next token with callback
    #[allow(dead_code)]
    pub fn generate_next_token(&mut self) -> Result<Option<String>> {
        if let Some(token_text) = self.streaming.generate_next_token()? {
            // Call token callback
            if let Some(ref callback) = self.on_token {
                callback(&token_text);
            }

            Ok(Some(token_text))
        } else {
            // Call completion callback
            if let Some(ref callback) = self.on_complete {
                callback();
            }

            Ok(None)
        }
    }

    /// Generate until completion with callbacks
    #[allow(dead_code)]
    pub fn generate_until_complete(&mut self) -> Result<String> {
        let mut full_text = String::new();

        while self.streaming.should_continue() {
            if let Some(token_text) = self.generate_next_token()? {
                full_text.push_str(&token_text);
            } else {
                break;
            }
        }

        Ok(full_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenerationConfig, TransformerConfig};

    #[test]
    fn test_streaming_inference_creation() {
        // This is a basic test to ensure the struct can be created
        // In a real test, you'd need a proper model and tokenizer
        let config = TransformerConfig::default();
        let model = Model::new(config.clone());
        let tokenizer = Tokenizer::new(
            std::collections::HashMap::new(),
            Vec::new(),
            wasm_chord_core::tokenizer::SpecialTokens::default(),
        );
        let gen_config = GenerationConfig::default();

        let streaming = StreamingInference::new(model, tokenizer, gen_config, None);
        assert_eq!(streaming.token_count(), 0);
        assert!(streaming.should_continue());
    }
}
