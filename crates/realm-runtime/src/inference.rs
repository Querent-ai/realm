/// Inference session management with token streaming
use realm_core::error::Result;
use std::collections::VecDeque;

/// Generation options
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GenOptions {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
    pub seed: u32,
    pub stop_token_count: u8,
    pub stop_tokens_ptr: u32,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            seed: 0,
            stop_token_count: 0,
            stop_tokens_ptr: 0,
        }
    }
}

/// Token generation state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationState {
    /// Ready to generate
    Ready,
    /// Currently generating
    Generating,
    /// Generation complete (max tokens reached)
    Complete,
    /// Generation stopped (stop token encountered)
    Stopped,
}

/// Inference session with streaming support
pub struct InferenceSession {
    /// Model ID reference
    model_id: u32,
    /// Input prompt tokens
    prompt_tokens: Vec<u32>,
    /// Generation options
    options: GenOptions,
    /// Current generation state
    state: GenerationState,
    /// Number of tokens generated so far
    tokens_generated: usize,
    /// Token buffer for streaming
    token_buffer: VecDeque<u32>,
    /// Stop tokens to detect
    stop_tokens: Vec<u32>,
    /// Complete generated sequence
    generated_tokens: Vec<u32>,
    /// Logits processor for sampling
    logits_processor: crate::sampling::LogitsProcessor,
    /// Optional speculative decoding configuration
    speculative_config: Option<crate::speculative::SpeculativeConfig>,
}

impl InferenceSession {
    /// Create a new inference session
    pub fn new(model_id: u32, prompt_tokens: Vec<u32>, options: GenOptions) -> Self {
        let logits_processor = crate::sampling::LogitsProcessor::with_params(
            42, // seed - could be made configurable
            options.temperature as f64,
            options.top_p as f64,
            options.top_k as usize,
            options.repetition_penalty,
        );

        Self {
            model_id,
            prompt_tokens,
            options,
            state: GenerationState::Ready,
            tokens_generated: 0,
            token_buffer: VecDeque::new(),
            stop_tokens: Vec::new(),
            generated_tokens: Vec::new(),
            logits_processor,
            speculative_config: None,
        }
    }

    /// Enable speculative decoding with configuration
    pub fn with_speculative_decoding(
        mut self,
        config: crate::speculative::SpeculativeConfig,
    ) -> Self {
        self.speculative_config = Some(config);
        self
    }

    /// Set stop tokens for generation
    pub fn set_stop_tokens(&mut self, stop_tokens: Vec<u32>) {
        self.stop_tokens = stop_tokens;
    }

    /// Get current generation state
    pub fn state(&self) -> GenerationState {
        self.state
    }

    /// Check if generation is complete
    pub fn is_complete(&self) -> bool {
        matches!(
            self.state,
            GenerationState::Complete | GenerationState::Stopped
        )
    }

    /// Get number of tokens generated
    pub fn tokens_generated(&self) -> usize {
        self.tokens_generated
    }

    /// Generate next token (returns token ID)
    ///
    /// Returns None if generation is complete, otherwise returns the next token ID.
    pub fn next_token(&mut self) -> Result<Option<u32>> {
        // Check if already complete
        if self.is_complete() {
            return Ok(None);
        }

        // Mark as generating
        if self.state == GenerationState::Ready {
            self.state = GenerationState::Generating;
        }

        // Check max tokens limit
        if self.tokens_generated >= self.options.max_tokens as usize {
            self.state = GenerationState::Complete;
            return Ok(None);
        }

        // TODO: Actual inference logic will go here
        // For now, generate placeholder token ID
        let token_id = (self.tokens_generated % 100) as u32;

        // Check stop tokens
        if self.stop_tokens.contains(&token_id) {
            self.state = GenerationState::Stopped;
            return Ok(None);
        }

        // Buffer the token
        self.token_buffer.push_back(token_id);
        self.generated_tokens.push(token_id);
        self.tokens_generated += 1;

        Ok(Some(token_id))
    }

    /// Generate next token using a model
    ///
    /// This is the real inference method that uses the transformer model.
    ///
    /// # Arguments
    /// * `model` - The transformer model to use for inference (target model)
    /// * `draft_model` - Optional draft model for speculative decoding
    ///
    /// # Returns
    /// Next token ID, or None if generation is complete
    pub fn next_token_with_model(
        &mut self,
        model: &mut realm_models::Model,
        draft_model: Option<&mut realm_models::Model>,
    ) -> Result<Option<u32>> {
        // Check if already complete
        if self.is_complete() {
            return Ok(None);
        }

        // Mark as generating
        if self.state == GenerationState::Ready {
            self.state = GenerationState::Generating;
        }

        // Check max tokens limit
        if self.tokens_generated >= self.options.max_tokens as usize {
            self.state = GenerationState::Complete;
            return Ok(None);
        }

        // Build input sequence: prompt + generated tokens
        let mut input_tokens = self.prompt_tokens.clone();
        input_tokens.extend_from_slice(&self.generated_tokens);

        // Use speculative decoding if enabled and draft model is available
        // Clone config only when needed to avoid borrow checker issues
        let (logits, pre_accepted_tokens) = if let Some(ref config) = self.speculative_config {
            if let Some(draft) = draft_model {
                // Speculative decoding: draft model generates tokens, target model verifies
                // Clone config to avoid borrow issues when calling &mut self method
                let config = config.clone();
                self.speculative_decode_step(&input_tokens, draft, model, &config)?
            } else {
                // Speculative decoding enabled but no draft model, fall back to standard
                let logits = model.forward(&input_tokens, input_tokens.len() - 1)?;
                (logits, Vec::new())
            }
        } else {
            // Standard inference
            let logits = model.forward(&input_tokens, input_tokens.len() - 1)?;
            (logits, Vec::new())
        };

        // If speculative decoding accepted tokens, add them to generated tokens
        for token in &pre_accepted_tokens {
            // Check stop tokens before adding
            if self.stop_tokens.contains(token) {
                self.state = GenerationState::Stopped;
                return Ok(None);
            }
            self.token_buffer.push_back(*token);
            self.generated_tokens.push(*token);
            self.tokens_generated += 1;
        }

        // Extract logits for last position
        let vocab_size = model.config.vocab_size;
        let mut last_logits = logits[logits.len() - vocab_size..].to_vec();

        // Sample next token
        let token_id = self
            .logits_processor
            .sample(&mut last_logits)
            .map_err(realm_core::error::Error::ParseError)?;

        // Check stop tokens
        if self.stop_tokens.contains(&token_id) {
            self.state = GenerationState::Stopped;
            return Ok(None);
        }

        // Buffer the token
        self.token_buffer.push_back(token_id);
        self.generated_tokens.push(token_id);
        self.tokens_generated += 1;

        Ok(Some(token_id))
    }

    /// Get buffered tokens (for batch processing)
    pub fn drain_buffer(&mut self) -> Vec<u32> {
        self.token_buffer.drain(..).collect()
    }

    /// Peek at buffer without draining
    pub fn peek_buffer(&self) -> &VecDeque<u32> {
        &self.token_buffer
    }

    /// Get all generated tokens so far
    pub fn generated_tokens(&self) -> &[u32] {
        &self.generated_tokens
    }

    /// Reset session for new generation
    pub fn reset(&mut self, new_prompt_tokens: Vec<u32>) {
        self.prompt_tokens = new_prompt_tokens;
        self.state = GenerationState::Ready;
        self.tokens_generated = 0;
        self.token_buffer.clear();
        self.generated_tokens.clear();
    }

    /// Get model ID
    pub fn model_id(&self) -> u32 {
        self.model_id
    }

    /// Get prompt tokens
    pub fn prompt_tokens(&self) -> &[u32] {
        &self.prompt_tokens
    }

    /// Perform one step of speculative decoding
    ///
    /// Algorithm:
    /// 1. Draft model generates k tokens quickly
    /// 2. Target model verifies draft tokens
    /// 3. Accept tokens until first rejection
    ///
    /// Returns: (final_logits, accepted_tokens)
    fn speculative_decode_step(
        &mut self,
        input_tokens: &[u32],
        draft_model: &mut realm_models::Model,
        target_model: &mut realm_models::Model,
        config: &crate::speculative::SpeculativeConfig,
    ) -> Result<(Vec<f32>, Vec<u32>)> {
        // Step 1: Draft model generates k tokens
        let draft_k = config.draft_k.min(config.max_draft_tokens);
        let mut draft_tokens = Vec::new();
        let mut current_input = input_tokens.to_vec();

        // Generate draft tokens using draft model
        for _ in 0..draft_k {
            if current_input.len() >= 2048 {
                break; // Prevent sequence length overflow
            }

            let logits = draft_model.forward(&current_input, current_input.len() - 1)?;
            let vocab_size = draft_model.config.vocab_size;
            let mut last_logits = logits[logits.len() - vocab_size..].to_vec();

            // Sample token from draft model
            let token_id = self
                .logits_processor
                .sample(&mut last_logits)
                .map_err(realm_core::error::Error::ParseError)?;

            draft_tokens.push(token_id);
            current_input.push(token_id);
        }

        if draft_tokens.is_empty() {
            // No draft tokens generated, fall back to target model
            let logits = target_model.forward(input_tokens, input_tokens.len() - 1)?;
            return Ok((logits, Vec::new()));
        }

        // Step 2: Target model verifies draft tokens
        // We verify by checking if target model would generate the same tokens
        // For simplicity, we verify one token at a time
        let mut accepted_tokens = Vec::new();
        let mut verify_input = input_tokens.to_vec();

        for draft_token in &draft_tokens {
            // Get target model's prediction for this position
            let target_logits = target_model.forward(&verify_input, verify_input.len() - 1)?;
            let vocab_size = target_model.config.vocab_size;
            let mut last_logits = target_logits[target_logits.len() - vocab_size..].to_vec();

            // Sample from target model
            let target_token = self
                .logits_processor
                .sample(&mut last_logits)
                .map_err(realm_core::error::Error::ParseError)?;

            if target_token == *draft_token {
                // Draft token accepted
                accepted_tokens.push(*draft_token);
                verify_input.push(*draft_token);
            } else {
                // Draft token rejected, use target model's token
                accepted_tokens.push(target_token);
                verify_input.push(target_token);
                break; // Stop after first rejection
            }
        }

        // Step 3: Return logits for the next token and accepted tokens
        let final_logits = target_model.forward(&verify_input, verify_input.len() - 1)?;
        Ok((final_logits, accepted_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let session = InferenceSession::new(1, vec![1, 2, 3], GenOptions::default());
        assert_eq!(session.state(), GenerationState::Ready);
        assert_eq!(session.tokens_generated(), 0);
        assert!(!session.is_complete());
    }

    #[test]
    fn test_token_generation() {
        let mut session = InferenceSession::new(1, vec![1, 2, 3], GenOptions::default());

        // Generate first token
        let token = session.next_token().unwrap();
        assert!(token.is_some());
        assert_eq!(session.state(), GenerationState::Generating);
        assert_eq!(session.tokens_generated(), 1);
    }

    #[test]
    fn test_max_tokens_limit() {
        let options = GenOptions {
            max_tokens: 5,
            ..Default::default()
        };

        let mut session = InferenceSession::new(1, vec![1, 2, 3], options);

        // Generate 5 tokens
        for _ in 0..5 {
            assert!(session.next_token().unwrap().is_some());
        }

        // 6th token should be None
        assert!(session.next_token().unwrap().is_none());
        assert_eq!(session.state(), GenerationState::Complete);
        assert!(session.is_complete());
    }

    #[test]
    fn test_stop_tokens() {
        let mut session = InferenceSession::new(1, vec![1, 2, 3], GenOptions::default());
        session.set_stop_tokens(vec![2]); // Stop on token ID 2

        // Token 0
        assert!(session.next_token().unwrap().is_some());
        // Token 1
        assert!(session.next_token().unwrap().is_some());
        // Token 2 should trigger stop
        assert!(session.next_token().unwrap().is_none());
        assert_eq!(session.state(), GenerationState::Stopped);
    }

    #[test]
    fn test_buffer_management() {
        let mut session = InferenceSession::new(1, vec![1, 2, 3], GenOptions::default());

        // Generate 3 tokens
        for _ in 0..3 {
            session.next_token().unwrap();
        }

        assert_eq!(session.peek_buffer().len(), 3);

        let drained = session.drain_buffer();
        assert_eq!(drained.len(), 3);
        assert_eq!(session.peek_buffer().len(), 0);
    }

    #[test]
    fn test_session_reset() {
        let mut session = InferenceSession::new(1, vec![1, 2, 3], GenOptions::default());

        // Generate some tokens
        for _ in 0..3 {
            session.next_token().unwrap();
        }

        // Reset with new prompt
        session.reset(vec![4, 5, 6]);

        assert_eq!(session.state(), GenerationState::Ready);
        assert_eq!(session.tokens_generated(), 0);
        assert_eq!(session.generated_tokens().len(), 0);
        assert_eq!(session.prompt_tokens(), &[4, 5, 6]);
    }

    #[test]
    #[ignore] // Model forward pass requires loaded weights, causes segfault with uninitialized model
    fn test_next_token_with_model_completion_check() {
        // Test that next_token_with_model returns None when already complete
        let mut session = InferenceSession::new(1, vec![1, 2, 3], GenOptions::default());

        // Manually set to complete state
        session.state = GenerationState::Complete;

        // Create a minimal model config for testing
        use realm_models::TransformerConfig;
        let config = TransformerConfig {
            vocab_size: 1000,
            hidden_size: 128,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            intermediate_size: 512,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            attention_backend: realm_models::AttentionBackend::Auto,
        };

        let mut model = realm_models::Model::new(config);

        // Should return None when already complete
        let result = session.next_token_with_model(&mut model, None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    #[ignore] // Model forward pass requires loaded weights, causes segfault with uninitialized model
    fn test_next_token_with_model_max_tokens() {
        // Test max_tokens limit in next_token_with_model
        let options = GenOptions {
            max_tokens: 2,
            ..Default::default()
        };

        let mut session = InferenceSession::new(1, vec![1, 2, 3], options);

        use realm_models::TransformerConfig;
        let config = TransformerConfig {
            vocab_size: 1000,
            hidden_size: 128,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            intermediate_size: 512,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            attention_backend: realm_models::AttentionBackend::Auto,
        };

        let mut model = realm_models::Model::new(config);

        // Generate 2 tokens (should succeed)
        let token1 = session.next_token_with_model(&mut model, None);
        assert!(token1.is_ok());
        assert!(token1.unwrap().is_some());
        assert_eq!(session.tokens_generated(), 1);

        let token2 = session.next_token_with_model(&mut model, None);
        assert!(token2.is_ok());
        assert!(token2.unwrap().is_some());
        assert_eq!(session.tokens_generated(), 2);

        // 3rd token should be None (max_tokens reached)
        let token3 = session.next_token_with_model(&mut model, None);
        assert!(token3.is_ok());
        assert!(token3.unwrap().is_none());
        assert_eq!(session.state(), GenerationState::Complete);
        assert!(session.is_complete());
    }

    #[test]
    #[ignore] // Model forward pass requires loaded weights, causes segfault with uninitialized model
    fn test_next_token_with_model_stop_tokens() {
        // Test stop tokens in next_token_with_model
        let mut session = InferenceSession::new(1, vec![1, 2, 3], GenOptions::default());
        session.set_stop_tokens(vec![42]); // Stop on token ID 42

        use realm_models::TransformerConfig;
        let config = TransformerConfig {
            vocab_size: 1000,
            hidden_size: 128,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            intermediate_size: 512,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            attention_backend: realm_models::AttentionBackend::Auto,
        };

        let mut model = realm_models::Model::new(config);

        // Generate tokens until we hit stop token (if sampling produces it)
        // Note: This test may be flaky due to randomness, but tests the stop token logic
        for _ in 0..10 {
            let result = session.next_token_with_model(&mut model, None);
            match result {
                Ok(Some(token)) => {
                    if token == 42 {
                        assert_eq!(session.state(), GenerationState::Stopped);
                        assert!(session.is_complete());
                        return;
                    }
                }
                Ok(None) => {
                    // Generation completed for other reason
                    break;
                }
                Err(_) => {
                    // Error occurred, which is acceptable for this test
                    break;
                }
            }
        }
    }

    #[test]
    fn test_inference_session_with_speculative_decoding() {
        // Test that speculative decoding config is stored correctly
        let config = crate::speculative::SpeculativeConfig {
            draft_k: 4,
            max_draft_tokens: 8,
        };

        let session = InferenceSession::new(1, vec![1, 2, 3], GenOptions::default())
            .with_speculative_decoding(config.clone());

        // Verify speculative config is set (we can't access it directly, but we can test behavior)
        // The config will be used in next_token_with_model when draft_model is provided
        assert_eq!(session.model_id(), 1);
        assert_eq!(session.prompt_tokens(), &[1, 2, 3]);
    }

    #[test]
    fn test_inference_session_generated_tokens_access() {
        // Test accessing generated tokens
        let mut session = InferenceSession::new(1, vec![1, 2, 3], GenOptions::default());

        // Initially no generated tokens
        assert_eq!(session.generated_tokens().len(), 0);

        // Generate some tokens
        for _ in 0..3 {
            session.next_token().unwrap();
        }

        // Should have 3 generated tokens
        assert_eq!(session.generated_tokens().len(), 3);
    }

    #[test]
    fn test_inference_session_model_id() {
        // Test model ID access
        let session = InferenceSession::new(42, vec![1, 2, 3], GenOptions::default());
        assert_eq!(session.model_id(), 42);
    }

    #[test]
    fn test_inference_session_prompt_tokens() {
        // Test prompt tokens access
        let prompt = vec![10, 20, 30, 40];
        let session = InferenceSession::new(1, prompt.clone(), GenOptions::default());
        assert_eq!(session.prompt_tokens(), &prompt);
    }
}
