use std::collections::VecDeque;
/// Inference session management with token streaming
use wasm_chord_core::error::Result;

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
        }
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
        matches!(self.state, GenerationState::Complete | GenerationState::Stopped)
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
    /// * `model` - The transformer model to use for inference
    ///
    /// # Returns
    /// Next token ID, or None if generation is complete
    pub fn next_token_with_model(
        &mut self,
        model: &mut crate::transformer::Model,
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

        // Run model forward pass
        let logits = model.forward(&input_tokens, input_tokens.len() - 1)?;

        // Extract logits for last position
        let vocab_size = model.config.vocab_size;
        let mut last_logits = logits[logits.len() - vocab_size..].to_vec();

        // Sample next token
        let token_id = self
            .logits_processor
            .sample(&mut last_logits)
            .map_err(wasm_chord_core::error::Error::ParseError)?;

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
        let options = GenOptions { max_tokens: 5, ..Default::default() };

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
}
