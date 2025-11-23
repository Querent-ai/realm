//! Speculative decoding integration for RuntimeManager
//!
//! This module provides integration between the speculative decoding framework
//! and the actual generation path in RuntimeManager.

use anyhow::{anyhow, Context, Result};
use realm_runtime::speculative::{DraftModel, SpeculativeConfig, SpeculativeDecoder, TargetModel};
use std::sync::{Arc, Mutex};

use crate::runtime_manager::TenantRuntime;
use crate::tokenization_helpers::{decode_tokens, encode_text};

/// Wrapper for draft model that uses WASM generation
pub struct DraftModelWrapper {
    runtime: Arc<Mutex<TenantRuntime>>,
    #[allow(dead_code)]
    model_id: u32, // Draft model ID in model storage (for future use)
}

impl DraftModelWrapper {
    pub fn new(runtime: Arc<Mutex<TenantRuntime>>, model_id: u32) -> Self {
        Self { runtime, model_id }
    }
}

impl DraftModel for DraftModelWrapper {
    fn generate_draft(&mut self, prompt: &[u32], k: usize) -> realm_core::error::Result<Vec<u32>> {
        // Decode prompt tokens to text using target model's tokenizer
        // (We use target model tokenizer since both models should use same tokenizer)
        let target_model_id = {
            let runtime_guard = self.runtime.lock().unwrap();
            runtime_guard.model_id().ok_or_else(|| {
                realm_core::error::Error::Runtime("Target model ID not available".to_string())
            })?
        };

        // Decode prompt tokens to text
        let prompt_text = decode_tokens(target_model_id, prompt)
            .map_err(|e| realm_core::error::Error::Runtime(format!("Decode failed: {}", e)))?;

        // Generate using draft model
        let mut runtime = self.runtime.lock().map_err(|e| {
            realm_core::error::Error::Runtime(format!("Failed to acquire runtime lock: {}", e))
        })?;
        let generated_text = runtime.generate(prompt_text).map_err(|e| {
            realm_core::error::Error::Runtime(format!("Draft generation failed: {}", e))
        })?;

        // Get draft model ID for tokenization
        let draft_model_id = runtime.draft_model_id().ok_or_else(|| {
            realm_core::error::Error::Runtime("Draft model ID not available".to_string())
        })?;
        drop(runtime);

        // Encode generated text back to tokens
        let generated_tokens = encode_text(draft_model_id, &generated_text)
            .map_err(|e| realm_core::error::Error::Runtime(format!("Encode failed: {}", e)))?;

        // Return first k tokens
        Ok(generated_tokens.into_iter().take(k).collect())
    }
}

/// Wrapper for target model that uses WASM generation
pub struct TargetModelWrapper {
    runtime: Arc<Mutex<TenantRuntime>>,
    #[allow(dead_code)]
    model_id: u32, // Target model ID in model storage (for future use)
}

impl TargetModelWrapper {
    pub fn new(runtime: Arc<Mutex<TenantRuntime>>, model_id: u32) -> Self {
        Self { runtime, model_id }
    }
}

impl TargetModel for TargetModelWrapper {
    fn verify_draft(
        &mut self,
        prompt: &[u32],
        draft_tokens: &[u32],
    ) -> realm_core::error::Result<Vec<u32>> {
        // Verify draft tokens using target model
        // In speculative decoding, we check if target model would generate the same tokens
        let target_model_id = {
            let runtime_guard = self.runtime.lock().unwrap();
            runtime_guard.model_id().ok_or_else(|| {
                realm_core::error::Error::Runtime("Target model ID not available".to_string())
            })?
        };

        // Build full sequence: prompt + draft tokens
        let mut full_sequence = prompt.to_vec();
        full_sequence.extend_from_slice(draft_tokens);
        let full_text = decode_tokens(target_model_id, &full_sequence)
            .map_err(|e| realm_core::error::Error::Runtime(format!("Decode failed: {}", e)))?;

        // Generate using target model to verify
        let mut runtime = self.runtime.lock().map_err(|e| {
            realm_core::error::Error::Runtime(format!("Failed to acquire runtime lock: {}", e))
        })?;
        let target_generated = runtime.generate(full_text).map_err(|e| {
            realm_core::error::Error::Runtime(format!("Target verification failed: {}", e))
        })?;
        drop(runtime);

        // Encode target model's generation
        let target_tokens = encode_text(target_model_id, &target_generated)
            .map_err(|e| realm_core::error::Error::Runtime(format!("Encode failed: {}", e)))?;

        // Compare draft tokens with target model's predictions
        // Accept tokens until first mismatch
        let mut accepted = Vec::new();
        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            if i < target_tokens.len() && target_tokens[i] == draft_token {
                accepted.push(draft_token);
            } else {
                // First mismatch - stop accepting
                break;
            }
        }

        // If all draft tokens were accepted, we can use them all
        // Otherwise, we return only the accepted prefix
        Ok(accepted)
    }
}

/// Generate with speculative decoding if draft model is configured
pub fn generate_with_speculative_decoding(
    runtime: Arc<Mutex<TenantRuntime>>,
    prompt: String,
    max_tokens: usize,
) -> Result<String> {
    // Check if draft model is configured and get model IDs
    let (target_model_id, draft_model_id) = {
        let runtime_guard = runtime
            .lock()
            .map_err(|e| anyhow!("Failed to acquire runtime lock: {}", e))?;
        if runtime_guard.draft_model_config().is_none() {
            drop(runtime_guard);
            // No draft model, use standard generation
            let mut runtime = runtime
                .lock()
                .map_err(|e| anyhow!("Failed to acquire runtime lock: {}", e))?;
            return runtime.generate(prompt);
        }
        let target_id = runtime_guard.model_id();
        let draft_id = runtime_guard.draft_model_id();
        (target_id, draft_id)
    };

    // Use speculative decoding
    let config = SpeculativeConfig {
        draft_k: 4,
        max_draft_tokens: 8,
    };

    let target_model_id = target_model_id
        .ok_or_else(|| anyhow::anyhow!("Target model ID not available for speculative decoding"))?;
    let draft_model_id = draft_model_id
        .ok_or_else(|| anyhow::anyhow!("Draft model ID not available for speculative decoding"))?;

    let draft_model = DraftModelWrapper::new(runtime.clone(), draft_model_id);
    let target_model = TargetModelWrapper::new(runtime.clone(), target_model_id);

    let mut decoder = SpeculativeDecoder::new(draft_model, target_model, config);

    // Tokenize prompt using target model's tokenizer
    let prompt_tokens = encode_text(target_model_id, &prompt)
        .context("Failed to tokenize prompt for speculative decoding")?;

    // Generate with speculative decoding
    let generated_tokens = decoder
        .generate(&prompt_tokens, max_tokens)
        .map_err(|e| anyhow::anyhow!("Speculative decoding failed: {}", e))?;

    // Detokenize using target model's tokenizer
    let result = decode_tokens(target_model_id, &generated_tokens)
        .context("Failed to detokenize speculative decoding result")?;

    Ok(result)
}
