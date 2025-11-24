//! Tokenization helper functions using host functions

use anyhow::{Context, Result};
use realm_runtime::model_storage::get_global_model_storage;

/// Encode text to tokens using host function (via model storage tokenizer)
pub fn encode_text(model_id: u32, text: &str) -> Result<Vec<u32>> {
    let storage = get_global_model_storage().lock();
    let model = storage
        .get_model(model_id)
        .with_context(|| format!("Model {} not found", model_id))?;

    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("No tokenizer for model {}", model_id))?;

    tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))
}

/// Decode tokens to text using host function (via model storage tokenizer)
pub fn decode_tokens(model_id: u32, tokens: &[u32]) -> Result<String> {
    let storage = get_global_model_storage().lock();
    let model = storage
        .get_model(model_id)
        .with_context(|| format!("Model {} not found", model_id))?;

    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("No tokenizer for model {}", model_id))?;

    tokenizer
        .decode(tokens, false)
        .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))
}
