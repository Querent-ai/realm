//! Integration helpers for LoRA, Speculative Decoding, and Continuous Batching
//!
//! This module provides helper functions to complete the integration of
//! advanced features into the runtime manager.

use anyhow::{Context, Result};
use realm_models::Model;
use realm_runtime::lora::LoRAManager;
use realm_runtime::lora_integration::apply_lora_to_model;
use std::path::PathBuf;
use tracing::info;

/// Apply LoRA adapter to a model if configured
///
/// This should be called after loading a model when LoRA adapter is configured.
/// For WASM-loaded models, this would need to be done via host functions.
/// For native models, this can be called directly.
pub fn apply_lora_if_configured(
    model: &mut Model,
    lora_manager: &LoRAManager,
    adapter_id: Option<&str>,
) -> Result<()> {
    if let Some(adapter_id) = adapter_id {
        info!("Applying LoRA adapter '{}' to model", adapter_id);
        apply_lora_to_model(model, lora_manager, adapter_id)
            .context("Failed to apply LoRA adapter")?;
    }
    Ok(())
}

/// Load draft model for speculative decoding
///
/// Returns the loaded draft model if configured, None otherwise.
pub fn load_draft_model_if_configured(
    draft_model_path: Option<&PathBuf>,
) -> Result<Option<Model>> {
    if let Some(path) = draft_model_path {
        info!("Loading draft model for speculative decoding: {:?}", path);
        
        // Read model file
        let model_bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read draft model: {:?}", path))?;
        
        // Parse GGUF header to get config
        use realm_core::formats::gguf::GGUFParser;
        use realm_core::tensor_loader::TensorLoader;
        use std::io::Cursor;
        
        let cursor = Cursor::new(&model_bytes);
        let mut parser = GGUFParser::new(cursor);
        let meta = parser.parse_header()
            .context("Failed to parse draft model GGUF header")?;
        
        // Extract config
        let config_data = parser.extract_config()
            .ok_or_else(|| anyhow::anyhow!("Failed to extract config from draft model"))?;
        let config: realm_models::TransformerConfig = config_data.into();
        
        // Load model
        let mut model = Model::new(config);
        let data_offset = parser.tensor_data_offset()
            .context("Failed to get tensor data offset")?;
        let mut tensor_loader = TensorLoader::new(data_offset);
        
        model.load_from_gguf(&mut tensor_loader, &mut parser, None, None)
            .context("Failed to load draft model weights")?;
        
        info!("Draft model loaded successfully");
        Ok(Some(model))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_lora_if_configured_none() {
        let mut model = Model::new(realm_models::TransformerConfig::default());
        let manager = LoRAManager::new();
        let result = apply_lora_if_configured(&mut model, &manager, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_draft_model_if_configured_none() {
        let result = load_draft_model_if_configured(None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }
}

