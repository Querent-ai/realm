//! LoRA integration helper for model layers
//!
//! This module provides helper functions to apply LoRA adapters to model weights
//! during the forward pass. LoRA is applied via realm-runtime::lora module.

use realm_core::error::Result;

/// Helper to apply LoRA to attention weights if adapter is available
///
/// This is a placeholder that will be called from layer forward pass
/// when LoRA manager is available at runtime.
pub fn apply_lora_to_attention_weights(
    _base_weights: &[f32],
    _layer_name: &str,
    _lora_manager: Option<&dyn std::any::Any>, // Erased LoRAManager
) -> Result<Option<Vec<f32>>> {
    // LoRA integration is handled at runtime layer via realm-runtime::lora
    // This function provides a hook for future integration
    Ok(None)
}

/// Helper to apply LoRA to FFN weights if adapter is available
pub fn apply_lora_to_ffn_weights(
    _base_weights: &[f32],
    _layer_name: &str,
    _lora_manager: Option<&dyn std::any::Any>, // Erased LoRAManager
) -> Result<Option<Vec<f32>>> {
    // LoRA integration is handled at runtime layer via realm-runtime::lora
    // This function provides a hook for future integration
    Ok(None)
}

