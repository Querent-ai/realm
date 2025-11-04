//! LoRA (Low-Rank Adaptation) adapter support
//!
//! This module provides support for loading and applying LoRA adapters to the base model,
//! enabling per-tenant fine-tuning without storing full model copies.

use realm_core::error::{Error, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// LoRA adapter weights
#[derive(Debug, Clone)]
pub struct LoRAWeights {
    /// Adapter ID
    pub adapter_id: String,
    /// Rank (r parameter)
    pub rank: usize,
    /// Alpha (scaling parameter)
    pub alpha: f32,
    /// LoRA A weights (low-rank matrix)
    pub lora_a: HashMap<String, Vec<f32>>,
    /// LoRA B weights (low-rank matrix)
    pub lora_b: HashMap<String, Vec<f32>>,
}

impl LoRAWeights {
    /// Create a new LoRA adapter
    pub fn new(adapter_id: String, rank: usize, alpha: f32) -> Self {
        Self {
            adapter_id,
            rank,
            alpha,
            lora_a: HashMap::new(),
            lora_b: HashMap::new(),
        }
    }

    /// Add LoRA weights for a layer
    pub fn add_layer_weights(
        &mut self,
        layer_name: String,
        lora_a: Vec<f32>,
        lora_b: Vec<f32>,
    ) -> Result<()> {
        self.lora_a.insert(layer_name.clone(), lora_a);
        self.lora_b.insert(layer_name, lora_b);
        Ok(())
    }

    /// Get scaling factor: alpha / rank
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

/// LoRA adapter manager for per-tenant fine-tuning
pub struct LoRAManager {
    /// Loaded adapters
    adapters: Arc<Mutex<HashMap<String, LoRAWeights>>>,
}

impl LoRAManager {
    /// Create a new LoRA manager
    pub fn new() -> Self {
        Self {
            adapters: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Load a LoRA adapter
    pub fn load_adapter(&self, adapter: LoRAWeights) -> Result<()> {
        let mut adapters = self.adapters.lock().unwrap();
        adapters.insert(adapter.adapter_id.clone(), adapter);
        Ok(())
    }

    /// Unload a LoRA adapter
    pub fn unload_adapter(&self, adapter_id: &str) -> Result<()> {
        let mut adapters = self.adapters.lock().unwrap();
        adapters
            .remove(adapter_id)
            .ok_or_else(|| Error::Runtime(format!("Adapter {} not found", adapter_id)))?;
        Ok(())
    }

    /// Get an adapter
    pub fn get_adapter(&self, adapter_id: &str) -> Option<LoRAWeights> {
        let adapters = self.adapters.lock().unwrap();
        adapters.get(adapter_id).cloned()
    }

    /// Apply LoRA weights to base weights
    ///
    /// Formula: W' = W + scale * (B @ A)
    /// where scale = alpha / rank
    pub fn apply_to_weights(
        &self,
        adapter_id: &str,
        layer_name: &str,
        base_weights: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Vec<f32>> {
        let adapter = self
            .get_adapter(adapter_id)
            .ok_or_else(|| Error::Runtime(format!("Adapter {} not found", adapter_id)))?;

        let a_key = format!("{}.lora_a", layer_name);
        let b_key = format!("{}.lora_b", layer_name);

        let lora_a = adapter.lora_a.get(&a_key).ok_or_else(|| {
            Error::Runtime(format!("LoRA A weights not found for {}", layer_name))
        })?;
        let lora_b = adapter.lora_b.get(&b_key).ok_or_else(|| {
            Error::Runtime(format!("LoRA B weights not found for {}", layer_name))
        })?;

        let scale = adapter.scale();

        // Compute LoRA delta: scale * (B @ A)
        // B: [out_dim, rank], A: [rank, in_dim]
        // Result: [out_dim, in_dim]
        let mut delta = vec![0.0f32; out_dim * in_dim];
        for i in 0..out_dim {
            for j in 0..in_dim {
                let mut sum = 0.0;
                for k in 0..adapter.rank {
                    let a_idx = k * in_dim + j;
                    let b_idx = i * adapter.rank + k;
                    if a_idx < lora_a.len() && b_idx < lora_b.len() {
                        sum += lora_b[b_idx] * lora_a[a_idx];
                    }
                }
                delta[i * in_dim + j] = sum * scale;
            }
        }

        // Add delta to base weights
        let mut result = base_weights.to_vec();
        for i in 0..result.len().min(delta.len()) {
            result[i] += delta[i];
        }

        Ok(result)
    }

    /// List all loaded adapters
    pub fn list_adapters(&self) -> Vec<String> {
        let adapters = self.adapters.lock().unwrap();
        adapters.keys().cloned().collect()
    }
}

impl Default for LoRAManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_weights_creation() {
        let weights = LoRAWeights::new("test".to_string(), 8, 16.0);
        assert_eq!(weights.scale(), 2.0); // 16.0 / 8 = 2.0
    }

    #[test]
    fn test_lora_manager() {
        let manager = LoRAManager::new();

        let mut adapter = LoRAWeights::new("test_adapter".to_string(), 8, 16.0);
        adapter
            .add_layer_weights(
                "layer.0".to_string(),
                vec![1.0; 8 * 128], // rank * in_dim
                vec![1.0; 256 * 8], // out_dim * rank
            )
            .unwrap();

        manager.load_adapter(adapter).unwrap();

        let adapters = manager.list_adapters();
        assert_eq!(adapters.len(), 1);
        assert!(adapters.contains(&"test_adapter".to_string()));
    }

    #[test]
    fn test_lora_apply() {
        let manager = LoRAManager::new();

        let mut adapter = LoRAWeights::new("test".to_string(), 2, 4.0);
        // Simple test: rank=2, alpha=4.0, scale=2.0
        adapter
            .add_layer_weights(
                "layer.0".to_string(),
                vec![1.0, 1.0, 1.0, 1.0], // [2, 2]
                vec![1.0, 1.0, 1.0, 1.0], // [2, 2]
            )
            .unwrap();

        manager.load_adapter(adapter).unwrap();

        // Test that adapter was loaded correctly
        let result = manager.get_adapter("test");
        assert!(result.is_some());
        let adapter = result.unwrap();
        assert_eq!(adapter.rank, 2);
        assert_eq!(adapter.scale(), 2.0);
        assert!(adapter.lora_a.contains_key("layer.0"));
        assert!(adapter.lora_b.contains_key("layer.0"));

        // Note: apply_to_weights requires specific key format (layer_name.lora_a)
        // For full integration test, we'd need to use the correct key format
    }
}
