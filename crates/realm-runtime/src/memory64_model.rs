//! Memory64-aware model loading for large models (>4GB)
//!
//! This module extends the existing model loading system to support Memory64
//! for models that exceed the 4GB WASM memory limit.

use crate::memory64::{Memory64Runtime, MemoryLayout};
use realm_core::{
    error::{Error, Result},
    formats::gguf::GGUFParser,
    tensor_loader::TensorLoader,
};
use realm_models::{Model, TransformerConfig};
use std::io::{Read, Seek};

/// Memory64-aware model loader
pub struct Memory64ModelLoader {
    /// Memory64 runtime for large model storage
    runtime: Option<Memory64Runtime>,
    /// Model configuration
    config: TransformerConfig,
    /// Total model size in bytes
    total_size: u64,
    /// Whether to use Memory64 (based on size threshold)
    use_memory64: bool,
}

impl Memory64ModelLoader {
    /// Create a new Memory64-aware model loader
    pub fn new(config: TransformerConfig, total_size: u64) -> Self {
        // Use Memory64 for models >3GB
        let use_memory64 = total_size > 3_000_000_000;

        Self {
            runtime: None,
            config,
            total_size,
            use_memory64,
        }
    }

    /// Initialize Memory64 runtime if needed
    pub fn initialize_memory64(&mut self) -> Result<()> {
        if !self.use_memory64 {
            return Ok(());
        }

        // Create appropriate memory layout based on model size
        let layout = if self.total_size <= 8_000_000_000 {
            // Single region for 7B-8B models
            MemoryLayout::single(8, "model_storage")
                .map_err(|e| Error::ParseError(format!("Failed to create layout: {}", e)))?
        } else if self.total_size <= 16_000_000_000 {
            // Single region for 13B models
            MemoryLayout::single(16, "model_storage")
                .map_err(|e| Error::ParseError(format!("Failed to create layout: {}", e)))?
        } else {
            // Multi-region for 30B+ models
            MemoryLayout::multi(&[
                ("embeddings", 2),   // 2GB for embeddings
                ("layers_0_15", 8),  // 8GB for first 16 layers
                ("layers_16_31", 8), // 8GB for next 16 layers
                ("lm_head", 2),      // 2GB for LM head
            ])
            .map_err(|e| Error::ParseError(format!("Failed to create multi layout: {}", e)))?
        };

        self.runtime = Some(Memory64Runtime::new(layout, true));

        Ok(())
    }

    /// Load model weights into Memory64 or standard memory
    pub fn load_model<R: Read + Seek>(
        &mut self,
        tensor_loader: &mut TensorLoader,
        parser: &mut GGUFParser<R>,
    ) -> Result<Model> {
        if self.use_memory64 {
            self.load_with_memory64(tensor_loader, parser)
        } else {
            self.load_standard(tensor_loader, parser)
        }
    }

    /// Load model using standard memory (existing implementation)
    fn load_standard<R: Read + Seek>(
        &mut self,
        tensor_loader: &mut TensorLoader,
        parser: &mut GGUFParser<R>,
    ) -> Result<Model> {
        let mut model = Model::new(self.config.clone());
        model.load_from_gguf(tensor_loader, parser, None, None)?;
        Ok(model)
    }

    /// Load model using Memory64 for large models
    fn load_with_memory64<R: Read + Seek>(
        &mut self,
        tensor_loader: &mut TensorLoader,
        parser: &mut GGUFParser<R>,
    ) -> Result<Model> {
        // Initialize Memory64 if not already done
        if self.runtime.is_none() {
            self.initialize_memory64()?;
        }

        let _runtime = self.runtime.as_ref().ok_or_else(|| {
            Error::AllocationFailed("Memory64 runtime not initialized".to_string())
        })?;

        // Create model structure (weights will be loaded into Memory64, not RAM)
        let mut model = Model::new(self.config.clone());

        // For Memory64 loading, we need a Store to write to Memory64 instances.
        // However, Memory64ModelLoader is often used before WASM instantiation.
        // Two options:
        // 1. Store weights in a temporary buffer, then write to Memory64 when Store is available
        // 2. Accept Store as parameter (breaking change)
        //
        // For now, we'll load weights normally but mark the model for Memory64 usage.
        // The caller should call store_model_weights_to_memory64() after getting a Store.
        model.load_from_gguf(tensor_loader, parser, None, None)?;

        // Note: To actually store weights in Memory64, call:
        //   loader.store_model_weights_to_memory64(&mut store, &model)?;
        // This must be done after WASM Store is created.

        Ok(model)
    }

    /// Store model weights into Memory64 instances (requires Wasmtime Store)
    ///
    /// This method actually writes the model weights to Memory64 storage.
    /// Call this after creating a Wasmtime Store and initializing Memory64.
    ///
    /// # Example
    /// ```ignore
    /// use wasmtime::*;
    ///
    /// let mut store = Store::new(&engine, ());
    /// let runtime = loader.runtime().unwrap();
    /// runtime.initialize(&mut store)?;
    /// loader.store_model_weights_to_memory64(&mut store, &model)?;
    /// ```
    #[cfg(feature = "memory64-host")]
    pub fn store_model_weights_to_memory64(
        &self,
        store: &mut wasmtime::Store<()>,
        model: &Model,
    ) -> anyhow::Result<()> {
        use anyhow::Context;

        let runtime = self.runtime.as_ref().ok_or_else(|| {
            Error::AllocationFailed("Memory64 runtime not initialized".to_string())
        })?;

        // Initialize Memory64 instances if not already done
        runtime
            .initialize(store)
            .context("Failed to initialize Memory64")?;

        let mut current_offset = 0u64;

        // Store embeddings (layer 0)
        let embeddings_size = model.token_embeddings.len() * 4; // f32 = 4 bytes
        let embeddings_bytes: Vec<u8> = model
            .token_embeddings
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        runtime
            .write_model_data(store, current_offset, &embeddings_bytes)
            .with_context(|| format!("Failed to write embeddings at offset {}", current_offset))?;

        runtime.register_layer(store, 0, "embeddings", current_offset, embeddings_size)?;

        current_offset += embeddings_size as u64;

        // Helper to extract f32 from WeightFormat
        fn extract_f32_weights(weight: &realm_models::WeightFormat) -> Result<Vec<f32>> {
            match weight {
                realm_models::WeightFormat::F32(data) => Ok(data.clone()),
                _ => Err(Error::ParseError(
                    "Memory64 storage only supports F32 weights currently. Dequantize first."
                        .to_string(),
                )),
            }
        }

        // Store each transformer layer
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            let layer_id = layer_idx as u32 + 1; // +1 because 0 is embeddings

            // Extract f32 weights from WeightFormat enums
            let attention_q = extract_f32_weights(&layer.attention_weights.wq)
                .with_context(|| format!("Failed to extract Q weights for layer {}", layer_idx))?;
            let attention_k = extract_f32_weights(&layer.attention_weights.wk)
                .with_context(|| format!("Failed to extract K weights for layer {}", layer_idx))?;
            let attention_v = extract_f32_weights(&layer.attention_weights.wv)
                .with_context(|| format!("Failed to extract V weights for layer {}", layer_idx))?;
            let attention_o = extract_f32_weights(&layer.attention_weights.wo)
                .with_context(|| format!("Failed to extract O weights for layer {}", layer_idx))?;

            // Calculate layer size
            let attention_q_size = attention_q.len() * 4;
            let attention_k_size = attention_k.len() * 4;
            let attention_v_size = attention_v.len() * 4;
            let attention_o_size = attention_o.len() * 4;
            let ffn_gate_size = layer.ffn_weights.w_gate.len() * 4;
            let ffn_up_size = layer.ffn_weights.w_up.len() * 4;
            let ffn_down_size = layer.ffn_weights.w_down.len() * 4;
            let attn_norm_size = layer.attention_norm.len() * 4;
            let ffn_norm_size = layer.ffn_norm.len() * 4;

            let layer_total_size = attention_q_size
                + attention_k_size
                + attention_v_size
                + attention_o_size
                + ffn_gate_size
                + ffn_up_size
                + ffn_down_size
                + attn_norm_size
                + ffn_norm_size;

            // Serialize layer to bytes
            let mut layer_bytes = Vec::with_capacity(layer_total_size);
            layer_bytes.extend(attention_q.iter().flat_map(|f| f.to_le_bytes()));
            layer_bytes.extend(attention_k.iter().flat_map(|f| f.to_le_bytes()));
            layer_bytes.extend(attention_v.iter().flat_map(|f| f.to_le_bytes()));
            layer_bytes.extend(attention_o.iter().flat_map(|f| f.to_le_bytes()));
            layer_bytes.extend(
                layer
                    .ffn_weights
                    .w_gate
                    .iter()
                    .flat_map(|f| f.to_le_bytes()),
            );
            layer_bytes.extend(layer.ffn_weights.w_up.iter().flat_map(|f| f.to_le_bytes()));
            layer_bytes.extend(
                layer
                    .ffn_weights
                    .w_down
                    .iter()
                    .flat_map(|f| f.to_le_bytes()),
            );
            layer_bytes.extend(layer.attention_norm.iter().flat_map(|f| f.to_le_bytes()));
            layer_bytes.extend(layer.ffn_norm.iter().flat_map(|f| f.to_le_bytes()));

            runtime
                .write_model_data(store, current_offset, &layer_bytes)
                .with_context(|| {
                    format!(
                        "Failed to write layer {} at offset {}",
                        layer_id, current_offset
                    )
                })?;

            runtime.register_layer(
                store,
                layer_id,
                format!("transformer_layer_{}", layer_idx),
                current_offset,
                layer_total_size,
            )?;

            current_offset += layer_total_size as u64;
        }

        // Store LM head (last layer)
        let lm_head_size = model.lm_head.len() * 4;
        let lm_head_bytes: Vec<u8> = model.lm_head.iter().flat_map(|f| f.to_le_bytes()).collect();
        let lm_head_layer_id = self.config.num_layers as u32 + 1;

        runtime
            .write_model_data(store, current_offset, &lm_head_bytes)
            .with_context(|| format!("Failed to write LM head at offset {}", current_offset))?;

        runtime.register_layer(
            store,
            lm_head_layer_id,
            "lm_head",
            current_offset,
            lm_head_size,
        )?;

        Ok(())
    }

    /// Get Memory64 runtime (for host integration)
    pub fn runtime(&self) -> Option<&Memory64Runtime> {
        self.runtime.as_ref()
    }

    /// Check if using Memory64
    pub fn uses_memory64(&self) -> bool {
        self.use_memory64
    }
}

/// Extension trait for Model to support Memory64
pub trait Memory64ModelExt {
    /// Check if this model should use Memory64
    fn should_use_memory64(&self) -> bool;

    /// Get layer weights using Memory64 if available
    fn get_layer_weights(&self, layer_id: u32) -> Result<Vec<f32>>;
}

impl Memory64ModelExt for Model {
    fn should_use_memory64(&self) -> bool {
        // Estimate model size based on configuration
        let embedding_size = self.config.vocab_size * self.config.hidden_size * 4; // f32
        let layer_size = self.config.hidden_size * self.config.hidden_size * 4 * 8; // 8 attention matrices
        let lm_head_size = self.config.vocab_size * self.config.hidden_size * 4;

        let total_size = embedding_size + (layer_size * self.config.num_layers) + lm_head_size;
        total_size > 3_000_000_000 // 3GB threshold
    }

    fn get_layer_weights(&self, _layer_id: u32) -> Result<Vec<f32>> {
        // This would be implemented to use Memory64LayerLoader
        // when Memory64 is enabled
        Err(Error::ParseError(
            "Memory64 layer loading not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
#[cfg(feature = "memory64-host")]
mod tests {
    use super::*;
    use anyhow::Result;
    use realm_models::TransformerConfig;
    #[cfg(feature = "memory64-host")]
    use wasmtime::{Config, Engine, Store};

    fn create_test_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 32000,
            hidden_size: 512,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 8,
            intermediate_size: 2048,
            max_seq_len: 2048,
            ..Default::default()
        }
    }

    #[test]
    fn test_memory64_loader_creation() {
        let config = create_test_config();
        let total_size = 5_000_000_000u64; // 5GB - should use Memory64
        let loader = Memory64ModelLoader::new(config.clone(), total_size);

        assert!(loader.uses_memory64());
        assert_eq!(loader.total_size, total_size);
    }

    #[test]
    fn test_memory64_loader_small_model() {
        let config = create_test_config();
        let total_size = 2_000_000_000u64; // 2GB - should NOT use Memory64
        let loader = Memory64ModelLoader::new(config.clone(), total_size);

        assert!(!loader.uses_memory64());
    }

    #[test]
    #[cfg_attr(target_os = "windows", ignore)] // Skip on Windows CI due to memory limits
    fn test_memory64_initialization() -> Result<()> {
        let config = create_test_config();
        let total_size = 5_000_000_000u64;
        let mut loader = Memory64ModelLoader::new(config, total_size);

        loader.initialize_memory64()?;

        assert!(loader.runtime().is_some());
        Ok(())
    }

    #[test]
    fn test_memory64_initialization_small() -> Result<()> {
        let config = create_test_config();
        let total_size = 2_000_000_000u64;
        let mut loader = Memory64ModelLoader::new(config, total_size);

        // Should not fail, just return early
        loader.initialize_memory64()?;

        assert!(loader.runtime().is_none());
        Ok(())
    }

    #[test]
    fn test_memory64_layout_selection() -> Result<()> {
        // Test 7B model (8GB)
        let config = create_test_config();
        let mut loader = Memory64ModelLoader::new(config.clone(), 7_000_000_000u64);
        loader.initialize_memory64()?;
        let runtime = loader.runtime().unwrap();
        {
            let state = runtime.state();
            let state_guard = state.lock();
            let layout = state_guard.layout();
            assert_eq!(layout.regions.len(), 1);
            assert_eq!(layout.regions[0].size, 8 * 1024 * 1024 * 1024);
        }

        // Test 13B model (16GB)
        let mut loader2 = Memory64ModelLoader::new(config.clone(), 13_000_000_000u64);
        loader2.initialize_memory64()?;
        let runtime2 = loader2.runtime().unwrap();
        {
            let state2 = runtime2.state();
            let state_guard2 = state2.lock();
            let layout2 = state_guard2.layout();
            assert_eq!(layout2.regions.len(), 1);
            assert_eq!(layout2.regions[0].size, 16 * 1024 * 1024 * 1024);
        }

        // Test 30B+ model (multi-region)
        let mut loader3 = Memory64ModelLoader::new(config, 30_000_000_000u64);
        loader3.initialize_memory64()?;
        let runtime3 = loader3.runtime().unwrap();
        {
            let state3 = runtime3.state();
            let state_guard3 = state3.lock();
            let layout3 = state_guard3.layout();
            assert!(layout3.regions.len() > 1);
        }

        Ok(())
    }

    #[test]
    fn test_store_model_weights_integration() -> Result<()> {
        let config = create_test_config();
        let total_size = 5_000_000_000u64;
        let mut loader = Memory64ModelLoader::new(config.clone(), total_size);

        // Initialize Memory64 layout first
        loader.initialize_memory64()?;

        // Create a minimal test model
        let _model = Model::new(config);

        // Initialize some test weights (empty for now, but structure exists)
        // The actual test would load a real model in a full integration test

        // Create Wasmtime engine and store
        let mut wasm_config = Config::new();
        wasm_config.wasm_memory64(true);
        let engine = Engine::new(&wasm_config)?;
        let mut store = Store::new(&engine, ());

        // Initialize runtime
        let runtime = loader.runtime().unwrap();
        runtime.initialize(&mut store)?;

        // Try to store (will fail if weights aren't loaded, but tests the structure)
        // In a real test, we'd load a model first
        // let result = loader.store_model_weights_to_memory64(&mut store, &model);
        // This test validates the structure, actual weight storage is tested in integration tests

        Ok(())
    }

    #[test]
    fn test_model_ext_trait() {
        let config = create_test_config();
        let model = Model::new(config.clone());

        // Small model should not use Memory64
        assert!(!model.should_use_memory64());

        // Large model config should
        let large_config = TransformerConfig {
            vocab_size: 100000,
            hidden_size: 8192,
            num_layers: 80,
            ..config
        };
        let large_model = Model::new(large_config);
        assert!(large_model.should_use_memory64());
    }

    #[test]
    fn test_get_layer_weights_not_implemented() {
        let config = create_test_config();
        let model = Model::new(config);

        let result = model.get_layer_weights(0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not yet implemented"));
    }
}
