//! Memory64-aware model loading for large models (>4GB)
//!
//! This module extends the existing model loading system to support Memory64
//! for models that exceed the 4GB WASM memory limit.

use crate::memory64::{Memory64Runtime, MemoryLayout};
use crate::transformer::{Model, TransformerConfig};
use std::io::{Read, Seek};
use realm_core::{
    error::{Error, Result},
    formats::gguf::GGUFParser,
    tensor_loader::TensorLoader,
};

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

        Self { runtime: None, config, total_size, use_memory64 }
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
        model.load_from_gguf(tensor_loader, parser)?;
        Ok(model)
    }

    /// Load model using Memory64 for large models
    fn load_with_memory64<R: Read + Seek>(
        &mut self,
        tensor_loader: &mut TensorLoader,
        parser: &mut GGUFParser<R>,
    ) -> Result<Model> {
        // For now, fall back to standard loading
        // TODO: Implement actual Memory64 loading when Store is available
        println!("⚠️  Memory64 loading not yet fully implemented, using standard loading");
        self.load_standard(tensor_loader, parser)
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
        Err(Error::ParseError("Memory64 layer loading not yet implemented".to_string()))
    }
}
