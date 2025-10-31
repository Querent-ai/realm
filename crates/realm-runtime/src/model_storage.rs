//! Host-Side Model Storage for WASM
//!
//! This module implements the innovative host-side storage system that keeps
//! quantized model weights in the HOST runtime instead of WASM memory.
//!
//! ## Architecture
//!
//! ```text
//! WASM Module (Lightweight)          HOST Runtime (Heavy Storage)
//! ┌──────────────────────┐          ┌────────────────────────────┐
//! │ ModelHandle {        │          │ ModelStorage:              │
//! │   id: 42             │◄────────►│   models: HashMap<u32, _>  │
//! │   config: {...}      │          │   ┌──────────────────────┐ │
//! │ }                    │          │   │ Model 42:            │ │
//! │                      │          │   │   quantized weights  │ │
//! │ Activations: ~10MB   │          │   │   637MB (Q4_K_M)     │ │
//! │ KV Cache: ~20MB      │          │   └──────────────────────┘ │
//! │                      │          │                            │
//! │ Total: ~50MB         │          │   LRU Cache (dequantized): │
//! └──────────────────────┘          │   ~500MB configurable      │
//!                                    └────────────────────────────┘
//! ```
//!
//! ## Benefits
//!
//! - **Memory Efficiency**: Quantized weights stay compressed (4-8x savings)
//! - **Scalability**: WASM memory stays small (~50MB), supports 70B+ models
//! - **Multi-Tenant**: Multiple WASM instances share single HOST model
//! - **Performance**: LRU cache for frequently accessed dequantized tensors

use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use realm_core::formats::gguf::ModelMeta;
use realm_core::tensor::DataType;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tracing::{debug, info};

/// Global model ID counter
static NEXT_MODEL_ID: AtomicU32 = AtomicU32::new(1);

/// Quantized tensor stored in host memory
#[derive(Clone, Debug)]
pub struct QuantizedTensor {
    /// Raw quantized data (Q4_K_M, Q8_0, etc.)
    pub data: Vec<u8>,
    /// Data type
    pub dtype: DataType,
    /// Tensor shape [dim1, dim2, ...]
    pub shape: Vec<u64>,
    /// Original name from GGUF
    pub name: String,
}

impl QuantizedTensor {
    /// Create a new quantized tensor
    pub fn new(data: Vec<u8>, dtype: DataType, shape: Vec<u64>, name: String) -> Self {
        Self {
            data,
            dtype,
            shape,
            name,
        }
    }

    /// Get tensor size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get tensor element count
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// Complete model stored in host memory
#[derive(Clone, Debug)]
pub struct StoredModel {
    /// Unique model ID
    pub id: u32,
    /// Model metadata from GGUF
    pub metadata: ModelMeta,
    /// All tensors (quantized)
    pub tensors: HashMap<String, QuantizedTensor>,
    /// Total size in bytes
    pub total_size: usize,
}

impl StoredModel {
    /// Create a new stored model
    pub fn new(id: u32, metadata: ModelMeta) -> Self {
        Self {
            id,
            metadata,
            tensors: HashMap::new(),
            total_size: 0,
        }
    }

    /// Add a tensor to the model
    pub fn add_tensor(&mut self, tensor: QuantizedTensor) {
        let size = tensor.size_bytes();
        self.tensors.insert(tensor.name.clone(), tensor);
        self.total_size += size;
    }

    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&QuantizedTensor> {
        self.tensors.get(name)
    }

    /// Get tensor count
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

/// Global model storage
pub struct ModelStorage {
    /// Stored models by ID
    models: Arc<Mutex<HashMap<u32, StoredModel>>>,
}

impl ModelStorage {
    /// Create a new model storage
    pub fn new() -> Self {
        Self {
            models: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Store a complete model from GGUF bytes
    ///
    /// # Arguments
    ///
    /// * `gguf_bytes` - Complete GGUF file bytes
    ///
    /// # Returns
    ///
    /// Model ID handle on success
    pub fn store_model(&self, gguf_bytes: &[u8]) -> Result<u32> {
        info!("Storing model from {} bytes of GGUF data", gguf_bytes.len());

        // Parse GGUF header and metadata
        let cursor = std::io::Cursor::new(gguf_bytes);
        let mut parser = realm_core::formats::gguf::GGUFParser::new(cursor);
        let metadata = parser
            .parse_header()
            .context("Failed to parse GGUF header")?;

        // Allocate new model ID
        let model_id = NEXT_MODEL_ID.fetch_add(1, Ordering::SeqCst);
        info!("Allocated model ID: {}", model_id);

        // Create stored model
        let mut stored = StoredModel::new(model_id, metadata.clone());

        // Get tensor data offset
        let data_offset = parser
            .tensor_data_offset()
            .context("Failed to get tensor data offset")?;

        debug!(
            "Processing {} tensors with data offset {}",
            metadata.tensors.len(),
            data_offset
        );

        // Extract and store each tensor (keeping quantized format)
        for tensor_desc in &metadata.tensors {
            let tensor_start = data_offset + tensor_desc.offset;
            let tensor_size_bytes = tensor_desc.size_bytes;
            let tensor_end = tensor_start + tensor_size_bytes as u64;

            if tensor_end > gguf_bytes.len() as u64 {
                return Err(anyhow!(
                    "Tensor '{}' extends beyond GGUF file bounds",
                    tensor_desc.name
                ));
            }

            // Extract quantized data (no dequantization!)
            let tensor_data = gguf_bytes[tensor_start as usize..tensor_end as usize].to_vec();

            // Get shape as Vec<u64>
            let shape_u64: Vec<u64> = tensor_desc.shape.dims().iter().map(|&d| d as u64).collect();

            let quantized_tensor = QuantizedTensor::new(
                tensor_data,
                tensor_desc.dtype,
                shape_u64,
                tensor_desc.name.clone(),
            );

            debug!(
                "Stored tensor '{}': {} bytes, dtype={:?}, shape={:?}",
                tensor_desc.name,
                quantized_tensor.size_bytes(),
                tensor_desc.dtype,
                tensor_desc.shape
            );

            stored.add_tensor(quantized_tensor);
        }

        info!(
            "Model {} stored: {} tensors, {:.2} MB total",
            model_id,
            stored.tensor_count(),
            stored.total_size as f64 / 1024.0 / 1024.0
        );

        // Store in global map
        self.models.lock().insert(model_id, stored);

        Ok(model_id)
    }

    /// Get a reference to a stored model
    pub fn get_model(&self, model_id: u32) -> Result<StoredModel> {
        self.models
            .lock()
            .get(&model_id)
            .cloned()
            .ok_or_else(|| anyhow!("Model {} not found", model_id))
    }

    /// Get a specific tensor from a model
    pub fn get_tensor(&self, model_id: u32, tensor_name: &str) -> Result<QuantizedTensor> {
        let models = self.models.lock();
        let model = models
            .get(&model_id)
            .ok_or_else(|| anyhow!("Model {} not found", model_id))?;

        model
            .get_tensor(tensor_name)
            .cloned()
            .ok_or_else(|| anyhow!("Tensor '{}' not found in model {}", tensor_name, model_id))
    }

    /// Remove a model from storage
    pub fn remove_model(&self, model_id: u32) -> Result<()> {
        self.models
            .lock()
            .remove(&model_id)
            .ok_or_else(|| anyhow!("Model {} not found", model_id))?;

        info!("Model {} removed from storage", model_id);
        Ok(())
    }

    /// Get total storage size across all models
    pub fn total_size(&self) -> usize {
        self.models.lock().values().map(|m| m.total_size).sum()
    }

    /// Get number of stored models
    pub fn model_count(&self) -> usize {
        self.models.lock().len()
    }
}

impl Default for ModelStorage {
    fn default() -> Self {
        Self::new()
    }
}

// Global model storage instance
lazy_static::lazy_static! {
    /// Global singleton for host-side model storage
    pub static ref GLOBAL_MODEL_STORAGE: ModelStorage = ModelStorage::new();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_tensor_creation() {
        let tensor = QuantizedTensor::new(
            vec![0u8; 1024],
            DataType::Q4_K,
            vec![32, 32],
            "test.weight".to_string(),
        );

        assert_eq!(tensor.size_bytes(), 1024);
        assert_eq!(tensor.element_count(), 1024);
        assert_eq!(tensor.name, "test.weight");
    }

    #[test]
    fn test_model_storage() {
        let storage = ModelStorage::new();
        assert_eq!(storage.model_count(), 0);
        assert_eq!(storage.total_size(), 0);
    }
}
