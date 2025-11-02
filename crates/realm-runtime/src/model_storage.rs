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
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tracing::{debug, info};

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

    /// Extract TransformerConfig from metadata
    pub fn extract_config(&self) -> realm_models::TransformerConfig {
        let arch = &self.metadata.architecture;

        realm_models::TransformerConfig {
            vocab_size: self
                .metadata
                .metadata
                .get(&format!("{}.vocab_size", arch))
                .or_else(|| self.metadata.metadata.get("vocab_size"))
                .and_then(|v| v.as_u32())
                .unwrap_or(32000) as usize,

            hidden_size: self
                .metadata
                .metadata
                .get(&format!("{}.embedding_length", arch))
                .or_else(|| self.metadata.metadata.get(&format!("{}.hidden_size", arch)))
                .and_then(|v| v.as_u32())
                .unwrap_or(2048) as usize,

            num_layers: self
                .metadata
                .metadata
                .get(&format!("{}.block_count", arch))
                .or_else(|| self.metadata.metadata.get(&format!("{}.num_layers", arch)))
                .and_then(|v| v.as_u32())
                .unwrap_or(22) as usize,

            num_heads: self
                .metadata
                .metadata
                .get(&format!("{}.attention.head_count", arch))
                .or_else(|| self.metadata.metadata.get(&format!("{}.num_heads", arch)))
                .and_then(|v| v.as_u32())
                .unwrap_or(32) as usize,

            num_kv_heads: self
                .metadata
                .metadata
                .get(&format!("{}.attention.head_count_kv", arch))
                .or_else(|| {
                    self.metadata
                        .metadata
                        .get(&format!("{}.num_kv_heads", arch))
                })
                .and_then(|v| v.as_u32())
                .unwrap_or(4) as usize,

            intermediate_size: self
                .metadata
                .metadata
                .get(&format!("{}.feed_forward_length", arch))
                .or_else(|| {
                    self.metadata
                        .metadata
                        .get(&format!("{}.intermediate_size", arch))
                })
                .and_then(|v| v.as_u32())
                .unwrap_or(5632) as usize,

            max_seq_len: self
                .metadata
                .metadata
                .get(&format!("{}.context_length", arch))
                .or_else(|| self.metadata.metadata.get(&format!("{}.max_seq_len", arch)))
                .and_then(|v| v.as_u32())
                .unwrap_or(2048) as usize,

            rms_norm_eps: self
                .metadata
                .metadata
                .get(&format!("{}.attention.layer_norm_rms_epsilon", arch))
                .or_else(|| self.metadata.metadata.get("rms_norm_eps"))
                .and_then(|v| v.as_f32())
                .unwrap_or(1e-5),

            rope_theta: self
                .metadata
                .metadata
                .get(&format!("{}.rope.freq_base", arch))
                .or_else(|| self.metadata.metadata.get("rope_theta"))
                .and_then(|v| v.as_f32())
                .unwrap_or(10000.0),

            attention_backend: realm_models::AttentionBackend::Auto,
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
    /// * `model_id` - Optional model ID. If None, generates a deterministic ID from model hash.
    ///   If Some, uses provided ID (must be unique or match existing model).
    ///
    /// # Returns
    ///
    /// Model ID handle on success
    pub fn store_model(&self, gguf_bytes: &[u8], model_id: Option<u32>) -> Result<u32> {
        info!("Storing model from {} bytes of GGUF data", gguf_bytes.len());

        // Parse GGUF header and metadata
        let cursor = std::io::Cursor::new(gguf_bytes);
        let mut parser = realm_core::formats::gguf::GGUFParser::new(cursor);
        let metadata = parser
            .parse_header()
            .context("Failed to parse GGUF header")?;

        // Determine model ID
        let model_id = if let Some(requested_id) = model_id {
            // Consumer-provided ID: validate uniqueness
            let models = self.models.lock();
            if models.contains_key(&requested_id) {
                // Check if it's the same model (by hash)
                let existing = models.get(&requested_id).unwrap();
                let model_hash = Self::compute_model_hash(gguf_bytes)?;
                let existing_hash = Self::compute_model_hash_for_stored(existing)?;

                if model_hash == existing_hash {
                    // Same model, reuse ID (model sharing)
                    info!(
                        "Reusing existing model ID {} (model hash match)",
                        requested_id
                    );
                    requested_id
                } else {
                    return Err(anyhow!(
                        "Model ID {} already exists with different model. Use a different ID or remove existing model.",
                        requested_id
                    ));
                }
            } else {
                info!("Using consumer-provided model ID: {}", requested_id);
                requested_id
            }
        } else {
            // Auto-generate deterministic ID from model hash
            let model_hash = Self::compute_model_hash(gguf_bytes)?;
            let hash_based_id = Self::hash_to_model_id(&model_hash);

            // Check if model already exists with this hash-based ID
            let models = self.models.lock();
            if let Some(existing) = models.get(&hash_based_id) {
                let existing_hash = Self::compute_model_hash_for_stored(existing)?;
                if existing_hash == model_hash {
                    info!(
                        "Model already stored with ID {} (hash match)",
                        hash_based_id
                    );
                    return Ok(hash_based_id);
                }
            }

            info!(
                "Generated deterministic model ID {} from hash",
                hash_based_id
            );
            hash_based_id
        };

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

    /// Compute deterministic hash from GGUF bytes (uses metadata + tensor count)
    fn compute_model_hash(gguf_bytes: &[u8]) -> Result<u64> {
        use std::io::Cursor;
        let cursor = Cursor::new(gguf_bytes);
        let mut parser = realm_core::formats::gguf::GGUFParser::new(cursor);
        let metadata = parser
            .parse_header()
            .context("Failed to parse GGUF header for hash computation")?;

        let mut hasher = DefaultHasher::new();

        // Hash model metadata (name, architecture, size, etc.)
        metadata.architecture.hash(&mut hasher);
        metadata.tensors.len().hash(&mut hasher);

        // Hash tensor names and shapes (deterministic model signature)
        for tensor in &metadata.tensors {
            tensor.name.hash(&mut hasher);
            tensor.shape.dims().len().hash(&mut hasher);
            for &dim in tensor.shape.dims() {
                dim.hash(&mut hasher);
            }
            // Hash dtype manually since it doesn't implement Hash
            std::mem::discriminant(&tensor.dtype).hash(&mut hasher);
        }

        Ok(hasher.finish())
    }

    /// Compute hash for already-stored model
    fn compute_model_hash_for_stored(model: &StoredModel) -> Result<u64> {
        let mut hasher = DefaultHasher::new();

        // Use metadata from stored model
        model.metadata.architecture.hash(&mut hasher);
        model.metadata.tensors.len().hash(&mut hasher);

        for tensor in &model.metadata.tensors {
            tensor.name.hash(&mut hasher);
            tensor.shape.dims().len().hash(&mut hasher);
            for &dim in tensor.shape.dims() {
                dim.hash(&mut hasher);
            }
            // Hash dtype manually since it doesn't implement Hash
            std::mem::discriminant(&tensor.dtype).hash(&mut hasher);
        }

        Ok(hasher.finish())
    }

    /// Convert hash to model ID (u32 range, avoid 0)
    fn hash_to_model_id(hash: &u64) -> u32 {
        // Use lower 32 bits, ensure non-zero
        let id = (*hash as u32).max(1);
        if id == 0 {
            1 // Fallback for edge case
        } else {
            id
        }
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
