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
use realm_core::Tokenizer;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tracing::{debug, info, warn};

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
    /// LoRA adapter ID (if applied)
    pub lora_adapter_id: Option<String>,
    /// Tokenizer for this model (for tokenization host functions)
    pub tokenizer: Option<Tokenizer>,
    /// Original GGUF bytes (for loading Model instances for inference)
    /// This is optional - if None, Model loading will fail
    pub gguf_bytes: Option<Vec<u8>>,
}

impl StoredModel {
    /// Create a new stored model
    pub fn new(id: u32, metadata: ModelMeta) -> Self {
        Self {
            id,
            metadata,
            tensors: HashMap::new(),
            total_size: 0,
            lora_adapter_id: None,
            tokenizer: None,
            gguf_bytes: None,
        }
    }

    /// Set GGUF bytes (for Model loading)
    pub fn set_gguf_bytes(&mut self, bytes: Vec<u8>) {
        self.gguf_bytes = Some(bytes);
    }

    /// Get GGUF bytes
    pub fn gguf_bytes(&self) -> Option<&[u8]> {
        self.gguf_bytes.as_deref()
    }

    /// Load a realm_models::Model instance from stored GGUF bytes
    ///
    /// This creates a fully-loaded Model instance that can be used for inference.
    /// Requires GGUF bytes to be stored (via set_gguf_bytes).
    pub fn load_model_instance(&self) -> Result<realm_models::Model> {
        use realm_core::formats::gguf::GGUFParser;
        use realm_core::tensor_loader::TensorLoader;
        use std::io::Cursor;

        let gguf_bytes = self
            .gguf_bytes
            .as_ref()
            .ok_or_else(|| anyhow!("No GGUF bytes stored for model {}", self.id))?;

        // Parse GGUF header
        let cursor = Cursor::new(gguf_bytes);
        let mut parser = GGUFParser::new(cursor);
        parser
            .parse_header()
            .context("Failed to parse GGUF header")?;

        // Extract config
        let config_data = parser
            .extract_config()
            .ok_or_else(|| anyhow!("Failed to extract config"))?;
        let config: realm_models::TransformerConfig = config_data.into();

        // Get tensor data offset
        let data_offset = parser
            .tensor_data_offset()
            .context("Failed to get tensor data offset")?;

        // Create tensor loader and register all tensors
        let mut tensor_loader = TensorLoader::new(data_offset);
        for tensor_desc in &self.metadata.tensors {
            tensor_loader.register_tensor(
                tensor_desc.name.clone(),
                tensor_desc.clone(),
                tensor_desc.offset,
            );
        }

        // Re-parse for loading (parser needs to be reset)
        let cursor = Cursor::new(gguf_bytes);
        let mut parser = GGUFParser::new(cursor);
        parser
            .parse_header()
            .context("Failed to re-parse GGUF header")?;

        // Create and load model
        let mut model = realm_models::Model::new(config);
        model
            .load_from_gguf(&mut tensor_loader, &mut parser, None, None)
            .context("Failed to load model weights")?;

        Ok(model)
    }

    /// Set tokenizer for this model
    pub fn set_tokenizer(&mut self, tokenizer: Tokenizer) {
        self.tokenizer = Some(tokenizer);
    }

    /// Get tokenizer for this model
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// Set LoRA adapter ID
    pub fn set_lora_adapter(&mut self, adapter_id: String) {
        let adapter_id_for_log = adapter_id.clone();
        self.lora_adapter_id = Some(adapter_id);
        info!(
            "Model {}: LoRA adapter '{}' marked for application",
            self.id, adapter_id_for_log
        );
    }

    /// Get LoRA adapter ID
    pub fn lora_adapter_id(&self) -> Option<&str> {
        self.lora_adapter_id.as_deref()
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
                .or_else(|| self.metadata.metadata.get("max_seq_len"))
                .and_then(|v| v.as_u32())
                .unwrap_or(2048) as usize,

            rope_theta: self
                .metadata
                .metadata
                .get(&format!("{}.rope.theta", arch))
                .or_else(|| self.metadata.metadata.get("rope_theta"))
                .and_then(|v| v.as_f32())
                .unwrap_or(10000.0),
            attention_backend: realm_models::AttentionBackend::Auto,
            rms_norm_eps: 1e-5,
        }
    }

    /// Add a tensor to the model
    pub fn add_tensor(&mut self, name: String, tensor: QuantizedTensor) {
        let size = tensor.size_bytes();
        self.tensors.insert(name, tensor);
        self.total_size += size;
    }

    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&QuantizedTensor> {
        self.tensors.get(name)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Get tensor count
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

/// Global model storage (thread-safe)
pub struct ModelStorage {
    /// Stored models indexed by ID
    models: HashMap<u32, StoredModel>,
    /// Next available model ID
    next_id: u32,
    /// Model ID counter (for auto-generation)
    counter: u32,
    /// Cached loaded Model instances (for inference)
    /// This avoids reloading models on every inference request
    /// Uses Arc<Mutex<>> to allow sharing across threads while dropping storage lock
    model_cache: HashMap<u32, Arc<Mutex<realm_models::Model>>>,
}

impl Default for ModelStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelStorage {
    /// Create a new model storage
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            next_id: 1,
            counter: 1,
            model_cache: HashMap::new(),
        }
    }

    /// Get or load a Model instance for inference
    ///
    /// This will use the cached instance if available, otherwise load from GGUF bytes.
    /// Returns Arc<Mutex<Model>> so it can be shared across threads and the storage lock
    /// can be dropped before inference.
    pub fn get_model_for_inference(
        &mut self,
        model_id: u32,
    ) -> Result<Arc<Mutex<realm_models::Model>>> {
        // Check cache first
        if let Some(cached) = self.model_cache.get(&model_id) {
            debug!("Using cached Model instance (model_id: {})", model_id);
            return Ok(cached.clone());
        }

        // Load from stored model
        let stored_model = self
            .get_model(model_id)
            .context("Model not found in storage")?;

        let model = stored_model
            .load_model_instance()
            .context("Failed to load model instance from GGUF bytes")?;

        info!(
            "Loaded Model instance for inference (model_id: {}) - caching for future requests",
            model_id
        );
        let model_arc = Arc::new(Mutex::new(model));
        self.model_cache.insert(model_id, model_arc.clone());

        Ok(model_arc)
    }

    /// Clear model cache (useful for memory management)
    pub fn clear_model_cache(&mut self) {
        let count = self.model_cache.len();
        self.model_cache.clear();
        info!("Cleared {} cached Model instances", count);
    }

    /// Remove a model from cache (called when model is removed)
    pub fn remove_from_cache(&mut self, model_id: u32) {
        self.model_cache.remove(&model_id);
    }

    /// Store a model from GGUF bytes
    pub fn store_model(&mut self, gguf_bytes: &[u8], requested_id: Option<u32>) -> Result<u32> {
        use realm_core::formats::gguf::GGUFParser;
        use std::io::Cursor;

        let cursor = Cursor::new(gguf_bytes);
        let mut parser = GGUFParser::new(cursor);
        let meta = parser
            .parse_header()
            .context("Failed to parse GGUF header")?;

        // Determine model ID
        let model_id = if let Some(id) = requested_id {
            if id == 0 {
                // Auto-generate ID from hash
                let mut hasher = DefaultHasher::new();
                gguf_bytes.hash(&mut hasher);
                let hash = hasher.finish();
                ((hash % (u32::MAX as u64)) as u32).max(1)
            } else {
                id
            }
        } else {
            // Use counter
            let id = self.counter;
            self.counter += 1;
            id
        };

        // Check if model already exists
        if self.models.contains_key(&model_id) {
            debug!("Model {} already exists, returning existing ID", model_id);
            return Ok(model_id);
        }

        // Create stored model
        let mut stored_model = StoredModel::new(model_id, meta.clone());

        // Store GGUF bytes for Model loading (needed for inference)
        stored_model.set_gguf_bytes(gguf_bytes.to_vec());
        debug!(
            "GGUF bytes stored for model {} ({} bytes)",
            model_id,
            gguf_bytes.len()
        );

        // Create and store tokenizer from metadata
        match Tokenizer::from_gguf(&meta) {
            Ok(tokenizer) => {
                stored_model.set_tokenizer(tokenizer);
                debug!("Tokenizer stored for model {}", model_id);
            }
            Err(e) => {
                warn!("Failed to create tokenizer for model {}: {}", model_id, e);
                // Continue without tokenizer - model can still be used for inference
            }
        }

        // Extract and store all tensors
        let _data_offset = parser
            .tensor_data_offset()
            .context("Failed to get tensor data offset")?;

        for tensor_desc in &meta.tensors {
            // Read tensor data
            let offset = tensor_desc.offset as usize;
            let size = tensor_desc.size_bytes;

            if offset + size > gguf_bytes.len() {
                warn!(
                    "Tensor {} offset {} + size {} exceeds GGUF data length {}",
                    tensor_desc.name,
                    offset,
                    size,
                    gguf_bytes.len()
                );
                continue;
            }

            let tensor_data = gguf_bytes[offset..offset + size].to_vec();
            let tensor = QuantizedTensor::new(
                tensor_data,
                tensor_desc.dtype,
                tensor_desc.shape.dims().iter().map(|&d| d as u64).collect(),
                tensor_desc.name.clone(),
            );

            stored_model.add_tensor(tensor_desc.name.clone(), tensor);
        }

        info!(
            "Stored model {} ({} tensors, {} bytes)",
            model_id,
            stored_model.tensor_count(),
            stored_model.total_size
        );

        self.models.insert(model_id, stored_model);
        self.next_id = self.next_id.max(model_id + 1);

        Ok(model_id)
    }

    /// Get a model by ID
    pub fn get_model(&self, model_id: u32) -> Result<&StoredModel> {
        self.models
            .get(&model_id)
            .ok_or_else(|| anyhow!("Model {} not found", model_id))
    }

    /// Get a mutable model by ID
    pub fn get_model_mut(&mut self, model_id: u32) -> Result<&mut StoredModel> {
        self.models
            .get_mut(&model_id)
            .ok_or_else(|| anyhow!("Model {} not found", model_id))
    }

    /// Remove a model
    pub fn remove_model(&mut self, model_id: u32) -> Result<()> {
        self.models
            .remove(&model_id)
            .ok_or_else(|| anyhow!("Model {} not found", model_id))?;
        self.remove_from_cache(model_id);
        info!("Removed model {}", model_id);
        Ok(())
    }

    /// List all model IDs
    pub fn list_models(&self) -> Vec<u32> {
        self.models.keys().copied().collect()
    }

    /// Get model info (tensor count, total size)
    pub fn get_model_info(&self, model_id: u32) -> Result<(usize, usize)> {
        let model = self.get_model(model_id)?;
        Ok((model.tensor_count(), model.total_size))
    }

    /// Set LoRA adapter for a model
    pub fn set_lora_adapter(&mut self, model_id: u32, adapter_id: String) -> Result<()> {
        let model = self.get_model_mut(model_id)?;
        model.set_lora_adapter(adapter_id);
        Ok(())
    }
}

/// Global model storage instance (thread-safe)
/// NOTE: We use a lazy static pattern since HashMap::new() cannot be called in const context
use std::sync::OnceLock;

static GLOBAL_MODEL_STORAGE_INNER: OnceLock<Mutex<ModelStorage>> = OnceLock::new();

/// Get the global model storage instance
pub fn get_global_model_storage() -> &'static Mutex<ModelStorage> {
    GLOBAL_MODEL_STORAGE_INNER.get_or_init(|| Mutex::new(ModelStorage::new()))
}
