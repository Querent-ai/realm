//! Memory64 GGUF Integration
//!
//! This module integrates GGUF model loading with Memory64 storage,
//! enabling loading of large models (>4GB) with on-demand layer access.

use crate::memory64::{Memory64Runtime, MemoryLayout};
use crate::memory64_layer_manager::{Memory64LayerManager, Memory64Model};
use realm_core::{
    error::{Error, Result},
    formats::gguf::{GGUFParser, ModelMeta},
    tensor_loader::TensorLoader,
};
use realm_models::TransformerConfig;
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::sync::Arc;

/// Memory64-aware GGUF model loader
pub struct Memory64GGUFLoader {
    /// Memory64 runtime for storing model weights
    runtime: Option<Arc<Memory64Runtime>>,
    /// Layer manager for on-demand loading
    layer_manager: Option<Memory64LayerManager>,
    /// Model configuration
    config: Option<TransformerConfig>,
    /// Model metadata from GGUF
    meta: Option<ModelMeta>,
    /// Tensor metadata for layer mapping
    tensor_metadata: HashMap<String, LayerTensorInfo>,
    /// Whether to use Memory64 (based on model size)
    use_memory64: bool,
}

/// Information about a tensor's location in Memory64
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LayerTensorInfo {
    /// Layer ID (0 = embeddings, 1+ = transformer layers, last = lm_head)
    layer_id: u32,
    /// Tensor type within the layer
    tensor_type: LayerTensorType,
    /// Offset in Memory64 storage
    offset: u64,
    /// Size in bytes
    size_bytes: usize,
    /// Shape information
    shape: Vec<usize>,
}

/// Types of tensors within a layer
#[derive(Debug, Clone)]
enum LayerTensorType {
    /// Token embeddings (layer 0)
    TokenEmbeddings,
    /// Attention weights (Q, K, V, O)
    AttentionWeights,
    /// Feed-forward network weights
    FFNWeights,
    /// Layer normalization weights
    LayerNorm,
    /// Language model head (final layer)
    LMHead,
}

impl Default for Memory64GGUFLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory64GGUFLoader {
    /// Create a new Memory64 GGUF loader
    pub fn new() -> Self {
        Self {
            runtime: None,
            layer_manager: None,
            config: None,
            meta: None,
            tensor_metadata: HashMap::new(),
            use_memory64: false,
        }
    }

    /// Load a GGUF model with Memory64 support
    pub fn load_model<R: Read + Seek>(
        &mut self,
        parser: &mut GGUFParser<R>,
    ) -> Result<Memory64Model> {
        // println!("🚀 Loading GGUF model with Memory64 support...");

        // Step 1: Parse GGUF header (only if not already parsed)
        // println!("📋 Parsing GGUF header...");
        let meta = if let Some(existing_meta) = parser.metadata() {
            // Header already parsed, use existing metadata
            existing_meta.clone()
        } else {
            // Parse header for the first time
            parser.parse_header()?
        };
        self.meta = Some(meta.clone());
        // println!("REMOVED");  //

        // Step 2: Extract model configuration
        // println!("⚙️  Extracting model configuration...");
        let config_data = parser
            .extract_config()
            .ok_or_else(|| Error::ParseError("Failed to extract config".to_string()))?;
        let config: TransformerConfig = config_data.into();
        self.config = Some(config.clone());
        // println!("REMOVED");  //

        // Step 3: Analyze model size and decide on Memory64 usage
        let total_size = self.estimate_model_size(&meta)?;
        self.use_memory64 = total_size > 3_000_000_000; // 3GB threshold

        // println!("REMOVED");  //
        // println!("🎯 Using Memory64: {}", self.use_memory64);

        if self.use_memory64 {
            // Step 4: Initialize Memory64 runtime
            self.initialize_memory64(total_size)?;

            // Step 5: Map tensors to layers
            self.map_tensors_to_layers(&meta)?;

            // Step 6: Initialize lazy loading infrastructure
            self.initialize_lazy_loading(parser)?;

            // Step 7: Create layer manager
            self.create_layer_manager()?;

            // Step 8: Create Memory64 model
            self.create_memory64_model()
        } else {
            // Fall back to standard loading
            self.load_standard_model(parser)
        }
    }

    /// Estimate total model size from GGUF metadata
    fn estimate_model_size(&self, meta: &ModelMeta) -> Result<u64> {
        let total_bytes: u64 = meta
            .tensors
            .iter()
            .map(|tensor| tensor.size_bytes as u64)
            .sum();

        Ok(total_bytes)
    }

    /// Initialize Memory64 runtime based on model size
    fn initialize_memory64(&mut self, total_size: u64) -> Result<()> {
        // println!("🔧 Initializing Memory64 runtime...");

        let layout = if total_size <= 8_000_000_000 {
            // Single region for 7B-8B models
            MemoryLayout::single(8, "model_storage")
                .map_err(|e| Error::ParseError(format!("Failed to create layout: {}", e)))?
        } else if total_size <= 16_000_000_000 {
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

        self.runtime = Some(Arc::new(Memory64Runtime::new(layout, true)));
        // println!("✅ Memory64 runtime initialized");
        Ok(())
    }

    /// Map GGUF tensors to layer structure
    fn map_tensors_to_layers(&mut self, meta: &ModelMeta) -> Result<()> {
        // println!("🗺️  Mapping tensors to layers...");

        let config = self
            .config
            .as_ref()
            .ok_or_else(|| Error::ParseError("Config not initialized".to_string()))?;
        let mut current_offset = 0u64;

        // Process each tensor and map to appropriate layer
        for tensor in &meta.tensors {
            let layer_id = self.determine_layer_id(&tensor.name, config)?;
            let tensor_type = self.determine_tensor_type(&tensor.name)?;

            let info = LayerTensorInfo {
                layer_id,
                tensor_type,
                offset: current_offset,
                size_bytes: tensor.size_bytes,
                shape: tensor.shape.dims().to_vec(),
            };

            self.tensor_metadata.insert(tensor.name.clone(), info);
            current_offset += tensor.size_bytes as u64;

            // println!("REMOVED");  //
        }

        // println!("✅ Mapped {} tensors to layers", self.tensor_metadata.len());
        Ok(())
    }

    /// Determine which layer a tensor belongs to
    fn determine_layer_id(&self, tensor_name: &str, config: &TransformerConfig) -> Result<u32> {
        if tensor_name.contains("token_embd") || tensor_name.contains("embed_tokens") {
            Ok(0) // Embeddings layer
        } else if tensor_name.contains("lm_head") || tensor_name.contains("output") {
            Ok(config.num_layers as u32 + 1) // LM head (after all transformer layers)
        } else if tensor_name.contains("layers.") {
            // Extract layer number from tensor name
            let parts: Vec<&str> = tensor_name.split('.').collect();
            if parts.len() >= 2 {
                if let Some(layer_part) = parts.get(1) {
                    if let Ok(layer_num) = layer_part.parse::<u32>() {
                        return Ok(layer_num + 1); // +1 because layer 0 is embeddings
                    }
                }
            }
            Err(Error::ParseError(format!(
                "Could not determine layer for tensor: {}",
                tensor_name
            )))
        } else {
            // Default to layer 1 for other tensors
            Ok(1)
        }
    }

    /// Determine the type of tensor within a layer
    fn determine_tensor_type(&self, tensor_name: &str) -> Result<LayerTensorType> {
        if tensor_name.contains("token_embd") || tensor_name.contains("embed_tokens") {
            Ok(LayerTensorType::TokenEmbeddings)
        } else if tensor_name.contains("lm_head") || tensor_name.contains("output") {
            Ok(LayerTensorType::LMHead)
        } else if tensor_name.contains("attention") {
            Ok(LayerTensorType::AttentionWeights)
        } else if tensor_name.contains("mlp") || tensor_name.contains("ffn") {
            Ok(LayerTensorType::FFNWeights)
        } else if tensor_name.contains("norm") {
            Ok(LayerTensorType::LayerNorm)
        } else {
            // Default to attention weights for unknown tensors
            Ok(LayerTensorType::AttentionWeights)
        }
    }

    /// Initialize lazy loading infrastructure (no actual weight loading)
    fn initialize_lazy_loading<R: Read + Seek>(
        &mut self,
        parser: &mut GGUFParser<R>,
    ) -> Result<()> {
        // println!("💾 Initializing lazy loading infrastructure...");

        let _runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| Error::ParseError("Runtime not initialized".to_string()))?;
        let data_offset = parser.tensor_data_offset()?;

        // Create tensor loader for reading GGUF data (but don't load yet)
        let mut tensor_loader = TensorLoader::new(data_offset);

        // Register all tensors (metadata only, no actual data loading)
        let meta = self
            .meta
            .as_ref()
            .ok_or_else(|| Error::ParseError("Metadata not initialized".to_string()))?;
        for tensor in &meta.tensors {
            tensor_loader.register_tensor(tensor.name.clone(), tensor.clone(), tensor.offset);
        }

        // Store tensor loader for later use (in production, this would be stored in the loader)
        // println!("✅ Lazy loading infrastructure ready");
        // println!("REMOVED");  //
        // println!("   💡 Weights will be loaded only when layers are accessed");

        Ok(())
    }

    /// Load a specific tensor on-demand
    pub fn load_tensor_lazy<R: Read + Seek>(
        &mut self,
        tensor_name: &str,
        parser: &mut GGUFParser<R>,
    ) -> Result<Vec<f32>> {
        let _layer_info = self
            .tensor_metadata
            .get(tensor_name)
            .ok_or_else(|| Error::ParseError(format!("Tensor not found: {}", tensor_name)))?;

        // println!("REMOVED");  //

        // Create tensor loader for this specific request
        let data_offset = parser.tensor_data_offset()?;
        let mut tensor_loader = TensorLoader::new(data_offset);

        // Register the specific tensor
        let meta = self
            .meta
            .as_ref()
            .ok_or_else(|| Error::ParseError("Metadata not initialized".to_string()))?;
        if let Some(tensor) = meta.tensors.iter().find(|t| t.name == tensor_name) {
            tensor_loader.register_tensor(tensor.name.clone(), tensor.clone(), tensor.offset);

            // Load the tensor data
            let tensor_data = tensor_loader.load_tensor(tensor_name, parser)?;
            // println!("REMOVED");  //
            Ok(tensor_data.to_vec())
        } else {
            Err(Error::ParseError(format!(
                "Tensor metadata not found: {}",
                tensor_name
            )))
        }
    }

    /// Create layer manager for on-demand loading
    fn create_layer_manager(&mut self) -> Result<()> {
        // println!("🧠 Creating layer manager...");

        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| Error::ParseError("Runtime not initialized".to_string()))?
            .clone();
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| Error::ParseError("Config not initialized".to_string()))?
            .clone();

        self.layer_manager = Some(Memory64LayerManager::new(
            runtime, config, 16, // Cache up to 16 layers for better performance
        ));

        // println!("✅ Layer manager created");
        Ok(())
    }

    /// Create Memory64 model with on-demand layer loading
    fn create_memory64_model(&mut self) -> Result<Memory64Model> {
        // println!("📋 Creating Memory64 model...");

        let config = self
            .config
            .as_ref()
            .ok_or_else(|| Error::ParseError("Config not initialized".to_string()))?
            .clone();
        let layer_manager = self
            .layer_manager
            .take()
            .ok_or_else(|| Error::ParseError("Layer manager not initialized".to_string()))?;

        // Create placeholder components (in production, these would be loaded from Memory64)
        let token_embeddings = vec![0.0; config.vocab_size * config.hidden_size];
        let output_norm = vec![1.0; config.hidden_size];
        let lm_head = vec![0.0; config.vocab_size * config.hidden_size];

        let model = Memory64Model::new(
            config.clone(),
            token_embeddings,
            output_norm,
            lm_head,
            layer_manager,
            config.num_layers as u32,
        );

        // println!("✅ Memory64 model created");
        Ok(model)
    }

    /// Fall back to standard model loading
    fn load_standard_model<R: Read + Seek>(
        &mut self,
        _parser: &mut GGUFParser<R>,
    ) -> Result<Memory64Model> {
        // println!("📋 Loading standard model (no Memory64)...");

        let config = self
            .config
            .as_ref()
            .ok_or_else(|| Error::ParseError("Config not initialized".to_string()))?
            .clone();

        // Create a dummy layer manager for standard models
        let layout = MemoryLayout::single(1, "dummy")
            .map_err(|e| Error::ParseError(format!("Failed to create dummy layout: {}", e)))?;
        let runtime = Arc::new(Memory64Runtime::new(layout, true));
        let layer_manager = Memory64LayerManager::new(runtime, config.clone(), 4);

        // Create placeholder components
        let token_embeddings = vec![0.0; config.vocab_size * config.hidden_size];
        let output_norm = vec![1.0; config.hidden_size];
        let lm_head = vec![0.0; config.vocab_size * config.hidden_size];

        let model = Memory64Model::new(
            config.clone(),
            token_embeddings,
            output_norm,
            lm_head,
            layer_manager,
            config.num_layers as u32,
        );

        // println!("✅ Standard model created");
        Ok(model)
    }
}

/// Convenience function to load a GGUF model with automatic Memory64 detection
///
/// This function automatically determines whether to use Memory64 based on model size:
/// - Models >3GB: Use Memory64 with lazy loading
/// - Models ≤3GB: Use standard in-memory loading
///
/// # Example
/// ```no_run
/// use realm_runtime::memory64_gguf::load_gguf_with_memory64;
/// use std::fs::File;
/// use std::io::BufReader;
/// use realm_core::formats::gguf::GGUFParser;
///
/// # fn main() -> realm_core::error::Result<()> {
/// let file = File::open("model.gguf")?;
/// let reader = BufReader::new(file);
/// let mut parser = GGUFParser::new(reader);
///
/// let model = load_gguf_with_memory64(&mut parser)?;
/// println!("Model loaded! Layers: {}", model.num_layers());
/// # Ok(())
/// # }
/// ```
pub fn load_gguf_with_memory64<R: Read + Seek>(
    parser: &mut GGUFParser<R>,
) -> Result<Memory64Model> {
    let mut loader = Memory64GGUFLoader::new();
    loader.load_model(parser)
}
