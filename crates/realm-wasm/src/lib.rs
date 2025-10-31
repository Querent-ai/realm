//! Realm WASM - WASM orchestrator module for Realm.ai
//!
//! This module provides:
//! - Customer-facing API (generate, loadModel, etc.)
//! - Inference orchestration
//! - Host function imports (calls to native runtime)

use realm_core::{GGUFParser, Tokenizer};
use realm_models::{GenerationConfig, Model, TransformerConfig};
use std::io::Cursor;
use wasm_bindgen::prelude::*;

// ========================================
// Host Function Imports
// ========================================

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
extern "C" {
    /// Store a model from GGUF bytes in HOST memory
    /// Returns: model_id on success (> 0), negative on error
    fn realm_store_model(gguf_ptr: *const u8, gguf_len: u32) -> i32;

    /// Get tensor data from HOST storage
    /// Returns: actual tensor size on success, negative on error
    fn realm_get_tensor(
        model_id: u32,
        tensor_name_ptr: *const u8,
        tensor_name_len: u32,
        out_ptr: *mut u8,
        out_max_len: u32,
    ) -> i32;

    /// Get model metadata (tensor count, total size)
    /// Returns: 0 on success, negative on error
    fn realm_get_model_info(
        model_id: u32,
        out_tensor_count_ptr: *mut u32,
        out_total_size_ptr: *mut u64,
    ) -> i32;

    /// Remove model from storage (cleanup)
    /// Returns: 0 on success, negative on error
    fn realm_remove_model(model_id: u32) -> i32;
}

/// Generation configuration for WASM API
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmGenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

#[wasm_bindgen]
impl WasmGenerationConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
        }
    }
}

impl Default for WasmGenerationConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl From<WasmGenerationConfig> for GenerationConfig {
    fn from(config: WasmGenerationConfig) -> Self {
        GenerationConfig {
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            repetition_penalty: config.repetition_penalty,
        }
    }
}

/// Realm WASM instance
#[wasm_bindgen]
pub struct Realm {
    /// Lightweight model structure (no weights in WASM!)
    model: Option<Model>,
    /// Tokenizer for encoding/decoding
    tokenizer: Option<Tokenizer>,
    /// Model ID handle for HOST storage
    /// TODO: Use this in generate() to call realm_get_tensor() for weights
    #[allow(dead_code)]
    model_id: Option<u32>,
    /// Transformer config
    transformer_config: Option<TransformerConfig>,
    /// Generation config
    config: WasmGenerationConfig,
}

#[wasm_bindgen]
impl Realm {
    /// Create a new Realm instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Realm, JsError> {
        // Set up better panic messages for WASM debugging
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();

        Ok(Realm {
            model: None,
            tokenizer: None,
            model_id: None,
            transformer_config: None,
            config: WasmGenerationConfig::new(),
        })
    }

    /// Load a model from GGUF bytes (HOST-side storage version)
    ///
    /// # Arguments
    /// * `model_bytes` - Raw GGUF file bytes (Uint8Array from JavaScript)
    ///
    /// # Architecture
    /// This function stores the model in HOST memory, not WASM memory:
    /// 1. Parse GGUF header/metadata in WASM (lightweight)
    /// 2. Call HOST function to store quantized weights in HOST
    /// 3. Keep only model_id handle + config in WASM (~4 bytes + config)
    ///
    /// Before: 637MB quantized → 2.5GB f32 in WASM → OOM
    /// After:  637MB stays in HOST, WASM has 4-byte handle → ~50MB total
    #[wasm_bindgen(js_name = loadModel)]
    pub fn load_model(&mut self, model_bytes: &[u8]) -> Result<(), JsError> {
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("loadModel: received {} bytes", model_bytes.len()).into());

        // Parse GGUF header for metadata (lightweight operation)
        let cursor = Cursor::new(model_bytes);
        let mut parser = GGUFParser::new(cursor);

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"loadModel: parsing header...".into());

        let meta = parser
            .parse_header()
            .map_err(|e| JsError::new(&format!("Failed to parse GGUF header: {}", e)))?;

        // Extract config
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"loadModel: extracting config...".into());

        let config_data = parser
            .extract_config()
            .ok_or_else(|| JsError::new("Failed to extract config from GGUF"))?;
        let config: TransformerConfig = config_data.into();

        // Create tokenizer from GGUF metadata
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"loadModel: creating tokenizer...".into());

        let tokenizer = Tokenizer::from_gguf(&meta)
            .map_err(|e| JsError::new(&format!("Failed to create tokenizer: {}", e)))?;

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(
            &format!(
                "loadModel: tokenizer created, vocab_size={}",
                tokenizer.vocab_size()
            )
            .into(),
        );

        // ========================================
        // INNOVATION: Store model in HOST, not WASM
        // ========================================

        #[cfg(target_arch = "wasm32")]
        {
            web_sys::console::log_1(&"loadModel: calling HOST to store model...".into());

            // Call HOST function to store model
            let model_id =
                unsafe { realm_store_model(model_bytes.as_ptr(), model_bytes.len() as u32) };

            if model_id < 0 {
                return Err(JsError::new(&format!(
                    "Failed to store model in HOST (error code: {})",
                    model_id
                )));
            }

            web_sys::console::log_1(
                &format!("loadModel: model stored in HOST with ID {}", model_id).into(),
            );

            // Get model info from HOST
            let mut tensor_count: u32 = 0;
            let mut total_size: u64 = 0;

            let result = unsafe {
                realm_get_model_info(
                    model_id as u32,
                    &mut tensor_count as *mut u32,
                    &mut total_size as *mut u64,
                )
            };

            if result < 0 {
                return Err(JsError::new(&format!(
                    "Failed to get model info (error code: {})",
                    result
                )));
            }

            web_sys::console::log_1(
                &format!(
                    "loadModel: model has {} tensors, {:.2} MB total in HOST",
                    tensor_count,
                    total_size as f64 / 1024.0 / 1024.0
                )
                .into(),
            );

            // Create lightweight model structure (NO WEIGHTS!)
            let model = Model::new(config.clone());

            // Store everything in WASM (minimal memory usage)
            self.model = Some(model);
            self.tokenizer = Some(tokenizer);
            self.model_id = Some(model_id as u32);
            self.transformer_config = Some(config);

            web_sys::console::log_1(
                &"loadModel: SUCCESS! Model handle stored in WASM, weights in HOST".into(),
            );

            Ok(())
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Non-WASM builds: fall back to old behavior
            let mut model = Model::new(config.clone());

            let data_offset = parser
                .tensor_data_offset()
                .map_err(|e| JsError::new(&format!("Failed to get tensor data offset: {}", e)))?;
            let mut tensor_loader = realm_core::TensorLoader::new(data_offset);

            for tensor_desc in meta.tensors.iter() {
                tensor_loader.register_tensor(
                    tensor_desc.name.clone(),
                    tensor_desc.clone(),
                    tensor_desc.offset,
                );
            }

            let cursor = Cursor::new(model_bytes);
            let mut parser = GGUFParser::new(cursor);
            parser
                .parse_header()
                .map_err(|e| JsError::new(&format!("Failed to parse header again: {}", e)))?;

            model
                .load_from_gguf(&mut tensor_loader, &mut parser)
                .map_err(|e| JsError::new(&format!("Failed to load model weights: {}", e)))?;

            self.model = Some(model);
            self.tokenizer = Some(tokenizer);
            self.transformer_config = Some(config);

            Ok(())
        }
    }

    /// Check if model is loaded
    #[wasm_bindgen(js_name = isLoaded)]
    pub fn is_loaded(&self) -> bool {
        self.model.is_some() && self.tokenizer.is_some()
    }

    /// Set generation configuration
    #[wasm_bindgen(js_name = setConfig)]
    pub fn set_config(&mut self, config: WasmGenerationConfig) {
        self.config = config;
    }

    /// Generate text from a prompt
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    ///
    /// # Returns
    /// Generated text response
    pub fn generate(&mut self, prompt: String) -> Result<String, JsError> {
        // Check if model is loaded
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| JsError::new("Model not loaded. Call loadModel() first."))?;

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| JsError::new("Tokenizer not loaded."))?;

        // Convert config
        let gen_config: GenerationConfig = self.config.clone().into();

        // Generate text
        let response = model
            .generate(&prompt, tokenizer, &gen_config)
            .map_err(|e| JsError::new(&format!("Generation failed: {}", e)))?;

        Ok(response)
    }

    /// Get model vocabulary size
    #[wasm_bindgen(js_name = vocabSize)]
    pub fn vocab_size(&self) -> Result<usize, JsError> {
        self.tokenizer
            .as_ref()
            .map(|t| t.vocab_size())
            .ok_or_else(|| JsError::new("Tokenizer not loaded"))
    }

    /// Get model configuration as JSON string
    #[wasm_bindgen(js_name = getModelConfig)]
    pub fn get_model_config(&self) -> Result<String, JsError> {
        self.model
            .as_ref()
            .map(|m| serde_json::to_string_pretty(&m.config).unwrap_or_else(|_| "{}".to_string()))
            .ok_or_else(|| JsError::new("Model not loaded"))
    }
}

impl Default for Realm {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

// Host function imports (to be called from WASM)
// These are provided by realm-runtime when running in a host environment
#[link(wasm_import_module = "realm_host")]
extern "C" {
    /// Matrix multiplication via Candle (host-side GPU)
    pub fn candle_matmul(
        a_ptr: *const f32,
        a_len: u32,
        b_ptr: *const f32,
        b_len: u32,
        m: u32,
        k: u32,
        n: u32,
        result_ptr: *mut f32,
    ) -> i32;

    /// Load layer from Memory64 into WASM memory
    pub fn memory64_load_layer(
        model_id: u32,
        layer_id: u32,
        buffer_ptr: *mut u8,
        buffer_len: u32,
    ) -> i32;

    /// Store layer to Memory64 from WASM memory
    pub fn memory64_store_layer(
        model_id: u32,
        layer_id: u32,
        buffer_ptr: *const u8,
        buffer_len: u32,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realm_creation() {
        let realm = Realm::new();
        assert!(realm.is_ok());
    }

    #[test]
    fn test_config() {
        let mut realm = Realm::new().unwrap();
        let mut config = WasmGenerationConfig::new();
        config.max_tokens = 200;
        config.temperature = 0.8;
        realm.set_config(config.clone());
        assert_eq!(realm.config.max_tokens, 200);
        assert_eq!(realm.config.temperature, 0.8);
    }

    #[test]
    fn test_not_loaded() {
        let realm = Realm::new().unwrap();
        assert!(!realm.is_loaded());
        // Skip generate() test on non-WASM targets since it calls wasm-bindgen imports
        #[cfg(target_arch = "wasm32")]
        {
            let mut realm_mut = realm;
            let result = realm_mut.generate("test".to_string());
            assert!(result.is_err());
        }
    }
}
