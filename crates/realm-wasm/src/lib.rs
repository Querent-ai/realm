//! Realm WASM - WASM orchestrator module for Realm.ai
//!
//! This module provides:
//! - Customer-facing API (generate, loadModel, etc.)
//! - Inference orchestration
//! - Host function imports (calls to native runtime)
//!
//! Memory: wasm-bindgen automatically exports WASM memory
//! Access via: wasm.memory or wasm.__wbindgen_memory()

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
    /// Parameters: gguf_ptr, gguf_len, model_id (0 = auto-generate, > 0 = use provided)
    /// Returns: model_id on success (> 0), negative on error
    fn realm_store_model(gguf_ptr: *const u8, gguf_len: u32, model_id: u32) -> i32;

    /// Get tensor data from HOST storage (DEPRECATED - use realm_forward_layer instead)
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

    /// Forward through a complete transformer layer (HOST-SIDE COMPUTATION)
    /// This is THE KEY function - weights never enter WASM!
    /// Parameters:
    ///   - model_id: Model ID in HOST storage
    ///   - layer_idx: Layer index (0..num_layers-1)
    ///   - hidden_states_ptr: Input hidden states in WASM memory
    ///   - hidden_states_len: Number of f32 elements in hidden_states
    ///   - position: Position for KV cache
    ///   - out_ptr: Output buffer in WASM memory
    /// Returns: Bytes written on success, negative on error
    fn realm_forward_layer(
        model_id: u32,
        layer_idx: u32,
        hidden_states_ptr: *const f32,
        hidden_states_len: u32,
        position: u32,
        out_ptr: *mut f32,
    ) -> i32;

    /// Embed tokens (HOST-SIDE COMPUTATION)
    /// Loads embeddings from HOST and returns hidden states
    /// Parameters:
    ///   - model_id: Model ID in HOST storage
    ///   - token_ids_ptr: Token IDs array in WASM memory
    ///   - token_count: Number of tokens
    ///   - out_ptr: Output buffer for hidden states (vocab_size * hidden_size * f32)
    /// Returns: Bytes written on success, negative on error
    fn realm_embed_tokens(
        model_id: u32,
        token_ids_ptr: *const u32,
        token_count: u32,
        out_ptr: *mut f32,
    ) -> i32;

    /// Compute logits from hidden states (HOST-SIDE COMPUTATION)
    /// Applies final norm + LM head projection
    /// Parameters:
    ///   - model_id: Model ID in HOST storage
    ///   - hidden_states_ptr: Hidden states in WASM memory
    ///   - hidden_size: Size of hidden dimension
    ///   - out_ptr: Output buffer for logits (vocab_size * f32)
    /// Returns: Bytes written on success, negative on error
    fn realm_compute_logits(
        model_id: u32,
        hidden_states_ptr: *const f32,
        hidden_size: u32,
        out_ptr: *mut f32,
    ) -> i32;
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
    #[allow(dead_code)] // Used for HOST-side computation via FFI
    model_id: Option<u32>,
    /// Transformer config
    transformer_config: Option<TransformerConfig>,
    /// Generation config
    config: WasmGenerationConfig,
    /// KV caches for each layer (persisted across forward passes)
    #[cfg(target_arch = "wasm32")]
    kv_caches: Option<Vec<realm_models::KVCache>>,
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
            #[cfg(target_arch = "wasm32")]
            kv_caches: None,
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
            // Use model_id = 0 to auto-generate deterministic ID from model hash
            // Consumer can pass custom ID via optional parameter in future
            let model_id =
                unsafe { realm_store_model(model_bytes.as_ptr(), model_bytes.len() as u32, 0) };

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

            // Initialize KV caches for each layer
            let head_dim = config.hidden_size / config.num_heads;
            let mut kv_caches = Vec::new();
            for _ in 0..config.num_layers {
                kv_caches.push(realm_models::KVCache::new(
                    config.max_seq_len,
                    config.num_kv_heads,
                    head_dim,
                ));
            }

            // Store everything in WASM (minimal memory usage)
            self.model = Some(model);
            self.tokenizer = Some(tokenizer);
            self.model_id = Some(model_id as u32);
            self.transformer_config = Some(config);
            self.kv_caches = Some(kv_caches);

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

    /// Generate text from a prompt (HOST-side storage version)
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    ///
    /// # Returns
    /// Generated text response
    ///
    /// # Architecture
    /// This function loads weights from HOST storage on-demand during inference.
    /// Each layer's weights are loaded from HOST, used for forward pass, then freed.
    /// This keeps WASM memory usage low (~50MB vs 2.5GB+).
    pub fn generate(&mut self, prompt: String) -> Result<String, JsError> {
        #[cfg(target_arch = "wasm32")]
        {
            // WASM path: Use HOST storage for weights
            self.generate_with_host_storage(prompt)
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Non-WASM path: Use traditional model loading
            let model = self
                .model
                .as_mut()
                .ok_or_else(|| JsError::new("Model not loaded. Call loadModel() first."))?;

            let tokenizer = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| JsError::new("Tokenizer not loaded."))?;

            let gen_config: GenerationConfig = self.config.clone().into();

            let response = model
                .generate(&prompt, tokenizer, &gen_config)
                .map_err(|e| JsError::new(&format!("Generation failed: {}", e)))?;

            Ok(response)
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn generate_with_host_storage(&mut self, prompt: String) -> Result<String, JsError> {
        let model_id = self
            .model_id
            .ok_or_else(|| JsError::new("Model not loaded in HOST. Call loadModel() first."))?;
        let config = self
            .transformer_config
            .as_ref()
            .ok_or_else(|| JsError::new("Config not loaded."))?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| JsError::new("Tokenizer not loaded."))?;
        let gen_config: GenerationConfig = self.config.clone().into();

        web_sys::console::log_1(&format!("generate: starting with prompt '{}'", prompt).into());

        // Note: Backends should be initialized in HOST, not WASM
        // For now, we'll create minimal CPU backend in WASM
        // TODO: Refactor to use HOST-provided backends via FFI
        use realm_compute_cpu::CandleCpuBackend;
        let cpu_backend = CandleCpuBackend::new()
            .map_err(|e| JsError::new(&format!("Failed to create CPU backend: {}", e)))?;
        use realm_compute_cpu::CandleNeuralOpsBackend;
        let candle_backend = CandleNeuralOpsBackend::new();

        // Get or initialize KV caches
        if let Some(ref mut kv_caches) = self.kv_caches {
            // Clear KV caches for new generation
            for cache in kv_caches.iter_mut() {
                cache.clear();
            }
        } else {
            return Err(JsError::new(
                "KV caches not initialized. Call loadModel() first.",
            ));
        }

        // Create logits processor
        let mut logits_processor = realm_models::LogitsProcessor::with_params(
            42,
            gen_config.temperature as f64,
            gen_config.top_p as f64,
            gen_config.top_k,
            gen_config.repetition_penalty,
        );

        // Encode prompt
        let mut tokens = tokenizer
            .encode(&prompt, true)
            .map_err(|e| JsError::new(&format!("Tokenization failed: {}", e)))?;

        if tokens.is_empty() {
            return Err(JsError::new("Empty token sequence"));
        }

        let num_prompt_tokens = tokens.len();
        web_sys::console::log_1(
            &format!("generate: encoded {} prompt tokens", num_prompt_tokens).into(),
        );

        // Get config (needed for helper functions)
        let config = self
            .transformer_config
            .as_ref()
            .ok_or_else(|| JsError::new("Transformer config not loaded"))?;

        // PREFILL PHASE: Process prompt tokens
        let chunk_size = 8;
        let mut prefill_logits = Vec::new();

        for chunk_start in (0..num_prompt_tokens).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(num_prompt_tokens);
            let chunk = &tokens[chunk_start..chunk_end];

            // Embed tokens using HOST function (no 262MB loading!)
            let mut hidden_states = Self::embed_tokens_via_host(model_id, chunk, config)?;

            // Forward through layers
            // KV caches are now managed in HOST, but we keep the Vec for compatibility
            // The actual cache is stored in HOST via realm_forward_layer
            let _kv_caches = self.kv_caches.as_mut().unwrap();
            for layer_idx in 0..config.num_layers {
                hidden_states = Self::forward_layer_with_host_weights_static(
                    model_id,
                    layer_idx,
                    &hidden_states,
                    chunk_start,
                    config,
                    &candle_backend,
                    &cpu_backend,
                    &mut _kv_caches[layer_idx], // Cache updated in HOST, this is just for type compatibility
                )?;
            }

            // Get logits for last token using HOST function (no LM head loading!)
            let seq_len = chunk.len();
            let last_hidden_start = (seq_len - 1) * config.hidden_size;
            let last_hidden =
                &hidden_states[last_hidden_start..last_hidden_start + config.hidden_size];

            let logits = Self::compute_logits_via_host(model_id, last_hidden, config)?;
            prefill_logits.extend(logits);
        }

        // Get logits for last token
        let mut last_logits = prefill_logits[(prefill_logits.len() - config.vocab_size)..].to_vec();

        // Sample first token
        let next = logits_processor
            .sample(&mut last_logits)
            .map_err(|e| JsError::new(&format!("Sampling failed: {}", e)))?;
        tokens.push(next);

        // DECODE PHASE: Generate tokens one at a time
        let mut generated = 1;
        while generated < gen_config.max_tokens {
            let last_token = tokens[tokens.len() - 1];

            // Embed last token using HOST function (no 262MB loading!)
            let token_array = [last_token];
            let mut hidden_states = Self::embed_tokens_via_host(model_id, &token_array, config)?;

            // Forward through layers
            let cache_position = num_prompt_tokens + generated - 1;
            // KV caches are now managed in HOST via realm_forward_layer
            let _kv_caches = self.kv_caches.as_mut().unwrap();
            for layer_idx in 0..config.num_layers {
                hidden_states = Self::forward_layer_with_host_weights_static(
                    model_id,
                    layer_idx,
                    &hidden_states,
                    cache_position,
                    config,
                    &candle_backend,
                    &cpu_backend,
                    &mut _kv_caches[layer_idx], // Cache updated in HOST, this is just for type compatibility
                )?;
            }

            // Compute logits using HOST function (no LM head loading!)
            last_logits = Self::compute_logits_via_host(model_id, &hidden_states, config)?;

            // Sample next token
            let next = logits_processor
                .sample(&mut last_logits)
                .map_err(|e| JsError::new(&format!("Sampling failed: {}", e)))?;

            if next == tokenizer.special_tokens().eos_token_id {
                break;
            }

            tokens.push(next);
            generated += 1;
        }

        // Decode tokens to text
        let response = tokenizer
            .decode(&tokens, true)
            .map_err(|e| JsError::new(&format!("Decoding failed: {}", e)))?;

        web_sys::console::log_1(&format!("generate: generated {} tokens", generated).into());
        Ok(response)
    }

    #[cfg(target_arch = "wasm32")]
    /// Embed token IDs using HOST function (no 262MB weight loading!)
    fn embed_tokens_via_host(
        model_id: u32,
        token_ids: &[u32],
        config: &TransformerConfig,
    ) -> Result<Vec<f32>, JsError> {
        let hidden_size = config.hidden_size;

        let seq_len = token_ids.len();
        let output_size = seq_len * hidden_size;
        let mut output = vec![0.0f32; output_size];

        // Copy token_ids to a Vec to ensure it's in WASM linear memory
        // (slices might be stack-allocated and have invalid pointers for FFI)
        let token_ids_vec = token_ids.to_vec();

        web_sys::console::log_1(
            &format!(
                "embed_tokens_via_host: token_ids_vec.as_ptr()={:?}, len={}, output.as_ptr()={:?}",
                token_ids_vec.as_ptr() as usize,
                token_ids_vec.len(),
                output.as_ptr() as usize
            )
            .into(),
        );

        let bytes_written = unsafe {
            realm_embed_tokens(
                model_id,
                token_ids_vec.as_ptr(),
                token_ids_vec.len() as u32,
                output.as_mut_ptr(),
            )
        };

        if bytes_written < 0 {
            return Err(JsError::new(&format!(
                "realm_embed_tokens failed (error: {})",
                bytes_written
            )));
        }

        let expected_bytes = output_size * 4; // f32 = 4 bytes
        if bytes_written as usize != expected_bytes {
            return Err(JsError::new(&format!(
                "realm_embed_tokens size mismatch: expected {} bytes, got {}",
                expected_bytes, bytes_written
            )));
        }

        Ok(output)
    }

    #[cfg(target_arch = "wasm32")]
    /// Compute logits using HOST function (no LM head weight loading!)
    fn compute_logits_via_host(
        model_id: u32,
        hidden_state: &[f32],
        config: &TransformerConfig,
    ) -> Result<Vec<f32>, JsError> {
        let vocab_size = config.vocab_size;

        let mut logits = vec![0.0f32; vocab_size];

        let bytes_written = unsafe {
            realm_compute_logits(
                model_id,
                hidden_state.as_ptr(),
                hidden_state.len() as u32,
                logits.as_mut_ptr(),
            )
        };

        if bytes_written < 0 {
            return Err(JsError::new(&format!(
                "realm_compute_logits failed (error: {})",
                bytes_written
            )));
        }

        let expected_bytes = vocab_size * 4; // f32 = 4 bytes
        if bytes_written as usize != expected_bytes {
            return Err(JsError::new(&format!(
                "realm_compute_logits size mismatch: expected {} bytes, got {}",
                expected_bytes, bytes_written
            )));
        }

        Ok(logits)
    }

    #[cfg(target_arch = "wasm32")]
    fn forward_layer_with_host_weights_static(
        model_id: u32,
        layer_idx: usize,
        hidden_states: &[f32],
        position: usize,
        _config: &TransformerConfig,
        _candle_backend: &realm_compute_cpu::CandleNeuralOpsBackend,
        _cpu_backend: &dyn realm_compute_cpu::CpuBackendTrait,
        _kv_cache: &mut realm_models::KVCache,
    ) -> Result<Vec<f32>, JsError> {
        // NEW ARCHITECTURE: Call HOST function - weights NEVER enter WASM!
        // This solves the 262MB memory issue - weights stay quantized in HOST

        let hidden_states_len = hidden_states.len() as u32;
        let mut output = vec![0.0f32; hidden_states.len()];

        let bytes_written = unsafe {
            realm_forward_layer(
                model_id,
                layer_idx as u32,
                hidden_states.as_ptr(),
                hidden_states_len,
                position as u32,
                output.as_mut_ptr(),
            )
        };

        if bytes_written < 0 {
            return Err(JsError::new(&format!(
                "realm_forward_layer failed for layer {} (error: {})",
                layer_idx, bytes_written
            )));
        }

        let expected_bytes = hidden_states.len() * 4; // f32 = 4 bytes
        if bytes_written as usize != expected_bytes {
            return Err(JsError::new(&format!(
                "realm_forward_layer size mismatch: expected {} bytes, got {}",
                expected_bytes, bytes_written
            )));
        }

        Ok(output)
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
