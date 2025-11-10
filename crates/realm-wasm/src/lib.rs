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
// Logging Helpers (Server vs Web)
// ========================================

#[cfg(all(target_arch = "wasm32", feature = "server"))]
macro_rules! wasm_log {
    ($($arg:tt)*) => {
        #[cfg(feature = "tracing")]
        tracing::debug!($($arg)*);
    };
}

#[cfg(all(target_arch = "wasm32", not(feature = "server")))]
macro_rules! wasm_log {
    ($($arg:tt)*) => {
        web_sys::console::log_1(&format!($($arg)*).into());
    };
}

#[cfg(not(target_arch = "wasm32"))]
macro_rules! wasm_log {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}

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
    #[allow(dead_code)] // Called by host, not from Rust
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

    /// Get model metadata (config + tokenizer info) as JSON
    /// Parameters: model_id, out_ptr, out_max_len
    /// Returns: number of bytes written on success, negative on error
    fn realm_get_model_metadata(model_id: u32, out_ptr: *mut u8, out_max_len: u32) -> i32;

    /// Remove model from storage (cleanup)
    /// Returns: 0 on success, negative on error
    #[allow(dead_code)] // Called by host, not from Rust
    fn realm_remove_model(model_id: u32) -> i32;

    /// Set LoRA adapter for a model
    /// Parameters: model_id, adapter_id_ptr, adapter_id_len (WASM memory offsets)
    /// Returns: 0 on success, negative on error
    #[allow(dead_code)] // Called by host, not from Rust
    fn realm_set_lora_adapter(model_id: u32, adapter_id_ptr: *const u8, adapter_id_len: u32)
        -> i32;

    /// Encode text to token IDs
    /// Parameters: model_id, text_ptr, text_len, out_ptr, out_max_len
    /// Returns: number of tokens written on success, negative on error
    #[allow(dead_code)] // Called by host, not from Rust
    fn realm_encode_tokens(
        model_id: u32,
        text_ptr: *const u8,
        text_len: u32,
        out_ptr: *mut u32,
        out_max_len: u32,
    ) -> i32;

    /// Decode token IDs to text
    /// Parameters: model_id, token_ids_ptr, token_ids_len, out_ptr, out_max_len
    /// Returns: number of bytes written on success, negative on error
    #[allow(dead_code)] // Called by host, not from Rust
    fn realm_decode_tokens(
        model_id: u32,
        token_ids_ptr: *const u32,
        token_ids_len: u32,
        out_ptr: *mut u8,
        out_max_len: u32,
    ) -> i32;

    /// Store draft model for speculative decoding
    /// Parameters: gguf_ptr, gguf_len, draft_model_id (WASM memory offsets)
    /// Returns: draft_model_id on success (> 0), negative on error
    #[allow(dead_code)]
    fn realm_store_draft_model(gguf_ptr: *const u8, gguf_len: u32, draft_model_id: u32) -> i32;

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

    /// Stream a token as it's generated (REAL TOKEN-BY-TOKEN STREAMING)
    /// This enables real-time token streaming via host function callbacks
    /// Parameters:
    ///   - token_ptr: Pointer to token string in WASM memory
    ///   - token_len: Length of token string in bytes
    /// Returns: 0 on success, negative on error
    fn realm_stream_token(token_ptr: *const u8, token_len: u32) -> i32;
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
    #[allow(dead_code)]
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
    ///
    /// Returns a fully-initialized Realm instance (Pattern 1: constructor returns instance)
    /// This ensures wasm-bindgen generates the correct signature: () -> u32
    #[wasm_bindgen(constructor)]
    pub fn new() -> Realm {
        // Set up better panic messages for WASM debugging (web mode only)
        #[cfg(all(target_arch = "wasm32", not(feature = "server")))]
        console_error_panic_hook::set_once();

        Realm {
            model: None,
            tokenizer: None,
            model_id: None,
            transformer_config: None,
            config: WasmGenerationConfig::new(),
            #[cfg(target_arch = "wasm32")]
            kv_caches: None,
        }
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
        wasm_log!("loadModel: received {} bytes", model_bytes.len());

        // Parse GGUF header for metadata (lightweight operation)
        let cursor = Cursor::new(model_bytes);
        let mut parser = GGUFParser::new(cursor);

        wasm_log!("loadModel: parsing header...");

        let meta = parser
            .parse_header()
            .map_err(|e| JsError::new(&format!("Failed to parse GGUF header: {}", e)))?;

        // Extract config
        wasm_log!("loadModel: extracting config...");

        let config_data = parser
            .extract_config()
            .ok_or_else(|| JsError::new("Failed to extract config from GGUF"))?;
        let _config: TransformerConfig = config_data.into();

        // Create tokenizer from GGUF metadata
        wasm_log!("loadModel: creating tokenizer...");

        let tokenizer = Tokenizer::from_gguf(&meta)
            .map_err(|e| JsError::new(&format!("Failed to create tokenizer: {}", e)))?;

        wasm_log!(
            "loadModel: tokenizer created, vocab_size={}",
            tokenizer.vocab_size()
        );

        // ========================================
        // INNOVATION: Store model in HOST, not WASM
        // ========================================

        #[cfg(target_arch = "wasm32")]
        {
            wasm_log!("loadModel: calling HOST to store model...");

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

            wasm_log!("loadModel: model stored in HOST with ID {}", model_id);

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

            wasm_log!(
                "loadModel: model has {} tensors, {:.2} MB total in HOST",
                tensor_count,
                total_size as f64 / 1024.0 / 1024.0
            );

            // Create lightweight model structure (NO WEIGHTS!)
            let model = Model::new(_config.clone());

            // Initialize KV caches for each layer
            let head_dim = _config.hidden_size / _config.num_heads;
            let mut kv_caches = Vec::new();
            for _ in 0.._config.num_layers {
                kv_caches.push(realm_models::KVCache::new(
                    _config.max_seq_len,
                    _config.num_kv_heads,
                    head_dim,
                ));
            }

            // Store everything in WASM (minimal memory usage)
            self.model = Some(model);
            self.tokenizer = Some(tokenizer);
            self.model_id = Some(model_id as u32);
            self.transformer_config = Some(_config);
            self.kv_caches = Some(kv_caches);

            wasm_log!("loadModel: SUCCESS! Model handle stored in WASM, weights in HOST");

            Ok(())
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Non-WASM builds: fall back to old behavior
            Err(JsError::new(
                "loadModel not available in non-WASM environment",
            ))
        }
    }

    /// Load a model by ID from HOST storage (no model bytes needed!)
    ///
    /// This is the correct way to load models when they're already stored in HOST:
    /// 1. Get config and tokenizer info from HOST storage via host function
    /// 2. Initialize Realm instance with config/tokenizer
    /// 3. Set model_id - model weights stay in HOST
    ///
    /// # Arguments
    /// * `model_id` - Model ID in HOST storage
    ///
    /// # Architecture
    /// Models are stored in HOST memory. This function gets metadata from HOST
    /// and initializes the WASM Realm instance without needing model bytes.
    #[wasm_bindgen(js_name = loadModelById)]
    pub fn load_model_by_id(&mut self, model_id: u32) -> Result<(), JsError> {
        wasm_log!(
            "🚀 loadModelById ENTRY: loading model ID {} from HOST storage",
            model_id
        );

        #[cfg(target_arch = "wasm32")]
        {
            // Get model metadata from HOST storage
            // Allocate buffer for JSON metadata (max 4KB should be enough)
            const METADATA_BUFFER_SIZE: u32 = 4096;
            let mut metadata_buffer = vec![0u8; METADATA_BUFFER_SIZE as usize];
            let metadata_ptr = metadata_buffer.as_mut_ptr();

            wasm_log!("📞 loadModelById: About to call realm_get_model_metadata for model_id {}, buffer_ptr={:p}, buffer_size={}", model_id, metadata_ptr, METADATA_BUFFER_SIZE);

            let bytes_written =
                unsafe { realm_get_model_metadata(model_id, metadata_ptr, METADATA_BUFFER_SIZE) };

            wasm_log!(
                "✅ loadModelById: realm_get_model_metadata returned {} bytes",
                bytes_written
            );

            if bytes_written < 0 {
                wasm_log!(
                    "loadModelById: realm_get_model_metadata failed with error code: {}",
                    bytes_written
                );
                return Err(JsError::new(&format!(
                    "Failed to get model metadata from HOST (error code: {})",
                    bytes_written
                )));
            }

            wasm_log!(
                "loadModelById: received {} bytes of metadata",
                bytes_written
            );

            // Read JSON from buffer
            let json_bytes = &metadata_buffer[..bytes_written as usize];
            let json_str = std::str::from_utf8(json_bytes)
                .map_err(|e| JsError::new(&format!("Invalid UTF-8 in metadata: {}", e)))?;

            // Parse JSON
            let metadata: serde_json::Value = serde_json::from_str(json_str)
                .map_err(|e| JsError::new(&format!("Failed to parse metadata JSON: {}", e)))?;

            // Extract config
            let config_obj = metadata
                .get("config")
                .ok_or_else(|| JsError::new("Missing 'config' in metadata"))?;

            let config = TransformerConfig {
                vocab_size: config_obj
                    .get("vocab_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32000) as usize,
                hidden_size: config_obj
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2048) as usize,
                num_layers: config_obj
                    .get("num_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(22) as usize,
                num_heads: config_obj
                    .get("num_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32) as usize,
                num_kv_heads: config_obj
                    .get("num_kv_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4) as usize,
                intermediate_size: config_obj
                    .get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(5632) as usize,
                max_seq_len: config_obj
                    .get("max_seq_len")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2048) as usize,
                rope_theta: config_obj
                    .get("rope_theta")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(10000.0) as f32,
                rms_norm_eps: config_obj
                    .get("rms_norm_eps")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1e-5) as f32,
                attention_backend: realm_models::AttentionBackend::Auto,
            };

            wasm_log!("loadModelById: extracted config from HOST storage");

            // Get tokenizer info
            let _has_tokenizer = metadata
                .get("has_tokenizer")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            // Create tokenizer - we'll get it from HOST when needed via host functions
            // For now, we'll create a placeholder or get it from HOST storage
            // Tokenizer is stored in HOST, so we don't need to recreate it here
            // We'll use host functions (realm_encode_tokens, realm_decode_tokens) for tokenization

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
            // Tokenizer is in HOST, we'll use host functions for tokenization
            self.tokenizer = None; // Will use host functions instead
            self.model_id = Some(model_id);
            self.transformer_config = Some(config);
            self.kv_caches = Some(kv_caches);

            wasm_log!(
                "loadModelById: SUCCESS! Model {} initialized from HOST storage (weights stay in HOST)",
                model_id
            );

            Ok(())
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Non-WASM fallback (should not be called)
            Err(JsError::new(
                "loadModelById not available in non-WASM environment",
            ))
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
            self.generate_with_callback(prompt, None)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.generate_with_callback(prompt, None)
        }
    }

    /// Generate text with streaming callback (HOST-side storage version)
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `callback` - Optional callback function(token_text: string, token_id: u32) -> bool
    ///                Returns false to stop generation, true to continue
    ///
    /// # Returns
    /// Generated text response
    #[allow(unused_variables)] // callback is only used in WASM mode
    fn generate_with_callback(
        &mut self,
        prompt: String,
        callback: Option<js_sys::Function>,
    ) -> Result<String, JsError> {
        #[cfg(target_arch = "wasm32")]
        {
            // WASM path: Use HOST storage for weights
            self.generate_with_host_storage_internal(prompt, callback)
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Non-WASM path: Use traditional model loading
            // Callback is only used in WASM mode
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
    fn generate_with_host_storage_internal(
        &mut self,
        prompt: String,
        callback: Option<js_sys::Function>,
    ) -> Result<String, JsError> {
        let model_id = self
            .model_id
            .ok_or_else(|| JsError::new("Model not loaded in HOST. Call loadModel() first."))?;
        let _config = self
            .transformer_config
            .as_ref()
            .ok_or_else(|| JsError::new("Config not loaded."))?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| JsError::new("Tokenizer not loaded."))?;
        let gen_config: GenerationConfig = self.config.clone().into();

        wasm_log!("generate: starting with prompt '{}'", prompt);

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
        wasm_log!("generate: encoded {} prompt tokens", num_prompt_tokens);

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

        // Stream first token via host function (real token-by-token streaming)
        let token_text = tokenizer
            .decode(&[next], false)
            .map_err(|e| JsError::new(&format!("Decoding failed: {}", e)))?;

        #[cfg(target_arch = "wasm32")]
        {
            // Call realm_stream_token host function for real streaming
            let token_bytes = token_text.as_bytes();
            let token_ptr = token_bytes.as_ptr();
            let token_len = token_bytes.len() as u32;
            let _ = unsafe { realm_stream_token(token_ptr, token_len) };
        }

        // Also support JavaScript callback for compatibility
        if let Some(ref cb) = callback {
            let this = JsValue::null();
            let token_js = JsValue::from_str(&token_text);
            let token_id_js = JsValue::from_f64(next as f64);
            if let Ok(result) = cb.call2(&this, &token_js, &token_id_js) {
                if !result.as_bool().unwrap_or(true) {
                    // Callback returned false, stop generation
                    return Ok(token_text);
                }
            }
        }

        // DECODE PHASE: Generate tokens one at a time
        let mut generated = 1;
        let mut accumulated_text = String::new();
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

            // Stream token via host function (real token-by-token streaming)
            let token_text = tokenizer
                .decode(&[next], false)
                .map_err(|e| JsError::new(&format!("Decoding failed: {}", e)))?;
            accumulated_text.push_str(&token_text);

            #[cfg(target_arch = "wasm32")]
            {
                // Call realm_stream_token host function for real streaming
                let token_bytes = token_text.as_bytes();
                let token_ptr = token_bytes.as_ptr();
                let token_len = token_bytes.len() as u32;
                let _ = unsafe { realm_stream_token(token_ptr, token_len) };
            }

            // Also support JavaScript callback for compatibility
            if let Some(ref cb) = callback {
                let this = JsValue::null();
                let token_js = JsValue::from_str(&token_text);
                let token_id_js = JsValue::from_f64(next as f64);
                if let Ok(result) = cb.call2(&this, &token_js, &token_id_js) {
                    if !result.as_bool().unwrap_or(true) {
                        // Callback returned false, stop generation
                        break;
                    }
                }
            }
        }

        // Decode tokens to text
        let response = tokenizer
            .decode(&tokens, true)
            .map_err(|e| JsError::new(&format!("Decoding failed: {}", e)))?;

        wasm_log!("generate: generated {} tokens", generated);
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

        wasm_log!(
            "embed_tokens_via_host: token_ids_vec.as_ptr()={:?}, len={}, output.as_ptr()={:?}",
            token_ids_vec.as_ptr() as usize,
            token_ids_vec.len(),
            output.as_ptr() as usize
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

#[allow(clippy::derivable_impls)] // Conditional logic based on target_arch prevents derivation
impl Default for Realm {
    fn default() -> Self {
        // Use wasm_bindgen constructor
        #[cfg(target_arch = "wasm32")]
        {
            Self::new() // No longer returns Result, so no expect needed
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Non-WASM fallback
            Self {
                model: None,
                tokenizer: None,
                model_id: None,
                transformer_config: None,
                config: WasmGenerationConfig::new(),
            }
        }
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
        // Constructor now returns Realm directly (not Result)
        assert!(realm.model.is_none());
        assert!(realm.tokenizer.is_none());
    }

    #[test]
    fn test_config() {
        let mut realm = Realm::new();
        let mut config = WasmGenerationConfig::new();
        config.max_tokens = 200;
        config.temperature = 0.8;
        realm.set_config(config.clone());
        assert_eq!(realm.config.max_tokens, 200);
        assert_eq!(realm.config.temperature, 0.8);
    }

    #[test]
    fn test_not_loaded() {
        let realm = Realm::new();
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
