//! WebAssembly bindings for browser usage
//!
//! Note: Using Arc<Mutex<T>> for WASM is intentional for consistency with potential
//! future multi-threaded WASM support (when SharedArrayBuffer is available).
#![allow(clippy::arc_with_non_send_sync)]

use crate::{
    realm_models::GenerationConfig, ChatMessage, ChatRole, ChatTemplate, Model as RustModel,
    TransformerConfig,
};
use realm_core::{GGUFParser, TensorLoader, Tokenizer};
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

/// JavaScript-compatible Model wrapper
///
/// Note: This is for browser usage only. For large models (>4GB), use the native
/// Memory64 runtime instead of the browser version.
#[wasm_bindgen]
pub struct WasmModel {
    model: Arc<Mutex<RustModel>>,
    tokenizer: Arc<Tokenizer>,
    config: Arc<Mutex<realm_models::GenerationConfig>>,
}

#[wasm_bindgen]
impl WasmModel {
    /// Load model from GGUF bytes (browser-compatible)
    ///
    /// Note: This is limited to models <4GB due to browser WASM memory constraints.
    /// For larger models, use the native Memory64 runtime.
    #[wasm_bindgen(constructor)]
    pub fn new(gguf_bytes: &[u8]) -> Result<WasmModel, JsValue> {
        web_sys::console::log_1(&"🚀 Loading model for browser...".into());

        // Parse GGUF
        let cursor = Cursor::new(gguf_bytes);
        let mut parser = GGUFParser::new(cursor);
        let meta = parser
            .parse_header()
            .map_err(|e| JsValue::from_str(&format!("Failed to parse GGUF: {}", e)))?;

        let config_data = parser
            .extract_config()
            .ok_or_else(|| JsValue::from_str("Failed to extract config"))?;
        let config: TransformerConfig = config_data.into();

        // Check model size
        let total_size: u64 = meta.tensors.iter().map(|t| t.size_bytes as u64).sum();
        let size_gb = total_size as f64 / 1_000_000_000.0;

        web_sys::console::log_1(&format!("📊 Model size: {:.2} GB", size_gb).into());

        // Browser limitation: models must be <4GB
        if total_size > 3_500_000_000 {
            // 3.5GB safety margin
            return Err(JsValue::from_str(&format!(
                "Model too large ({:.2} GB) for browser WASM memory. Browser limit is ~4GB. \
                For large models, use the native Memory64 runtime instead.",
                size_gb
            )));
        }

        // Load tokenizer
        let tokenizer = Tokenizer::from_gguf(&meta)
            .map_err(|e| JsValue::from_str(&format!("Failed to load tokenizer: {}", e)))?;

        // Load weights using standard method
        let mut model = RustModel::new(config.clone());
        let data_offset = parser
            .tensor_data_offset()
            .map_err(|e| JsValue::from_str(&format!("Failed to get data offset: {}", e)))?;
        let mut tensor_loader = TensorLoader::new(data_offset);

        for tensor_desc in meta.tensors.iter() {
            tensor_loader.register_tensor(
                tensor_desc.name.clone(),
                tensor_desc.clone(),
                tensor_desc.offset,
            );
        }

        // Re-parse for loading
        let cursor = Cursor::new(gguf_bytes);
        let mut parser = GGUFParser::new(cursor);
        parser
            .parse_header()
            .map_err(|e| JsValue::from_str(&format!("Failed to re-parse: {}", e)))?;

        model
            .load_from_gguf(&mut tensor_loader, &mut parser)
            .map_err(|e| JsValue::from_str(&format!("Failed to load weights: {}", e)))?;

        web_sys::console::log_1(&"✅ Browser model loaded successfully!".into());

        Ok(WasmModel {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            config: Arc::new(Mutex::new(realm_models::GenerationConfig::default())),
        })
    }

    /// Set generation configuration
    pub fn set_config(
        &mut self,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repetition_penalty: f32,
    ) {
        let mut config = self
            .config
            .lock()
            .expect("Config mutex poisoned - thread panicked while holding lock");
        *config = realm_models::GenerationConfig {
            max_tokens,
            temperature,
            top_p,
            top_k: top_k as usize,
            repetition_penalty,
        };
    }

    /// Generate text (blocking)
    pub fn generate(&mut self, prompt: &str) -> Result<String, JsValue> {
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let config = self.config.clone();

        let mut model_guard = model
            .lock()
            .expect("Model mutex poisoned - thread panicked while holding lock");
        let config_guard = config
            .lock()
            .expect("Config mutex poisoned - thread panicked while holding lock");

        model_guard
            .generate(prompt, &tokenizer, &config_guard)
            .map_err(|e| JsValue::from_str(&format!("Generation failed: {}", e)))
    }

    /// Generate with streaming callback
    /// callback: function(token_text: string) -> boolean (continue?)
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        callback: &js_sys::Function,
    ) -> Result<String, JsValue> {
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let config = self.config.clone();

        let mut model_guard = model.lock().unwrap();
        let config_guard = config.lock().unwrap();
        let this = JsValue::null();

        model_guard
            .generate_stream(
                prompt,
                &tokenizer,
                &config_guard,
                |_token_id, token_text| {
                    let token_js = JsValue::from_str(token_text);
                    if let Ok(result) = callback.call1(&this, &token_js) {
                        result.as_bool().unwrap_or(true)
                    } else {
                        false
                    }
                },
            )
            .map_err(|e| JsValue::from_str(&format!("Streaming generation failed: {}", e)))
    }

    /// Generate with async iterator (returns AsyncTokenStream)
    /// Usage: for await (const token of model.generate_async(prompt)) { ... }
    pub fn generate_async(&self, prompt: String) -> AsyncTokenStream {
        AsyncTokenStream {
            prompt,
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            config: self.config.clone(),
            tokens: Vec::new(),
            current_index: 0,
            is_complete: false,
        }
    }

    /// Initialize GPU backend asynchronously (if available)
    #[cfg(feature = "webgpu")]
    pub async fn init_gpu_async(&self) -> Result<(), JsValue> {
        let model = self.model.clone();

        // Since init_gpu is sync, we use JsFuture to yield to event loop
        let promise = js_sys::Promise::resolve(&JsValue::undefined());
        wasm_bindgen_futures::JsFuture::from(promise).await.ok();

        // Now do GPU init
        let mut model_guard = model.lock().unwrap();
        model_guard
            .init_gpu()
            .map_err(|e| JsValue::from_str(&format!("GPU initialization failed: {}", e)))?;

        web_sys::console::log_1(&"✅ GPU backend initialized successfully".into());
        Ok(())
    }

    /// Initialize GPU backend (blocking fallback)
    #[cfg(feature = "webgpu")]
    pub fn init_gpu(&self) -> Result<(), JsValue> {
        let model = self.model.clone();
        let mut model_guard = model.lock().unwrap();
        model_guard
            .init_gpu()
            .map_err(|e| JsValue::from_str(&format!("GPU initialization failed: {}", e)))
    }

    /// Check if GPU is available
    #[cfg(feature = "webgpu")]
    pub fn is_gpu_available() -> bool {
        realm_compute_gpu::GpuBackend::is_available()
    }

    /// Get model info
    pub fn get_model_info(&self) -> Result<JsValue, JsValue> {
        let model_guard = self.model.lock().unwrap();
        let info = js_sys::Object::new();

        js_sys::Reflect::set(
            &info,
            &"vocab_size".into(),
            &model_guard.config.vocab_size.into(),
        )?;
        js_sys::Reflect::set(
            &info,
            &"hidden_size".into(),
            &model_guard.config.hidden_size.into(),
        )?;
        js_sys::Reflect::set(
            &info,
            &"num_layers".into(),
            &model_guard.config.num_layers.into(),
        )?;
        js_sys::Reflect::set(
            &info,
            &"num_heads".into(),
            &model_guard.config.num_heads.into(),
        )?;
        js_sys::Reflect::set(
            &info,
            &"max_seq_len".into(),
            &model_guard.config.max_seq_len.into(),
        )?;
        js_sys::Reflect::set(&info, &"is_browser_model".into(), &true.into())?;

        Ok(info.into())
    }
}

/// Async token stream for streaming generation
#[wasm_bindgen]
pub struct AsyncTokenStream {
    prompt: String,
    model: Arc<Mutex<RustModel>>,
    tokenizer: Arc<Tokenizer>,
    config: Arc<Mutex<realm_models::GenerationConfig>>,
    tokens: Vec<String>,
    current_index: usize,
    is_complete: bool,
}

#[wasm_bindgen]
impl AsyncTokenStream {
    /// Get next token (async iterator protocol)
    /// Returns {value: string, done: boolean}
    pub async fn next(&mut self) -> Result<JsValue, JsValue> {
        // If we haven't generated tokens yet, do the full generation
        if self.tokens.is_empty() && !self.is_complete {
            let model = self.model.clone();
            let tokenizer = self.tokenizer.clone();
            let config = self.config.clone();
            let prompt = self.prompt.clone();

            // Yield to event loop
            let promise = js_sys::Promise::resolve(&JsValue::undefined());
            wasm_bindgen_futures::JsFuture::from(promise).await.ok();

            // Generate tokens
            let mut model_guard = model.lock().unwrap();
            let config_guard = config.lock().unwrap();

            match model_guard.generate(&prompt, &tokenizer, &config_guard) {
                Ok(result) => {
                    self.tokens = result
                        .chars()
                        .map(|c| c.to_string())
                        .collect::<Vec<String>>();
                }
                Err(_) => {
                    self.tokens = vec!["Error generating tokens".to_string()];
                }
            }

            self.is_complete = true;
        }

        // Return the next token
        if self.current_index < self.tokens.len() {
            let token = self.tokens[self.current_index].clone();
            self.current_index += 1;

            let result = js_sys::Object::new();
            js_sys::Reflect::set(&result, &"value".into(), &token.into())?;
            js_sys::Reflect::set(&result, &"done".into(), &false.into())?;
            Ok(result.into())
        } else {
            let result = js_sys::Object::new();
            js_sys::Reflect::set(&result, &"value".into(), &JsValue::undefined())?;
            js_sys::Reflect::set(&result, &"done".into(), &true.into())?;
            Ok(result.into())
        }
    }

    /// Get async iterator (simplified version)
    pub fn get_async_iterator(&self) -> AsyncTokenStream {
        AsyncTokenStream {
            prompt: self.prompt.clone(),
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            config: self.config.clone(),
            tokens: Vec::new(),
            current_index: 0,
            is_complete: false,
        }
    }

    /// Check if stream is complete
    pub fn is_complete(&self) -> bool {
        self.is_complete && self.current_index >= self.tokens.len()
    }

    /// Get total token count
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

/// Format chat messages with template
#[wasm_bindgen]
pub fn format_chat(
    system: Option<String>,
    user: String,
    template_type: &str,
) -> Result<String, JsValue> {
    let mut messages = Vec::new();

    if let Some(sys) = system {
        messages.push(ChatMessage {
            role: ChatRole::System,
            content: sys,
        });
    }

    messages.push(ChatMessage {
        role: ChatRole::User,
        content: user,
    });

    let template = match template_type {
        "chatml" => ChatTemplate::ChatML,
        "llama2" => ChatTemplate::Llama2,
        "alpaca" => ChatTemplate::Alpaca,
        _ => ChatTemplate::ChatML,
    };

    template
        .format(&messages)
        .map_err(|e| JsValue::from_str(&format!("Template formatting failed: {}", e)))
}

/// Get library version
#[wasm_bindgen]
pub fn version() -> String {
    crate::VERSION.to_string()
}
