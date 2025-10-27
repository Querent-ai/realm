//! Realm WASM - WASM orchestrator module for Realm.ai
//!
//! This module provides:
//! - Customer-facing API (generate, loadModel, etc.)
//! - Inference orchestration
//! - Host function imports (calls to native runtime)

use wasm_bindgen::prelude::*;

/// Realm WASM instance
#[wasm_bindgen]
pub struct Realm {
    max_tokens: u32,
    temperature: f32,
}

#[wasm_bindgen]
impl Realm {
    /// Create a new Realm instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Realm, JsError> {
        Ok(Realm {
            max_tokens: 100,
            temperature: 0.7,
        })
    }

    /// Set max tokens for generation
    pub fn set_max_tokens(&mut self, max_tokens: u32) {
        self.max_tokens = max_tokens;
    }

    /// Set temperature for sampling
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }

    /// Generate text from a prompt
    pub fn generate(&self, prompt: String) -> Result<String, JsError> {
        // TODO: Implement actual generation
        // This will:
        // 1. Tokenize the prompt
        // 2. Call host functions for matmul operations
        // 3. Sample tokens
        // 4. Decode back to text

        Ok(format!(
            "Generated response for: {} (max_tokens: {}, temp: {})",
            prompt, self.max_tokens, self.temperature
        ))
    }

    /// Load a model (delegates to host)
    pub fn load_model(&mut self, _model_path: String) -> Result<(), JsError> {
        // TODO: Call host function to load model into Memory64
        // Removed console.log to avoid wasm-bindgen dependencies in simple test
        Ok(())
    }
}

// Host function imports (to be called from WASM)
// These are provided by realm-runtime
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
        realm.set_max_tokens(200);
        realm.set_temperature(0.8);
        assert_eq!(realm.max_tokens, 200);
        assert_eq!(realm.temperature, 0.8);
    }
}
