//! Core inference primitives for wasm-chord
//!
//! This crate provides the fundamental building blocks for neural network inference:
//! - Tensor types and operations
//! - Model format parsing (GGUF)
//! - Quantization/dequantization primitives
//! - Memory management abstractions

pub mod error;
pub mod formats;
pub mod memory;
pub mod quant;
pub mod tensor;
pub mod tensor_loader;
pub mod tokenizer;

pub use error::{Error, Result};
pub use formats::gguf::{GGUFParser, MetadataValue, ModelMeta};
pub use tensor::{DataType, Shape, Tensor, TensorDesc};
pub use tensor_loader::{TensorLoader, TensorMetadata};
pub use tokenizer::{SpecialTokens, Tokenizer};

/// Core version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Transformer configuration data (for cross-crate usage)
#[derive(Debug, Clone)]
pub struct TransformerConfigData {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl TransformerConfigData {
    /// Convert to runtime TransformerConfig (requires runtime crate to import)
    pub fn to_config(&self) -> Self {
        self.clone()
    }
}
