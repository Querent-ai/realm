//! Transformer configuration structs

use crate::attention_stub::AttentionBackend;
use serde::{Deserialize, Serialize};

/// Transformer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct TransformerConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Intermediate (FFN) size
    pub intermediate_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta
    pub rope_theta: f32,
    /// Attention implementation (Standard, Flash, Auto)
    pub attention_backend: AttentionBackend,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            num_layers: 22,
            num_heads: 32,
            num_kv_heads: 4, // GQA: Grouped Query Attention
            intermediate_size: 5632,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_backend: AttentionBackend::Auto, // Auto-select best attention
        }
    }
}

impl From<realm_core::TransformerConfigData> for TransformerConfig {
    fn from(data: realm_core::TransformerConfigData) -> Self {
        Self {
            vocab_size: data.vocab_size,
            hidden_size: data.hidden_size,
            num_layers: data.num_layers,
            num_heads: data.num_heads,
            num_kv_heads: data.num_kv_heads,
            intermediate_size: data.intermediate_size,
            max_seq_len: data.max_seq_len,
            rms_norm_eps: data.rms_norm_eps,
            rope_theta: data.rope_theta,
            attention_backend: AttentionBackend::Auto, // Auto-select best attention
        }
    }
}

/// Generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = greedy/deterministic)
    pub temperature: f32,
    /// Nucleus sampling threshold
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: usize,
    /// Repetition penalty
    pub repetition_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
        }
    }
}
