//! Transformer architecture implementation
//!
//! Implements the core transformer components for LLM inference.

mod attention;
mod attention_stub;
mod config;
mod ffn;
mod kv_cache;
mod layer;
mod matmul_dispatch;
mod model;
mod sampling;
mod weight_format;

pub use attention::{AttentionWeights, MultiHeadAttention};
pub use attention_stub::AttentionBackend;
pub use config::{GenerationConfig, TransformerConfig};
pub use ffn::{FFNWeights, FeedForward};
pub use kv_cache::KVCache;
pub use layer::TransformerLayer;
pub use matmul_dispatch::*;
pub use model::Model;
pub use sampling::*;
pub use weight_format::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCache::new(2048, 4, 64);
        assert_eq!(cache.current_seq_len, 0);
        assert_eq!(cache.max_seq_len, 2048);
        assert_eq!(cache.num_kv_heads, 4);
        assert_eq!(cache.head_dim, 64);
    }

    #[test]
    fn test_transformer_config() {
        let config = TransformerConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 22);
    }

    #[test]
    fn test_attention_weights_creation() {
        let config = TransformerConfig::default();
        let weights = AttentionWeights::new(&config);

        assert_eq!(weights.wq.len(), config.hidden_size * config.hidden_size);
        assert_eq!(weights.wo.len(), config.hidden_size * config.hidden_size);
    }

    #[test]
    fn test_ffn_weights_creation() {
        let config = TransformerConfig::default();
        let weights = FFNWeights::new(&config);

        assert_eq!(
            weights.w_gate.len(),
            config.hidden_size * config.intermediate_size
        );
        assert_eq!(
            weights.w_down.len(),
            config.intermediate_size * config.hidden_size
        );
    }

    #[test]
    fn test_model_creation() {
        let config = TransformerConfig::default();
        let model = Model::new(config.clone());

        assert_eq!(model.layers.len(), config.num_layers);
        assert_eq!(model.kv_caches.len(), config.num_layers);
        assert_eq!(
            model.token_embeddings.len(),
            config.vocab_size * config.hidden_size
        );
        assert_eq!(model.lm_head.len(), config.hidden_size * config.vocab_size);
    }

    #[test]
    #[ignore] // TODO: Fix attention implementation
    fn test_model_forward_pass() {
        // Create a tiny config for testing
        let config = TransformerConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_backend: crate::AttentionBackend::Auto,
        };

        let mut model = Model::new(config);

        // Test forward pass with a single token
        let input_tokens = vec![1];
        let result = model.forward(&input_tokens, 0);

        assert!(result.is_ok());
        let logits = result.unwrap();
        // In inference mode, we only return logits for the last token
        assert_eq!(logits.len(), model.config.vocab_size);
    }

    #[test]
    fn test_model_sampling_greedy() {
        let config = TransformerConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 16,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_backend: crate::AttentionBackend::Auto,
        };

        let model = Model::new(config);

        // Create test logits (favor token 5)
        let mut logits = vec![0.0; 10];
        logits[5] = 10.0;

        // Greedy sampling (temperature = 0)
        let sampled = model.sample(&logits, 0.0, 1.0, 0).unwrap();
        assert_eq!(sampled, 5);

        // Greedy with temperature = 1.0
        let sampled = model.sample(&logits, 1.0, 1.0, 0).unwrap();
        assert_eq!(sampled, 5);
    }

    #[test]
    fn test_model_sampling_top_k() {
        let config = TransformerConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 16,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_backend: crate::AttentionBackend::Auto,
        };

        let model = Model::new(config);

        // Create logits with known distribution
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 0.3, 0.2, 0.1, 0.0];

        // Top-k = 3 should only consider indices 2, 3, 4
        let sampled = model.sample(&logits, 1.0, 1.0, 3).unwrap();
        assert!((2..=4).contains(&sampled));
    }

    #[test]
    fn test_model_sampling_top_p() {
        let config = TransformerConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 16,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_backend: crate::AttentionBackend::Auto,
        };

        let model = Model::new(config);

        // Create logits with exponential distribution
        let mut logits = vec![0.0; 10];
        logits[0] = 10.0; // Highest probability
        logits[1] = 5.0;
        logits[2] = 2.0;
        logits[3] = 1.0;
        // Rest are much lower

        // Top-p = 0.9 should focus on top few tokens
        let sampled = model.sample(&logits, 1.0, 0.9, 0).unwrap();
        assert!(sampled <= 3, "sampled token should be in nucleus");
    }

    #[test]
    fn test_model_sampling_temperature() {
        let config = TransformerConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 16,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_backend: crate::AttentionBackend::Auto,
        };

        let model = Model::new(config);

        let mut logits = vec![1.0; 10];
        logits[5] = 2.0;

        // Low temperature should still prefer token 5
        let sampled = model.sample(&logits, 0.1, 1.0, 0).unwrap();
        assert_eq!(sampled, 5);

        // High temperature makes distribution more uniform
        let sampled = model.sample(&logits, 10.0, 1.0, 0).unwrap();
        // Should still work (may or may not be 5)
        assert!(sampled < 10);
    }

    #[test]
    #[ignore] // TODO: Fix attention implementation
    fn test_attention_computation() {
        // Small config for testing
        let config = TransformerConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_backend: crate::AttentionBackend::Auto,
        };

        let attention = MultiHeadAttention::new(config.clone());

        let seq_len = 3;
        let head_dim = config.hidden_size / config.num_heads; // 16

        // Create test Q, K, V
        let q = vec![0.1; seq_len * config.num_heads * head_dim];
        let k = vec![0.2; seq_len * config.num_kv_heads * head_dim];
        let v = vec![0.3; seq_len * config.num_kv_heads * head_dim];

        let result = attention.compute_attention(&q, &k, &v, seq_len, 0);

        assert!(result.is_ok());
        let output = result.unwrap();

        // Check output shape
        assert_eq!(output.len(), seq_len * config.num_heads * head_dim);

        // Output should not be all zeros (attention was computed)
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 0.0);
    }

    #[test]
    #[ignore] // TODO: Fix stack overflow
    fn test_attention_causal_masking() {
        // Test that causal masking works correctly
        let config = TransformerConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4, // Standard MHA for simplicity
            intermediate_size: 128,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_backend: crate::AttentionBackend::Auto,
        };

        let attention = MultiHeadAttention::new(config.clone());

        let seq_len = 4;
        let head_dim = config.hidden_size / config.num_heads;

        // Create distinct Q, K, V values
        let mut q = vec![0.0; seq_len * config.num_heads * head_dim];
        let mut k = vec![0.0; seq_len * config.num_kv_heads * head_dim];
        let mut v = vec![0.0; seq_len * config.num_kv_heads * head_dim];

        // Set distinct values for each position
        for i in 0..seq_len {
            for h in 0..config.num_heads {
                for d in 0..head_dim {
                    q[(i * config.num_heads + h) * head_dim + d] = (i + 1) as f32;
                }
            }
            for h in 0..config.num_kv_heads {
                for d in 0..head_dim {
                    k[(i * config.num_kv_heads + h) * head_dim + d] = (i + 1) as f32;
                    v[(i * config.num_kv_heads + h) * head_dim + d] = (i + 1) as f32 * 10.0;
                }
            }
        }

        let result = attention.compute_attention(&q, &k, &v, seq_len, 0).unwrap();

        // Check that output shape is correct
        assert_eq!(result.len(), seq_len * config.num_heads * head_dim);

        // Verify values are reasonable (not NaN, not zero)
        for &val in &result {
            assert!(val.is_finite());
        }
    }

    #[test]
    #[ignore] // TODO: Fix attention implementation
    fn test_attention_with_gqa() {
        // Test Grouped Query Attention
        let config = TransformerConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 1,
            num_heads: 8,
            num_kv_heads: 2, // GQA: 4 query heads per KV head
            intermediate_size: 128,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_backend: crate::AttentionBackend::Auto,
        };

        let attention = MultiHeadAttention::new(config.clone());

        let seq_len = 2;
        let head_dim = config.hidden_size / config.num_heads; // 8

        let q = vec![1.0; seq_len * config.num_heads * head_dim];
        let k = vec![0.5; seq_len * config.num_kv_heads * head_dim];
        let v = vec![2.0; seq_len * config.num_kv_heads * head_dim];

        let result = attention.compute_attention(&q, &k, &v, seq_len, 0);

        assert!(result.is_ok());
        let output = result.unwrap();

        // Verify correct output shape
        assert_eq!(output.len(), seq_len * config.num_heads * head_dim);

        // All values should be finite
        for &val in &output {
            assert!(val.is_finite());
            assert!(val >= 0.0); // Since V is all positive and weights are normalized
        }
    }
}
