//! Transformer layer implementation

use realm_core::error::Result;

#[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
use realm_compute_gpu::GpuBackendTrait;

use super::{
    AttentionWeights, FFNWeights, FeedForward, KVCache, MultiHeadAttention, TransformerConfig,
};

/// Single transformer layer
#[allow(dead_code)]
pub struct TransformerLayer {
    pub attention: MultiHeadAttention,
    pub ffn: FeedForward,
    pub attention_weights: AttentionWeights,
    pub ffn_weights: FFNWeights,
    pub attention_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
}

#[allow(dead_code)]
impl TransformerLayer {
    pub fn new(config: &TransformerConfig) -> Self {
        Self {
            attention: MultiHeadAttention::new(config.clone()),
            ffn: FeedForward::new(config.clone()),
            attention_weights: AttentionWeights::new(config),
            ffn_weights: FFNWeights::new(config),
            attention_norm: vec![1.0; config.hidden_size],
            ffn_norm: vec![1.0; config.hidden_size],
        }
    }

    pub fn forward(
        &self,
        hidden_states: &[f32],
        kv_cache: &mut KVCache,
        position: usize,
        cpu_backend: Option<&dyn realm_compute_cpu::CpuBackendTrait>,
        #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))] gpu: Option<
            &dyn GpuBackendTrait,
        >,
    ) -> Result<Vec<f32>> {
        // Pre-norm architecture (like LLaMA)

        // Calculate sequence length from input dimensions (unused but kept for clarity)
        let _seq_len = hidden_states.len() / self.attention_norm.len();

        // 1. Attention block with residual
        let normed = self.rms_norm(hidden_states, &self.attention_norm)?;
        let attn_output = self.attention.forward(
            &normed,
            &self.attention_weights,
            kv_cache,
            position,
            cpu_backend,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            gpu,
        )?;

        // Add residual connection
        let hidden: Vec<f32> = hidden_states
            .iter()
            .zip(attn_output.iter())
            .map(|(a, b)| a + b)
            .collect();

        // 2. FFN block with residual
        let normed = self.rms_norm(&hidden, &self.ffn_norm)?;
        let ffn_output = self.ffn.forward(
            &normed,
            &self.ffn_weights,
            cpu_backend,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            gpu,
        )?;

        // Add residual connection
        let final_hidden: Vec<f32> = hidden
            .iter()
            .zip(ffn_output.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(final_hidden)
    }

    pub fn rms_norm(&self, input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
        let hidden_size = weight.len();
        let seq_len = input.len() / hidden_size;
        let mut output = vec![0.0; input.len()];

        for seq in 0..seq_len {
            let offset = seq * hidden_size;
            let slice = &input[offset..offset + hidden_size];

            // Compute RMS exactly like llama.cpp:
            // 1. Sum of squares
            // 2. Mean = sum / n
            // 3. RMS = sqrt(mean + eps)
            let sum_sq: f32 = slice.iter().map(|&x| x * x).sum();
            let mean = sum_sq / hidden_size as f32;
            let rms = (mean + self.attention.config.rms_norm_eps).sqrt();

            // Normalize and scale by weight
            for i in 0..hidden_size {
                output[offset + i] = (slice[i] / rms) * weight[i];
            }
        }

        Ok(output)
    }
}
