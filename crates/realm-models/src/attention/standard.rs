// Standard O(N²) attention implementation
//
// This is the baseline implementation that:
// - Materializes the full N² attention matrix
// - Uses more memory than Flash Attention
// - But is simpler and always available

use super::{config::StandardAttentionConfig, Attention};
use realm_core::error::Result;

/// Standard attention implementation
///
/// Computes attention using the formula:
/// ```text
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
/// ```
///
/// Memory complexity: O(batch * num_heads * seq_len²)
/// Time complexity: O(batch * num_heads * seq_len² * head_dim)
pub struct StandardAttention {
    config: StandardAttentionConfig,
}

impl StandardAttention {
    /// Create a new standard attention instance
    pub fn new() -> Self {
        Self { config: StandardAttentionConfig::default() }
    }

    /// Create with custom configuration
    pub fn with_config(config: StandardAttentionConfig) -> Self {
        Self { config }
    }

    /// Compute Q @ K^T with scaling
    #[allow(clippy::too_many_arguments)]
    fn compute_scores(
        &self,
        q: &[f32],
        k: &[f32],
        batch_size: usize,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let total_size = batch_size * num_heads * seq_len_q * seq_len_k;
        let mut scores = vec![0.0; total_size];

        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len_q {
                    for j in 0..seq_len_k {
                        let mut dot = 0.0f32;

                        // Compute dot product between Q[i] and K[j]
                        for d in 0..head_dim {
                            let q_idx = ((b * num_heads + h) * seq_len_q + i) * head_dim + d;
                            let k_idx = ((b * num_heads + h) * seq_len_k + j) * head_dim + d;
                            dot += q[q_idx] * k[k_idx];
                        }

                        let score_idx = ((b * num_heads + h) * seq_len_q + i) * seq_len_k + j;
                        scores[score_idx] = dot * scale;
                    }
                }
            }
        }

        scores
    }

    /// Apply softmax to attention scores
    fn apply_softmax(
        &self,
        scores: &mut [f32],
        mask: Option<&[f32]>,
        batch_size: usize,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
    ) {
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len_q {
                    let row_start = ((b * num_heads + h) * seq_len_q + i) * seq_len_k;
                    let row_end = row_start + seq_len_k;

                    // Apply mask if provided
                    // Support both simple 2D masks [seq_len_q, seq_len_k] and batched masks
                    if let Some(mask_data) = mask {
                        let mask_len = mask_data.len();
                        let simple_2d_size = seq_len_q * seq_len_k;

                        for j in 0..seq_len_k {
                            let mask_idx = if mask_len == simple_2d_size {
                                // Simple 2D mask: [seq_len_q, seq_len_k]
                                i * seq_len_k + j
                            } else {
                                // Batched mask: [batch, num_heads, seq_len_q, seq_len_k]
                                ((b * num_heads + h) * seq_len_q + i) * seq_len_k + j
                            };

                            if mask_idx < mask_len && mask_data[mask_idx] == 0.0 {
                                scores[row_start + j] = f32::NEG_INFINITY;
                            }
                        }
                    }

                    // Find max for numerical stability
                    let max_score =
                        scores[row_start..row_end].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                    // Compute exp and sum
                    let mut sum = 0.0f32;
                    for j in 0..seq_len_k {
                        let idx = row_start + j;
                        if scores[idx].is_finite() {
                            scores[idx] = (scores[idx] - max_score).exp();
                            sum += scores[idx];
                        } else {
                            scores[idx] = 0.0;
                        }
                    }

                    // Normalize
                    if sum > 0.0 {
                        for j in 0..seq_len_k {
                            scores[row_start + j] /= sum;
                        }
                    }
                }
            }
        }
    }

    /// Compute attention_weights @ V
    #[allow(clippy::too_many_arguments)]
    fn apply_attention_to_values(
        &self,
        attention_weights: &[f32],
        v: &[f32],
        batch_size: usize,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let output_size = batch_size * num_heads * seq_len_q * head_dim;
        let mut output = vec![0.0; output_size];

        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len_q {
                    for d in 0..head_dim {
                        let mut sum = 0.0f32;

                        for j in 0..seq_len_k {
                            let attn_idx = ((b * num_heads + h) * seq_len_q + i) * seq_len_k + j;
                            let v_idx = ((b * num_heads + h) * seq_len_k + j) * head_dim + d;
                            sum += attention_weights[attn_idx] * v[v_idx];
                        }

                        let out_idx = ((b * num_heads + h) * seq_len_q + i) * head_dim + d;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        output
    }
}

impl Attention for StandardAttention {
    fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
        batch_size: usize,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        // Get softmax scale
        let scale = self.config.get_softmax_scale(head_dim);

        // 1. Compute Q @ K^T / sqrt(d)
        let mut scores =
            self.compute_scores(q, k, batch_size, num_heads, seq_len_q, seq_len_k, head_dim, scale);

        // 2. Apply softmax (with optional mask)
        self.apply_softmax(&mut scores, mask, batch_size, num_heads, seq_len_q, seq_len_k);

        // 3. Multiply by V
        let output = self.apply_attention_to_values(
            &scores, v, batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
        );

        Ok(output)
    }

    fn name(&self) -> &str {
        "StandardAttention"
    }

    fn is_available(&self) -> bool {
        true // Always available
    }

    fn estimated_memory(&self, seq_len: usize, head_dim: usize, num_heads: usize) -> usize {
        // Q, K, V, scores, output
        let qkv_size = 3 * seq_len * head_dim * num_heads * 4; // FP32
        let scores_size = seq_len * seq_len * num_heads * 4; // Attention matrix
        let output_size = seq_len * head_dim * num_heads * 4;

        qkv_size + scores_size + output_size
    }
}

impl Default for StandardAttention {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_attention_basic() {
        let attn = StandardAttention::new();

        // Simple test: 1 batch, 1 head, 2 positions, 4 dimensions
        let batch_size = 1;
        let num_heads = 1;
        let seq_len = 2;
        let head_dim = 4;

        // Create simple Q, K, V
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2x4
        let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2x4
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2x4

        let output = attn
            .forward(&q, &k, &v, None, batch_size, num_heads, seq_len, seq_len, head_dim)
            .unwrap();

        assert_eq!(output.len(), batch_size * num_heads * seq_len * head_dim);

        // Output should be a weighted combination of V
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_standard_attention_with_mask() {
        let attn = StandardAttention::new();

        let batch_size = 1;
        let num_heads = 1;
        let seq_len = 3;
        let head_dim = 2;

        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2

        // Causal mask: only attend to previous positions
        let mask = vec![
            1.0, 0.0, 0.0, // Position 0 can only see position 0
            1.0, 1.0, 0.0, // Position 1 can see 0,1
            1.0, 1.0, 1.0, // Position 2 can see 0,1,2
        ];

        let output = attn
            .forward(&q, &k, &v, Some(&mask), batch_size, num_heads, seq_len, seq_len, head_dim)
            .unwrap();

        assert_eq!(output.len(), batch_size * num_heads * seq_len * head_dim);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_memory_estimation() {
        let attn = StandardAttention::new();

        let seq_len = 1024;
        let head_dim = 64;
        let num_heads = 8;

        let mem = attn.estimated_memory(seq_len, head_dim, num_heads);

        // Should estimate O(N²) memory for attention matrix
        assert!(mem > 1_000_000); // At least 1 MB for this config
    }
}
