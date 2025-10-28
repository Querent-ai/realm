use realm_core::error::Result;

/// Simple attention trait stub
pub trait Attention: Send + Sync {
    #[allow(dead_code)]
    fn name(&self) -> &str;

    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<Vec<f32>>;
}

pub fn create_attention(_backend: AttentionBackend) -> Box<dyn Attention> {
    Box::new(StandardAttention)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionBackend {
    Standard,
    Flash,
    Auto,
}

struct StandardAttention;

impl Attention for StandardAttention {
    fn name(&self) -> &str {
        "StandardAttention"
    }

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
        // Softmax scale: 1/sqrt(head_dim)
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 1. Compute Q @ K^T / sqrt(d)
        let mut scores = vec![0.0; batch_size * num_heads * seq_len_q * seq_len_k];

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

        // 2. Apply softmax (with optional mask)
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len_q {
                    let row_start = ((b * num_heads + h) * seq_len_q + i) * seq_len_k;
                    let row_end = row_start + seq_len_k;

                    // Apply mask if provided
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
                    let max_score = scores[row_start..row_end]
                        .iter()
                        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

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

        // 3. Compute attention_weights @ V
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
                            sum += scores[attn_idx] * v[v_idx];
                        }

                        let out_idx = ((b * num_heads + h) * seq_len_q + i) * head_dim + d;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        Ok(output)
    }
}
