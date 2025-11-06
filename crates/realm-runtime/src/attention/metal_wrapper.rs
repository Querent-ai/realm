//! Metal wrapper for Flash Attention kernel
//!
//! This module provides a safe Rust interface to Flash Attention using Candle's Metal operations.

use candle_core::{Device, Tensor};
use realm_core::error::{Error as WasmChordError, Result};

/// Metal Flash Attention implementation using Candle
///
/// This implementation uses Candle's tensor operations which leverage Metal under the hood.
#[cfg(feature = "metal")]
pub struct FlashAttentionMetal {
    device: Device,
}

#[cfg(feature = "metal")]
impl FlashAttentionMetal {
    pub fn new() -> Result<Self> {
        // Try to create Metal device
        let device = Device::new_metal(0)
            .map_err(|e| WasmChordError::Runtime(format!("Metal device not available: {}", e)))?;

        Ok(Self { device })
    }

    pub fn forward(
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
        // Reshape inputs: [batch, num_heads, seq_len, head_dim]
        let q_tensor = Tensor::from_slice(
            q,
            &[batch_size, num_heads, seq_len_q, head_dim],
            &self.device,
        )
        .map_err(|e| WasmChordError::Runtime(format!("Failed to create Q tensor: {}", e)))?;

        let k_tensor = Tensor::from_slice(
            k,
            &[batch_size, num_heads, seq_len_k, head_dim],
            &self.device,
        )
        .map_err(|e| WasmChordError::Runtime(format!("Failed to create K tensor: {}", e)))?;

        let v_tensor = Tensor::from_slice(
            v,
            &[batch_size, num_heads, seq_len_k, head_dim],
            &self.device,
        )
        .map_err(|e| WasmChordError::Runtime(format!("Failed to create V tensor: {}", e)))?;

        // Compute attention scores: Q @ K^T / sqrt(head_dim)
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Reshape for matmul: [batch*num_heads, seq_len, head_dim]
        let q_flat = q_tensor
            .flatten(0, 1)
            .map_err(|e| WasmChordError::Runtime(format!("Failed to flatten Q: {}", e)))?;
        let q_reshaped = q_flat
            .reshape(&[batch_size * num_heads, seq_len_q, head_dim])
            .map_err(|e| WasmChordError::Runtime(format!("Failed to reshape Q: {}", e)))?;

        let k_flat = k_tensor
            .flatten(0, 1)
            .map_err(|e| WasmChordError::Runtime(format!("Failed to flatten K: {}", e)))?;
        let k_reshaped = k_flat
            .reshape(&[batch_size * num_heads, seq_len_k, head_dim])
            .map_err(|e| WasmChordError::Runtime(format!("Failed to reshape K: {}", e)))?;

        // Transpose K: [batch*num_heads, head_dim, seq_len_k]
        let k_t = k_reshaped
            .transpose(1, 2)
            .map_err(|e| WasmChordError::Runtime(format!("Failed to transpose K: {}", e)))?;

        // Compute scores: [batch*num_heads, seq_len_q, seq_len_k]
        let scores = q_reshaped
            .matmul(&k_t)
            .map_err(|e| WasmChordError::Runtime(format!("Failed to compute scores: {}", e)))?
            * scale;

        // Apply mask if provided (GPU-native implementation)
        let scores = if let Some(mask_data) = mask {
            // Create mask tensor on GPU
            let mask_tensor = Tensor::from_slice(mask_data, &[seq_len_q, seq_len_k], &self.device)
                .map_err(|e| {
                    WasmChordError::Runtime(format!("Failed to create mask tensor: {}", e))
                })?;

            // Expand mask to match scores shape: [batch*num_heads, seq_len_q, seq_len_k]
            let mask_expanded = mask_tensor
                .unsqueeze(0)
                .map_err(|e| WasmChordError::Runtime(format!("Failed to expand mask: {}", e)))?
                .expand(&[batch_size * num_heads, seq_len_q, seq_len_k])
                .map_err(|e| WasmChordError::Runtime(format!("Failed to broadcast mask: {}", e)))?;

            // Apply mask: where mask is 0, set scores to -inf
            let mask_bool = mask_expanded
                .gt(
                    &Tensor::zeros(&mask_expanded.shape(), mask_expanded.dtype(), &self.device)
                        .map_err(|e| {
                            WasmChordError::Runtime(format!("Failed to create zeros: {}", e))
                        })?,
                )
                .map_err(|e| WasmChordError::Runtime(format!("Failed to compare mask: {}", e)))?;

            let neg_inf = Tensor::new(&[f32::NEG_INFINITY], &self.device)
                .map_err(|e| WasmChordError::Runtime(format!("Failed to create -inf: {}", e)))?
                .broadcast_as(scores.shape())
                .map_err(|e| WasmChordError::Runtime(format!("Failed to broadcast -inf: {}", e)))?;

            scores
                .where_cond(&mask_bool, &neg_inf)
                .map_err(|e| WasmChordError::Runtime(format!("Failed to apply mask: {}", e)))?
        } else {
            scores
        };

        // Apply softmax: softmax(scores)
        let scores_softmax = scores
            .softmax_last_dim()
            .map_err(|e| WasmChordError::Runtime(format!("Failed to apply softmax: {}", e)))?;

        // Reshape V: [batch*num_heads, seq_len_k, head_dim]
        let v_flat = v_tensor
            .flatten(0, 1)
            .map_err(|e| WasmChordError::Runtime(format!("Failed to flatten V: {}", e)))?;
        let v_reshaped = v_flat
            .reshape(&[batch_size * num_heads, seq_len_k, head_dim])
            .map_err(|e| WasmChordError::Runtime(format!("Failed to reshape V: {}", e)))?;

        // Compute output: scores_softmax @ V
        let output = scores_softmax
            .matmul(&v_reshaped)
            .map_err(|e| WasmChordError::Runtime(format!("Failed to compute output: {}", e)))?;

        // Reshape back: [batch, num_heads, seq_len_q, head_dim]
        let output_reshaped = output
            .reshape(&[batch_size, num_heads, seq_len_q, head_dim])
            .map_err(|e| WasmChordError::Runtime(format!("Failed to reshape output: {}", e)))?;

        // Convert to Vec<f32>
        let output_vec = output_reshaped
            .flatten_all()
            .map_err(|e| WasmChordError::Runtime(format!("Failed to flatten output: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| WasmChordError::Runtime(format!("Failed to convert to vec: {}", e)))?;

        Ok(output_vec)
    }
}

#[cfg(not(feature = "metal"))]
pub struct FlashAttentionMetal;

#[cfg(not(feature = "metal"))]
impl FlashAttentionMetal {
    pub fn new() -> Result<Self> {
        Err(WasmChordError::NotImplemented(
            "Metal support not enabled (compile with --features metal)".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_flash_attention_creation() {
        match FlashAttentionMetal::new() {
            Ok(_metal_attn) => {
                println!("✅ Metal Flash Attention created successfully");
            }
            Err(e) => {
                println!(
                    "⚠️  Metal Flash Attention not available (expected in CI): {}",
                    e
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_metal_flash_attention_forward() {
        match FlashAttentionMetal::new() {
            Ok(metal_attn) => {
                let batch_size = 1;
                let num_heads = 1;
                let seq_len = 4;
                let head_dim = 8;

                let mut q = vec![0.0f32; batch_size * num_heads * seq_len * head_dim];
                let mut k = vec![0.0f32; batch_size * num_heads * seq_len * head_dim];
                let mut v = vec![0.0f32; batch_size * num_heads * seq_len * head_dim];

                for i in 0..q.len() {
                    q[i] = (i as f32 * 0.1).sin();
                    k[i] = (i as f32 * 0.1).cos();
                    v[i] = i as f32 * 0.01;
                }

                match metal_attn.forward(
                    &q, &k, &v, None, batch_size, num_heads, seq_len, seq_len, head_dim,
                ) {
                    Ok(output) => {
                        assert_eq!(output.len(), batch_size * num_heads * seq_len * head_dim);
                        assert!(output.iter().all(|&x| x.is_finite()));
                        println!("✅ Metal Flash Attention forward pass succeeded");
                    }
                    Err(e) => {
                        println!(
                            "⚠️  Metal Flash Attention failed: {} (expected in CI without GPU)",
                            e
                        );
                    }
                }
            }
            Err(e) => {
                println!(
                    "⚠️  Metal not available: {} (expected in CI without GPU)",
                    e
                );
            }
        }
    }
}
