use crate::attention_stub::{create_attention, Attention};

use realm_compute_cpu::{matmul_f32, matmul_transposed};
use realm_core::error::Result;

#[cfg(feature = "webgpu")]
use realm_gpu::GpuBackend;

use super::{KVCache, TransformerConfig};
use crate::matmul_dispatch::dispatch_matmul;
use crate::weight_format::WeightFormat;

#[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
use realm_gpu::GpuBackendTrait;

/// Multi-head attention layer
#[allow(dead_code)]
pub struct MultiHeadAttention {
    pub config: TransformerConfig,
    head_dim: usize,
    attention_impl: Box<dyn Attention>,
}

#[allow(dead_code)]
impl MultiHeadAttention {
    pub fn new(config: TransformerConfig) -> Self {
        let head_dim = config.hidden_size / config.num_heads;
        let attention_impl = create_attention(config.attention_backend);

        Self {
            config,
            head_dim,
            attention_impl,
        }
    }

    /// Helper: matrix multiplication with GPU/CPU fallback
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        transposed_b: bool,
        #[cfg(feature = "webgpu")] gpu: Option<&GpuBackend>,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "webgpu")]
        if let Some(gpu) = gpu {
            if !transposed_b {
                if let Ok(result) = gpu.matmul(a, b, m as u32, k as u32, n as u32) {
                    return Ok(result);
                }
            }
        }

        // CPU fallback
        let mut result = vec![0.0; m * n];
        if transposed_b {
            matmul_transposed(a, b, &mut result, m, k, n)?;
        } else {
            matmul_f32(a, b, &mut result, m, k, n)?;
        }
        Ok(result)
    }

    /// Apply attention with KV caching
    ///
    /// # Arguments
    /// * `hidden_states` - Input [batch, seq_len, hidden_size]
    /// * `weights` - Model weights (wq, wk, wv, wo)
    /// * `kv_cache` - KV cache for this layer
    /// * `position` - Current position in sequence
    /// * `gpu` - Optional GPU backend
    ///
    /// # Returns
    /// Output [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &[f32],
        weights: &AttentionWeights,
        kv_cache: &mut KVCache,
        position: usize,
        cpu_backend: Option<&dyn realm_compute_cpu::CpuBackendTrait>,
        #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))] gpu: Option<
            &dyn GpuBackendTrait,
        >,
    ) -> Result<Vec<f32>> {
        let seq_len = hidden_states.len() / self.config.hidden_size;
        let hidden_size = self.config.hidden_size;

        // Project to Q, K, V
        // Q projection: [seq_len, hidden_size] x [hidden_size, hidden_size]
        let q = dispatch_matmul(
            hidden_states,
            &weights.wq,
            seq_len,
            hidden_size,
            hidden_size,
            cpu_backend,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            gpu,
        )?;

        // K projection: [seq_len, hidden_size] x [hidden_size, num_kv_heads * head_dim]
        let k = dispatch_matmul(
            hidden_states,
            &weights.wk,
            seq_len,
            hidden_size,
            self.config.num_kv_heads * self.head_dim,
            cpu_backend,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            gpu,
        )?;

        // V projection
        let v = dispatch_matmul(
            hidden_states,
            &weights.wv,
            seq_len,
            hidden_size,
            self.config.num_kv_heads * self.head_dim,
            cpu_backend,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            gpu,
        )?;

        // Apply RoPE (Rotary Position Embedding)
        // Apply different positions for each token in the sequence
        let mut q = q;
        let mut k = k;
        self.apply_rope(&mut q, position, seq_len, self.config.num_heads)?;
        self.apply_rope(&mut k, position, seq_len, self.config.num_kv_heads)?;

        // MINIMAL KV CACHE: Use a very simple approach that's guaranteed to work
        let output = if std::env::var("DISABLE_KV").is_ok() {
            // Bypass cache: attend only to current K/V (useful for isolating KV issues)
            self.compute_attention(&q, &k, &v, seq_len, position)?
        } else {
            // Use KV cache properly - this is the key fix!
            // Append new K/V to cache and get all cached K/V for attention
            let (cached_k, cached_v) = kv_cache.append(&k, &v)?;
            // Use cached K/V for attention computation
            self.compute_attention(&q, &cached_k, &cached_v, seq_len, position)?
        };

        // Output projection
        let result = dispatch_matmul(
            &output,
            &weights.wo,
            seq_len,
            hidden_size,
            hidden_size,
            cpu_backend,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            gpu,
        )?;

        Ok(result)
    }

    /// Apply RoPE (Rotary Position Embedding) correctly for multiple tokens
    ///
    /// # Arguments
    /// * `tensor` - Q or K tensor with shape [seq_len, num_heads, head_dim]
    /// * `start_pos` - Starting position in the sequence
    /// * `seq_len` - Number of tokens in this batch
    /// * `num_heads` - Number of heads (num_heads for Q, num_kv_heads for K/V)
    pub fn apply_rope(
        &self,
        tensor: &mut [f32],
        start_pos: usize,
        seq_len: usize,
        num_heads: usize,
    ) -> Result<()> {
        let head_dim = self.head_dim;
        let debug_rope = std::env::var("DEBUG_ROPE").is_ok();

        // For each token in the sequence
        for seq_idx in 0..seq_len {
            let token_pos = start_pos + seq_idx;

            if debug_rope && seq_idx == 0 {
                eprintln!(
                    "  RoPE: token_pos={}, head_dim={}, theta={}",
                    token_pos, head_dim, self.config.rope_theta
                );
            }

            // For each head
            for head in 0..num_heads {
                let base_idx = (seq_idx * num_heads + head) * head_dim;

                // Apply RoPE using INTERLEAVED pairing: (0,1), (2,3), (4,5), ...
                // This matches llama2.c implementation
                for i in (0..head_dim).step_by(2) {
                    let idx0 = base_idx + i;
                    let idx1 = base_idx + i + 1;

                    // RoPE frequency calculation
                    // Since we step_by(2), i goes 0,2,4,6,... so we use i (not 2*i)
                    // freq = 1.0 / (theta ^ (i / head_dim))
                    let freq = 1.0 / self.config.rope_theta.powf(i as f32 / head_dim as f32);
                    let angle = token_pos as f32 * freq;

                    let cos = angle.cos();
                    let sin = angle.sin();

                    // Rotate the pair
                    let x0 = tensor[idx0];
                    let x1 = tensor[idx1];

                    let new_x0 = x0 * cos - x1 * sin;
                    let new_x1 = x0 * sin + x1 * cos;

                    tensor[idx0] = new_x0;
                    tensor[idx1] = new_x1;

                    if debug_rope && head == 0 && i < 4 {
                        eprintln!(
                            "    head={}, i={}, freq={:.6}, angle={:.6}, cos={:.6}, sin={:.6}",
                            head, i, freq, angle, cos, sin
                        );
                        eprintln!(
                            "      ({:.6}, {:.6}) -> ({:.6}, {:.6})",
                            x0, x1, new_x0, new_x1
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimized dot product with manual loop unrolling
    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = 0.0;

        // Process 4 elements at a time (loop unrolling)
        let chunks = len / 4;

        for i in 0..chunks {
            let idx = i * 4;
            sum += a[idx] * b[idx];
            sum += a[idx + 1] * b[idx + 1];
            sum += a[idx + 2] * b[idx + 2];
            sum += a[idx + 3] * b[idx + 3];
        }

        // Handle remaining elements
        for i in (chunks * 4)..len {
            sum += a[i] * b[i];
        }

        sum
    }

    /// Repeat K/V tensors for Grouped Query Attention (GQA)
    ///
    /// Expands K/V from num_kv_heads to num_heads by repeating each KV head n_rep times.
    /// Input shape: [seq_len, num_kv_heads, head_dim]
    /// Output shape: [seq_len, num_heads, head_dim]
    fn repeat_kv(&self, kv: &[f32], seq_len: usize, n_rep: usize) -> Vec<f32> {
        if n_rep == 1 {
            return kv.to_vec();
        }

        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.head_dim;
        let mut output = vec![0.0; seq_len * num_kv_heads * n_rep * head_dim];

        for seq_idx in 0..seq_len {
            for kv_h in 0..num_kv_heads {
                // Read the KV head once
                let kv_offset = (seq_idx * num_kv_heads + kv_h) * head_dim;
                let kv_head = &kv[kv_offset..kv_offset + head_dim];

                // Repeat it n_rep times
                for rep in 0..n_rep {
                    let out_h = kv_h * n_rep + rep;
                    let out_offset = (seq_idx * num_kv_heads * n_rep + out_h) * head_dim;
                    output[out_offset..out_offset + head_dim].copy_from_slice(kv_head);
                }
            }
        }

        output
    }

    /// Compute scaled dot-product attention using the configured attention backend
    ///
    /// Supports multiple attention implementations:
    /// - Standard: Traditional O(N²) attention
    /// - Flash: IO-aware Flash Attention (3-4x faster, O(N) memory)
    ///
    /// Process:
    /// 1. Repeat K/V for GQA (if num_heads != num_kv_heads)
    /// 2. Build causal mask
    /// 3. Call attention implementation (Standard or Flash)
    /// 4. Return output
    pub fn compute_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        position: usize,
    ) -> Result<Vec<f32>> {
        let head_dim = self.head_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let n_rep = num_heads / num_kv_heads;

        // Q shape: [seq_len, num_heads, head_dim]
        // K, V shape: [kv_seq_len, num_kv_heads, head_dim]
        let kv_seq_len = k.len() / (num_kv_heads * head_dim);

        // Repeat K/V to match num_heads (for GQA/MQA)
        let k_repeated = if n_rep > 1 {
            self.repeat_kv(k, kv_seq_len, n_rep)
        } else {
            k.to_vec()
        };
        let v_repeated = if n_rep > 1 {
            self.repeat_kv(v, kv_seq_len, n_rep)
        } else {
            v.to_vec()
        };

        // Build causal mask: mask[i, j] = 0.0 if j > position + i (masked), 1.0 if allowed
        // Attention implementations expect 0.0 = masked, non-zero = allowed
        let mut mask = vec![1.0; seq_len * kv_seq_len];
        for i in 0..seq_len {
            let query_abs_pos = position + i;
            for j in 0..kv_seq_len {
                if j > query_abs_pos {
                    mask[i * kv_seq_len + j] = 0.0; // Mask future tokens
                }
            }
        }

        // Call the attention implementation (Flash or Standard)
        // Input layout: [seq_len, num_heads, head_dim]
        // Attention trait expects: [batch=1, num_heads, seq_len, head_dim]
        // We need to reshape: transpose from [seq_len, num_heads, head_dim] to [num_heads, seq_len, head_dim]

        let q_transposed = self.transpose_for_attention(q, seq_len, num_heads, head_dim);
        let k_transposed =
            self.transpose_for_attention(&k_repeated, kv_seq_len, num_heads, head_dim);
        let v_transposed =
            self.transpose_for_attention(&v_repeated, kv_seq_len, num_heads, head_dim);

        // Call attention (batch_size=1 for single sequence)
        let output_transposed = self.attention_impl.forward(
            &q_transposed,
            &k_transposed,
            &v_transposed,
            Some(&mask),
            1, // batch_size
            num_heads,
            seq_len,
            kv_seq_len,
            head_dim,
        )?;

        // Transpose back: [num_heads, seq_len, head_dim] -> [seq_len, num_heads, head_dim]
        let output =
            self.transpose_from_attention(&output_transposed, seq_len, num_heads, head_dim);

        Ok(output)
    }

    /// Transpose from [seq_len, num_heads, head_dim] to [num_heads, seq_len, head_dim]
    fn transpose_for_attention(
        &self,
        tensor: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; seq_len * num_heads * head_dim];
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src_idx = (s * num_heads + h) * head_dim + d;
                    let dst_idx = (h * seq_len + s) * head_dim + d;
                    output[dst_idx] = tensor[src_idx];
                }
            }
        }
        output
    }

    /// Transpose from [num_heads, seq_len, head_dim] to [seq_len, num_heads, head_dim]
    fn transpose_from_attention(
        &self,
        tensor: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; seq_len * num_heads * head_dim];
        for h in 0..num_heads {
            for s in 0..seq_len {
                for d in 0..head_dim {
                    let src_idx = (h * seq_len + s) * head_dim + d;
                    let dst_idx = (s * num_heads + h) * head_dim + d;
                    output[dst_idx] = tensor[src_idx];
                }
            }
        }
        output
    }
}

/// Attention weight matrices
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AttentionWeights {
    /// Query projection [hidden_size, hidden_size]
    pub wq: WeightFormat,
    /// Key projection [hidden_size, num_kv_heads * head_dim]
    pub wk: WeightFormat,
    /// Value projection [hidden_size, num_kv_heads * head_dim]
    pub wv: WeightFormat,
    /// Output projection [hidden_size, hidden_size]
    pub wo: WeightFormat,
}

#[allow(dead_code)]
impl AttentionWeights {
    pub fn new(config: &TransformerConfig) -> Self {
        let hidden_size = config.hidden_size;
        let kv_size = config.num_kv_heads * (hidden_size / config.num_heads);

        Self {
            wq: WeightFormat::F32(vec![0.0; hidden_size * hidden_size]),
            wk: WeightFormat::F32(vec![0.0; hidden_size * kv_size]),
            wv: WeightFormat::F32(vec![0.0; hidden_size * kv_size]),
            wo: WeightFormat::F32(vec![0.0; hidden_size * hidden_size]),
        }
    }
}
