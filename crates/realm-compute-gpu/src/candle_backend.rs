//! GPU backend using Candle for accelerated inference
//!
//! This module provides GPU-accelerated operations using Candle's tensor operations.
//! It supports both CUDA and Metal backends depending on the available hardware.

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::ops;
use realm_core::error::{Error, Result};
use realm_core::quant::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_K};

use crate::GpuBackendTrait;

/// GPU backend for accelerated tensor operations
pub struct CandleGpuBackend {
    device: Device,
}

impl CandleGpuBackend {
    /// Create a new GPU backend
    ///
    /// This will automatically detect the best available GPU backend:
    /// 1. CUDA (if available and cuda feature is enabled)
    /// 2. Metal (if available and metal feature is enabled)
    /// 3. CPU (fallback)
    pub fn new() -> CandleResult<Self> {
        let device = Self::select_device()?;
        Ok(Self { device })
    }

    /// Select the best available device
    fn select_device() -> CandleResult<Device> {
        // Try CUDA first if available
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                println!("🚀 Using CUDA GPU acceleration");
                return Ok(device);
            }
        }

        // Try Metal if available
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                println!("🚀 Using Metal GPU acceleration");
                return Ok(device);
            }
        }

        // Fallback to CPU
        println!("⚠️  No GPU available, using CPU");
        Ok(Device::Cpu)
    }

    /// Get the device reference
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Matrix multiplication: C = A @ B
    ///
    /// # Arguments
    /// * `a` - Left matrix [m, k]
    /// * `b` - Right matrix [k, n]
    ///
    /// # Returns
    /// Result matrix [m, n]
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> CandleResult<Tensor> {
        a.matmul(b)
    }

    /// Matrix multiplication with transposed B: C = A @ B^T
    ///
    /// # Arguments
    /// * `a` - Left matrix [m, k]
    /// * `b` - Right matrix [n, k] (will be transposed to [k, n])
    ///
    /// # Returns
    /// Result matrix [m, n]
    pub fn matmul_transposed(&self, a: &Tensor, b: &Tensor) -> CandleResult<Tensor> {
        let b_t = b.t()?;
        a.matmul(&b_t)
    }

    /// RMS normalization
    ///
    /// # Arguments
    /// * `x` - Input tensor [..., hidden_size]
    /// * `weight` - Normalization weights [hidden_size]
    /// * `eps` - Small epsilon for numerical stability
    ///
    /// # Returns
    /// Normalized tensor with same shape as input
    pub fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> CandleResult<Tensor> {
        ops::rms_norm(x, weight, eps)
    }

    /// Scaled dot-product attention
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, heads, seq_len, head_dim]
    /// * `k` - Key tensor [batch, heads, seq_len, head_dim]
    /// * `v` - Value tensor [batch, heads, seq_len, head_dim]
    /// * `scale` - Scaling factor (usually 1/sqrt(head_dim))
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Attention output [batch, heads, seq_len, head_dim]
    pub fn scaled_dot_product_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // Compute attention scores: Q @ K^T
        let scores = q.matmul(&k.t()?)?;

        // Scale the scores
        let scale_tensor = Tensor::new(&[scale], &self.device)?;
        let scores = scores.broadcast_mul(&scale_tensor)?;

        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            scores.broadcast_add(mask)?
        } else {
            scores
        };

        // Apply softmax
        let attn_weights = ops::softmax_last_dim(&scores)?;

        // Apply attention to values: attn_weights @ V
        attn_weights.matmul(v)
    }

    /// SiLU (Swish) activation function
    ///
    /// SiLU(x) = x * sigmoid(x)
    pub fn silu(&self, x: &Tensor) -> CandleResult<Tensor> {
        ops::silu(x)
    }

    /// Softmax activation function
    ///
    /// Applies softmax along the last dimension
    pub fn softmax(&self, x: &Tensor) -> CandleResult<Tensor> {
        ops::softmax_last_dim(x)
    }

    /// Element-wise addition
    pub fn add(&self, a: &Tensor, b: &Tensor) -> CandleResult<Tensor> {
        a.broadcast_add(b)
    }

    /// Element-wise multiplication
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> CandleResult<Tensor> {
        a.broadcast_mul(b)
    }

    /// Rotary Position Embedding (RoPE)
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, heads, seq_len, head_dim]
    /// * `cos` - Cosine values for RoPE
    /// * `sin` - Sine values for RoPE
    ///
    /// # Returns
    /// Rotated tensor with same shape as input
    pub fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        // RoPE implementation using Candle's operations
        // This is a simplified version - full RoPE would need more complex indexing

        // Split x into even and odd parts
        let x_even = x.narrow(3, 0, x.dim(3)? / 2)?;
        let x_odd = x.narrow(3, x.dim(3)? / 2, x.dim(3)? / 2)?;

        // Apply rotation
        let rotated_even = x_even
            .broadcast_mul(cos)?
            .broadcast_sub(&x_odd.broadcast_mul(sin)?)?;
        let rotated_odd = x_even
            .broadcast_mul(sin)?
            .broadcast_add(&x_odd.broadcast_mul(cos)?)?;

        // Concatenate back
        Tensor::cat(&[&rotated_even, &rotated_odd], 3)
    }

    /// Convert f32 slice to Candle tensor
    pub fn f32_to_tensor(&self, data: &[f32], shape: &[usize]) -> CandleResult<Tensor> {
        Tensor::from_slice(data, shape, &self.device)
    }

    /// Convert Candle tensor to f32 slice
    pub fn tensor_to_f32(&self, tensor: &Tensor) -> CandleResult<Vec<f32>> {
        // Flatten the tensor to 1D if needed
        let flat_tensor = if tensor.dims().len() > 1 {
            tensor.flatten_all()?
        } else {
            tensor.clone()
        };
        flat_tensor.to_vec1::<f32>()
    }
}

impl Default for CandleGpuBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create GPU backend")
    }
}

impl GpuBackendTrait for CandleGpuBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: u32, k: u32, n: u32) -> Result<Vec<f32>> {
        // Convert slices to tensors
        let a_tensor = self
            .f32_to_tensor(a, &[m as usize, k as usize])
            .map_err(|e| Error::Runtime(format!("Candle GPU tensor creation failed: {}", e)))?;
        let b_tensor = self
            .f32_to_tensor(b, &[k as usize, n as usize])
            .map_err(|e| Error::Runtime(format!("Candle GPU tensor creation failed: {}", e)))?;

        // Perform matrix multiplication
        let result_tensor = self
            .matmul(&a_tensor, &b_tensor)
            .map_err(|e| Error::Runtime(format!("Candle GPU matmul failed: {}", e)))?;

        // Convert back to f32 slice
        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Candle tensor conversion failed: {}", e)))
    }
    fn fused_dequant_matmul_q4k(
        &self,
        _blocks: &[BlockQ4_K],
        _input: &[f32],
        _batch_size: usize,
        _n: usize,
        _k: usize,
    ) -> Result<Vec<f32>> {
        // TODO: Implement GPU fused Q4_K kernel using Candle
        Err(Error::Runtime(
            "Candle GPU Q4_K fused kernel not implemented yet".to_string(),
        ))
    }

    fn fused_dequant_matmul_q5k(
        &self,
        _blocks: &[BlockQ5_K],
        _input: &[f32],
        _batch_size: usize,
        _n: usize,
        _k: usize,
    ) -> Result<Vec<f32>> {
        // TODO: Implement GPU fused Q5_K kernel using Candle
        Err(Error::Runtime(
            "Candle GPU Q5_K fused kernel not implemented yet".to_string(),
        ))
    }

    fn fused_dequant_matmul_q6k(
        &self,
        _blocks: &[BlockQ6_K],
        _input: &[f32],
        _batch_size: usize,
        _n: usize,
        _k: usize,
    ) -> Result<Vec<f32>> {
        // TODO: Implement GPU fused Q6_K kernel using Candle
        Err(Error::Runtime(
            "Candle GPU Q6_K fused kernel not implemented yet".to_string(),
        ))
    }

    fn fused_dequant_matmul_q8k(
        &self,
        _blocks: &[BlockQ8_K],
        _input: &[f32],
        _batch_size: usize,
        _n: usize,
        _k: usize,
    ) -> Result<Vec<f32>> {
        // TODO: Implement GPU fused Q8_K kernel using Candle
        Err(Error::Runtime(
            "Candle GPU Q8_K fused kernel not implemented yet".to_string(),
        ))
    }

    fn name(&self) -> &'static str {
        match self.device {
            Device::Cuda(_) => "CUDA",
            Device::Metal(_) => "Metal",
            Device::Cpu => "CPU",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_creation() {
        let backend = CandleGpuBackend::new();
        assert!(backend.is_ok());
    }

    #[test]
    fn test_matmul() {
        let backend = CandleGpuBackend::new().unwrap();

        // Create test matrices
        let a = backend
            .f32_to_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap();
        let b = backend
            .f32_to_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2])
            .unwrap();

        // Perform matrix multiplication
        let result = backend.matmul(&a, &b).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        // Expected result: [[19, 22], [43, 50]]
        assert_eq!(result_vec, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_rms_norm() {
        let backend = CandleGpuBackend::new().unwrap();

        // Create test input
        let x = backend.f32_to_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let weight = backend.f32_to_tensor(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();

        // Perform RMS normalization
        let result = backend.rms_norm(&x, &weight, 1e-5).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        // Check that result has same length as input
        assert_eq!(result_vec.len(), 4);

        // Check that values are reasonable (not NaN or inf)
        for val in &result_vec {
            assert!(
                val.is_finite(),
                "RMS norm produced non-finite value: {}",
                val
            );
        }

        // RMS norm should reduce the magnitude
        let input_rms: f32 = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let output_rms: f32 = result_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            output_rms < input_rms * 2.0,
            "Output RMS {} not reasonable compared to input RMS {}",
            output_rms,
            input_rms
        );
    }
}
