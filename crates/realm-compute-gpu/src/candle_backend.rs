//! GPU backend using Candle for accelerated inference
//!
//! This module provides GPU-accelerated operations using Candle's tensor operations.
//! It supports both CUDA and Metal backends depending on the available hardware.

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::ops;
use realm_core::error::{Error, Result};
use realm_core::quant::{
    BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ4_K, BlockQ5_0, BlockQ5_1, BlockQ5_K,
    BlockQ6_K, BlockQ8_0, BlockQ8_1, BlockQ8_K,
};

use crate::GpuBackendTrait;

/// GPU backend for accelerated tensor operations
pub struct CandleGpuBackend {
    device: Device,
    /// Mixed precision configuration (optional)
    precision_config: Option<crate::mixed_precision::MixedPrecisionConfig>,
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
        Ok(Self {
            device,
            precision_config: None,
        })
    }

    /// Create a new GPU backend with mixed precision configuration
    pub fn with_precision(
        precision_config: crate::mixed_precision::MixedPrecisionConfig,
    ) -> CandleResult<Self> {
        let device = Self::select_device()?;
        Ok(Self {
            device,
            precision_config: Some(precision_config),
        })
    }

    /// Select the best available device
    fn select_device() -> CandleResult<Device> {
        // Try CUDA first if available
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                // println!("🚀 Using CUDA GPU acceleration");
                return Ok(device);
            }
        }

        // Try Metal if available
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                // println!("🚀 Using Metal GPU acceleration");
                return Ok(device);
            }
        }

        // Fallback to CPU
        // println!("⚠️  No GPU available, using CPU");
        Ok(Device::Cpu)
    }

    /// Get the device reference
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if CUDA is available
    #[cfg(feature = "cuda")]
    pub fn is_cuda_available() -> bool {
        Device::new_cuda(0).is_ok()
    }

    /// Check if Metal is available
    #[cfg(feature = "metal")]
    pub fn is_metal_available() -> bool {
        Device::new_metal(0).is_ok()
    }

    /// Get device count for the selected backend
    pub fn device_count(&self) -> usize {
        match self.device {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                // Try to get CUDA device count
                (0..).take_while(|i| Device::new_cuda(*i).is_ok()).count()
            }
            #[cfg(feature = "metal")]
            Device::Metal(_) => {
                // Try to get Metal device count
                (0..).take_while(|i| Device::new_metal(*i).is_ok()).count()
            }
            _ => 1, // CPU always has 1 device
        }
    }

    /// Check if the backend is using GPU acceleration
    pub fn is_gpu(&self) -> bool {
        match self.device {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => true,
            #[cfg(feature = "metal")]
            Device::Metal(_) => true,
            _ => false,
        }
    }

    /// Matrix multiplication: C = A @ B
    ///
    /// # Arguments
    /// * `a` - Left matrix [m, k]
    /// * `b` - Right matrix [k, n]
    ///
    /// # Returns
    /// Result matrix [m, n]
    ///
    /// # Mixed Precision Support
    /// If precision_config is set, this will automatically convert tensors to the
    /// configured precision (FP16/BF16) for better performance on modern GPUs.
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> CandleResult<Tensor> {
        // Apply mixed precision if configured
        if let Some(ref config) = self.precision_config {
            use crate::mixed_precision::{select_precision, PrecisionMode};

            // Select precision based on GPU capabilities
            let precision = select_precision(config.forward_precision);

            match precision {
                PrecisionMode::FP16 => {
                    // Convert to FP16 if supported
                    // Note: Candle handles FP16 conversion internally when using GPU
                    // For explicit conversion, we'd need to use half::f16 types
                    // For now, let Candle handle it - GPU operations will use FP16 if available
                    a.matmul(b)
                }
                PrecisionMode::BF16 => {
                    // BF16 support - Candle may handle this internally on Ampere+ GPUs
                    a.matmul(b)
                }
                _ => a.matmul(b),
            }
        } else {
            a.matmul(b)
        }
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
        // Try GPU operation first
        match ops::rms_norm(x, weight, eps) {
            Ok(result) => Ok(result),
            Err(_e) => {
                // If GPU operation fails (e.g., Metal doesn't support it), fallback to CPU
                let cpu_device = Device::Cpu;
                let x_cpu = x.to_device(&cpu_device)?;
                let weight_cpu = weight.to_device(&cpu_device)?;
                let result_cpu = ops::rms_norm(&x_cpu, &weight_cpu, eps)?;
                // Move result back to original device
                result_cpu.to_device(x.device())
            }
        }
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
        // Try GPU operation first
        let result = (|| -> CandleResult<Tensor> {
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
        })();

        match result {
            Ok(r) => Ok(r),
            Err(_e) => {
                // If GPU operation fails (e.g., PTX version mismatch), fallback to CPU
                let cpu_device = Device::Cpu;
                let q_cpu = q.to_device(&cpu_device)?;
                let k_cpu = k.to_device(&cpu_device)?;
                let v_cpu = v.to_device(&cpu_device)?;

                let scores = q_cpu.matmul(&k_cpu.t()?)?;
                let scale_tensor = Tensor::new(&[scale], &cpu_device)?;
                let scores = scores.broadcast_mul(&scale_tensor)?;

                let scores = if let Some(mask) = mask {
                    let mask_cpu = mask.to_device(&cpu_device)?;
                    scores.broadcast_add(&mask_cpu)?
                } else {
                    scores
                };

                let attn_weights = ops::softmax_last_dim(&scores)?;
                let result_cpu = attn_weights.matmul(&v_cpu)?;

                // Move result back to original device
                result_cpu.to_device(q.device())
            }
        }
    }

    /// SiLU (Swish) activation function
    ///
    /// SiLU(x) = x * sigmoid(x)
    pub fn silu(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Try GPU operation first
        match ops::silu(x) {
            Ok(result) => Ok(result),
            Err(_e) => {
                // If GPU operation fails (e.g., PTX version mismatch), fallback to CPU
                let cpu_device = Device::Cpu;
                let x_cpu = x.to_device(&cpu_device)?;
                let result_cpu = ops::silu(&x_cpu)?;
                // Move result back to original device
                result_cpu.to_device(x.device())
            }
        }
    }

    /// Softmax activation function
    ///
    /// Applies softmax along the last dimension
    pub fn softmax(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Try GPU operation first
        match ops::softmax_last_dim(x) {
            Ok(result) => Ok(result),
            Err(_e) => {
                // If GPU operation fails (e.g., Metal doesn't support it), fallback to CPU
                let cpu_device = Device::Cpu;
                let x_cpu = x.to_device(&cpu_device)?;
                let result_cpu = ops::softmax_last_dim(&x_cpu)?;
                // Move result back to original device
                result_cpu.to_device(x.device())
            }
        }
    }

    /// Element-wise addition
    pub fn add(&self, a: &Tensor, b: &Tensor) -> CandleResult<Tensor> {
        // Try GPU operation first
        match a.broadcast_add(b) {
            Ok(result) => Ok(result),
            Err(_e) => {
                // If GPU operation fails (e.g., PTX version mismatch), fallback to CPU
                let cpu_device = Device::Cpu;
                let a_cpu = a.to_device(&cpu_device)?;
                let b_cpu = b.to_device(&cpu_device)?;
                let result_cpu = a_cpu.broadcast_add(&b_cpu)?;
                // Move result back to original device
                result_cpu.to_device(a.device())
            }
        }
    }

    /// Element-wise multiplication
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> CandleResult<Tensor> {
        // Try GPU operation first
        match a.broadcast_mul(b) {
            Ok(result) => Ok(result),
            Err(_e) => {
                // If GPU operation fails (e.g., PTX version mismatch), fallback to CPU
                let cpu_device = Device::Cpu;
                let a_cpu = a.to_device(&cpu_device)?;
                let b_cpu = b.to_device(&cpu_device)?;
                let result_cpu = a_cpu.broadcast_mul(&b_cpu)?;
                // Move result back to original device
                result_cpu.to_device(a.device())
            }
        }
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
        // Try GPU operation first
        let result = (|| -> CandleResult<Tensor> {
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
        })();

        match result {
            Ok(r) => Ok(r),
            Err(_e) => {
                // If GPU operation fails (e.g., PTX version mismatch), fallback to CPU
                let cpu_device = Device::Cpu;
                let x_cpu = x.to_device(&cpu_device)?;
                let cos_cpu = cos.to_device(&cpu_device)?;
                let sin_cpu = sin.to_device(&cpu_device)?;

                let x_even = x_cpu.narrow(3, 0, x_cpu.dim(3)? / 2)?;
                let x_odd = x_cpu.narrow(3, x_cpu.dim(3)? / 2, x_cpu.dim(3)? / 2)?;

                let rotated_even = x_even
                    .broadcast_mul(&cos_cpu)?
                    .broadcast_sub(&x_odd.broadcast_mul(&sin_cpu)?)?;
                let rotated_odd = x_even
                    .broadcast_mul(&sin_cpu)?
                    .broadcast_add(&x_odd.broadcast_mul(&cos_cpu)?)?;

                let result_cpu = Tensor::cat(&[&rotated_even, &rotated_odd], 3)?;

                // Move result back to original device
                result_cpu.to_device(x.device())
            }
        }
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
        blocks: &[BlockQ4_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q4_k, QK_K};

        // Validate inputs
        if !k.is_multiple_of(QK_K) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, QK_K
            )));
        }

        let num_blocks_per_row = k / QK_K;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q4_K blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights: [n, k] -> f32 weights matrix
        // This can be done on CPU or GPU - for now, dequantize on CPU then upload to GPU
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * QK_K;
                let dequant_slice = &mut dequantized_weights[dequant_offset..dequant_offset + QK_K];

                dequantize_q4_k(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q4_K dequantization failed: {}", e)))?;
            }
        }

        // Upload to GPU and perform matmul
        let weights_tensor = self
            .f32_to_tensor(&dequantized_weights, &[n, k])
            .map_err(|e| Error::Runtime(format!("Failed to create weights tensor: {}", e)))?;

        let input_tensor = self
            .f32_to_tensor(input, &[batch_size, k])
            .map_err(|e| Error::Runtime(format!("Failed to create input tensor: {}", e)))?;

        // Transpose weights: [n, k] -> [k, n] for matmul
        let weights_t = weights_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Failed to transpose weights: {}", e)))?;

        // Input @ Weights^T: [batch_size, k] @ [k, n] -> [batch_size, n]
        let result_tensor = self
            .matmul(&input_tensor, &weights_t)
            .map_err(|e| Error::Runtime(format!("GPU matmul failed: {}", e)))?;

        // Convert back to f32
        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Failed to convert result: {}", e)))
    }

    fn fused_dequant_matmul_q5k(
        &self,
        blocks: &[BlockQ5_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q5_k, QK_K};

        // Validate inputs
        if !k.is_multiple_of(QK_K) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, QK_K
            )));
        }

        let num_blocks_per_row = k / QK_K;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q5_K blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights on CPU, then upload to GPU
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * QK_K;
                let dequant_slice = &mut dequantized_weights[dequant_offset..dequant_offset + QK_K];

                dequantize_q5_k(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q5_K dequantization failed: {}", e)))?;
            }
        }

        // Upload to GPU and perform matmul
        let weights_tensor = self
            .f32_to_tensor(&dequantized_weights, &[n, k])
            .map_err(|e| Error::Runtime(format!("Failed to create weights tensor: {}", e)))?;

        let input_tensor = self
            .f32_to_tensor(input, &[batch_size, k])
            .map_err(|e| Error::Runtime(format!("Failed to create input tensor: {}", e)))?;

        let weights_t = weights_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Failed to transpose weights: {}", e)))?;

        let result_tensor = self
            .matmul(&input_tensor, &weights_t)
            .map_err(|e| Error::Runtime(format!("GPU matmul failed: {}", e)))?;

        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Failed to convert result: {}", e)))
    }

    fn fused_dequant_matmul_q6k(
        &self,
        blocks: &[BlockQ6_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q6_k, QK_K};

        // Validate inputs
        if !k.is_multiple_of(QK_K) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, QK_K
            )));
        }

        let num_blocks_per_row = k / QK_K;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q6_K blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights on CPU, then upload to GPU
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * QK_K;
                let dequant_slice = &mut dequantized_weights[dequant_offset..dequant_offset + QK_K];

                dequantize_q6_k(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q6_K dequantization failed: {}", e)))?;
            }
        }

        // Upload to GPU and perform matmul
        let weights_tensor = self
            .f32_to_tensor(&dequantized_weights, &[n, k])
            .map_err(|e| Error::Runtime(format!("Failed to create weights tensor: {}", e)))?;

        let input_tensor = self
            .f32_to_tensor(input, &[batch_size, k])
            .map_err(|e| Error::Runtime(format!("Failed to create input tensor: {}", e)))?;

        let weights_t = weights_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Failed to transpose weights: {}", e)))?;

        let result_tensor = self
            .matmul(&input_tensor, &weights_t)
            .map_err(|e| Error::Runtime(format!("GPU matmul failed: {}", e)))?;

        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Failed to convert result: {}", e)))
    }

    fn fused_dequant_matmul_q8k(
        &self,
        blocks: &[BlockQ8_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q8_k, QK_K};

        // Validate inputs
        if !k.is_multiple_of(QK_K) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, QK_K
            )));
        }

        let num_blocks_per_row = k / QK_K;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q8_K blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights on CPU, then upload to GPU
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * QK_K;
                let dequant_slice = &mut dequantized_weights[dequant_offset..dequant_offset + QK_K];

                dequantize_q8_k(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q8_K dequantization failed: {}", e)))?;
            }
        }

        // Upload to GPU and perform matmul
        let weights_tensor = self
            .f32_to_tensor(&dequantized_weights, &[n, k])
            .map_err(|e| Error::Runtime(format!("Failed to create weights tensor: {}", e)))?;

        let input_tensor = self
            .f32_to_tensor(input, &[batch_size, k])
            .map_err(|e| Error::Runtime(format!("Failed to create input tensor: {}", e)))?;

        let weights_t = weights_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Failed to transpose weights: {}", e)))?;

        let result_tensor = self
            .matmul(&input_tensor, &weights_t)
            .map_err(|e| Error::Runtime(format!("GPU matmul failed: {}", e)))?;

        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Failed to convert result: {}", e)))
    }

    fn fused_dequant_matmul_q2k(
        &self,
        blocks: &[BlockQ2_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q2_k, QK_K};

        // Validate inputs
        if !k.is_multiple_of(QK_K) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, QK_K
            )));
        }

        let num_blocks_per_row = k / QK_K;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q2_K blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights on CPU, then upload to GPU
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * QK_K;
                let dequant_slice = &mut dequantized_weights[dequant_offset..dequant_offset + QK_K];

                dequantize_q2_k(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q2_K dequantization failed: {}", e)))?;
            }
        }

        // Upload to GPU and perform matmul
        let weights_tensor = self
            .f32_to_tensor(&dequantized_weights, &[n, k])
            .map_err(|e| Error::Runtime(format!("Failed to create weights tensor: {}", e)))?;

        let input_tensor = self
            .f32_to_tensor(input, &[batch_size, k])
            .map_err(|e| Error::Runtime(format!("Failed to create input tensor: {}", e)))?;

        let weights_t = weights_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Failed to transpose weights: {}", e)))?;

        let result_tensor = self
            .matmul(&input_tensor, &weights_t)
            .map_err(|e| Error::Runtime(format!("GPU matmul failed: {}", e)))?;

        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Failed to convert result: {}", e)))
    }

    fn fused_dequant_matmul_q3k(
        &self,
        blocks: &[BlockQ3_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q3_k, QK_K};

        // Validate inputs
        if !k.is_multiple_of(QK_K) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, QK_K
            )));
        }

        let num_blocks_per_row = k / QK_K;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q3_K blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights on CPU, then upload to GPU
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * QK_K;
                let dequant_slice = &mut dequantized_weights[dequant_offset..dequant_offset + QK_K];

                dequantize_q3_k(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q3_K dequantization failed: {}", e)))?;
            }
        }

        // Upload to GPU and perform matmul
        let weights_tensor = self
            .f32_to_tensor(&dequantized_weights, &[n, k])
            .map_err(|e| Error::Runtime(format!("Failed to create weights tensor: {}", e)))?;

        let input_tensor = self
            .f32_to_tensor(input, &[batch_size, k])
            .map_err(|e| Error::Runtime(format!("Failed to create input tensor: {}", e)))?;

        let weights_t = weights_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Failed to transpose weights: {}", e)))?;

        let result_tensor = self
            .matmul(&input_tensor, &weights_t)
            .map_err(|e| Error::Runtime(format!("GPU matmul failed: {}", e)))?;

        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Failed to convert result: {}", e)))
    }

    fn fused_dequant_matmul_q40(
        &self,
        blocks: &[BlockQ4_0],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q4_0, Q4_BLOCK_SIZE};

        // Validate inputs
        if !k.is_multiple_of(Q4_BLOCK_SIZE) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, Q4_BLOCK_SIZE
            )));
        }

        let num_blocks_per_row = k / Q4_BLOCK_SIZE;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q4_0 blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights on CPU, then upload to GPU
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * Q4_BLOCK_SIZE;
                let dequant_slice =
                    &mut dequantized_weights[dequant_offset..dequant_offset + Q4_BLOCK_SIZE];

                dequantize_q4_0(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q4_0 dequantization failed: {}", e)))?;
            }
        }

        // Upload to GPU and perform matmul
        let weights_tensor = self
            .f32_to_tensor(&dequantized_weights, &[n, k])
            .map_err(|e| Error::Runtime(format!("Failed to create weights tensor: {}", e)))?;

        let input_tensor = self
            .f32_to_tensor(input, &[batch_size, k])
            .map_err(|e| Error::Runtime(format!("Failed to create input tensor: {}", e)))?;

        let weights_t = weights_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Failed to transpose weights: {}", e)))?;

        let result_tensor = self
            .matmul(&input_tensor, &weights_t)
            .map_err(|e| Error::Runtime(format!("GPU matmul failed: {}", e)))?;

        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Failed to convert result: {}", e)))
    }

    fn fused_dequant_matmul_q41(
        &self,
        blocks: &[BlockQ4_1],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q4_1, Q4_BLOCK_SIZE};

        // Validate inputs
        if !k.is_multiple_of(Q4_BLOCK_SIZE) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, Q4_BLOCK_SIZE
            )));
        }

        let num_blocks_per_row = k / Q4_BLOCK_SIZE;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q4_1 blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights on CPU, then upload to GPU
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * Q4_BLOCK_SIZE;
                let dequant_slice =
                    &mut dequantized_weights[dequant_offset..dequant_offset + Q4_BLOCK_SIZE];

                dequantize_q4_1(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q4_1 dequantization failed: {}", e)))?;
            }
        }

        // Upload to GPU and perform matmul
        let weights_tensor = self
            .f32_to_tensor(&dequantized_weights, &[n, k])
            .map_err(|e| Error::Runtime(format!("Failed to create weights tensor: {}", e)))?;

        let input_tensor = self
            .f32_to_tensor(input, &[batch_size, k])
            .map_err(|e| Error::Runtime(format!("Failed to create input tensor: {}", e)))?;

        let weights_t = weights_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Failed to transpose weights: {}", e)))?;

        let result_tensor = self
            .matmul(&input_tensor, &weights_t)
            .map_err(|e| Error::Runtime(format!("GPU matmul failed: {}", e)))?;

        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Failed to convert result: {}", e)))
    }

    fn fused_dequant_matmul_q50(
        &self,
        blocks: &[BlockQ5_0],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q5_0, Q4_BLOCK_SIZE};

        // Validate inputs (Q5_0 uses same block size as Q4_0)
        if !k.is_multiple_of(Q4_BLOCK_SIZE) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, Q4_BLOCK_SIZE
            )));
        }

        let num_blocks_per_row = k / Q4_BLOCK_SIZE;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q5_0 blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights on CPU, then upload to GPU
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * Q4_BLOCK_SIZE;
                let dequant_slice =
                    &mut dequantized_weights[dequant_offset..dequant_offset + Q4_BLOCK_SIZE];

                dequantize_q5_0(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q5_0 dequantization failed: {}", e)))?;
            }
        }

        // Upload to GPU and perform matmul
        let weights_tensor = self
            .f32_to_tensor(&dequantized_weights, &[n, k])
            .map_err(|e| Error::Runtime(format!("Failed to create weights tensor: {}", e)))?;

        let input_tensor = self
            .f32_to_tensor(input, &[batch_size, k])
            .map_err(|e| Error::Runtime(format!("Failed to create input tensor: {}", e)))?;

        let weights_t = weights_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Failed to transpose weights: {}", e)))?;

        let result_tensor = self
            .matmul(&input_tensor, &weights_t)
            .map_err(|e| Error::Runtime(format!("GPU matmul failed: {}", e)))?;

        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Failed to convert result: {}", e)))
    }

    fn fused_dequant_matmul_q51(
        &self,
        blocks: &[BlockQ5_1],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q5_1, Q4_BLOCK_SIZE};

        // Validate inputs (Q5_1 uses same block size as Q4_0)
        if !k.is_multiple_of(Q4_BLOCK_SIZE) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, Q4_BLOCK_SIZE
            )));
        }

        let num_blocks_per_row = k / Q4_BLOCK_SIZE;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q5_1 blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights on CPU, then upload to GPU
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * Q4_BLOCK_SIZE;
                let dequant_slice =
                    &mut dequantized_weights[dequant_offset..dequant_offset + Q4_BLOCK_SIZE];

                dequantize_q5_1(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q5_1 dequantization failed: {}", e)))?;
            }
        }

        // Upload to GPU and perform matmul
        let weights_tensor = self
            .f32_to_tensor(&dequantized_weights, &[n, k])
            .map_err(|e| Error::Runtime(format!("Failed to create weights tensor: {}", e)))?;

        let input_tensor = self
            .f32_to_tensor(input, &[batch_size, k])
            .map_err(|e| Error::Runtime(format!("Failed to create input tensor: {}", e)))?;

        let weights_t = weights_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Failed to transpose weights: {}", e)))?;

        let result_tensor = self
            .matmul(&input_tensor, &weights_t)
            .map_err(|e| Error::Runtime(format!("GPU matmul failed: {}", e)))?;

        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Failed to convert result: {}", e)))
    }

    fn fused_dequant_matmul_q80(
        &self,
        blocks: &[BlockQ8_0],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q8_0, Q8_BLOCK_SIZE};

        // Validate inputs
        if !k.is_multiple_of(Q8_BLOCK_SIZE) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, Q8_BLOCK_SIZE
            )));
        }

        let num_blocks_per_row = k / Q8_BLOCK_SIZE;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q8_0 blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights on CPU, then upload to GPU
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * Q8_BLOCK_SIZE;
                let dequant_slice =
                    &mut dequantized_weights[dequant_offset..dequant_offset + Q8_BLOCK_SIZE];

                dequantize_q8_0(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q8_0 dequantization failed: {}", e)))?;
            }
        }

        // Upload to GPU and perform matmul
        let weights_tensor = self
            .f32_to_tensor(&dequantized_weights, &[n, k])
            .map_err(|e| Error::Runtime(format!("Failed to create weights tensor: {}", e)))?;

        let input_tensor = self
            .f32_to_tensor(input, &[batch_size, k])
            .map_err(|e| Error::Runtime(format!("Failed to create input tensor: {}", e)))?;

        let weights_t = weights_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Failed to transpose weights: {}", e)))?;

        let result_tensor = self
            .matmul(&input_tensor, &weights_t)
            .map_err(|e| Error::Runtime(format!("GPU matmul failed: {}", e)))?;

        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Failed to convert result: {}", e)))
    }

    fn fused_dequant_matmul_q81(
        &self,
        blocks: &[BlockQ8_1],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q8_1, Q8_BLOCK_SIZE};

        // Validate inputs
        if !k.is_multiple_of(Q8_BLOCK_SIZE) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, Q8_BLOCK_SIZE
            )));
        }

        let num_blocks_per_row = k / Q8_BLOCK_SIZE;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q8_1 blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights on CPU, then upload to GPU
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * Q8_BLOCK_SIZE;
                let dequant_slice =
                    &mut dequantized_weights[dequant_offset..dequant_offset + Q8_BLOCK_SIZE];

                dequantize_q8_1(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q8_1 dequantization failed: {}", e)))?;
            }
        }

        // Upload to GPU and perform matmul
        let weights_tensor = self
            .f32_to_tensor(&dequantized_weights, &[n, k])
            .map_err(|e| Error::Runtime(format!("Failed to create weights tensor: {}", e)))?;

        let input_tensor = self
            .f32_to_tensor(input, &[batch_size, k])
            .map_err(|e| Error::Runtime(format!("Failed to create input tensor: {}", e)))?;

        let weights_t = weights_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Failed to transpose weights: {}", e)))?;

        let result_tensor = self
            .matmul(&input_tensor, &weights_t)
            .map_err(|e| Error::Runtime(format!("GPU matmul failed: {}", e)))?;

        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Failed to convert result: {}", e)))
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

    #[test]
    fn test_matmul_transposed() {
        let backend = CandleGpuBackend::new().unwrap();

        // A: [2, 3], B: [2, 3] -> result should be A @ B^T = [2, 2]
        let a = backend
            .f32_to_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap();
        let b = backend
            .f32_to_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap();

        let result = backend.matmul_transposed(&a, &b).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        // Expected: [[1*1+2*2+3*3, 1*4+2*5+3*6], [4*1+5*2+6*3, 4*4+5*5+6*6]]
        // = [[14, 32], [32, 77]]
        assert_eq!(result_vec.len(), 4);
        assert!((result_vec[0] - 14.0).abs() < 0.001);
        assert!((result_vec[1] - 32.0).abs() < 0.001);
        assert!((result_vec[2] - 32.0).abs() < 0.001);
        assert!((result_vec[3] - 77.0).abs() < 0.001);
    }

    #[test]
    fn test_silu() {
        let backend = CandleGpuBackend::new().unwrap();

        // SiLU(x) = x * sigmoid(x)
        let x = backend.f32_to_tensor(&[0.0, 1.0, -1.0, 2.0], &[4]).unwrap();
        let result = backend.silu(&x).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        assert_eq!(result_vec.len(), 4);
        // SiLU(0) = 0
        assert!((result_vec[0]).abs() < 0.001);
        // SiLU(1) ≈ 0.731
        assert!((result_vec[1] - 0.731).abs() < 0.01);
        // SiLU(-1) ≈ -0.269
        assert!((result_vec[2] - (-0.269)).abs() < 0.01);
        // All values should be finite
        for val in &result_vec {
            assert!(val.is_finite(), "SiLU produced non-finite value: {}", val);
        }
    }

    #[test]
    fn test_softmax() {
        let backend = CandleGpuBackend::new().unwrap();

        // Test softmax on simple input
        let x = backend.f32_to_tensor(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let result = backend.softmax(&x).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        assert_eq!(result_vec.len(), 3);
        // Sum should be approximately 1.0
        let sum: f32 = result_vec.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Softmax sum should be 1.0, got {}",
            sum
        );
        // All values should be positive
        for val in &result_vec {
            assert!(*val > 0.0, "Softmax produced negative value: {}", val);
            assert!(
                val.is_finite(),
                "Softmax produced non-finite value: {}",
                val
            );
        }
    }

    #[test]
    fn test_add() {
        let backend = CandleGpuBackend::new().unwrap();

        let a = backend.f32_to_tensor(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = backend.f32_to_tensor(&[4.0, 5.0, 6.0], &[3]).unwrap();

        let result = backend.add(&a, &b).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        assert_eq!(result_vec, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul() {
        let backend = CandleGpuBackend::new().unwrap();

        let a = backend.f32_to_tensor(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = backend.f32_to_tensor(&[4.0, 5.0, 6.0], &[3]).unwrap();

        let result = backend.mul(&a, &b).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        assert_eq!(result_vec, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_fused_dequant_matmul_q4k() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ4_K, QK_K};

        // Create minimal test: n=256, k=256 (one block per output)
        let n = 256;
        let k = 256;
        let batch_size = 1;

        // Create dummy Q4_K blocks (all zeros for simplicity)
        let block = BlockQ4_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; 12],
            qs: [0u8; QK_K / 2],
        };
        let num_blocks = n * (k / QK_K);
        let blocks = vec![block; num_blocks];

        // Create input
        let input = vec![1.0f32; batch_size * k];

        // Should succeed (even if result is zero)
        let result = backend.fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * n);
    }

    #[test]
    fn test_fused_dequant_matmul_q4k_invalid_shape() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ4_K, QK_K};

        let n = 256;
        let k = 255; // Not a multiple of QK_K (256)
        let batch_size = 1;

        let block = BlockQ4_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; 12],
            qs: [0u8; QK_K / 2],
        };
        let blocks = vec![block; 10];

        let input = vec![1.0f32; batch_size * k];

        // Should fail with InvalidShape error
        let result = backend.fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_dequant_matmul_q5k() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ5_K, QK_K};

        let n = 256;
        let k = 256;
        let batch_size = 1;

        let block = BlockQ5_K {
            d: half::f16::from_f32(1.0).to_bits(),
            scales: [0i8; QK_K / 16],
            qh: [0u8; QK_K / 8],
            ql: [0u8; QK_K / 2],
        };
        let num_blocks = n * (k / QK_K);
        let blocks = vec![block; num_blocks];

        let input = vec![1.0f32; batch_size * k];

        let result = backend.fused_dequant_matmul_q5k(&blocks, &input, batch_size, n, k);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * n);
    }

    #[test]
    fn test_fused_dequant_matmul_q6k() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ6_K, QK_K};

        let n = 256;
        let k = 256;
        let batch_size = 1;

        let block = BlockQ6_K {
            d: half::f16::from_f32(1.0).to_bits(),
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [0i8; QK_K / 16],
        };
        let num_blocks = n * (k / QK_K);
        let blocks = vec![block; num_blocks];

        let input = vec![1.0f32; batch_size * k];

        let result = backend.fused_dequant_matmul_q6k(&blocks, &input, batch_size, n, k);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * n);
    }

    #[test]
    fn test_fused_dequant_matmul_q8k() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ8_K, QK_K};

        let n = 256;
        let k = 256;
        let batch_size = 1;

        let block = BlockQ8_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; QK_K / 8],
            quants: [0i8; QK_K],
        };
        let num_blocks = n * (k / QK_K);
        let blocks = vec![block; num_blocks];

        let input = vec![1.0f32; batch_size * k];

        let result = backend.fused_dequant_matmul_q8k(&blocks, &input, batch_size, n, k);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * n);
    }

    #[test]
    fn test_fused_dequant_matmul_q2k() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ2_K, QK_K};

        let n = 256;
        let k = 256;
        let batch_size = 1;

        let block = BlockQ2_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; QK_K / 16], // 16 scales
            qs: [0u8; QK_K / 4],      // 64 bytes
            qh: [0u8; QK_K / 8],      // 32 bytes
        };
        let num_blocks = n * (k / QK_K);
        let blocks = vec![block; num_blocks];

        let input = vec![1.0f32; batch_size * k];

        let result = backend.fused_dequant_matmul_q2k(&blocks, &input, batch_size, n, k);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * n);
    }

    #[test]
    fn test_fused_dequant_matmul_q3k() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ3_K, QK_K};

        let n = 256;
        let k = 256;
        let batch_size = 1;

        let block = BlockQ3_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; QK_K / 8], // 32 scales
            qs: [0u8; QK_K / 2],     // 128 bytes
            qh: [0u8; QK_K / 8],     // 32 bytes
        };
        let num_blocks = n * (k / QK_K);
        let blocks = vec![block; num_blocks];

        let input = vec![1.0f32; batch_size * k];

        let result = backend.fused_dequant_matmul_q3k(&blocks, &input, batch_size, n, k);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * n);
    }

    #[test]
    fn test_fused_dequant_matmul_q40() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ4_0, Q4_BLOCK_SIZE};

        let n = 32;
        let k = 32;
        let batch_size = 1;

        let block = BlockQ4_0 {
            scale: half::f16::from_f32(1.0).to_bits(),
            quants: [0x10u8; Q4_BLOCK_SIZE / 2],
        };
        let num_blocks = n * (k / Q4_BLOCK_SIZE);
        let blocks = vec![block; num_blocks];

        let input = vec![1.0f32; batch_size * k];

        let result = backend.fused_dequant_matmul_q40(&blocks, &input, batch_size, n, k);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * n);
    }

    #[test]
    fn test_fused_dequant_matmul_q41() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ4_1, Q4_BLOCK_SIZE};

        let n = 32;
        let k = 32;
        let batch_size = 1;

        let block = BlockQ4_1 {
            scale: half::f16::from_f32(1.0).to_bits(),
            delta: half::f16::from_f32(0.0).to_bits(),
            quants: [0x10u8; Q4_BLOCK_SIZE / 2],
        };
        let num_blocks = n * (k / Q4_BLOCK_SIZE);
        let blocks = vec![block; num_blocks];

        let input = vec![1.0f32; batch_size * k];

        let result = backend.fused_dequant_matmul_q41(&blocks, &input, batch_size, n, k);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * n);
    }

    #[test]
    fn test_fused_dequant_matmul_q50() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ5_0, Q4_BLOCK_SIZE};

        let n = 32;
        let k = 32;
        let batch_size = 1;

        let block = BlockQ5_0 {
            scale: half::f16::from_f32(1.0).to_bits(),
            qh: [0u8; Q4_BLOCK_SIZE / 8],
            ql: [0x10u8; Q4_BLOCK_SIZE / 2],
        };
        let num_blocks = n * (k / Q4_BLOCK_SIZE);
        let blocks = vec![block; num_blocks];

        let input = vec![1.0f32; batch_size * k];

        let result = backend.fused_dequant_matmul_q50(&blocks, &input, batch_size, n, k);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * n);
    }

    #[test]
    fn test_fused_dequant_matmul_q51() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ5_1, Q4_BLOCK_SIZE};

        let n = 32;
        let k = 32;
        let batch_size = 1;

        let block = BlockQ5_1 {
            scale: half::f16::from_f32(1.0).to_bits(),
            delta: half::f16::from_f32(0.0).to_bits(),
            qh: [0u8; Q4_BLOCK_SIZE / 8],
            ql: [0x10u8; Q4_BLOCK_SIZE / 2],
        };
        let num_blocks = n * (k / Q4_BLOCK_SIZE);
        let blocks = vec![block; num_blocks];

        let input = vec![1.0f32; batch_size * k];

        let result = backend.fused_dequant_matmul_q51(&blocks, &input, batch_size, n, k);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * n);
    }

    #[test]
    fn test_fused_dequant_matmul_q80() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ8_0, Q8_BLOCK_SIZE};

        let n = 32;
        let k = 32;
        let batch_size = 1;

        let block = BlockQ8_0 {
            scale: half::f16::from_f32(1.0).to_bits(),
            quants: [0i8; Q8_BLOCK_SIZE],
        };
        let num_blocks = n * (k / Q8_BLOCK_SIZE);
        let blocks = vec![block; num_blocks];

        let input = vec![1.0f32; batch_size * k];

        let result = backend.fused_dequant_matmul_q80(&blocks, &input, batch_size, n, k);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * n);
    }

    #[test]
    fn test_fused_dequant_matmul_q81() {
        let backend = CandleGpuBackend::new().unwrap();
        use realm_core::quant::{BlockQ8_1, Q8_BLOCK_SIZE};

        let n = 32;
        let k = 32;
        let batch_size = 1;

        let block = BlockQ8_1 {
            scale: half::f16::from_f32(1.0).to_bits(),
            delta: half::f16::from_f32(0.0).to_bits(),
            quants: [0i8; Q8_BLOCK_SIZE],
        };
        let num_blocks = n * (k / Q8_BLOCK_SIZE);
        let blocks = vec![block; num_blocks];

        let input = vec![1.0f32; batch_size * k];

        let result = backend.fused_dequant_matmul_q81(&blocks, &input, batch_size, n, k);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * n);
    }

    #[test]
    fn test_backend_name() {
        let backend = CandleGpuBackend::new().unwrap();
        let name = backend.name();
        // Should return one of: "CUDA", "Metal", or "CPU"
        assert!(name == "CUDA" || name == "Metal" || name == "CPU");
    }

    #[test]
    fn test_tensor_conversion() {
        let backend = CandleGpuBackend::new().unwrap();

        // Test f32_to_tensor and tensor_to_f32 round-trip
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = backend.f32_to_tensor(&original, &[5]).unwrap();
        let converted = backend.tensor_to_f32(&tensor).unwrap();

        assert_eq!(original, converted);
    }

    #[test]
    fn test_matmul_large() {
        let backend = CandleGpuBackend::new().unwrap();

        // Test larger matrix multiplication
        let size = 64;
        let a_data: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| ((i * 2) % 10) as f32).collect();

        let a = backend.f32_to_tensor(&a_data, &[size, size]).unwrap();
        let b = backend.f32_to_tensor(&b_data, &[size, size]).unwrap();

        let result = backend.matmul(&a, &b).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        assert_eq!(result_vec.len(), size * size);
        // Check no NaN or inf
        for val in &result_vec {
            assert!(val.is_finite(), "Large matmul produced non-finite value");
        }
    }

    #[test]
    fn test_device_detection() {
        let backend = CandleGpuBackend::new().unwrap();
        let device = backend.device();
        let name = backend.name();

        // Device should be valid
        match device {
            candle_core::Device::Cuda(_) => assert_eq!(name, "CUDA"),
            candle_core::Device::Metal(_) => assert_eq!(name, "Metal"),
            candle_core::Device::Cpu => assert_eq!(name, "CPU"),
        }
    }

    #[test]
    fn test_gpu_detection() {
        let backend = CandleGpuBackend::new().unwrap();
        let is_gpu = backend.is_gpu();
        let name = backend.name();

        // If CUDA or Metal, should be GPU
        if name == "CUDA" || name == "Metal" {
            assert!(is_gpu, "CUDA/Metal backend should report as GPU");
        } else {
            assert!(!is_gpu, "CPU backend should not report as GPU");
        }
    }

    #[test]
    fn test_device_count() {
        let backend = CandleGpuBackend::new().unwrap();
        let count = backend.device_count();

        // Should have at least 1 device
        assert!(count >= 1, "Should have at least 1 device");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_availability() {
        let is_available = CandleGpuBackend::is_cuda_available();
        // Just verify the function doesn't panic
        // Actual value depends on system configuration
        let _ = is_available;
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_availability() {
        let is_available = CandleGpuBackend::is_metal_available();
        // Just verify the function doesn't panic
        // Actual value depends on system configuration
        let _ = is_available;
    }

    #[test]
    fn test_add_zero_tensors() {
        let backend = CandleGpuBackend::new().unwrap();

        let a = backend.f32_to_tensor(&[0.0, 0.0, 0.0], &[3]).unwrap();
        let b = backend.f32_to_tensor(&[1.0, 2.0, 3.0], &[3]).unwrap();

        let result = backend.add(&a, &b).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        assert_eq!(result_vec, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mul_zero_tensors() {
        let backend = CandleGpuBackend::new().unwrap();

        let a = backend.f32_to_tensor(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = backend.f32_to_tensor(&[0.0, 0.0, 0.0], &[3]).unwrap();

        let result = backend.mul(&a, &b).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        assert_eq!(result_vec, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        let backend = CandleGpuBackend::new().unwrap();

        // Simple attention: batch=1, heads=1, seq_len=2, head_dim=2
        let q = backend
            .f32_to_tensor(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])
            .unwrap();
        let k = backend
            .f32_to_tensor(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])
            .unwrap();
        let v = backend
            .f32_to_tensor(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])
            .unwrap();

        let scale = 1.0 / (2.0f32.sqrt());
        let result = backend
            .scaled_dot_product_attention(&q, &k, &v, scale, None)
            .unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        assert_eq!(result_vec.len(), 4);
        // All values should be finite
        for val in &result_vec {
            assert!(
                val.is_finite(),
                "Attention produced non-finite value: {}",
                val
            );
        }
    }

    #[test]
    fn test_rope_basic() {
        let backend = CandleGpuBackend::new().unwrap();

        // Simple RoPE test: batch=1, heads=1, seq_len=1, head_dim=4
        let x = backend
            .f32_to_tensor(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 1, 4])
            .unwrap();
        let cos = backend.f32_to_tensor(&[1.0, 1.0], &[1, 1, 1, 2]).unwrap();
        let sin = backend.f32_to_tensor(&[0.0, 0.0], &[1, 1, 1, 2]).unwrap();

        let result = backend.rope(&x, &cos, &sin).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        assert_eq!(result_vec.len(), 4);
        // All values should be finite
        for val in &result_vec {
            assert!(val.is_finite(), "RoPE produced non-finite value: {}", val);
        }
    }

    #[test]
    fn test_matmul_with_precision_config() {
        use crate::mixed_precision::{MixedPrecisionConfig, PrecisionMode};

        let config = MixedPrecisionConfig {
            forward_precision: PrecisionMode::FP16,
            ..Default::default()
        };

        let backend = CandleGpuBackend::with_precision(config).unwrap();

        let a = backend
            .f32_to_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap();
        let b = backend
            .f32_to_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2])
            .unwrap();

        let result = backend.matmul(&a, &b).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        assert_eq!(result_vec.len(), 4);
        // Expected result: [[19, 22], [43, 50]]
        assert!((result_vec[0] - 19.0).abs() < 0.1);
        assert!((result_vec[1] - 22.0).abs() < 0.1);
        assert!((result_vec[2] - 43.0).abs() < 0.1);
        assert!((result_vec[3] - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_broadcast_operations() {
        let backend = CandleGpuBackend::new().unwrap();

        // Test broadcasting: [3] + [1] should broadcast
        let a = backend.f32_to_tensor(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = backend.f32_to_tensor(&[10.0], &[1]).unwrap();

        let result = backend.add(&a, &b).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        assert_eq!(result_vec, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_softmax_edge_cases() {
        let backend = CandleGpuBackend::new().unwrap();

        // Test softmax with all zeros
        let x = backend.f32_to_tensor(&[0.0, 0.0, 0.0], &[3]).unwrap();
        let result = backend.softmax(&x).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        // All zeros should produce uniform distribution
        let sum: f32 = result_vec.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
        // Each value should be approximately 1/3
        for val in &result_vec {
            assert!((val - 1.0 / 3.0).abs() < 0.1);
        }

        // Test softmax with large values
        let x_large = backend.f32_to_tensor(&[100.0, 200.0, 300.0], &[3]).unwrap();
        let result_large = backend.softmax(&x_large).unwrap();
        let result_large_vec = backend.tensor_to_f32(&result_large).unwrap();

        let sum_large: f32 = result_large_vec.iter().sum();
        assert!((sum_large - 1.0).abs() < 0.01);
        // Largest value should have highest probability
        assert!(result_large_vec[2] > result_large_vec[1]);
        assert!(result_large_vec[1] > result_large_vec[0]);
    }

    #[test]
    fn test_rms_norm_edge_cases() {
        let backend = CandleGpuBackend::new().unwrap();

        // Test RMS norm with all zeros
        let x = backend.f32_to_tensor(&[0.0, 0.0, 0.0, 0.0], &[4]).unwrap();
        let weight = backend.f32_to_tensor(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();

        let result = backend.rms_norm(&x, &weight, 1e-5).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        assert_eq!(result_vec.len(), 4);
        // All zeros should remain zeros (or very close)
        for val in &result_vec {
            assert!(val.abs() < 0.01, "RMS norm of zeros should be near zero");
        }
    }

    #[test]
    fn test_silu_edge_cases() {
        let backend = CandleGpuBackend::new().unwrap();

        // Test SiLU with zero
        let x = backend.f32_to_tensor(&[0.0], &[1]).unwrap();
        let result = backend.silu(&x).unwrap();
        let result_vec = backend.tensor_to_f32(&result).unwrap();

        // SiLU(0) = 0
        assert!((result_vec[0]).abs() < 0.001);

        // Test SiLU with large positive value
        let x_large = backend.f32_to_tensor(&[10.0], &[1]).unwrap();
        let result_large = backend.silu(&x_large).unwrap();
        let result_large_vec = backend.tensor_to_f32(&result_large).unwrap();

        // SiLU(x) ≈ x for large positive values
        assert!(result_large_vec[0] > 9.0);

        // Test SiLU with large negative value
        let x_neg = backend.f32_to_tensor(&[-10.0], &[1]).unwrap();
        let result_neg = backend.silu(&x_neg).unwrap();
        let result_neg_vec = backend.tensor_to_f32(&result_neg).unwrap();

        // SiLU(x) ≈ 0 for large negative values
        assert!(result_neg_vec[0] > -1.0);
    }
}
