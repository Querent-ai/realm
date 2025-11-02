//! Candle CPU backend implementation
//!
//! This module provides a CPU backend using Candle's optimized CPU operations.

use candle_core::{Device, Result as CandleResult, Tensor};
use realm_core::error::{Error, Result};
use realm_core::quant::{
    BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ4_K, BlockQ5_0, BlockQ5_1, BlockQ5_K,
    BlockQ6_K, BlockQ8_0, BlockQ8_1, BlockQ8_K,
};

use super::cpu_backend_trait::CpuBackendTrait;

/// Candle CPU backend using optimized CPU operations
pub struct CandleCpuBackend {
    device: Device,
}

impl CandleCpuBackend {
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;
        Ok(Self { device })
    }

    /// Convert f32 slice to Candle tensor
    fn f32_to_tensor(&self, data: &[f32], shape: &[usize]) -> CandleResult<Tensor> {
        Tensor::from_slice(data, shape, &self.device)
    }

    /// Convert Candle tensor to f32 slice
    fn tensor_to_f32(&self, tensor: &Tensor) -> CandleResult<Vec<f32>> {
        let flat_tensor = if tensor.dims().len() > 1 {
            tensor.flatten_all()?
        } else {
            tensor.clone()
        };
        flat_tensor.to_vec1::<f32>()
    }
}

impl CpuBackendTrait for CandleCpuBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        // Convert slices to tensors
        let a_tensor = self
            .f32_to_tensor(a, &[m, k])
            .map_err(|e| Error::Runtime(format!("Candle CPU tensor creation failed: {}", e)))?;
        let b_tensor = self
            .f32_to_tensor(b, &[k, n])
            .map_err(|e| Error::Runtime(format!("Candle CPU tensor creation failed: {}", e)))?;

        // Perform matrix multiplication using Tensor::matmul
        let result_tensor = a_tensor
            .matmul(&b_tensor)
            .map_err(|e| Error::Runtime(format!("Candle CPU matmul failed: {}", e)))?;

        // Convert back to f32 slice
        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Candle CPU tensor conversion failed: {}", e)))
    }

    fn matmul_transposed(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Convert slices to tensors
        let a_tensor = self
            .f32_to_tensor(a, &[m, k])
            .map_err(|e| Error::Runtime(format!("Candle CPU tensor creation failed: {}", e)))?;
        let b_tensor = self
            .f32_to_tensor(b, &[n, k]) // Note: b is already transposed
            .map_err(|e| Error::Runtime(format!("Candle CPU tensor creation failed: {}", e)))?;

        // Transpose b_tensor to get the correct shape for matmul
        let b_transposed = b_tensor
            .t()
            .map_err(|e| Error::Runtime(format!("Candle CPU transpose failed: {}", e)))?;

        // Perform matrix multiplication using Tensor::matmul
        let result_tensor = a_tensor
            .matmul(&b_transposed)
            .map_err(|e| Error::Runtime(format!("Candle CPU matmul failed: {}", e)))?;

        // Convert back to f32 slice
        self.tensor_to_f32(&result_tensor)
            .map_err(|e| Error::Runtime(format!("Candle CPU tensor conversion failed: {}", e)))
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

        // Dequantize weights to f32
        let num_blocks = blocks.len();
        let mut dequantized = Vec::with_capacity(num_blocks * QK_K);
        for block in blocks {
            let mut block_output = vec![0.0f32; QK_K];
            dequantize_q4_k(block, &mut block_output)?;
            dequantized.extend_from_slice(&block_output);
        }

        // Reshape: dequantized is [n, k], input is [batch_size, k]
        // Output should be [batch_size, n] = input @ dequantized^T
        self.matmul_transposed(input, &dequantized, batch_size, k, n)
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

        let num_blocks = blocks.len();
        let mut dequantized = Vec::with_capacity(num_blocks * QK_K);
        for block in blocks {
            let mut block_output = vec![0.0f32; QK_K];
            dequantize_q5_k(block, &mut block_output)?;
            dequantized.extend_from_slice(&block_output);
        }

        self.matmul_transposed(input, &dequantized, batch_size, k, n)
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

        let num_blocks = blocks.len();
        let mut dequantized = Vec::with_capacity(num_blocks * QK_K);
        for block in blocks {
            let mut block_output = vec![0.0f32; QK_K];
            dequantize_q6_k(block, &mut block_output)?;
            dequantized.extend_from_slice(&block_output);
        }

        self.matmul_transposed(input, &dequantized, batch_size, k, n)
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

        let num_blocks = blocks.len();
        let mut dequantized = Vec::with_capacity(num_blocks * QK_K);
        for block in blocks {
            let mut block_output = vec![0.0f32; QK_K];
            dequantize_q8_k(block, &mut block_output)?;
            dequantized.extend_from_slice(&block_output);
        }

        self.matmul_transposed(input, &dequantized, batch_size, k, n)
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

        let num_blocks = blocks.len();
        let mut dequantized = Vec::with_capacity(num_blocks * QK_K);
        for block in blocks {
            let mut block_output = vec![0.0f32; QK_K];
            dequantize_q2_k(block, &mut block_output)?;
            dequantized.extend_from_slice(&block_output);
        }

        self.matmul_transposed(input, &dequantized, batch_size, k, n)
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

        let num_blocks = blocks.len();
        let mut dequantized = Vec::with_capacity(num_blocks * QK_K);
        for block in blocks {
            let mut block_output = vec![0.0f32; QK_K];
            dequantize_q3_k(block, &mut block_output)?;
            dequantized.extend_from_slice(&block_output);
        }

        self.matmul_transposed(input, &dequantized, batch_size, k, n)
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

        let num_blocks = blocks.len();
        let mut dequantized = Vec::with_capacity(num_blocks * Q4_BLOCK_SIZE);
        for block in blocks {
            let mut block_output = vec![0.0f32; Q4_BLOCK_SIZE];
            dequantize_q4_0(block, &mut block_output)?;
            dequantized.extend_from_slice(&block_output);
        }

        self.matmul_transposed(input, &dequantized, batch_size, k, n)
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

        let num_blocks = blocks.len();
        let mut dequantized = Vec::with_capacity(num_blocks * Q4_BLOCK_SIZE);
        for block in blocks {
            let mut block_output = vec![0.0f32; Q4_BLOCK_SIZE];
            dequantize_q4_1(block, &mut block_output)?;
            dequantized.extend_from_slice(&block_output);
        }

        self.matmul_transposed(input, &dequantized, batch_size, k, n)
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

        let num_blocks = blocks.len();
        let mut dequantized = Vec::with_capacity(num_blocks * Q4_BLOCK_SIZE);
        for block in blocks {
            let mut block_output = vec![0.0f32; Q4_BLOCK_SIZE];
            dequantize_q5_0(block, &mut block_output)?;
            dequantized.extend_from_slice(&block_output);
        }

        self.matmul_transposed(input, &dequantized, batch_size, k, n)
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

        let num_blocks = blocks.len();
        let mut dequantized = Vec::with_capacity(num_blocks * Q4_BLOCK_SIZE);
        for block in blocks {
            let mut block_output = vec![0.0f32; Q4_BLOCK_SIZE];
            dequantize_q5_1(block, &mut block_output)?;
            dequantized.extend_from_slice(&block_output);
        }

        self.matmul_transposed(input, &dequantized, batch_size, k, n)
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

        let num_blocks = blocks.len();
        let mut dequantized = Vec::with_capacity(num_blocks * Q8_BLOCK_SIZE);
        for block in blocks {
            let mut block_output = vec![0.0f32; Q8_BLOCK_SIZE];
            dequantize_q8_0(block, &mut block_output)?;
            dequantized.extend_from_slice(&block_output);
        }

        self.matmul_transposed(input, &dequantized, batch_size, k, n)
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

        let num_blocks = blocks.len();
        let mut dequantized = Vec::with_capacity(num_blocks * Q8_BLOCK_SIZE);
        for block in blocks {
            let mut block_output = vec![0.0f32; Q8_BLOCK_SIZE];
            dequantize_q8_1(block, &mut block_output)?;
            dequantized.extend_from_slice(&block_output);
        }

        self.matmul_transposed(input, &dequantized, batch_size, k, n)
    }

    fn name(&self) -> &'static str {
        "Candle CPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use realm_core::error::Result;
    use realm_core::quant::{Q4_BLOCK_SIZE, Q8_BLOCK_SIZE, QK_K};

    // Reuse helper functions from naive_backend tests - we'll inline them here
    fn create_test_q4_0_block(scale: f32, values: [u8; Q4_BLOCK_SIZE / 2]) -> BlockQ4_0 {
        BlockQ4_0 {
            scale: half::f16::from_f32(scale).to_bits(),
            quants: values,
        }
    }

    fn create_test_q4_1_block(
        scale: f32,
        delta: f32,
        values: [u8; Q4_BLOCK_SIZE / 2],
    ) -> BlockQ4_1 {
        BlockQ4_1 {
            scale: half::f16::from_f32(scale).to_bits(),
            delta: half::f16::from_f32(delta).to_bits(),
            quants: values,
        }
    }

    fn create_test_q5_0_block(scale: f32) -> BlockQ5_0 {
        let mut ql = [0u8; Q4_BLOCK_SIZE / 2];
        let _qh = [0u8; Q4_BLOCK_SIZE / 8];
        #[allow(clippy::needless_range_loop)] // Complex indexing with i+1
        for i in 0..Q4_BLOCK_SIZE / 2 {
            ql[i] = (i % 16) as u8 | (((i + 1) % 16) as u8) << 4;
        }
        BlockQ5_0 {
            scale: half::f16::from_f32(scale).to_bits(),
            qh: _qh,
            ql,
        }
    }

    fn create_test_q5_1_block(scale: f32, delta: f32) -> BlockQ5_1 {
        let mut ql = [0u8; Q4_BLOCK_SIZE / 2];
        let _qh = [0u8; Q4_BLOCK_SIZE / 8];
        #[allow(clippy::needless_range_loop)] // Complex indexing with i+1
        for i in 0..Q4_BLOCK_SIZE / 2 {
            ql[i] = (i % 16) as u8 | (((i + 1) % 16) as u8) << 4;
        }
        BlockQ5_1 {
            scale: half::f16::from_f32(scale).to_bits(),
            delta: half::f16::from_f32(delta).to_bits(),
            qh: _qh,
            ql,
        }
    }

    fn create_test_q8_0_block(scale: f32) -> BlockQ8_0 {
        let mut quants = [0i8; Q8_BLOCK_SIZE];
        for (i, quant) in quants.iter_mut().enumerate() {
            *quant = ((i % 127) as i8) - 64;
        }
        BlockQ8_0 {
            scale: half::f16::from_f32(scale).to_bits(),
            quants,
        }
    }

    fn create_test_q8_1_block(scale: f32, delta: f32) -> BlockQ8_1 {
        let mut quants = [0i8; Q8_BLOCK_SIZE];
        for (i, quant) in quants.iter_mut().enumerate() {
            *quant = ((i % 127) as i8) - 64;
        }
        BlockQ8_1 {
            scale: half::f16::from_f32(scale).to_bits(),
            delta: half::f16::from_f32(delta).to_bits(),
            quants,
        }
    }

    fn create_test_q4_k_block(d: f32, dmin: f32) -> BlockQ4_K {
        let mut scales = [0u8; 12];
        let mut qs = [0u8; QK_K / 2];
        for (i, scale) in scales.iter_mut().enumerate() {
            *scale = (16 + i) as u8;
        }
        #[allow(clippy::needless_range_loop)] // Complex indexing with i+1
        for i in 0..QK_K / 2 {
            qs[i] = ((i % 16) as u8) | (((i + 1) % 16) as u8) << 4;
        }
        BlockQ4_K {
            d: half::f16::from_f32(d).to_bits(),
            dmin: half::f16::from_f32(dmin).to_bits(),
            scales,
            qs,
        }
    }

    #[allow(dead_code)] // Used in tests, but clippy doesn't detect test usage
    fn create_test_q2_k_block(d: f32, dmin: f32) -> BlockQ2_K {
        let mut scales = [0u8; QK_K / 16];
        let mut qs = [0u8; QK_K / 4];
        let mut qh = [0u8; QK_K / 8];
        for (i, scale) in scales.iter_mut().enumerate() {
            *scale = (4 + i) as u8;
        }
        #[allow(clippy::needless_range_loop)] // Complex indexing with i+1
        for i in 0..QK_K / 4 {
            qs[i] = ((i % 4) as u8) | (((i + 1) % 4) as u8) << 2;
        }
        // Initialize qh properly - each byte contains 8 bits for 8 values
        for (i, qh_byte) in qh.iter_mut().enumerate() {
            *qh_byte = ((i % 2) * 0x55) as u8; // Pattern: 01010101 or 00000000
        }
        BlockQ2_K {
            d: half::f16::from_f32(d).to_bits(),
            dmin: half::f16::from_f32(dmin).to_bits(),
            scales,
            qh,
            qs,
        }
    }

    fn create_test_q3_k_block(d: f32, dmin: f32) -> BlockQ3_K {
        let mut scales = [0u8; QK_K / 8];
        let mut qs = [0u8; QK_K / 2];
        let _qh = [0u8; QK_K / 8];
        for (i, scale) in scales.iter_mut().enumerate() {
            *scale = (8 + i) as u8;
        }
        #[allow(clippy::needless_range_loop)] // Complex indexing with i+1
        for i in 0..QK_K / 2 {
            qs[i] = ((i % 4) as u8) | (((i + 1) % 4) as u8) << 2;
        }
        BlockQ3_K {
            d: half::f16::from_f32(d).to_bits(),
            dmin: half::f16::from_f32(dmin).to_bits(),
            scales,
            qh: _qh,
            qs,
        }
    }

    #[test]
    fn test_candle_matmul_basic() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2]
        let result = backend.matmul(&a, &b, 2, 2, 2)?;
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 3.0).abs() < 1e-5);
        assert!((result[3] - 4.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_candle_matmul_large() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let m = 3;
        let k = 4;
        let n = 2;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let result = backend.matmul(&a, &b, m, k, n)?;
        assert_eq!(result.len(), m * n);
        Ok(())
    }

    #[test]
    fn test_candle_matmul_transposed() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b_t = vec![1.0, 3.0, 2.0, 4.0]; // [2, 2] transposed
        let result = backend.matmul_transposed(&a, &b_t, 2, 2, 2)?;
        assert_eq!(result.len(), 4);
        assert!((result[0] - 7.0).abs() < 1e-4);
        assert!((result[1] - 10.0).abs() < 1e-4);
        assert!((result[2] - 15.0).abs() < 1e-4);
        assert!((result[3] - 22.0).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_candle_fused_q4k_basic() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let k = QK_K;
        let n = 1;
        let batch_size = 1;

        let blocks = vec![create_test_q4_k_block(0.5, 0.1)];
        let input: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let result = backend.fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        assert!(result[0].abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_candle_fused_q4k_batch() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let k = QK_K;
        let n = 2;
        let batch_size = 3;

        let mut blocks = Vec::new();
        for _ in 0..n {
            blocks.push(create_test_q4_k_block(0.5, 0.1));
        }

        let input: Vec<f32> = (0..batch_size * k)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.5)
            .collect();

        let result = backend.fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);

        for batch_idx in 0..batch_size {
            let batch_sum: f32 = (0..n).map(|i| result[batch_idx * n + i]).sum();
            assert!(batch_sum.abs() > 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_candle_fused_q40_basic() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let k = Q4_BLOCK_SIZE;
        let n = 1;
        let batch_size = 1;

        let mut quants = [0u8; Q4_BLOCK_SIZE / 2];
        for (i, quant) in quants.iter_mut().enumerate() {
            *quant = 0x10 + (i % 16) as u8;
        }
        let blocks = vec![create_test_q4_0_block(0.25, quants)];

        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

        let result = backend.fused_dequant_matmul_q40(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        assert!(result[0].abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_candle_fused_q41_basic() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let k = Q4_BLOCK_SIZE;
        let n = 1;
        let batch_size = 1;

        let mut quants = [0u8; Q4_BLOCK_SIZE / 2];
        for (i, quant) in quants.iter_mut().enumerate() {
            *quant = 0x10 + (i % 16) as u8;
        }
        let blocks = vec![create_test_q4_1_block(0.25, 0.1, quants)];

        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

        let result = backend.fused_dequant_matmul_q41(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        assert!(result[0].abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_candle_fused_q50_basic() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let k = Q4_BLOCK_SIZE;
        let n = 1;
        let batch_size = 1;

        let blocks = vec![create_test_q5_0_block(0.5)];
        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

        let result = backend.fused_dequant_matmul_q50(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        assert!(result[0].abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_candle_fused_q51_basic() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let k = Q4_BLOCK_SIZE;
        let n = 1;
        let batch_size = 1;

        let blocks = vec![create_test_q5_1_block(0.5, 0.1)];
        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

        let result = backend.fused_dequant_matmul_q51(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        assert!(result[0].abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_candle_fused_q80_basic() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let k = Q8_BLOCK_SIZE;
        let n = 1;
        let batch_size = 1;

        let blocks = vec![create_test_q8_0_block(0.25)];
        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

        let result = backend.fused_dequant_matmul_q80(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        assert!(result[0].abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_candle_fused_q81_basic() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let k = Q8_BLOCK_SIZE;
        let n = 1;
        let batch_size = 1;

        let blocks = vec![create_test_q8_1_block(0.25, 0.1)];
        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

        let result = backend.fused_dequant_matmul_q81(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        assert!(result[0].abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_candle_fused_q2k_basic() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let k = QK_K;
        let n = 1;
        let batch_size = 1;

        // Create a minimal valid Q2_K block with proper initialization
        let block = BlockQ2_K {
            d: half::f16::from_f32(0.5).to_bits(),
            dmin: half::f16::from_f32(0.1).to_bits(),
            scales: [0x44; QK_K / 16], // Simple scale pattern
            qh: [0u8; QK_K / 8],
            qs: [0x11; QK_K / 4], // Simple quant pattern
        };
        let blocks = vec![block];

        let input: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let result = backend.fused_dequant_matmul_q2k(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        Ok(())
    }

    #[test]
    fn test_candle_fused_q3k_basic() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let k = QK_K;
        let n = 1;
        let batch_size = 1;

        let blocks = vec![create_test_q3_k_block(0.5, 0.1)];
        let input: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let result = backend.fused_dequant_matmul_q3k(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        assert!(result[0].abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_candle_fused_q8k_basic() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let k = QK_K;
        let n = 1;
        let batch_size = 1;

        let mut quants = [0i8; QK_K];
        for (i, quant) in quants.iter_mut().enumerate() {
            *quant = ((i % 127) as i8) - 64;
        }
        let mut scales = [0u8; QK_K / 8];
        for (i, scale) in scales.iter_mut().enumerate() {
            *scale = (8 + i) as u8;
        }
        let block = BlockQ8_K {
            quants,
            scales,
            d: half::f16::from_f32(0.5).to_bits(),
            dmin: half::f16::from_f32(0.1).to_bits(),
        };
        let blocks = vec![block];

        let input: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let result = backend.fused_dequant_matmul_q8k(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        assert!(result[0].abs() > 1e-6);
        Ok(())
    }

    // Comparison tests between Naive and Candle backends
    #[test]
    fn test_compare_naive_candle_matmul() -> Result<()> {
        use crate::NaiveCpuBackend;

        let naive = NaiveCpuBackend::new();
        let candle = CandleCpuBackend::new()?;

        let m = 3;
        let k = 4;
        let n = 2;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();

        let naive_result = naive.matmul(&a, &b, m, k, n)?;
        let candle_result = candle.matmul(&a, &b, m, k, n)?;

        assert_eq!(naive_result.len(), candle_result.len());
        for i in 0..naive_result.len() {
            let diff = (naive_result[i] - candle_result[i]).abs();
            assert!(
                diff < 1e-4,
                "Mismatch at index {}: naive={}, candle={}",
                i,
                naive_result[i],
                candle_result[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_compare_naive_candle_fused_q80() -> Result<()> {
        use crate::NaiveCpuBackend;

        let naive = NaiveCpuBackend::new();
        let candle = CandleCpuBackend::new()?;

        let k = Q8_BLOCK_SIZE;
        let n = 2;
        let batch_size = 2;

        let mut blocks = Vec::new();
        for _ in 0..n {
            blocks.push(create_test_q8_0_block(0.25));
        }

        let input: Vec<f32> = (0..batch_size * k).map(|i| (i as f32) * 0.1).collect();

        let naive_result = naive.fused_dequant_matmul_q80(&blocks, &input, batch_size, n, k)?;
        let candle_result = candle.fused_dequant_matmul_q80(&blocks, &input, batch_size, n, k)?;

        assert_eq!(naive_result.len(), candle_result.len());
        for i in 0..naive_result.len() {
            let diff = (naive_result[i] - candle_result[i]).abs();
            let rel_error = if candle_result[i].abs() > 1e-6 {
                diff / candle_result[i].abs()
            } else {
                diff
            };
            assert!(
                rel_error < 1e-3,
                "Mismatch at index {}: naive={}, candle={}, rel_error={}",
                i,
                naive_result[i],
                candle_result[i],
                rel_error
            );
        }
        Ok(())
    }

    #[test]
    fn test_compare_naive_candle_fused_q40() -> Result<()> {
        use crate::NaiveCpuBackend;

        let naive = NaiveCpuBackend::new();
        let candle = CandleCpuBackend::new()?;

        let k = Q4_BLOCK_SIZE;
        let n = 2;
        let batch_size = 2;

        let mut blocks = Vec::new();
        for _ in 0..n {
            let mut quants = [0u8; Q4_BLOCK_SIZE / 2];
            for (j, quant) in quants.iter_mut().enumerate() {
                *quant = 0x10 + (j % 16) as u8;
            }
            blocks.push(create_test_q4_0_block(0.25, quants));
        }

        let input: Vec<f32> = (0..batch_size * k).map(|i| (i as f32) * 0.1).collect();

        let naive_result = naive.fused_dequant_matmul_q40(&blocks, &input, batch_size, n, k)?;
        let candle_result = candle.fused_dequant_matmul_q40(&blocks, &input, batch_size, n, k)?;

        assert_eq!(naive_result.len(), candle_result.len());
        for i in 0..naive_result.len() {
            let diff = (naive_result[i] - candle_result[i]).abs();
            let rel_error = if candle_result[i].abs() > 1e-6 {
                diff / candle_result[i].abs()
            } else {
                diff
            };
            assert!(
                rel_error < 1e-3,
                "Mismatch at index {}: naive={}, candle={}, rel_error={}",
                i,
                naive_result[i],
                candle_result[i],
                rel_error
            );
        }
        Ok(())
    }

    #[test]
    fn test_candle_fused_all_types_small_batch() -> Result<()> {
        let backend = CandleCpuBackend::new()?;
        let batch_size = 2;
        let n = 2;
        let k: usize = 64;

        // Test Q4_0
        let mut blocks_q40 = Vec::new();
        for _ in 0..n {
            for _ in 0..k.div_ceil(Q4_BLOCK_SIZE) {
                let mut quants = [0u8; Q4_BLOCK_SIZE / 2];
                for (j, quant) in quants.iter_mut().enumerate() {
                    *quant = 0x10 + (j % 16) as u8;
                }
                blocks_q40.push(create_test_q4_0_block(0.25, quants));
            }
        }
        let input_q40: Vec<f32> = (0..batch_size * k).map(|i| (i as f32) * 0.1).collect();
        let result_q40 =
            backend.fused_dequant_matmul_q40(&blocks_q40, &input_q40, batch_size, n, k)?;
        assert_eq!(result_q40.len(), batch_size * n);

        // Test Q8_0
        let mut blocks_q80 = Vec::new();
        for _ in 0..n {
            for _ in 0..k.div_ceil(Q8_BLOCK_SIZE) {
                blocks_q80.push(create_test_q8_0_block(0.25));
            }
        }
        let input_q80: Vec<f32> = (0..batch_size * k).map(|i| (i as f32) * 0.1).collect();
        let result_q80 =
            backend.fused_dequant_matmul_q80(&blocks_q80, &input_q80, batch_size, n, k)?;
        assert_eq!(result_q80.len(), batch_size * n);

        Ok(())
    }
}
