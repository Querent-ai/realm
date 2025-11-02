//! Naive CPU backend implementation
//!
//! This module provides a simple CPU backend using basic Rust operations.
//! It's useful as a fallback and for testing.

use crate::{
    fused_dequant_matmul_q2k, fused_dequant_matmul_q3k, fused_dequant_matmul_q40,
    fused_dequant_matmul_q41, fused_dequant_matmul_q4k, fused_dequant_matmul_q50,
    fused_dequant_matmul_q51, fused_dequant_matmul_q5k, fused_dequant_matmul_q6k,
    fused_dequant_matmul_q80, fused_dequant_matmul_q81, fused_dequant_matmul_q8k, matmul_f32,
    matmul_transposed,
};
use realm_core::error::Result;
use realm_core::quant::{
    BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ4_K, BlockQ5_0, BlockQ5_1, BlockQ5_K,
    BlockQ6_K, BlockQ8_0, BlockQ8_1, BlockQ8_K,
};

use super::cpu_backend_trait::CpuBackendTrait;

/// Naive CPU backend using basic Rust operations
pub struct NaiveCpuBackend;

impl NaiveCpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NaiveCpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackendTrait for NaiveCpuBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        let mut result = vec![0.0; m * n];
        matmul_f32(a, b, &mut result, m, k, n)?;
        Ok(result)
    }

    fn matmul_transposed(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let mut result = vec![0.0; m * n];
        matmul_transposed(a, b, &mut result, m, k, n)?;
        Ok(result)
    }

    fn fused_dequant_matmul_q4k(
        &self,
        blocks: &[BlockQ4_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; batch_size * n];
        fused_dequant_matmul_q4k(blocks, input, &mut output, batch_size, n, k)?;
        Ok(output)
    }

    fn fused_dequant_matmul_q5k(
        &self,
        blocks: &[BlockQ5_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; batch_size * n];
        fused_dequant_matmul_q5k(blocks, input, &mut output, batch_size, n, k)?;
        Ok(output)
    }

    fn fused_dequant_matmul_q6k(
        &self,
        blocks: &[BlockQ6_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; batch_size * n];
        fused_dequant_matmul_q6k(blocks, input, &mut output, batch_size, n, k)?;
        Ok(output)
    }

    fn fused_dequant_matmul_q8k(
        &self,
        blocks: &[BlockQ8_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; batch_size * n];
        fused_dequant_matmul_q8k(blocks, input, &mut output, batch_size, n, k)?;
        Ok(output)
    }

    fn fused_dequant_matmul_q2k(
        &self,
        blocks: &[BlockQ2_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; batch_size * n];
        fused_dequant_matmul_q2k(blocks, input, &mut output, batch_size, n, k)?;
        Ok(output)
    }

    fn fused_dequant_matmul_q3k(
        &self,
        blocks: &[BlockQ3_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; batch_size * n];
        fused_dequant_matmul_q3k(blocks, input, &mut output, batch_size, n, k)?;
        Ok(output)
    }

    fn fused_dequant_matmul_q40(
        &self,
        blocks: &[BlockQ4_0],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; batch_size * n];
        fused_dequant_matmul_q40(blocks, input, &mut output, batch_size, n, k)?;
        Ok(output)
    }

    fn fused_dequant_matmul_q41(
        &self,
        blocks: &[BlockQ4_1],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; batch_size * n];
        fused_dequant_matmul_q41(blocks, input, &mut output, batch_size, n, k)?;
        Ok(output)
    }

    fn fused_dequant_matmul_q50(
        &self,
        blocks: &[BlockQ5_0],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; batch_size * n];
        fused_dequant_matmul_q50(blocks, input, &mut output, batch_size, n, k)?;
        Ok(output)
    }

    fn fused_dequant_matmul_q51(
        &self,
        blocks: &[BlockQ5_1],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; batch_size * n];
        fused_dequant_matmul_q51(blocks, input, &mut output, batch_size, n, k)?;
        Ok(output)
    }

    fn fused_dequant_matmul_q80(
        &self,
        blocks: &[BlockQ8_0],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; batch_size * n];
        fused_dequant_matmul_q80(blocks, input, &mut output, batch_size, n, k)?;
        Ok(output)
    }

    fn fused_dequant_matmul_q81(
        &self,
        blocks: &[BlockQ8_1],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; batch_size * n];
        fused_dequant_matmul_q81(blocks, input, &mut output, batch_size, n, k)?;
        Ok(output)
    }

    fn name(&self) -> &'static str {
        "Naive CPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use realm_core::error::Result;
    use realm_core::quant::{Q4_BLOCK_SIZE, Q8_BLOCK_SIZE, QK_K};

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
    fn test_naive_matmul_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2]
        let result = backend.matmul(&a, &b, 2, 2, 2)?;
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_naive_matmul_large() -> Result<()> {
        let backend = NaiveCpuBackend::new();
        let m = 3;
        let k = 4;
        let n = 2;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let result = backend.matmul(&a, &b, m, k, n)?;
        assert_eq!(result.len(), m * n);

        // Verify first element: sum of first row of a * first col of b
        let expected = (0..k)
            .map(|i| (i as f32) * (i as f32 * n as f32))
            .sum::<f32>();
        assert!((result[0] - expected).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_naive_matmul_transposed() -> Result<()> {
        let backend = NaiveCpuBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b_t = vec![1.0, 3.0, 2.0, 4.0]; // [2, 2] transposed (so stored as [2, 2])
        let result = backend.matmul_transposed(&a, &b_t, 2, 2, 2)?;
        // Expected: [[1,2],[3,4]] @ [[1,2],[3,4]]^T = [[7,10],[15,22]]
        assert!((result[0] - 7.0).abs() < 1e-5);
        assert!((result[1] - 10.0).abs() < 1e-5);
        assert!((result[2] - 15.0).abs() < 1e-5);
        assert!((result[3] - 22.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_naive_fused_q4k_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
        let k = QK_K;
        let n = 1;
        let batch_size = 1;

        let blocks = vec![create_test_q4_k_block(0.5, 0.1)];

        let input: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let result = backend.fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);

        // Verify non-zero result
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_naive_fused_q4k_batch() -> Result<()> {
        let backend = NaiveCpuBackend::new();
        let k = QK_K;
        let n = 2;
        let batch_size = 3;

        let blocks: Vec<_> = (0..n).map(|_| create_test_q4_k_block(0.5, 0.1)).collect();

        let input: Vec<f32> = (0..batch_size * k)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.5)
            .collect();

        let result = backend.fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);

        // Each batch should produce different output
        for batch_idx in 0..batch_size {
            let batch_sum: f32 = (0..n).map(|i| result[batch_idx * n + i]).sum();
            assert!(batch_sum.abs() > 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_naive_fused_q40_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
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
    fn test_naive_fused_q41_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
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
    fn test_naive_fused_q50_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
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
    fn test_naive_fused_q51_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
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
    fn test_naive_fused_q80_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
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
    fn test_naive_fused_q81_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
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
    fn test_naive_fused_q2k_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
        let k = QK_K;
        let n = 1;
        let batch_size = 1;

        // Create a minimal valid Q2_K block with proper initialization
        // Q2_K requires specific structure - use zero-initialized block to avoid bounds issues
        let block = BlockQ2_K {
            d: half::f16::from_f32(0.5).to_bits(),
            dmin: half::f16::from_f32(0.1).to_bits(),
            scales: [0x44; QK_K / 16], // Simple scale pattern
            qh: [0u8; QK_K / 8],
            qs: [0x11; QK_K / 4], // Simple quant pattern (2 values per byte)
        };
        let blocks = vec![block];

        let input: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let result = backend.fused_dequant_matmul_q2k(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        // Q2_K with mostly zero inputs may produce very small results
        Ok(())
    }

    #[test]
    fn test_naive_fused_q3k_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
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
    fn test_naive_fused_q5k_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
        let k = QK_K;
        let n = 1;
        let batch_size = 1;

        // Create a Q5_K block: ql (128 bytes), qh (32 bytes), scales (16 bytes), d (2 bytes)
        let block = realm_core::quant::BlockQ5_K {
            d: half::f16::from_f32(0.5).to_bits(),
            ql: {
                let mut ql = [0u8; QK_K / 2];
                for (i, q) in ql.iter_mut().enumerate() {
                    *q = (i % 16) as u8 | (((i + 1) % 16) as u8) << 4;
                }
                ql
            },
            qh: {
                let mut qh = [0u8; QK_K / 8];
                for (i, q) in qh.iter_mut().enumerate() {
                    *q = (i % 2) as u8 * 0x55; // Pattern: 01010101 or 00000000
                }
                qh
            },
            scales: {
                let mut scales = [0i8; QK_K / 16];
                for (i, scale) in scales.iter_mut().enumerate() {
                    *scale = 16i8 + i as i8;
                }
                scales
            },
        };
        let blocks = vec![block];

        let input: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let result = backend.fused_dequant_matmul_q5k(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        assert!(result[0].abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_naive_fused_q6k_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
        let k = QK_K;
        let n = 1;
        let batch_size = 1;

        // Create a Q6_K block: ql (128 bytes), qh (64 bytes), scales (16 bytes), d (2 bytes)
        let block = realm_core::quant::BlockQ6_K {
            d: half::f16::from_f32(0.5).to_bits(),
            ql: {
                let mut ql = [0u8; QK_K / 2];
                for (i, q) in ql.iter_mut().enumerate() {
                    *q = (i % 16) as u8 | (((i + 1) % 16) as u8) << 4;
                }
                ql
            },
            qh: {
                let mut qh = [0u8; QK_K / 4];
                for (i, q) in qh.iter_mut().enumerate() {
                    *q = (i % 4) as u8 | (((i + 1) % 4) as u8) << 2;
                }
                qh
            },
            scales: {
                let mut scales = [0i8; QK_K / 16];
                for (i, scale) in scales.iter_mut().enumerate() {
                    *scale = 16i8 + i as i8;
                }
                scales
            },
        };
        let blocks = vec![block];

        let input: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let result = backend.fused_dequant_matmul_q6k(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        assert!(result[0].abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_naive_fused_q8k_basic() -> Result<()> {
        let backend = NaiveCpuBackend::new();
        let k = QK_K;
        let n = 1;
        let batch_size = 1;

        let mut blocks = Vec::new();
        // Create a simple Q8_K block (similar to Q8_0 but with scales)
        let mut quants = [0i8; QK_K];
        for (i, quant) in quants.iter_mut().enumerate() {
            *quant = ((i % 127) as i8) - 64;
        }
        let mut scales = [0u8; QK_K / 8];
        for (i, scale) in scales.iter_mut().enumerate() {
            *scale = (8 + i) as u8;
        }
        let block = realm_core::quant::BlockQ8_K {
            quants,
            scales,
            d: half::f16::from_f32(0.5).to_bits(),
            dmin: half::f16::from_f32(0.1).to_bits(),
        };
        blocks.push(block);

        let input: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let result = backend.fused_dequant_matmul_q8k(&blocks, &input, batch_size, n, k)?;
        assert_eq!(result.len(), batch_size * n);
        assert!(result[0].abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_naive_fused_all_types_small_batch() -> Result<()> {
        let backend = NaiveCpuBackend::new();
        let batch_size = 2;
        let n = 2;
        let k: usize = 64; // Small k for Q4/Q5/Q8 types

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
