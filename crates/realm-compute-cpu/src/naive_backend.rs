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
