//! CPU backend trait for unified CPU operations
//!
//! This trait defines the interface for CPU backends that support
//! matrix multiplication and other CPU operations.

use realm_core::error::Result;
use realm_core::quant::{
    BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ4_K, BlockQ5_0, BlockQ5_1, BlockQ5_K,
    BlockQ6_K, BlockQ8_0, BlockQ8_1, BlockQ8_K,
};

/// Trait for CPU backends that support matrix multiplication and fused operations
pub trait CpuBackendTrait: Send + Sync {
    /// Basic matrix multiplication: C = A @ B
    /// A: [M, K], B: [K, N] -> C: [M, N]
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>>;

    /// Transposed matrix multiplication: C = A @ B^T
    /// A: [M, K], B: [N, K] -> C: [M, N]
    fn matmul_transposed(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>>;

    /// Fused Q4_K dequantization + matrix multiplication
    fn fused_dequant_matmul_q4k(
        &self,
        blocks: &[BlockQ4_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>>;

    /// Fused Q5_K dequantization + matrix multiplication
    fn fused_dequant_matmul_q5k(
        &self,
        blocks: &[BlockQ5_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>>;

    /// Fused Q6_K dequantization + matrix multiplication
    fn fused_dequant_matmul_q6k(
        &self,
        blocks: &[BlockQ6_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>>;

    /// Fused Q8_K dequantization + matrix multiplication
    fn fused_dequant_matmul_q8k(
        &self,
        blocks: &[BlockQ8_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>>;

    /// Fused Q2_K dequantization + matrix multiplication
    fn fused_dequant_matmul_q2k(
        &self,
        blocks: &[BlockQ2_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>>;

    /// Fused Q3_K dequantization + matrix multiplication
    fn fused_dequant_matmul_q3k(
        &self,
        blocks: &[BlockQ3_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>>;

    /// Fused Q4_0 dequantization + matrix multiplication
    fn fused_dequant_matmul_q40(
        &self,
        blocks: &[BlockQ4_0],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>>;

    /// Fused Q4_1 dequantization + matrix multiplication
    fn fused_dequant_matmul_q41(
        &self,
        blocks: &[BlockQ4_1],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>>;

    /// Fused Q5_0 dequantization + matrix multiplication
    fn fused_dequant_matmul_q50(
        &self,
        blocks: &[BlockQ5_0],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>>;

    /// Fused Q5_1 dequantization + matrix multiplication
    fn fused_dequant_matmul_q51(
        &self,
        blocks: &[BlockQ5_1],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>>;

    /// Fused Q8_0 dequantization + matrix multiplication
    fn fused_dequant_matmul_q80(
        &self,
        blocks: &[BlockQ8_0],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>>;

    /// Fused Q8_1 dequantization + matrix multiplication
    fn fused_dequant_matmul_q81(
        &self,
        blocks: &[BlockQ8_1],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>>;

    /// Backend name for debugging
    fn name(&self) -> &'static str;
}
