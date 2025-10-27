//! CPU backend trait for unified CPU operations
//!
//! This trait defines the interface for CPU backends that support
//! matrix multiplication and other CPU operations.

use realm_core::error::Result;
use realm_core::quant::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_K};

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

    /// Backend name for debugging
    fn name(&self) -> &'static str;
}
