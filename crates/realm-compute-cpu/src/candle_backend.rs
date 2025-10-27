//! Candle-based CPU backend for optimized matrix operations
//!
//! This module provides optimized matrix multiplication using Candle's CPU kernels.
//! Candle uses the `gemm` crate which provides highly optimized GEMM implementations
//! in pure Rust, making it perfect for WebAssembly.

use candle_core::{Device, Tensor};
use realm_core::error::{Error, Result};

/// Candle-based matrix multiplication: C = A * B
///
/// # Arguments
/// * `a` - Matrix A [M, K] in row-major format
/// * `b` - Matrix B [K, N] in row-major format
/// * `c` - Output matrix C [M, N] in row-major format
/// * `m` - Number of rows in A
/// * `k` - Number of columns in A / rows in B
/// * `n` - Number of columns in B
pub fn matmul_f32_candle(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    // Validate input dimensions
    if a.len() != m * k {
        return Err(Error::InvalidShape(format!(
            "Matrix A size mismatch: expected {}, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(Error::InvalidShape(format!(
            "Matrix B size mismatch: expected {}, got {}",
            k * n,
            b.len()
        )));
    }
    if c.len() != m * n {
        return Err(Error::InvalidShape(format!(
            "Matrix C size mismatch: expected {}, got {}",
            m * n,
            c.len()
        )));
    }

    // Use CPU device (Candle will use optimized gemm crate)
    let device = Device::Cpu;

    // Create tensors from slices
    let a_tensor = Tensor::from_slice(a, (m, k), &device)
        .map_err(|e| Error::BackendError(format!("Failed to create tensor A: {}", e)))?;

    let b_tensor = Tensor::from_slice(b, (k, n), &device)
        .map_err(|e| Error::BackendError(format!("Failed to create tensor B: {}", e)))?;

    // Perform matrix multiplication using Candle's optimized matmul
    let c_tensor = a_tensor
        .matmul(&b_tensor)
        .map_err(|e| Error::BackendError(format!("Matmul failed: {}", e)))?;

    // Extract result as 1D vector
    let c_vec = c_tensor
        .flatten_all()
        .map_err(|e| Error::BackendError(format!("Flatten failed: {}", e)))?
        .to_vec1::<f32>()
        .map_err(|e| Error::BackendError(format!("Failed to extract result: {}", e)))?;

    c.copy_from_slice(&c_vec);
    Ok(())
}

/// Candle-based matrix multiplication with transposed B: C = A * B^T
///
/// # Arguments
/// * `a` - Matrix A [M, K] in row-major format
/// * `b_t` - Matrix B [N, K] in row-major format (will be accessed as transposed)
/// * `c` - Output matrix C [M, N] in row-major format
/// * `m` - Number of rows in A
/// * `k` - Number of columns in A (= columns in B^T)
/// * `n` - Number of rows in B (= columns in A * B^T result)
pub fn matmul_transposed_candle(
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    // Validate input dimensions
    if a.len() != m * k {
        return Err(Error::InvalidShape(format!(
            "Matrix A size mismatch: expected {}, got {}",
            m * k,
            a.len()
        )));
    }
    if b_t.len() != n * k {
        return Err(Error::InvalidShape(format!(
            "Matrix B^T size mismatch: expected {}, got {}",
            n * k,
            b_t.len()
        )));
    }
    if c.len() != m * n {
        return Err(Error::InvalidShape(format!(
            "Matrix C size mismatch: expected {}, got {}",
            m * n,
            c.len()
        )));
    }

    let device = Device::Cpu;

    // Create tensors
    let a_tensor = Tensor::from_slice(a, (m, k), &device)
        .map_err(|e| Error::BackendError(format!("Failed to create tensor A: {}", e)))?;

    // B is stored as [N, K] (row-major), we need to transpose it to [K, N]
    let b_tensor = Tensor::from_slice(b_t, (n, k), &device)
        .map_err(|e| Error::BackendError(format!("Failed to create tensor B: {}", e)))?;

    // Transpose B to get [K, N]
    let b_transposed = b_tensor
        .t()
        .map_err(|e| Error::BackendError(format!("Transpose failed: {}", e)))?;

    // Perform A * B^T (which is A * B after transpose)
    let c_tensor = a_tensor
        .matmul(&b_transposed)
        .map_err(|e| Error::BackendError(format!("Matmul failed: {}", e)))?;

    // Extract result
    let c_vec = c_tensor
        .flatten_all()
        .map_err(|e| Error::BackendError(format!("Flatten failed: {}", e)))?
        .to_vec1::<f32>()
        .map_err(|e| Error::BackendError(format!("Failed to extract result: {}", e)))?;

    c.copy_from_slice(&c_vec);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_matmul_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        matmul_f32_candle(&a, &b, &mut c, 2, 2, 2).unwrap();

        assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_candle_matmul_simple() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0; 4];

        matmul_f32_candle(&a, &b, &mut c, 2, 3, 2).unwrap();

        // Expected: [[22, 28], [49, 64]]
        assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_candle_matmul_transposed() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b_t = vec![1.0, 3.0, 2.0, 4.0];
        let mut c = vec![0.0; 4];

        matmul_transposed_candle(&a, &b_t, &mut c, 2, 2, 2).unwrap();

        // Expected: [[7, 10], [15, 22]]
        assert_eq!(c, vec![7.0, 10.0, 15.0, 22.0]);
    }
}
