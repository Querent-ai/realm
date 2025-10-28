//! Matmul dispatch for fused kernels
//!
//! This module provides intelligent dispatching to fused dequantization + matmul kernels
//! based on the weight format, enabling 2-4x speedups for quantized models.
//! Supports both CPU and GPU backends with automatic fallback.

use realm_compute_cpu::{
    fused_dequant_matmul_q4k, fused_dequant_matmul_q5k, fused_dequant_matmul_q6k,
    fused_dequant_matmul_q8k, matmul_transposed, CpuBackendTrait,
};
use realm_core::error::Result;

#[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
use realm_gpu::GpuBackendTrait;

use crate::weight_format::WeightFormat;

/// Dispatch matmul to appropriate kernel based on weight format and available backends
///
/// This function automatically selects the optimal implementation:
/// - **Quantized formats (Q4_K/Q5_K/Q6_K/Q8_K):** Fused dequant + matmul (2-4x faster)
/// - **F32:** Standard matmul
/// - **GPU backends:** When available, uses GPU acceleration
/// - **CPU backends:** Fallback or primary for CPU-only builds
/// - **Automatic fallback:** CPU if GPU fails or unavailable
///
/// # Arguments
/// * `input` - Input activations [batch_size, k]
/// * `weights` - Weight matrix (any format) [n, k] (stored transposed)
/// * `batch_size` - Batch size (m)
/// * `k` - Input dimension
/// * `n` - Output dimension
/// * `cpu_backend` - Optional CPU backend (NaiveCpuBackend, CandleCpuBackend, etc.)
/// * `gpu_backend` - Optional GPU backend (WebGPU/CUDA/Metal)
///
/// # Returns
/// Output activations [batch_size, n]
///
/// # Performance
/// - Q4_K CPU: ~8.6x faster than naive (measured)
/// - Q4_K GPU: ~10-20x faster (expected)
/// - Q5_K/Q6_K/Q8_K: ~2-4x faster
/// - F32: Baseline performance
pub fn dispatch_matmul(
    input: &[f32],
    weights: &WeightFormat,
    batch_size: usize,
    k: usize,
    n: usize,
    cpu_backend: Option<&dyn CpuBackendTrait>,
    #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))] gpu_backend: Option<
        &dyn GpuBackendTrait,
    >,
) -> Result<Vec<f32>> {
    // Debug: Log which path is taken
    if std::env::var("DEBUG_DISPATCH").is_ok() {
        eprintln!(
            "[dispatch_matmul] format={}, shape=[{}, {}, {}]",
            weights.format_name(),
            batch_size,
            k,
            n
        );
    }

    match weights {
        WeightFormat::F32(w) => {
            // Try CPU backend first, fallback to direct implementation
            if let Some(cpu) = cpu_backend {
                return cpu.matmul_transposed(input, w, batch_size, k, n);
            }

            // Direct implementation fallback
            let mut output = vec![0.0f32; batch_size * n];
            matmul_transposed(input, w, &mut output, batch_size, k, n)?;
            Ok(output)
        }
        WeightFormat::Q4K(blocks) => {
            // Try GPU first, fallback to CPU
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            if let Some(gpu) = gpu_backend {
                if let Ok(result) = gpu.fused_dequant_matmul_q4k(blocks, input, batch_size, n, k) {
                    return Ok(result);
                }
            }

            // Try CPU backend, fallback to direct implementation
            if let Some(cpu) = cpu_backend {
                return cpu.fused_dequant_matmul_q4k(blocks, input, batch_size, n, k);
            }

            // Direct implementation fallback
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q4k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q5K(blocks) => {
            // Try GPU first, fallback to CPU
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            if let Some(gpu) = gpu_backend {
                if let Ok(result) = gpu.fused_dequant_matmul_q5k(blocks, input, batch_size, n, k) {
                    return Ok(result);
                }
            }

            // Try CPU backend, fallback to direct implementation
            if let Some(cpu) = cpu_backend {
                return cpu.fused_dequant_matmul_q5k(blocks, input, batch_size, n, k);
            }

            // Direct implementation fallback
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q5k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q6K(blocks) => {
            // Try GPU first, fallback to CPU
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            if let Some(gpu) = gpu_backend {
                if let Ok(result) = gpu.fused_dequant_matmul_q6k(blocks, input, batch_size, n, k) {
                    return Ok(result);
                }
            }

            // Try CPU backend, fallback to direct implementation
            if let Some(cpu) = cpu_backend {
                return cpu.fused_dequant_matmul_q6k(blocks, input, batch_size, n, k);
            }

            // Direct implementation fallback
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q6k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q8K(blocks) => {
            // Try GPU first, fallback to CPU
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            if let Some(gpu) = gpu_backend {
                if let Ok(result) = gpu.fused_dequant_matmul_q8k(blocks, input, batch_size, n, k) {
                    return Ok(result);
                }
            }

            // Try CPU backend, fallback to direct implementation
            if let Some(cpu) = cpu_backend {
                return cpu.fused_dequant_matmul_q8k(blocks, input, batch_size, n, k);
            }

            // Direct implementation fallback
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q8k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_f32() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = WeightFormat::F32(vec![0.5; 8]); // [2, 4] transposed

        let output = dispatch_matmul(
            &input,
            &weights,
            1,
            4,
            2,
            None,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_dispatch_q4k() {
        let input = vec![0.5f32; 256];
        // Create 256 Q4_K blocks (one per output element)
        let block = realm_core::quant::BlockQ4_K {
            d: 0u16,           // f16 scale as u16
            dmin: 0u16,        // f16 min scale as u16
            scales: [0u8; 12], // Quantized scales
            qs: [0u8; 128],    // 4-bit quants (256/2 = 128 bytes)
        };
        let blocks = vec![block; 256]; // Need 256 blocks for 256x256 matmul
        let weights = WeightFormat::Q4K(blocks);

        let output = dispatch_matmul(
            &input,
            &weights,
            1,
            256,
            256,
            None,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();
        assert_eq!(output.len(), 256);
    }
}
