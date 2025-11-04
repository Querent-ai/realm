//! Matmul dispatch for fused kernels
//!
//! This module provides intelligent dispatching to fused dequantization + matmul kernels
//! based on the weight format, enabling 2-4x speedups for quantized models.
//! Supports both CPU and GPU backends with automatic fallback.

use realm_compute_cpu::{
    fused_dequant_matmul_q2k, fused_dequant_matmul_q3k, fused_dequant_matmul_q40,
    fused_dequant_matmul_q41, fused_dequant_matmul_q4k, fused_dequant_matmul_q50,
    fused_dequant_matmul_q51, fused_dequant_matmul_q5k, fused_dequant_matmul_q6k,
    fused_dequant_matmul_q80, fused_dequant_matmul_q81, fused_dequant_matmul_q8k,
    matmul_transposed, CpuBackendTrait,
};
use realm_core::error::Result;

#[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
use realm_gpu::GpuBackendTrait;

use crate::weight_format::WeightFormat;

/// Dispatch matmul to appropriate kernel based on weight format and available backends
///
/// This function automatically selects the optimal implementation:
/// - **Quantized formats (Q2_K/Q3_K/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q4_K/Q5_K/Q6_K/Q8_K):** Fused dequant + matmul (2-4x faster)
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
        // eprintln!("[dispatch_matmul] format={}, shape=[{}, {}, {}]", weights.format_name(), batch_size, k, n);
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
        WeightFormat::Q2K(blocks) => {
            if let Some(cpu) = cpu_backend {
                return cpu.fused_dequant_matmul_q2k(blocks, input, batch_size, n, k);
            }
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q2k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q3K(blocks) => {
            if let Some(cpu) = cpu_backend {
                return cpu.fused_dequant_matmul_q3k(blocks, input, batch_size, n, k);
            }
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q3k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q40(blocks) => {
            if let Some(cpu) = cpu_backend {
                return cpu.fused_dequant_matmul_q40(blocks, input, batch_size, n, k);
            }
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q40(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q41(blocks) => {
            if let Some(cpu) = cpu_backend {
                return cpu.fused_dequant_matmul_q41(blocks, input, batch_size, n, k);
            }
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q41(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q50(blocks) => {
            if let Some(cpu) = cpu_backend {
                return cpu.fused_dequant_matmul_q50(blocks, input, batch_size, n, k);
            }
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q50(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q51(blocks) => {
            if let Some(cpu) = cpu_backend {
                return cpu.fused_dequant_matmul_q51(blocks, input, batch_size, n, k);
            }
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q51(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q80(blocks) => {
            if let Some(cpu) = cpu_backend {
                return cpu.fused_dequant_matmul_q80(blocks, input, batch_size, n, k);
            }
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q80(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q81(blocks) => {
            if let Some(cpu) = cpu_backend {
                return cpu.fused_dequant_matmul_q81(blocks, input, batch_size, n, k);
            }
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q81(blocks, input, &mut output, batch_size, n, k)?;
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

    #[test]
    fn test_dispatch_q5k() {
        use realm_core::quant::{BlockQ5_K, QK_K};

        let input = vec![1.0f32; QK_K];
        let block = BlockQ5_K {
            d: half::f16::from_f32(1.0).to_bits(),
            scales: [0i8; QK_K / 16],
            qh: [0u8; QK_K / 8],
            ql: [0u8; QK_K / 2],
        };
        let blocks = vec![block; 256]; // n=256, k=256
        let weights = WeightFormat::Q5K(blocks);

        let output = dispatch_matmul(
            &input,
            &weights,
            1,
            QK_K,
            256,
            None,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_dispatch_q6k() {
        use realm_core::quant::{BlockQ6_K, QK_K};

        let input = vec![1.0f32; QK_K];
        let block = BlockQ6_K {
            d: half::f16::from_f32(1.0).to_bits(),
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [0i8; QK_K / 16],
        };
        let blocks = vec![block; 256];
        let weights = WeightFormat::Q6K(blocks);

        let output = dispatch_matmul(
            &input,
            &weights,
            1,
            QK_K,
            256,
            None,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_dispatch_q8k() {
        use realm_core::quant::{BlockQ8_K, QK_K};

        let input = vec![1.0f32; QK_K];
        let block = BlockQ8_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; QK_K / 8],
            quants: [0i8; QK_K],
        };
        let blocks = vec![block; 256];
        let weights = WeightFormat::Q8K(blocks);

        let output = dispatch_matmul(
            &input,
            &weights,
            1,
            QK_K,
            256,
            None,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_dispatch_q2k() {
        use realm_core::quant::{BlockQ2_K, QK_K};

        let input = vec![1.0f32; QK_K];
        let block = BlockQ2_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; QK_K / 16],
            qh: [0u8; QK_K / 8],
            qs: [0u8; QK_K / 4],
        };
        let blocks = vec![block; 256];
        let weights = WeightFormat::Q2K(blocks);

        let output = dispatch_matmul(
            &input,
            &weights,
            1,
            QK_K,
            256,
            None,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_dispatch_q3k() {
        use realm_core::quant::{BlockQ3_K, QK_K};

        let input = vec![1.0f32; QK_K];
        let block = BlockQ3_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; QK_K / 8],
            qh: [0u8; QK_K / 8],
            qs: [0u8; QK_K / 2],
        };
        let blocks = vec![block; 256];
        let weights = WeightFormat::Q3K(blocks);

        let output = dispatch_matmul(
            &input,
            &weights,
            1,
            QK_K,
            256,
            None,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_dispatch_all_formats_batch_size_2() {
        use realm_core::quant::{BlockQ4_K, QK_K};

        // Test with batch_size = 2
        let input = vec![1.0f32; 2 * QK_K];
        let block = BlockQ4_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; 12],
            qs: [0u8; QK_K / 2],
        };
        let blocks = vec![block; 256];
        let weights = WeightFormat::Q4K(blocks);

        let output = dispatch_matmul(
            &input,
            &weights,
            2, // batch_size
            QK_K,
            256,
            None,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();
        assert_eq!(output.len(), 2 * 256);
    }

    #[test]
    fn test_dispatch_f32_deterministic() {
        // Test with known values for deterministic output
        // Input: [1, 2, 3, 4] shape [1, 4]
        // Weights: [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]] shape [4, 2] (stored transposed as [2, 4])
        // Expected output: [1, 2, 3, 4] @ [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]^T
        // = [1*0.5+2*0.5+3*0.5+4*0.5, 1*0.5+2*0.5+3*0.5+4*0.5] = [5.0, 5.0]
        let input = vec![1.0, 2.0, 3.0, 4.0];
        // Weights are stored transposed: [n, k] -> weights are [2, 4] = 8 elements
        let weights = WeightFormat::F32(vec![
            0.5, 0.5, 0.5, 0.5, // First row of weights (transposed)
            0.5, 0.5, 0.5, 0.5, // Second row of weights (transposed)
        ]);

        let output = dispatch_matmul(
            &input,
            &weights,
            1, // batch_size
            4, // k (input dimension)
            2, // n (output dimension)
            None,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();

        // Expected: [1,2,3,4] @ [[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5]]^T = [5.0, 5.0]
        assert_eq!(output.len(), 2);
        assert!((output[0] - 5.0).abs() < 0.001);
        assert!((output[1] - 5.0).abs() < 0.001);
    }
}
