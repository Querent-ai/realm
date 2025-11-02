//! CPU backend for wasm-chord
//!
//! Provides SIMD-accelerated kernels for tensor operations on CPU.

pub mod candle_backend;
pub mod candle_cpu_backend;
pub mod candle_neural_ops_backend;
pub mod cpu_backend_trait;
pub mod fused;
pub mod gemm;
pub mod kernels;
pub mod naive_backend;

pub use candle_backend::{matmul_f32_candle, matmul_transposed_candle};
pub use candle_cpu_backend::CandleCpuBackend;
pub use candle_neural_ops_backend::CandleNeuralOpsBackend;
pub use cpu_backend_trait::CpuBackendTrait;
pub use naive_backend::NaiveCpuBackend;

// Re-export CandleGpuBackend from wasm-chord-gpu for convenience
// This allows code to use `realm_cpu::CandleGpuBackend` while the implementation lives in the GPU package
pub use fused::{
    fused_attention_score, fused_dequant_matmul_q2k, fused_dequant_matmul_q3k,
    fused_dequant_matmul_q40, fused_dequant_matmul_q41, fused_dequant_matmul_q4k,
    fused_dequant_matmul_q50, fused_dequant_matmul_q51, fused_dequant_matmul_q5k,
    fused_dequant_matmul_q6k, fused_dequant_matmul_q80, fused_dequant_matmul_q81,
    fused_dequant_matmul_q8k, fused_rmsnorm_linear, fused_swiglu_proj,
};
pub use gemm::{matmul_f32, matmul_transposed};
#[cfg(feature = "gpu")]
pub use realm_gpu::CandleGpuBackend;

use realm_core::error::Result;

/// CPU backend configuration
#[derive(Debug, Clone)]
pub struct CpuBackend {
    /// Number of threads to use (0 = auto-detect)
    pub num_threads: usize,
    /// Enable SIMD optimizations
    pub use_simd: bool,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self {
            num_threads: 0, // Auto-detect
            use_simd: true,
        }
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    pub fn init(&self) -> Result<()> {
        // Initialize rayon thread pool if specified
        if self.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_threads)
                .build_global()
                .map_err(|e| realm_core::error::Error::BackendError(e.to_string()))?;
        }
        Ok(())
    }
}
