// Backend trait for abstraction between CPU and GPU implementations

use realm_core::error::Result;

/// Backend trait for compute operations
pub trait Backend: Send + Sync {
    /// Matrix multiplication: C = A @ B
    /// A: [M, K], B: [K, N] -> C: [M, N]
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>>;

    /// RMS Normalization with weight scaling
    fn rmsnorm(&self, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>>;

    /// Softmax operation
    fn softmax(&self, logits: &mut [f32], dim_size: usize) -> Result<()>;

    /// Rotary Position Embedding
    fn rope(
        &self,
        tensor: &mut [f32],
        freqs_cos: &[f32],
        freqs_sin: &[f32],
        n_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> Result<()>;

    /// Element-wise operations
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;
    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;
    fn silu(&self, x: &[f32]) -> Result<Vec<f32>>;

    /// Backend name for debugging
    fn name(&self) -> &'static str;

    /// Check if backend is available
    fn is_available() -> bool
    where
        Self: Sized;
}

/// Backend selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Use CPU backend (always available)
    CPU,
    /// Use GPU backend (WebGPU/wgpu)
    GPU,
    /// Automatically select best available
    Auto,
}

impl BackendType {
    pub fn select() -> Self {
        #[cfg(any(feature = "cuda", feature = "metal", feature = "webgpu"))]
        {
            if crate::GpuBackend::is_available() {
                return Self::GPU;
            }
        }
        Self::CPU
    }
}
