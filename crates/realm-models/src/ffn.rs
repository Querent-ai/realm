//! Feed-forward network implementation

use realm_compute_cpu::{matmul_f32, matmul_transposed};
use realm_core::error::Result;

#[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
use realm_compute_gpu::GpuBackendTrait;

use super::TransformerConfig;

/// Feed-forward network layer
#[allow(dead_code)]
pub struct FeedForward {
    config: TransformerConfig,
}

#[allow(dead_code)]
impl FeedForward {
    pub fn new(config: TransformerConfig) -> Self {
        Self { config }
    }

    /// Helper: matrix multiplication with GPU/CPU fallback
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        transposed_b: bool,
        #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))] gpu: Option<
            &dyn GpuBackendTrait,
        >,
    ) -> Result<Vec<f32>> {
        // Try GPU first
        #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
        if let Some(gpu) = gpu {
            if !transposed_b {
                if let Ok(result) = gpu.matmul(a, b, m as u32, k as u32, n as u32) {
                    return Ok(result);
                }
            }
        }

        // CPU fallback
        let mut result = vec![0.0; m * n];
        if transposed_b {
            matmul_transposed(a, b, &mut result, m, k, n)?;
        } else {
            matmul_f32(a, b, &mut result, m, k, n)?;
        }
        Ok(result)
    }

    pub fn forward(
        &self,
        hidden_states: &[f32],
        weights: &FFNWeights,
        _cpu_backend: Option<&dyn realm_compute_cpu::CpuBackendTrait>,
        #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))] gpu: Option<
            &dyn GpuBackendTrait,
        >,
    ) -> Result<Vec<f32>> {
        let seq_len = hidden_states.len() / self.config.hidden_size;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        // Gate projection
        let mut gate = self.matmul(
            hidden_states,
            &weights.w_gate,
            seq_len,
            hidden_size,
            intermediate_size,
            true,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            gpu,
        )?;

        // Up projection
        let up = self.matmul(
            hidden_states,
            &weights.w_up,
            seq_len,
            hidden_size,
            intermediate_size,
            true,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            gpu,
        )?;

        // SwiGLU activation: silu(gate) * up
        // where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        for i in 0..gate.len() {
            let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
            let silu = gate[i] * sigmoid;
            gate[i] = silu * up[i];
        }

        // Down projection
        let output = self.matmul(
            &gate,
            &weights.w_down,
            seq_len,
            intermediate_size,
            hidden_size,
            true,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            gpu,
        )?;

        Ok(output)
    }
}

/// FFN weight matrices
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FFNWeights {
    /// Gate projection [hidden_size, intermediate_size]
    pub w_gate: Vec<f32>,
    /// Up projection [hidden_size, intermediate_size]
    pub w_up: Vec<f32>,
    /// Down projection [intermediate_size, hidden_size]
    pub w_down: Vec<f32>,
}

#[allow(dead_code)]
impl FFNWeights {
    pub fn new(config: &TransformerConfig) -> Self {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        Self {
            w_gate: vec![0.0; hidden_size * intermediate_size],
            w_up: vec![0.0; hidden_size * intermediate_size],
            w_down: vec![0.0; intermediate_size * hidden_size],
        }
    }
}
