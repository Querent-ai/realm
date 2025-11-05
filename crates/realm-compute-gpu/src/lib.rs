//! WebGPU backend for wasm-chord
//!
//! Provides GPU-accelerated compute kernels using WebGPU/wgpu and Candle.

use realm_core::error::{Error, Result};
use realm_core::quant::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_K};
use wgpu::util::DeviceExt;

// Export Candle GPU backend (available with cuda/metal features)
pub mod candle_backend;
pub use candle_backend::CandleGpuBackend;

// Export GPU backend trait
pub mod gpu_backend_trait;
pub use gpu_backend_trait::GpuBackendTrait;

// Export advanced GPU features
pub mod distributed;
pub mod fused_kernels;
pub mod mixed_precision;

pub use distributed::{
    create_model_shards, DistributedConfig, DistributedCoordinator, DistributionStrategy,
    GpuDevice, ModelShardConfig, NodeInfo,
};
pub use fused_kernels::{FusedKernelConfig, Precision};
pub use mixed_precision::{MixedPrecisionConfig, PrecisionMode};

/// GPU backend for accelerated inference
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matmul_pipeline: wgpu::ComputePipeline,
}

impl GpuBackend {
    /// Initialize GPU backend
    pub async fn new() -> Result<Self> {
        // Request GPU adapter
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| Error::Runtime("Failed to find GPU adapter".to_string()))?;

        // Create device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("wasm-chord GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| Error::Runtime(format!("Failed to create device: {}", e)))?;

        // Load and compile matmul shader
        let shader_source = include_str!("matmul.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create compute pipeline
        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul pipeline"),
            layout: None,
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            device,
            queue,
            matmul_pipeline,
        })
    }

    /// Matrix multiplication: C = A @ B
    /// A: [M, K], B: [K, N], C: [M, N]
    pub fn matmul(&self, a: &[f32], b: &[f32], m: u32, k: u32, n: u32) -> Result<Vec<f32>> {
        // Validate dimensions
        if a.len() != (m * k) as usize {
            return Err(Error::InvalidShape(format!(
                "Matrix A has wrong size: expected {}, got {}",
                m * k,
                a.len()
            )));
        }
        if b.len() != (k * n) as usize {
            return Err(Error::InvalidShape(format!(
                "Matrix B has wrong size: expected {}, got {}",
                k * n,
                b.len()
            )));
        }

        // Create GPU buffers
        let a_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matrix A"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let b_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matrix B"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matrix C"),
            size: (m * n * 4) as u64, // 4 bytes per f32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dimensions uniform buffer
        let dims = [m, k, n, 0u32]; // padding for alignment
        let dims_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dimensions"),
                contents: bytemuck::cast_slice(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group_layout = self.matmul_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        // Encode and submit compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.matmul_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups: ceil(M/16) x ceil(N/16)
            let workgroups_x = m.div_ceil(16);
            let workgroups_y = n.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy result to staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging buffer"),
            size: (m * n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, (m * n * 4) as u64);

        self.queue.submit(Some(encoder.finish()));

        // Read back result
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver)
            .map_err(|_| Error::Runtime("Failed to receive buffer".to_string()))?
            .map_err(|e| Error::Runtime(format!("Failed to map buffer: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Check if GPU is available
    pub fn is_available() -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            // For WASM, we need to check if WebGPU is available in the browser
            // This is a synchronous check that doesn't require async initialization

            // Check if the WebGPU API is available
            if let Some(window) = web_sys::window() {
                let navigator = window.navigator();
                // Check if 'gpu' property exists on navigator
                return js_sys::Reflect::has(&navigator, &"gpu".into()).unwrap_or(false);
            }
            false
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // For native platforms, we can assume GPU support
            true
        }
    }
}

impl GpuBackendTrait for GpuBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: u32, k: u32, n: u32) -> Result<Vec<f32>> {
        // Call the existing matmul method
        GpuBackend::matmul(self, a, b, m, k, n)
    }
    fn fused_dequant_matmul_q4k(
        &self,
        blocks: &[BlockQ4_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q4_k, QK_K};

        // Validate inputs
        if !k.is_multiple_of(QK_K) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, QK_K
            )));
        }

        let num_blocks_per_row = k / QK_K;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q4_K blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize weights on CPU: [n, k]
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * QK_K;
                let dequant_slice = &mut dequantized_weights[dequant_offset..dequant_offset + QK_K];

                dequantize_q4_k(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q4_K dequantization failed: {}", e)))?;
            }
        }

        // Transpose weights: [n, k] -> [k, n] for matmul
        let mut weights_transposed = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                weights_transposed[j * n + i] = dequantized_weights[i * k + j];
            }
        }

        // Perform matmul on GPU: input [batch_size, k] @ weights_transposed [k, n] -> [batch_size, n]
        self.matmul(
            input,
            &weights_transposed,
            batch_size as u32,
            k as u32,
            n as u32,
        )
    }

    fn fused_dequant_matmul_q5k(
        &self,
        blocks: &[BlockQ5_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q5_k, QK_K};

        if !k.is_multiple_of(QK_K) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, QK_K
            )));
        }

        let num_blocks_per_row = k / QK_K;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q5_K blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize on CPU, then GPU matmul
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * QK_K;
                let dequant_slice = &mut dequantized_weights[dequant_offset..dequant_offset + QK_K];

                dequantize_q5_k(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q5_K dequantization failed: {}", e)))?;
            }
        }

        // Transpose and matmul
        let mut weights_transposed = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                weights_transposed[j * n + i] = dequantized_weights[i * k + j];
            }
        }

        self.matmul(
            input,
            &weights_transposed,
            batch_size as u32,
            k as u32,
            n as u32,
        )
    }

    fn fused_dequant_matmul_q6k(
        &self,
        blocks: &[BlockQ6_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q6_k, QK_K};

        if !k.is_multiple_of(QK_K) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, QK_K
            )));
        }

        let num_blocks_per_row = k / QK_K;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q6_K blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize on CPU, then GPU matmul
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * QK_K;
                let dequant_slice = &mut dequantized_weights[dequant_offset..dequant_offset + QK_K];

                dequantize_q6_k(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q6_K dequantization failed: {}", e)))?;
            }
        }

        // Transpose and matmul
        let mut weights_transposed = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                weights_transposed[j * n + i] = dequantized_weights[i * k + j];
            }
        }

        self.matmul(
            input,
            &weights_transposed,
            batch_size as u32,
            k as u32,
            n as u32,
        )
    }

    fn fused_dequant_matmul_q8k(
        &self,
        blocks: &[BlockQ8_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::{dequantize_q8_k, QK_K};

        if !k.is_multiple_of(QK_K) {
            return Err(Error::InvalidShape(format!(
                "K dimension {} must be multiple of {}",
                k, QK_K
            )));
        }

        let num_blocks_per_row = k / QK_K;
        let expected_blocks = n * num_blocks_per_row;

        if blocks.len() != expected_blocks {
            return Err(Error::InvalidShape(format!(
                "Expected {} Q8_K blocks, got {}",
                expected_blocks,
                blocks.len()
            )));
        }

        // Dequantize on CPU, then GPU matmul
        let mut dequantized_weights = vec![0.0f32; n * k];

        for out_idx in 0..n {
            for k_block in 0..num_blocks_per_row {
                let block_idx = out_idx * num_blocks_per_row + k_block;
                let block = &blocks[block_idx];

                let dequant_offset = out_idx * k + k_block * QK_K;
                let dequant_slice = &mut dequantized_weights[dequant_offset..dequant_offset + QK_K];

                dequantize_q8_k(block, dequant_slice)
                    .map_err(|e| Error::Runtime(format!("Q8_K dequantization failed: {}", e)))?;
            }
        }

        // Transpose and matmul
        let mut weights_transposed = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                weights_transposed[j * n + i] = dequantized_weights[i * k + j];
            }
        }

        self.matmul(
            input,
            &weights_transposed,
            batch_size as u32,
            k as u32,
            n as u32,
        )
    }

    fn name(&self) -> &'static str {
        "WebGPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_simple() {
        pollster::block_on(async {
            // Try to initialize GPU - skip test if unavailable
            let gpu = match GpuBackend::new().await {
                Ok(gpu) => gpu,
                Err(e) => {
                    eprintln!("⚠️  GPU not available, skipping test: {}", e);
                    return; // Skip test gracefully
                }
            };

            // 2x3 @ 3x2 = 2x2
            #[rustfmt::skip]
            let a = vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ];

            #[rustfmt::skip]
            let b = vec![
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
            ];

            let result = gpu.matmul(&a, &b, 2, 3, 2).expect("Matmul failed");

            // Expected:
            // [1*1 + 2*3 + 3*5,  1*2 + 2*4 + 3*6]   = [22, 28]
            // [4*1 + 5*3 + 6*5,  4*2 + 5*4 + 6*6]   = [49, 64]
            assert_eq!(result.len(), 4);
            assert!((result[0] - 22.0).abs() < 0.001);
            assert!((result[1] - 28.0).abs() < 0.001);
            assert!((result[2] - 49.0).abs() < 0.001);
            assert!((result[3] - 64.0).abs() < 0.001);

            // println!("✅ GPU matmul test passed!");
        });
    }
}
