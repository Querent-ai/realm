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
        _blocks: &[BlockQ4_K],
        _input: &[f32],
        _batch_size: usize,
        _n: usize,
        _k: usize,
    ) -> Result<Vec<f32>> {
        // TODO: Implement GPU fused Q4_K kernel
        // For now, fall back to CPU implementation
        Err(Error::Runtime(
            "GPU Q4_K fused kernel not implemented yet".to_string(),
        ))
    }

    fn fused_dequant_matmul_q5k(
        &self,
        _blocks: &[BlockQ5_K],
        _input: &[f32],
        _batch_size: usize,
        _n: usize,
        _k: usize,
    ) -> Result<Vec<f32>> {
        // TODO: Implement GPU fused Q5_K kernel
        Err(Error::Runtime(
            "GPU Q5_K fused kernel not implemented yet".to_string(),
        ))
    }

    fn fused_dequant_matmul_q6k(
        &self,
        _blocks: &[BlockQ6_K],
        _input: &[f32],
        _batch_size: usize,
        _n: usize,
        _k: usize,
    ) -> Result<Vec<f32>> {
        // TODO: Implement GPU fused Q6_K kernel
        Err(Error::Runtime(
            "GPU Q6_K fused kernel not implemented yet".to_string(),
        ))
    }

    fn fused_dequant_matmul_q8k(
        &self,
        _blocks: &[BlockQ8_K],
        _input: &[f32],
        _batch_size: usize,
        _n: usize,
        _k: usize,
    ) -> Result<Vec<f32>> {
        // TODO: Implement GPU fused Q8_K kernel
        Err(Error::Runtime(
            "GPU Q8_K fused kernel not implemented yet".to_string(),
        ))
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
