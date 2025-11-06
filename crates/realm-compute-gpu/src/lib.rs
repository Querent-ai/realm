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
pub mod advanced_features_integration;
pub mod distributed;
pub mod fused_kernels;
pub mod mixed_precision;

pub use advanced_features_integration::{init_advanced_features, AdvancedGpuConfig};
pub use distributed::{
    create_model_shards, DistributedConfig, DistributedCoordinator, DistributionStrategy,
    GpuDevice, ModelShardConfig, NodeInfo,
};
pub use fused_kernels::{FusedKernelConfig, Precision};
pub use mixed_precision::{MixedPrecisionConfig, PrecisionMode};

/// GPU backend for accelerated inference using WebGPU/wgpu.
///
/// # Thread Safety
///
/// This struct implements `Send + Sync` via `unsafe impl` to work with `Arc<dyn GpuBackendTrait>`.
/// However, the underlying `wgpu` types (`Device`, `Queue`, `ComputePipeline`) are not `Send + Sync`
/// because the WebGPU spec hasn't finalized multi-threading support.
///
/// **Important Usage Constraints:**
/// - Create `GpuBackend` on the thread where it will be used (or before Arc sharing)
/// - Only share via `Arc<dyn GpuBackendTrait>`, never move directly between threads
/// - All wgpu operations are protected by `Mutex` for exclusive access
///
/// # Safety
///
/// This is safe because:
/// 1. **WASM**: Single-threaded execution, so no cross-thread movement occurs
/// 2. **Native**: Only `Arc` pointers move between threads, not the wgpu types themselves
/// 3. **Mutex**: Provides synchronization for concurrent access to wgpu resources
///
/// See the `unsafe impl Send + Sync` documentation below for detailed safety guarantees.
pub struct GpuBackend {
    device: std::sync::Mutex<wgpu::Device>,
    queue: std::sync::Mutex<wgpu::Queue>,
    matmul_pipeline: std::sync::Mutex<wgpu::ComputePipeline>,
    q4k_pipeline: std::sync::Mutex<wgpu::ComputePipeline>,
    q5k_pipeline: std::sync::Mutex<wgpu::ComputePipeline>,
    q6k_pipeline: std::sync::Mutex<wgpu::ComputePipeline>,
    q8k_pipeline: std::sync::Mutex<wgpu::ComputePipeline>,
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

        // Load and compile dequant shaders
        let q4k_shader_source = include_str!("q4k.wgsl");
        let q4k_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("q4k shader"),
            source: wgpu::ShaderSource::Wgsl(q4k_shader_source.into()),
        });
        let q4k_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q4k pipeline"),
            layout: None,
            module: &q4k_shader,
            entry_point: "main",
        });

        let q5k_shader_source = include_str!("q5k.wgsl");
        let q5k_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("q5k shader"),
            source: wgpu::ShaderSource::Wgsl(q5k_shader_source.into()),
        });
        let q5k_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q5k pipeline"),
            layout: None,
            module: &q5k_shader,
            entry_point: "main",
        });

        let q6k_shader_source = include_str!("q6k.wgsl");
        let q6k_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("q6k shader"),
            source: wgpu::ShaderSource::Wgsl(q6k_shader_source.into()),
        });
        let q6k_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q6k pipeline"),
            layout: None,
            module: &q6k_shader,
            entry_point: "main",
        });

        let q8k_shader_source = include_str!("q8k.wgsl");
        let q8k_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("q8k shader"),
            source: wgpu::ShaderSource::Wgsl(q8k_shader_source.into()),
        });
        let q8k_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q8k pipeline"),
            layout: None,
            module: &q8k_shader,
            entry_point: "main",
        });

        Ok(Self {
            device: std::sync::Mutex::new(device),
            queue: std::sync::Mutex::new(queue),
            matmul_pipeline: std::sync::Mutex::new(matmul_pipeline),
            q4k_pipeline: std::sync::Mutex::new(q4k_pipeline),
            q5k_pipeline: std::sync::Mutex::new(q5k_pipeline),
            q6k_pipeline: std::sync::Mutex::new(q6k_pipeline),
            q8k_pipeline: std::sync::Mutex::new(q8k_pipeline),
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

        // Lock device for thread-safe access
        let device = self
            .device
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU device: {}", e)))?;

        // Create GPU buffers
        let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matrix A"),
            contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matrix B"),
            contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let c_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matrix C"),
            size: (m * n * 4) as u64, // 4 bytes per f32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dimensions uniform buffer
        let dims = [m, k, n, 0u32]; // padding for alignment
        let dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dimensions"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let pipeline = self
            .matmul_pipeline
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU pipeline: {}", e)))?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups: ceil(M/16) x ceil(N/16)
            let workgroups_x = m.div_ceil(16);
            let workgroups_y = n.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy result to staging buffer
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging buffer"),
            size: (m * n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, (m * n * 4) as u64);

        let queue = self
            .queue
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU queue: {}", e)))?;
        queue.submit(Some(encoder.finish()));

        // Read back result
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver)
            .map_err(|_| Error::Runtime("Failed to receive buffer".to_string()))?
            .map_err(|e| Error::Runtime(format!("Failed to map buffer: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// GPU-native fused dequantization + matmul for Q4_K
    fn fused_dequant_matmul_q4k_gpu(
        &self,
        blocks: &[BlockQ4_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::QK_K;

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

        if input.len() != batch_size * k {
            return Err(Error::InvalidShape(format!(
                "Input size mismatch: expected {}, got {}",
                batch_size * k,
                input.len()
            )));
        }

        let device = self
            .device
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU device: {}", e)))?;

        // Upload blocks to GPU (convert f16 to f32 for d and dmin)
        let blocks_bytes: Vec<u8> = blocks
            .iter()
            .flat_map(|block| {
                // Convert BlockQ4_K to bytes, converting f16 to f32
                let d_f32 = half::f16::from_bits(block.d).to_f32();
                let dmin_f32 = half::f16::from_bits(block.dmin).to_f32();
                let mut bytes = Vec::with_capacity(std::mem::size_of::<BlockQ4_K>());
                bytes.extend_from_slice(bytemuck::bytes_of(&d_f32));
                bytes.extend_from_slice(bytemuck::bytes_of(&dmin_f32));
                bytes.extend_from_slice(bytemuck::cast_slice(&block.scales));
                bytes.extend_from_slice(bytemuck::cast_slice(&block.qs));
                bytes
            })
            .collect();

        let blocks_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q4k blocks"),
            contents: &blocks_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q4k input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q4k output"),
            size: (batch_size * n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Params uniform buffer
        let num_k_blocks = num_blocks_per_row as u32;
        let params = [
            batch_size as u32,
            n as u32,
            k as u32,
            num_k_blocks,
            0u32, // padding
        ];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q4k params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let pipeline = self
            .q4k_pipeline
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU pipeline: {}", e)))?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("q4k bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: blocks_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q4k encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("q4k pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: ceil(n/16) x ceil(batch_size/16)
            let workgroups_x = (n as u32).div_ceil(16);
            let workgroups_y = (batch_size as u32).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy result to staging buffer
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q4k staging"),
            size: (batch_size * n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (batch_size * n * 4) as u64,
        );

        let queue = self
            .queue
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU queue: {}", e)))?;
        queue.submit(Some(encoder.finish()));

        // Read back result
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver)
            .map_err(|_| Error::Runtime("Failed to receive buffer".to_string()))?
            .map_err(|e| Error::Runtime(format!("Failed to map buffer: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// GPU-native fused dequantization + matmul for Q5_K
    fn fused_dequant_matmul_q5k_gpu(
        &self,
        blocks: &[BlockQ5_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::QK_K;

        // Similar to Q4_K but with Q5_K block structure
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

        if input.len() != batch_size * k {
            return Err(Error::InvalidShape(format!(
                "Input size mismatch: expected {}, got {}",
                batch_size * k,
                input.len()
            )));
        }

        let device = self
            .device
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU device: {}", e)))?;

        // Upload blocks (convert f16 d to f32)
        let blocks_bytes: Vec<u8> = blocks
            .iter()
            .flat_map(|block| {
                let d_f32 = half::f16::from_bits(block.d).to_f32();
                let mut bytes = Vec::with_capacity(std::mem::size_of::<BlockQ5_K>());
                bytes.extend_from_slice(bytemuck::cast_slice(&block.ql));
                bytes.extend_from_slice(bytemuck::cast_slice(&block.qh));
                bytes.extend_from_slice(bytemuck::cast_slice(&block.scales));
                bytes.extend_from_slice(bytemuck::bytes_of(&d_f32));
                bytes
            })
            .collect();

        let blocks_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q5k blocks"),
            contents: &blocks_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q5k input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q5k output"),
            size: (batch_size * n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let num_k_blocks = num_blocks_per_row as u32;
        let params = [batch_size as u32, n as u32, k as u32, num_k_blocks, 0u32];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q5k params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pipeline = self
            .q5k_pipeline
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU pipeline: {}", e)))?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("q5k bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: blocks_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q5k encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("q5k pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (n as u32).div_ceil(16);
            let workgroups_y = (batch_size as u32).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q5k staging"),
            size: (batch_size * n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (batch_size * n * 4) as u64,
        );

        let queue = self
            .queue
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU queue: {}", e)))?;
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver)
            .map_err(|_| Error::Runtime("Failed to receive buffer".to_string()))?
            .map_err(|e| Error::Runtime(format!("Failed to map buffer: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// GPU-native fused dequantization + matmul for Q6_K
    fn fused_dequant_matmul_q6k_gpu(
        &self,
        blocks: &[BlockQ6_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::QK_K;

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

        if input.len() != batch_size * k {
            return Err(Error::InvalidShape(format!(
                "Input size mismatch: expected {}, got {}",
                batch_size * k,
                input.len()
            )));
        }

        let device = self
            .device
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU device: {}", e)))?;

        let blocks_bytes: Vec<u8> = blocks
            .iter()
            .flat_map(|block| {
                let d_f32 = half::f16::from_bits(block.d).to_f32();
                let mut bytes = Vec::with_capacity(std::mem::size_of::<BlockQ6_K>());
                bytes.extend_from_slice(bytemuck::cast_slice(&block.ql));
                bytes.extend_from_slice(bytemuck::cast_slice(&block.qh));
                bytes.extend_from_slice(bytemuck::cast_slice(&block.scales));
                bytes.extend_from_slice(bytemuck::bytes_of(&d_f32));
                bytes
            })
            .collect();

        let blocks_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q6k blocks"),
            contents: &blocks_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q6k input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q6k output"),
            size: (batch_size * n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let num_k_blocks = num_blocks_per_row as u32;
        let params = [batch_size as u32, n as u32, k as u32, num_k_blocks, 0u32];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q6k params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pipeline = self
            .q6k_pipeline
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU pipeline: {}", e)))?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("q6k bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: blocks_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q6k encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("q6k pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (n as u32).div_ceil(16);
            let workgroups_y = (batch_size as u32).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q6k staging"),
            size: (batch_size * n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (batch_size * n * 4) as u64,
        );

        let queue = self
            .queue
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU queue: {}", e)))?;
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver)
            .map_err(|_| Error::Runtime("Failed to receive buffer".to_string()))?
            .map_err(|e| Error::Runtime(format!("Failed to map buffer: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// GPU-native fused dequantization + matmul for Q8_K
    fn fused_dequant_matmul_q8k_gpu(
        &self,
        blocks: &[BlockQ8_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use realm_core::quant::QK_K;

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

        if input.len() != batch_size * k {
            return Err(Error::InvalidShape(format!(
                "Input size mismatch: expected {}, got {}",
                batch_size * k,
                input.len()
            )));
        }

        let device = self
            .device
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU device: {}", e)))?;

        let blocks_bytes: Vec<u8> = blocks
            .iter()
            .flat_map(|block| {
                let d_f32 = half::f16::from_bits(block.d).to_f32();
                let dmin_f32 = half::f16::from_bits(block.dmin).to_f32();
                let mut bytes = Vec::with_capacity(std::mem::size_of::<BlockQ8_K>());
                bytes.extend_from_slice(bytemuck::cast_slice(&block.quants));
                bytes.extend_from_slice(bytemuck::cast_slice(&block.scales));
                bytes.extend_from_slice(bytemuck::bytes_of(&d_f32));
                bytes.extend_from_slice(bytemuck::bytes_of(&dmin_f32));
                bytes
            })
            .collect();

        let blocks_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q8k blocks"),
            contents: &blocks_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q8k input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q8k output"),
            size: (batch_size * n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let num_k_blocks = num_blocks_per_row as u32;
        let params = [batch_size as u32, n as u32, k as u32, num_k_blocks, 0u32];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q8k params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pipeline = self
            .q8k_pipeline
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU pipeline: {}", e)))?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("q8k bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: blocks_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q8k encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("q8k pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (n as u32).div_ceil(16);
            let workgroups_y = (batch_size as u32).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q8k staging"),
            size: (batch_size * n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (batch_size * n * 4) as u64,
        );

        let queue = self
            .queue
            .lock()
            .map_err(|e| Error::Runtime(format!("Failed to lock GPU queue: {}", e)))?;
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
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

// SAFETY: We implement Send + Sync for GpuBackend despite containing non-Send wgpu types.
//
// Safety Guarantees:
// 1. WASM contexts: Single-threaded execution, so no actual cross-thread movement occurs.
// 2. Native contexts: GpuBackend is created on one thread and shared via Arc<dyn GpuBackendTrait>.
//    The Arc is Send + Sync, and we never move the actual wgpu::Device/Queue/Pipeline between threads.
//    Only the Arc pointer moves, which is safe. The Mutex wrappers ensure exclusive access.
//
// Important Constraints:
// - GpuBackend MUST be created on the thread where it will be used (or before Arc sharing)
// - GpuBackend MUST only be shared via Arc, never moved directly between threads
// - All access to wgpu types goes through Mutex::lock(), ensuring exclusive access
//
// This is safe because:
// - The wgpu types are never actually moved between threads (only Arc references move)
// - Mutex provides synchronization for concurrent access
// - WASM is single-threaded, so no cross-thread movement happens
//
// WARNING: Do NOT move GpuBackend directly between threads. Always use Arc.
unsafe impl Send for GpuBackend {}
unsafe impl Sync for GpuBackend {}

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
        // Use GPU-native implementation
        self.fused_dequant_matmul_q4k_gpu(blocks, input, batch_size, n, k)
    }

    fn fused_dequant_matmul_q5k(
        &self,
        blocks: &[BlockQ5_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        // Use GPU-native implementation
        self.fused_dequant_matmul_q5k_gpu(blocks, input, batch_size, n, k)
    }

    fn fused_dequant_matmul_q6k(
        &self,
        blocks: &[BlockQ6_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        // Use GPU-native implementation
        self.fused_dequant_matmul_q6k_gpu(blocks, input, batch_size, n, k)
    }

    fn fused_dequant_matmul_q8k(
        &self,
        blocks: &[BlockQ8_K],
        input: &[f32],
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        // Use GPU-native implementation
        self.fused_dequant_matmul_q8k_gpu(blocks, input, batch_size, n, k)
    }

    fn name(&self) -> &'static str {
        "WebGPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_backend_creation() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let backend = GpuBackend::new().await;
        assert!(backend.is_ok(), "Failed to create GPU backend");
    }

    #[tokio::test]
    async fn test_matmul() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let backend = GpuBackend::new().await.unwrap();

        // Simple 2x2 matmul
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [2, 2]
        let result = backend.matmul(&a, &b, 2, 2, 2).unwrap();

        // Expected: [[19, 22], [43, 50]]
        assert_eq!(result.len(), 4);
        assert!((result[0] - 19.0).abs() < 0.001);
        assert!((result[1] - 22.0).abs() < 0.001);
        assert!((result[2] - 43.0).abs() < 0.001);
        assert!((result[3] - 50.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_fused_dequant_matmul_q4k() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        use realm_core::quant::{BlockQ4_K, QK_K};

        let backend = GpuBackend::new().await.unwrap();

        // Create test blocks
        let n = 4;
        let k = QK_K; // 256
        let batch_size = 2;
        let num_blocks_per_row = k / QK_K; // 1
        let num_blocks = n * num_blocks_per_row; // 4

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let mut block = BlockQ4_K {
                d: half::f16::from_f32(1.0).to_bits(),
                dmin: half::f16::from_f32(0.0).to_bits(),
                scales: [0u8; 12],
                qs: [0u8; 128],
            };
            // Set some test values
            block.scales[0] = 1;
            block.qs[0] = 0x10; // Lower nibble = 0, upper nibble = 1
            blocks.push(block);
        }

        let input = vec![1.0f32; batch_size * k];

        let result = backend
            .fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k)
            .unwrap();

        assert_eq!(result.len(), batch_size * n);
        // Result should be non-zero (exact value depends on dequantization)
        assert!(result.iter().any(|&x| x != 0.0));
    }

    #[tokio::test]
    async fn test_fused_dequant_matmul_q5k() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        use realm_core::quant::{BlockQ5_K, QK_K};

        let backend = GpuBackend::new().await.unwrap();

        let n = 4;
        let k = QK_K;
        let batch_size = 2;
        let num_blocks_per_row = k / QK_K;
        let num_blocks = n * num_blocks_per_row;

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let mut block = BlockQ5_K {
                ql: [0u8; QK_K / 2],
                qh: [0u8; QK_K / 8],
                scales: [1i8; QK_K / 16],
                d: half::f16::from_f32(1.0).to_bits(),
            };
            block.ql[0] = 0x10;
            blocks.push(block);
        }

        let input = vec![1.0f32; batch_size * k];

        let result = backend
            .fused_dequant_matmul_q5k(&blocks, &input, batch_size, n, k)
            .unwrap();

        assert_eq!(result.len(), batch_size * n);
        assert!(result.iter().any(|&x| x != 0.0));
    }

    #[tokio::test]
    async fn test_fused_dequant_matmul_q6k() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        use realm_core::quant::{BlockQ6_K, QK_K};

        let backend = GpuBackend::new().await.unwrap();

        let n = 4;
        let k = QK_K;
        let batch_size = 2;
        let num_blocks_per_row = k / QK_K;
        let num_blocks = n * num_blocks_per_row;

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let mut block = BlockQ6_K {
                ql: [0u8; QK_K / 2],
                qh: [0u8; QK_K / 4],
                scales: [1i8; QK_K / 16],
                d: half::f16::from_f32(1.0).to_bits(),
            };
            block.ql[0] = 0x10;
            blocks.push(block);
        }

        let input = vec![1.0f32; batch_size * k];

        let result = backend
            .fused_dequant_matmul_q6k(&blocks, &input, batch_size, n, k)
            .unwrap();

        assert_eq!(result.len(), batch_size * n);
        assert!(result.iter().any(|&x| x != 0.0));
    }

    #[tokio::test]
    async fn test_fused_dequant_matmul_q8k() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        use realm_core::quant::{BlockQ8_K, QK_K};

        let backend = GpuBackend::new().await.unwrap();

        let n = 4;
        let k = QK_K;
        let batch_size = 2;
        let num_blocks_per_row = k / QK_K;
        let num_blocks = n * num_blocks_per_row;

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let mut block = BlockQ8_K {
                quants: [0i8; QK_K],
                scales: [1u8; QK_K / 8],
                d: half::f16::from_f32(1.0).to_bits(),
                dmin: half::f16::from_f32(0.0).to_bits(),
            };
            block.quants[0] = 10;
            blocks.push(block);
        }

        let input = vec![1.0f32; batch_size * k];

        let result = backend
            .fused_dequant_matmul_q8k(&blocks, &input, batch_size, n, k)
            .unwrap();

        assert_eq!(result.len(), batch_size * n);
        assert!(result.iter().any(|&x| x != 0.0));
    }

    #[tokio::test]
    async fn test_fused_dequant_matmul_all_formats() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        use realm_core::quant::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_K, QK_K};

        let backend = GpuBackend::new().await.unwrap();

        let n = 2;
        let k = QK_K;
        let batch_size = 1;
        let num_blocks = n;

        // Test Q4_K
        let q4k_blocks: Vec<BlockQ4_K> = (0..num_blocks)
            .map(|_| BlockQ4_K {
                d: half::f16::from_f32(1.0).to_bits(),
                dmin: half::f16::from_f32(0.0).to_bits(),
                scales: [1u8; 12],
                qs: [0x10u8; 128],
            })
            .collect();
        let input = vec![1.0f32; batch_size * k];
        let q4k_result = backend
            .fused_dequant_matmul_q4k(&q4k_blocks, &input, batch_size, n, k)
            .unwrap();
        assert_eq!(q4k_result.len(), batch_size * n);

        // Test Q5_K
        let q5k_blocks: Vec<BlockQ5_K> = (0..num_blocks)
            .map(|_| BlockQ5_K {
                ql: [0x10u8; QK_K / 2],
                qh: [0u8; QK_K / 8],
                scales: [1i8; QK_K / 16],
                d: half::f16::from_f32(1.0).to_bits(),
            })
            .collect();
        let q5k_result = backend
            .fused_dequant_matmul_q5k(&q5k_blocks, &input, batch_size, n, k)
            .unwrap();
        assert_eq!(q5k_result.len(), batch_size * n);

        // Test Q6_K
        let q6k_blocks: Vec<BlockQ6_K> = (0..num_blocks)
            .map(|_| BlockQ6_K {
                ql: [0x10u8; QK_K / 2],
                qh: [0u8; QK_K / 4],
                scales: [1i8; QK_K / 16],
                d: half::f16::from_f32(1.0).to_bits(),
            })
            .collect();
        let q6k_result = backend
            .fused_dequant_matmul_q6k(&q6k_blocks, &input, batch_size, n, k)
            .unwrap();
        assert_eq!(q6k_result.len(), batch_size * n);

        // Test Q8_K
        let q8k_blocks: Vec<BlockQ8_K> = (0..num_blocks)
            .map(|_| BlockQ8_K {
                quants: [10i8; QK_K],
                scales: [1u8; QK_K / 8],
                d: half::f16::from_f32(1.0).to_bits(),
                dmin: half::f16::from_f32(0.0).to_bits(),
            })
            .collect();
        let q8k_result = backend
            .fused_dequant_matmul_q8k(&q8k_blocks, &input, batch_size, n, k)
            .unwrap();
        assert_eq!(q8k_result.len(), batch_size * n);
    }
}
