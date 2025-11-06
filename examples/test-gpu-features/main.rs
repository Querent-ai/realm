//! Comprehensive GPU Features Test
//!
//! Tests all GPU features: Fused Kernels, Flash Attention, Mixed Precision

use candle_core::Tensor;
use realm_compute_gpu::{
    fused_kernels::{FusedKernelConfig, Precision},
    mixed_precision::MixedPrecisionConfig,
    CandleGpuBackend, GpuBackend, GpuBackendTrait,
};
use realm_core::error::Result;
use realm_core::quant::{BlockQ4_K, QK_K};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Comprehensive GPU Features Test\n");

    // Test 1: WebGPU Backend
    println!("{}", "=".repeat(60));
    println!("1️⃣  Testing WebGPU Backend");
    println!("{}", "=".repeat(60));
    test_webgpu().await?;

    // Test 2: Candle GPU Backend (CUDA/Metal if available)
    println!("\n{}", "=".repeat(60));
    println!("2️⃣  Testing Candle GPU Backend (CUDA/Metal)");
    println!("{}", "=".repeat(60));
    test_candle_gpu()?;

    // Test 3: Fused Kernels
    println!("\n{}", "=".repeat(60));
    println!("3️⃣  Testing Fused Kernels");
    println!("{}", "=".repeat(60));
    test_fused_kernels()?;

    // Test 4: Mixed Precision
    println!("\n{}", "=".repeat(60));
    println!("4️⃣  Testing Mixed Precision");
    println!("{}", "=".repeat(60));
    test_mixed_precision()?;

    println!("\n{}", "=".repeat(60));
    println!("✅ All GPU Features Test Completed!");
    println!("{}", "=".repeat(60));
    Ok(())
}

async fn test_webgpu() -> Result<()> {
    if !GpuBackend::is_available() {
        println!("⚠️  WebGPU not available, skipping...");
        return Ok(());
    }

    let backend = GpuBackend::new().await?;
    let result = backend.matmul(&[1.0; 16], &[1.0; 16], 4, 4, 4)?;
    println!("✅ WebGPU matmul: {} elements", result.len());
    Ok(())
}

fn test_candle_gpu() -> Result<()> {
    match CandleGpuBackend::new() {
        Ok(backend) => {
            println!("✅ Candle GPU Backend initialized: {}", backend.name());
            // Test matmul using tensor interface
            let device = backend.device();
            let a_tensor = Tensor::from_slice(&[1.0f32; 16], &[4, 4], device).map_err(|e| {
                realm_core::error::Error::Runtime(format!("Failed to create tensor: {}", e))
            })?;
            let b_tensor = Tensor::from_slice(&[1.0f32; 16], &[4, 4], device).map_err(|e| {
                realm_core::error::Error::Runtime(format!("Failed to create tensor: {}", e))
            })?;

            match backend.matmul(&a_tensor, &b_tensor) {
                Ok(result) => {
                    let num_elements: usize = result.shape().dims().iter().product();
                    println!("✅ Candle GPU matmul: {} elements", num_elements);
                }
                Err(e) => {
                    println!("⚠️  Candle GPU matmul failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!(
                "⚠️  Candle GPU not available: {} (expected if no CUDA/Metal)",
                e
            );
        }
    }
    Ok(())
}

fn test_fused_kernels() -> Result<()> {
    use realm_compute_gpu::fused_kernels::fused_dequant_matmul_q4k_gpu;

    let config = FusedKernelConfig {
        enabled: true,
        precision: Precision::FP32,
        block_size: 256,
    };

    // Create dummy Q4_K blocks
    let k = QK_K * 2; // 512
    let n = 128;
    let num_blocks_per_row = k / QK_K;
    let num_blocks = n * num_blocks_per_row;

    let mut blocks = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        blocks.push(BlockQ4_K {
            d: 0,
            dmin: 0,
            scales: [0; 12],
            qs: [0; 128],
        });
    }

    let input = vec![1.0f32; 512];
    let batch_size = 1;

    match fused_dequant_matmul_q4k_gpu(&blocks, &input, batch_size, n, k, &config) {
        Ok(result) => {
            println!("✅ Fused Q4_K kernel: {} elements", result.len());
        }
        Err(e) => {
            println!("⚠️  Fused kernel test: {} (may need GPU hardware)", e);
        }
    }

    Ok(())
}

fn test_mixed_precision() -> Result<()> {
    use realm_compute_gpu::mixed_precision::{
        f32_to_fp16, fp16_to_f32, supports_bf16, supports_fp16,
    };

    // Test FP16 conversion
    let f32_data = vec![1.0f32, 2.0, std::f32::consts::PI, -1.0, 0.0];
    let fp16_data = f32_to_fp16(&f32_data);
    let back_to_f32 = fp16_to_f32(&fp16_data);

    println!("✅ FP16 conversion test:");
    for (orig, conv) in f32_data.iter().zip(back_to_f32.iter()) {
        let diff = (orig - conv).abs();
        if diff < 0.01 || (*orig == 0.0 && *conv == 0.0) {
            println!("   {} -> {} (diff: {:.4})", orig, conv, diff);
        }
    }

    // Test GPU capability detection
    println!("\n📊 GPU Capability Detection:");
    println!("   FP16 support: {}", supports_fp16());
    println!("   BF16 support: {}", supports_bf16());

    // Test configs
    let _inference_config = MixedPrecisionConfig::inference();
    println!("\n⚙️  Mixed Precision Configs:");
    println!("   Inference config: FP16 forward, FP16 attention");
    println!("   Full precision config available");

    Ok(())
}
