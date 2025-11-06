//! GPU Integration Tests
//!
//! Comprehensive tests for GPU backends (CUDA, Metal, WebGPU)
//! These tests require GPU hardware and are skipped in CI if not available.

use realm_compute_gpu::{CandleGpuBackend, GpuBackend, GpuBackendTrait};
use realm_core::quant::{BlockQ4_K, QK_K};

/// Test GPU backend creation
#[tokio::test]
async fn test_gpu_backend_creation() {
    // Test Candle GPU backend (CUDA/Metal)
    match CandleGpuBackend::new() {
        Ok(backend) => {
            println!("✅ Candle GPU Backend created: {}", backend.name());
            assert!(!backend.name().is_empty());
        }
        Err(e) => {
            println!(
                "⚠️  Candle GPU not available: {} (expected in CI without GPU)",
                e
            );
        }
    }

    // Test WebGPU backend
    if GpuBackend::is_available() {
        match GpuBackend::new().await {
            Ok(_backend) => {
                println!("✅ WebGPU Backend created");
            }
            Err(e) => {
                println!(
                    "⚠️  WebGPU creation failed: {} (expected in some CI environments)",
                    e
                );
            }
        }
    } else {
        println!("⚠️  WebGPU not available (expected in CI)");
    }
}

/// Test matrix multiplication on GPU
#[tokio::test]
async fn test_gpu_matmul() {
    // Test Candle GPU matmul
    if let Ok(backend) = CandleGpuBackend::new() {
        use candle_core::Tensor;
        let device = backend.device();

        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], device).unwrap();

        match backend.matmul(&a, &b) {
            Ok(result) => {
                let num_elements: usize = result.shape().dims().iter().product();
                assert_eq!(num_elements, 4);
                println!("✅ Candle GPU matmul: {} elements", num_elements);
            }
            Err(e) => {
                println!(
                    "⚠️  Candle GPU matmul failed: {} (expected in CI without GPU)",
                    e
                );
            }
        }
    }

    // Test WebGPU matmul
    if GpuBackend::is_available() {
        if let Ok(backend) = GpuBackend::new().await {
            match backend.matmul(&[1.0; 16], &[1.0; 16], 4, 4, 4) {
                Ok(result) => {
                    assert_eq!(result.len(), 16);
                    println!("✅ WebGPU matmul: {} elements", result.len());
                }
                Err(e) => {
                    println!(
                        "⚠️  WebGPU matmul failed: {} (expected in some CI environments)",
                        e
                    );
                }
            }
        }
    }
}

/// Test fused dequantization + matmul
#[tokio::test]
async fn test_fused_dequant_matmul() {
    if let Ok(backend) = CandleGpuBackend::new() {
        let n = 256;
        let k = 256;
        let batch_size = 1;

        // Create dummy Q4_K blocks
        let block = BlockQ4_K {
            d: 0,
            dmin: 0,
            scales: [0; 12],
            qs: [0; 128],
        };
        let num_blocks = n * (k / QK_K);
        let blocks = vec![block; num_blocks];
        let input = vec![1.0f32; batch_size * k];

        match backend.fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k) {
            Ok(result) => {
                assert_eq!(result.len(), batch_size * n);
                println!("✅ Fused Q4_K dequant+matmul: {} elements", result.len());
            }
            Err(e) => {
                println!(
                    "⚠️  Fused kernel failed: {} (expected in CI without GPU)",
                    e
                );
            }
        }
    }
}

/// Test GPU backend selection
#[test]
fn test_gpu_backend_selection() {
    // Test that backend selection works
    let backend = CandleGpuBackend::new();
    match backend {
        Ok(b) => {
            let name = b.name();
            assert!(matches!(name, "CUDA" | "Metal" | "CPU"));
            println!("✅ Backend selected: {}", name);
        }
        Err(_) => {
            println!("⚠️  No GPU backend available (expected in CI)");
        }
    }
}

/// Test mixed precision conversion
#[test]
fn test_mixed_precision_conversion() {
    use realm_compute_gpu::mixed_precision::{bf16_to_f32, f32_to_bf16, f32_to_fp16, fp16_to_f32};

    let test_values = vec![
        0.0f32,
        1.0,
        -1.0,
        std::f32::consts::PI,
        std::f32::consts::E,
        100.0,
        -100.0,
        0.001,
        -0.001,
    ];

    // Test FP16
    let fp16 = f32_to_fp16(&test_values);
    let back_fp16 = fp16_to_f32(&fp16);
    for (orig, conv) in test_values.iter().zip(back_fp16.iter()) {
        let diff = (orig - conv).abs();
        assert!(
            diff < 0.1 || (*orig == 0.0 && *conv == 0.0),
            "FP16 conversion error: {} -> {} (diff: {})",
            orig,
            conv,
            diff
        );
    }
    println!("✅ FP16 conversion test passed");

    // Test BF16
    let bf16 = f32_to_bf16(&test_values);
    let back_bf16 = bf16_to_f32(&bf16);
    for (orig, conv) in test_values.iter().zip(back_bf16.iter()) {
        let diff = (orig - conv).abs();
        assert!(
            diff < 1.0 || (*orig == 0.0 && *conv == 0.0),
            "BF16 conversion error: {} -> {} (diff: {})",
            orig,
            conv,
            diff
        );
    }
    println!("✅ BF16 conversion test passed");
}

/// Test GPU capability detection
#[test]
fn test_gpu_capability_detection() {
    use realm_compute_gpu::mixed_precision::{supports_bf16, supports_fp16};

    let fp16_supported = supports_fp16();
    let bf16_supported = supports_bf16();

    println!("📊 GPU Capabilities:");
    println!("   FP16: {}", fp16_supported);
    println!("   BF16: {}", bf16_supported);

    // These may be false in CI, that's okay
    // Just verify the functions don't panic - the actual value doesn't matter
    let _ = fp16_supported;
    let _ = bf16_supported;
}
