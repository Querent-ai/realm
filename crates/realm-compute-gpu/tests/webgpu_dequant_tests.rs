//! Comprehensive WebGPU dequantization shader tests
//!
//! Tests GPU-native dequantization + matmul for all quantization formats
//! Compares results with CPU dequantization to ensure correctness

use realm_compute_gpu::{GpuBackend, GpuBackendTrait};
use realm_core::quant::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_K, QK_K};

// Note: CPU comparison tests would require realm-compute-cpu as a dev-dependency
// For now, we test GPU correctness by verifying:
// 1. Correct output dimensions
// 2. Non-zero results (when input is non-zero)
// 3. Error handling for invalid inputs
// 4. Various batch sizes and configurations

/// Test Q4_K with various batch sizes
#[tokio::test]
async fn test_q4k_various_batch_sizes() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let backend = match GpuBackend::new().await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("⚠️  WebGPU backend creation failed: {} (skipping)", e);
            return;
        }
    };
    let n = 4;
    let k = QK_K;
    let num_blocks = n;

    let blocks: Vec<BlockQ4_K> = (0..num_blocks)
        .map(|_| BlockQ4_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [1u8; 12],
            qs: [0x10u8; 128],
        })
        .collect();

    for batch_size in [1, 2, 4, 8] {
        let input = vec![1.0f32; batch_size * k];
        let result = backend
            .fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k)
            .unwrap();

        assert_eq!(result.len(), batch_size * n);
        assert!(result.iter().any(|&x| x != 0.0));
        println!(
            "✅ Q4_K batch_size={}: {} elements",
            batch_size,
            result.len()
        );
    }
}

/// Test Q4_K with multiple K blocks
#[tokio::test]
async fn test_q4k_multiple_k_blocks() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let backend = match GpuBackend::new().await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("⚠️  WebGPU backend creation failed: {} (skipping)", e);
            return;
        }
    };
    let n = 2;
    let k = QK_K * 2; // 2 blocks per row
    let batch_size = 1;
    let num_blocks_per_row = k / QK_K;
    let num_blocks = n * num_blocks_per_row;

    let blocks: Vec<BlockQ4_K> = (0..num_blocks)
        .map(|_| BlockQ4_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [1u8; 12],
            qs: [0x10u8; 128],
        })
        .collect();

    let input = vec![1.0f32; batch_size * k];
    let result = backend
        .fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k)
        .unwrap();

    assert_eq!(result.len(), batch_size * n);
    println!(
        "✅ Q4_K multiple K blocks (k={}): {} elements",
        k,
        result.len()
    );
}

/// Test Q5_K with various configurations
#[tokio::test]
async fn test_q5k_various_configs() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let backend = match GpuBackend::new().await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("⚠️  WebGPU backend creation failed: {} (skipping)", e);
            return;
        }
    };

    for (n, k, batch_size) in [(2, QK_K, 1), (4, QK_K, 2), (8, QK_K * 2, 1)] {
        let num_blocks_per_row = k / QK_K;
        let num_blocks = n * num_blocks_per_row;

        let blocks: Vec<BlockQ5_K> = (0..num_blocks)
            .map(|_| BlockQ5_K {
                ql: [0x10u8; QK_K / 2],
                qh: [0u8; QK_K / 8],
                scales: [1i8; QK_K / 16],
                d: half::f16::from_f32(1.0).to_bits(),
            })
            .collect();

        let input = vec![1.0f32; batch_size * k];
        let result = backend
            .fused_dequant_matmul_q5k(&blocks, &input, batch_size, n, k)
            .unwrap();

        assert_eq!(result.len(), batch_size * n);
        println!(
            "✅ Q5_K (n={}, k={}, batch={}): {} elements",
            n,
            k,
            batch_size,
            result.len()
        );
    }
}

/// Test Q6_K with various configurations
#[tokio::test]
async fn test_q6k_various_configs() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let backend = match GpuBackend::new().await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("⚠️  WebGPU backend creation failed: {} (skipping)", e);
            return;
        }
    };

    for (n, k, batch_size) in [(2, QK_K, 1), (4, QK_K, 2), (8, QK_K * 2, 1)] {
        let num_blocks_per_row = k / QK_K;
        let num_blocks = n * num_blocks_per_row;

        let blocks: Vec<BlockQ6_K> = (0..num_blocks)
            .map(|_| BlockQ6_K {
                ql: [0x10u8; QK_K / 2],
                qh: [0u8; QK_K / 4],
                scales: [1i8; QK_K / 16],
                d: half::f16::from_f32(1.0).to_bits(),
            })
            .collect();

        let input = vec![1.0f32; batch_size * k];
        let result = backend
            .fused_dequant_matmul_q6k(&blocks, &input, batch_size, n, k)
            .unwrap();

        assert_eq!(result.len(), batch_size * n);
        println!(
            "✅ Q6_K (n={}, k={}, batch={}): {} elements",
            n,
            k,
            batch_size,
            result.len()
        );
    }
}

/// Test Q8_K with various configurations
#[tokio::test]
async fn test_q8k_various_configs() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let backend = match GpuBackend::new().await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("⚠️  WebGPU backend creation failed: {} (skipping)", e);
            return;
        }
    };

    for (n, k, batch_size) in [(2, QK_K, 1), (4, QK_K, 2), (8, QK_K * 2, 1)] {
        let num_blocks_per_row = k / QK_K;
        let num_blocks = n * num_blocks_per_row;

        let blocks: Vec<BlockQ8_K> = (0..num_blocks)
            .map(|_| BlockQ8_K {
                quants: [10i8; QK_K],
                scales: [1u8; QK_K / 8],
                d: half::f16::from_f32(1.0).to_bits(),
                dmin: half::f16::from_f32(0.0).to_bits(),
            })
            .collect();

        let input = vec![1.0f32; batch_size * k];
        let result = backend
            .fused_dequant_matmul_q8k(&blocks, &input, batch_size, n, k)
            .unwrap();

        assert_eq!(result.len(), batch_size * n);
        println!(
            "✅ Q8_K (n={}, k={}, batch={}): {} elements",
            n,
            k,
            batch_size,
            result.len()
        );
    }
}

/// Test error handling - invalid K dimension
#[tokio::test]
async fn test_q4k_invalid_k_dimension() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let backend = match GpuBackend::new().await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("⚠️  WebGPU backend creation failed: {} (skipping)", e);
            return;
        }
    };
    let n = 4;
    let k = 100; // Not a multiple of QK_K
    let batch_size = 1;

    let blocks: Vec<BlockQ4_K> = vec![];
    let input = vec![1.0f32; batch_size * k];

    let result = backend.fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k);
    assert!(result.is_err());
    println!("✅ Q4_K correctly rejects invalid K dimension");
}

/// Test error handling - wrong number of blocks
#[tokio::test]
async fn test_q4k_wrong_block_count() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let backend = match GpuBackend::new().await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("⚠️  WebGPU backend creation failed: {} (skipping)", e);
            return;
        }
    };
    let n = 4;
    let k = QK_K;
    let batch_size = 1;
    let num_blocks = n - 1; // Wrong count

    let blocks: Vec<BlockQ4_K> = (0..num_blocks)
        .map(|_| BlockQ4_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; 12],
            qs: [0u8; 128],
        })
        .collect();

    let input = vec![1.0f32; batch_size * k];
    let result = backend.fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k);
    assert!(result.is_err());
    println!("✅ Q4_K correctly rejects wrong block count");
}

/// Test error handling - wrong input size
#[tokio::test]
async fn test_q4k_wrong_input_size() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let backend = GpuBackend::new().await.unwrap();
    let n = 4;
    let k = QK_K;
    let batch_size = 1;
    let num_blocks = n;

    let blocks: Vec<BlockQ4_K> = (0..num_blocks)
        .map(|_| BlockQ4_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; 12],
            qs: [0u8; 128],
        })
        .collect();

    let input = vec![1.0f32; batch_size * k - 1]; // Wrong size
    let result = backend.fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k);
    assert!(result.is_err());
    println!("✅ Q4_K correctly rejects wrong input size");
}

/// Test all formats with realistic values
#[tokio::test]
async fn test_all_formats_realistic_values() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let backend = GpuBackend::new().await.unwrap();
    let n = 8;
    let k = QK_K;
    let batch_size = 4;
    let num_blocks = n;

    // Q4_K with realistic scale values
    let q4k_blocks: Vec<BlockQ4_K> = (0..num_blocks)
        .map(|i| {
            let scale_val = (i % 64) as u8;
            let mut scales = [0u8; 12];
            scales[0] = scale_val;
            BlockQ4_K {
                d: half::f16::from_f32(0.5).to_bits(),
                dmin: half::f16::from_f32(0.0).to_bits(),
                scales,
                qs: [0x10u8; 128],
            }
        })
        .collect();
    let input = vec![0.5f32; batch_size * k];
    let q4k_result = backend
        .fused_dequant_matmul_q4k(&q4k_blocks, &input, batch_size, n, k)
        .unwrap();
    assert_eq!(q4k_result.len(), batch_size * n);
    println!("✅ Q4_K realistic values: {} elements", q4k_result.len());

    // Q5_K
    let q5k_blocks: Vec<BlockQ5_K> = (0..num_blocks)
        .map(|_| BlockQ5_K {
            ql: [0x10u8; QK_K / 2],
            qh: [0u8; QK_K / 8],
            scales: [2i8; QK_K / 16],
            d: half::f16::from_f32(0.5).to_bits(),
        })
        .collect();
    let q5k_result = backend
        .fused_dequant_matmul_q5k(&q5k_blocks, &input, batch_size, n, k)
        .unwrap();
    assert_eq!(q5k_result.len(), batch_size * n);
    println!("✅ Q5_K realistic values: {} elements", q5k_result.len());

    // Q6_K
    let q6k_blocks: Vec<BlockQ6_K> = (0..num_blocks)
        .map(|_| BlockQ6_K {
            ql: [0x10u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [2i8; QK_K / 16],
            d: half::f16::from_f32(0.5).to_bits(),
        })
        .collect();
    let q6k_result = backend
        .fused_dequant_matmul_q6k(&q6k_blocks, &input, batch_size, n, k)
        .unwrap();
    assert_eq!(q6k_result.len(), batch_size * n);
    println!("✅ Q6_K realistic values: {} elements", q6k_result.len());

    // Q8_K
    let q8k_blocks: Vec<BlockQ8_K> = (0..num_blocks)
        .map(|_| BlockQ8_K {
            quants: [20i8; QK_K],
            scales: [2u8; QK_K / 8],
            d: half::f16::from_f32(0.5).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
        })
        .collect();
    let q8k_result = backend
        .fused_dequant_matmul_q8k(&q8k_blocks, &input, batch_size, n, k)
        .unwrap();
    assert_eq!(q8k_result.len(), batch_size * n);
    println!("✅ Q8_K realistic values: {} elements", q8k_result.len());
}

/// Test large batch sizes
#[tokio::test]
async fn test_large_batch_sizes() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let backend = GpuBackend::new().await.unwrap();
    let n = 16;
    let k = QK_K;
    let num_blocks = n;

    for batch_size in [16, 32, 64] {
        let blocks: Vec<BlockQ4_K> = (0..num_blocks)
            .map(|_| BlockQ4_K {
                d: half::f16::from_f32(1.0).to_bits(),
                dmin: half::f16::from_f32(0.0).to_bits(),
                scales: [1u8; 12],
                qs: [0x10u8; 128],
            })
            .collect();

        let input = vec![1.0f32; batch_size * k];
        let result = backend
            .fused_dequant_matmul_q4k(&blocks, &input, batch_size, n, k)
            .unwrap();

        assert_eq!(result.len(), batch_size * n);
        println!(
            "✅ Q4_K large batch (batch={}): {} elements",
            batch_size,
            result.len()
        );
    }
}

/// Test all formats in sequence (stress test)
#[tokio::test]
async fn test_all_formats_sequence() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let backend = GpuBackend::new().await.unwrap();
    let n = 4;
    let k = QK_K;
    let batch_size = 2;
    let num_blocks = n;
    let input = vec![1.0f32; batch_size * k];

    // Run all formats in sequence
    let q4k_blocks: Vec<BlockQ4_K> = (0..num_blocks)
        .map(|_| BlockQ4_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [1u8; 12],
            qs: [0x10u8; 128],
        })
        .collect();
    backend
        .fused_dequant_matmul_q4k(&q4k_blocks, &input, batch_size, n, k)
        .unwrap();

    let q5k_blocks: Vec<BlockQ5_K> = (0..num_blocks)
        .map(|_| BlockQ5_K {
            ql: [0x10u8; QK_K / 2],
            qh: [0u8; QK_K / 8],
            scales: [1i8; QK_K / 16],
            d: half::f16::from_f32(1.0).to_bits(),
        })
        .collect();
    backend
        .fused_dequant_matmul_q5k(&q5k_blocks, &input, batch_size, n, k)
        .unwrap();

    let q6k_blocks: Vec<BlockQ6_K> = (0..num_blocks)
        .map(|_| BlockQ6_K {
            ql: [0x10u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [1i8; QK_K / 16],
            d: half::f16::from_f32(1.0).to_bits(),
        })
        .collect();
    backend
        .fused_dequant_matmul_q6k(&q6k_blocks, &input, batch_size, n, k)
        .unwrap();

    let q8k_blocks: Vec<BlockQ8_K> = (0..num_blocks)
        .map(|_| BlockQ8_K {
            quants: [10i8; QK_K],
            scales: [1u8; QK_K / 8],
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
        })
        .collect();
    backend
        .fused_dequant_matmul_q8k(&q8k_blocks, &input, batch_size, n, k)
        .unwrap();

    println!("✅ All formats executed successfully in sequence");
}
