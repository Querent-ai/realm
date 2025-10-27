use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use wasm_chord_core::quant::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_K};
use wasm_chord_cpu::fused::{
    fused_attention_score, fused_dequant_matmul_q4k, fused_dequant_matmul_q5k,
    fused_dequant_matmul_q6k, fused_dequant_matmul_q8k,
};

/// Benchmark traditional fused attention score (for comparison)
fn bench_traditional_attention(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("traditional_attention");
    group.sample_size(20);

    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for seq_len in [64, 128, 256, 512] {
        let q = vec![0.5f32; seq_len * head_dim];
        let k = vec![0.5f32; seq_len * head_dim];
        let empty_output = vec![0.0f32; 0]; // Unused but required by API

        group.bench_with_input(
            BenchmarkId::new("attention_score", seq_len),
            &seq_len,
            |bencher, _| {
                bencher.iter(|| {
                    fused_attention_score(
                        black_box(&q),
                        black_box(&k),
                        black_box(&empty_output),
                        seq_len,
                        head_dim,
                        scale,
                        true, // causal_mask
                    )
                    .unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Helper to create a Q4_K block for benchmarking
fn create_test_q4k_block() -> BlockQ4_K {
    use half::f16;

    BlockQ4_K {
        d: f16::from_f32(0.5).to_bits(),
        dmin: f16::from_f32(0.01).to_bits(),
        scales: [128u8; 12], // Mid-range scales
        qs: [0x55u8; 128],   // Alternating nibbles 0101...
    }
}

/// Helper to create a Q5_K block for benchmarking
fn create_test_q5k_block() -> BlockQ5_K {
    use half::f16;

    BlockQ5_K {
        d: f16::from_f32(0.5).to_bits(),
        ql: [0x55u8; 128],  // Lower 4 bits
        qh: [0xAAu8; 32],   // Upper 1 bit
        scales: [64i8; 16], // Mid-range scales
    }
}

/// Helper to create a Q6_K block for benchmarking
fn create_test_q6k_block() -> BlockQ6_K {
    use half::f16;

    BlockQ6_K {
        d: f16::from_f32(0.5).to_bits(),
        ql: [0x55u8; 128],  // Lower 4 bits
        qh: [0xAAu8; 64],   // Upper 2 bits
        scales: [64i8; 16], // Mid-range scales
    }
}

/// Helper to create a Q8_K block for benchmarking
fn create_test_q8k_block() -> BlockQ8_K {
    use half::f16;

    BlockQ8_K {
        d: f16::from_f32(0.5).to_bits(),
        dmin: f16::from_f32(0.01).to_bits(),
        quants: [64i8; 256], // Mid-range quantized values
        scales: [128u8; 32], // Mid-range scales
    }
}

/// Benchmark Q4_K fused dequant+matmul
fn bench_q4k_fused_kernel(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("q4k_fused_kernel");
    group.sample_size(30);

    // Transformer-like workloads (batch=1 for inference)
    let workloads = vec![
        ("qkv_proj", 1, 6144, 2048),      // Q+K+V projection (hidden → 3*hidden)
        ("attention_out", 1, 2048, 2048), // Attention output projection
        ("ffn_gate_up", 1, 11264, 2048),  // FFN gate+up (hidden → 2*intermediate)
        ("ffn_down", 1, 2048, 5632),      // FFN down (intermediate → hidden)
        ("lm_head", 1, 32000, 2048),      // Vocabulary projection
    ];

    for (name, batch_size, num_features, k) in workloads {
        // K must be multiple of 256 for Q4_K
        let k_aligned = ((k + 255) / 256) * 256;
        let num_blocks = (num_features * k_aligned) / 256;

        let blocks: Vec<BlockQ4_K> = (0..num_blocks).map(|_| create_test_q4k_block()).collect();

        let input = vec![0.5f32; batch_size * k_aligned];
        let mut output = vec![0.0f32; batch_size * num_features];

        group.bench_with_input(BenchmarkId::new("workload", name), name, |bencher, _| {
            bencher.iter(|| {
                fused_dequant_matmul_q4k(
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    batch_size,
                    num_features,
                    k_aligned,
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark Q4_K with different batch sizes
fn bench_q4k_batch_sizes(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("q4k_batch_sizes");
    group.sample_size(30);

    let num_features = 2048;
    let k = 2048;
    let num_blocks = (num_features * k) / 256;

    let blocks: Vec<BlockQ4_K> = (0..num_blocks).map(|_| create_test_q4k_block()).collect();

    for batch_size in [1, 4, 8, 16, 32] {
        let input = vec![0.5f32; batch_size * k];
        let mut output = vec![0.0f32; batch_size * num_features];

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    fused_dequant_matmul_q4k(
                        black_box(&blocks),
                        black_box(&input),
                        black_box(&mut output),
                        batch_size,
                        num_features,
                        k,
                    )
                    .unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Q4_K with different matrix sizes
fn bench_q4k_matrix_sizes(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("q4k_matrix_sizes");
    group.sample_size(20);

    // Square matrices of different sizes
    for size in [256, 512, 1024, 2048, 4096] {
        let num_blocks = (size * size) / 256;
        let blocks: Vec<BlockQ4_K> = (0..num_blocks).map(|_| create_test_q4k_block()).collect();

        let input = vec![0.5f32; size];
        let mut output = vec![0.0f32; size];

        group.bench_with_input(BenchmarkId::new("size", size), &size, |bencher, _| {
            bencher.iter(|| {
                fused_dequant_matmul_q4k(
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    1,
                    size,
                    size,
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark memory bandwidth impact
fn bench_memory_patterns(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("memory_patterns");
    group.sample_size(20);

    // Test cache-friendly vs cache-unfriendly access patterns
    let sizes = vec![
        ("L1_fit", 256),      // Fits in L1 (32KB typical)
        ("L2_fit", 1024),     // Fits in L2 (256KB typical)
        ("L3_fit", 2048),     // Fits in L3 (8MB typical)
        ("cache_miss", 4096), // Exceeds L3
    ];

    for (name, size) in sizes {
        let num_blocks = (size * size) / 256;
        let blocks: Vec<BlockQ4_K> = (0..num_blocks).map(|_| create_test_q4k_block()).collect();

        let input = vec![0.5f32; size];
        let mut output = vec![0.0f32; size];

        group.bench_with_input(BenchmarkId::new("pattern", name), name, |bencher, _| {
            bencher.iter(|| {
                fused_dequant_matmul_q4k(
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    1,
                    size,
                    size,
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark Q5_K fused dequant+matmul
fn bench_q5k_fused_kernel(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("q5k_fused_kernel");
    group.sample_size(30);

    // Same transformer-like workloads as Q4_K
    let workloads = vec![
        ("qkv_proj", 1, 6144, 2048),
        ("attention_out", 1, 2048, 2048),
        ("ffn_gate_up", 1, 11264, 2048),
        ("ffn_down", 1, 2048, 5632),
        ("lm_head", 1, 32000, 2048),
    ];

    for (name, batch_size, num_features, k) in workloads {
        let k_aligned = ((k + 255) / 256) * 256;
        let num_blocks = (num_features * k_aligned) / 256;

        let blocks: Vec<BlockQ5_K> = (0..num_blocks).map(|_| create_test_q5k_block()).collect();

        let input = vec![0.5f32; batch_size * k_aligned];
        let mut output = vec![0.0f32; batch_size * num_features];

        group.bench_with_input(BenchmarkId::new("workload", name), name, |bencher, _| {
            bencher.iter(|| {
                fused_dequant_matmul_q5k(
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    batch_size,
                    num_features,
                    k_aligned,
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark Q6_K fused dequant+matmul
fn bench_q6k_fused_kernel(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("q6k_fused_kernel");
    group.sample_size(30);

    // Same transformer-like workloads
    let workloads = vec![
        ("qkv_proj", 1, 6144, 2048),
        ("attention_out", 1, 2048, 2048),
        ("ffn_gate_up", 1, 11264, 2048),
        ("ffn_down", 1, 2048, 5632),
        ("lm_head", 1, 32000, 2048),
    ];

    for (name, batch_size, num_features, k) in workloads {
        let k_aligned = ((k + 255) / 256) * 256;
        let num_blocks = (num_features * k_aligned) / 256;

        let blocks: Vec<BlockQ6_K> = (0..num_blocks).map(|_| create_test_q6k_block()).collect();

        let input = vec![0.5f32; batch_size * k_aligned];
        let mut output = vec![0.0f32; batch_size * num_features];

        group.bench_with_input(BenchmarkId::new("workload", name), name, |bencher, _| {
            bencher.iter(|| {
                fused_dequant_matmul_q6k(
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    batch_size,
                    num_features,
                    k_aligned,
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark Q8_K fused dequant+matmul
fn bench_q8k_fused_kernel(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("q8k_fused_kernel");
    group.sample_size(30);

    let workloads = vec![
        ("qkv_proj", 1, 6144, 2048),
        ("attention_out", 1, 2048, 2048),
        ("ffn_gate_up", 1, 11264, 2048),
        ("ffn_down", 1, 2048, 5632),
        ("lm_head", 1, 32000, 2048),
    ];

    for (name, batch_size, num_features, k) in workloads {
        let k_aligned = ((k + 255) / 256) * 256;
        let num_blocks = (num_features * k_aligned) / 256;

        let blocks: Vec<BlockQ8_K> = (0..num_blocks).map(|_| create_test_q8k_block()).collect();

        let input = vec![0.5f32; batch_size * k_aligned];
        let mut output = vec![0.0f32; batch_size * num_features];

        group.bench_with_input(BenchmarkId::new("workload", name), name, |bencher, _| {
            bencher.iter(|| {
                fused_dequant_matmul_q8k(
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    batch_size,
                    num_features,
                    k_aligned,
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

/// Compare all quantization formats
fn bench_quantization_comparison(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("quant_format_comparison");
    group.sample_size(30);

    // Standard workload: 2048x2048 matrix (typical attention projection)
    let batch_size = 1;
    let num_features = 2048;
    let k = 2048;
    let num_blocks = (num_features * k) / 256;

    let q4k_blocks: Vec<BlockQ4_K> = (0..num_blocks).map(|_| create_test_q4k_block()).collect();
    let q5k_blocks: Vec<BlockQ5_K> = (0..num_blocks).map(|_| create_test_q5k_block()).collect();
    let q6k_blocks: Vec<BlockQ6_K> = (0..num_blocks).map(|_| create_test_q6k_block()).collect();
    let q8k_blocks: Vec<BlockQ8_K> = (0..num_blocks).map(|_| create_test_q8k_block()).collect();

    let input = vec![0.5f32; batch_size * k];
    let mut output = vec![0.0f32; batch_size * num_features];

    // Q4_K
    group.bench_function("Q4_K", |bencher| {
        bencher.iter(|| {
            fused_dequant_matmul_q4k(
                black_box(&q4k_blocks),
                black_box(&input),
                black_box(&mut output),
                batch_size,
                num_features,
                k,
            )
            .unwrap();
        });
    });

    // Q5_K
    group.bench_function("Q5_K", |bencher| {
        bencher.iter(|| {
            fused_dequant_matmul_q5k(
                black_box(&q5k_blocks),
                black_box(&input),
                black_box(&mut output),
                batch_size,
                num_features,
                k,
            )
            .unwrap();
        });
    });

    // Q6_K
    group.bench_function("Q6_K", |bencher| {
        bencher.iter(|| {
            fused_dequant_matmul_q6k(
                black_box(&q6k_blocks),
                black_box(&input),
                black_box(&mut output),
                batch_size,
                num_features,
                k,
            )
            .unwrap();
        });
    });

    // Q8_K
    group.bench_function("Q8_K", |bencher| {
        bencher.iter(|| {
            fused_dequant_matmul_q8k(
                black_box(&q8k_blocks),
                black_box(&input),
                black_box(&mut output),
                batch_size,
                num_features,
                k,
            )
            .unwrap();
        });
    });

    group.finish();
}

criterion_group!(attention_benches, bench_traditional_attention,);

criterion_group!(
    q4k_benches,
    bench_q4k_fused_kernel,
    bench_q4k_batch_sizes,
    bench_q4k_matrix_sizes,
);

criterion_group!(q5k_benches, bench_q5k_fused_kernel,);

criterion_group!(q6k_benches, bench_q6k_fused_kernel,);

criterion_group!(q8k_benches, bench_q8k_fused_kernel,);

criterion_group!(
    comparison_benches,
    bench_quantization_comparison,
    bench_memory_patterns,
);

criterion_main!(
    attention_benches,
    q4k_benches,
    q5k_benches,
    q6k_benches,
    q8k_benches,
    comparison_benches
);
