use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use realm_compute_cpu::{matmul_f32, matmul_transposed};

fn bench_gemm_small(criterion: &mut Criterion) {
    let m = 128;
    let k = 128;
    let n = 128;

    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut c = vec![0.0f32; m * n];

    criterion.bench_function("gemm_128x128x128", |bencher| {
        bencher.iter(|| {
            matmul_f32(
                std::hint::black_box(&a),
                std::hint::black_box(&b),
                std::hint::black_box(&mut c),
                m,
                k,
                n,
            )
            .unwrap();
        });
    });
}

fn bench_gemm_medium(criterion: &mut Criterion) {
    let m = 512;
    let k = 512;
    let n = 512;

    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut c = vec![0.0f32; m * n];

    criterion.bench_function("gemm_512x512x512", |bencher| {
        bencher.iter(|| {
            matmul_f32(
                std::hint::black_box(&a),
                std::hint::black_box(&b),
                std::hint::black_box(&mut c),
                m,
                k,
                n,
            )
            .unwrap();
        });
    });
}

fn bench_gemm_transposed(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("gemm_transposed");

    // Transformer-like shapes (single token × hidden → vocab)
    let shapes = vec![
        (1, 2048, 32000),  // TinyLlama LM head (single token)
        (1, 2048, 2048),   // Attention QKV projection
        (1, 2048, 5632),   // FFN gate/up
        (128, 2048, 2048), // Attention (seq_len=128)
    ];

    for (m, k, n) in shapes {
        let a = vec![1.0f32; m * k];
        let b_t = vec![1.0f32; n * k];
        let mut c = vec![0.0f32; m * n];

        group.bench_with_input(
            BenchmarkId::new("matmul_transposed", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bencher, _| {
                bencher.iter(|| {
                    matmul_transposed(
                        std::hint::black_box(&a),
                        std::hint::black_box(&b_t),
                        std::hint::black_box(&mut c),
                        m,
                        k,
                        n,
                    )
                    .unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_gemm_transformer_workload(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("transformer_workload");

    // Realistic transformer matmul patterns
    let workloads = vec![
        ("qkv_projection", 1, 2048, 6144),   // Q+K+V combined
        ("attention_output", 1, 2048, 2048), // Attention output
        ("ffn_gate_up", 1, 2048, 11264),     // Gate + Up combined
        ("ffn_down", 1, 5632, 2048),         // Down projection
        ("lm_head", 1, 2048, 32000),         // Vocabulary projection
    ];

    for (name, m, k, n) in workloads {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c = vec![0.0f32; m * n];

        group.bench_with_input(BenchmarkId::new("matmul", name), name, |bencher, _| {
            bencher.iter(|| {
                matmul_f32(
                    std::hint::black_box(&a),
                    std::hint::black_box(&b),
                    std::hint::black_box(&mut c),
                    m,
                    k,
                    n,
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

fn bench_gemm_batch_sizes(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("batch_sizes");

    // Hidden size typical for small models
    let hidden = 2048;
    let intermediate = 5632;

    for batch_size in [1, 8, 16, 32, 64] {
        let a = vec![1.0f32; batch_size * hidden];
        let b = vec![1.0f32; hidden * intermediate];
        let mut c = vec![0.0f32; batch_size * intermediate];

        group.bench_with_input(
            BenchmarkId::new("ffn_gate", batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    matmul_f32(
                        std::hint::black_box(&a),
                        std::hint::black_box(&b),
                        std::hint::black_box(&mut c),
                        batch_size,
                        hidden,
                        intermediate,
                    )
                    .unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gemm_small,
    bench_gemm_medium,
    bench_gemm_transposed,
    bench_gemm_transformer_workload,
    bench_gemm_batch_sizes
);
criterion_main!(benches);
