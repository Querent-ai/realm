use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use wasm_chord_runtime::{attention::AttentionBackend, MultiHeadAttention, TransformerConfig};

fn bench_attention_computation(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("attention_computation");
    group.sample_size(50); // Reduce sample size for expensive operations

    let config = TransformerConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        num_layers: 22,
        num_heads: 32,
        num_kv_heads: 4,
        intermediate_size: 5632,
        max_seq_len: 2048,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
    };

    let head_dim = config.hidden_size / config.num_heads;
    let attn = MultiHeadAttention::new(config.clone());

    for seq_len in [1, 16, 64, 128, 256] {
        let q = vec![1.0f32; seq_len * config.num_heads * head_dim];
        let k = vec![1.0f32; seq_len * config.num_kv_heads * head_dim];
        let v = vec![1.0f32; seq_len * config.num_kv_heads * head_dim];

        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &seq_len,
            |bencher, _| {
                bencher.iter(|| {
                    attn.compute_attention(black_box(&q), black_box(&k), black_box(&v), seq_len, 0)
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_attention_gqa_ratios(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("gqa_ratios");
    group.sample_size(50);

    let seq_len = 64;
    let hidden_size = 2048;
    let num_heads = 32;
    let head_dim = hidden_size / num_heads;

    // Different GQA ratios: 1:1 (MHA), 2:1, 4:1, 8:1, 16:1, 32:1 (MQA)
    for num_kv_heads in [1, 2, 4, 8, 16, 32] {
        let config = TransformerConfig {
            vocab_size: 32000,
            hidden_size,
            num_layers: 22,
            num_heads,
            num_kv_heads,
            intermediate_size: 5632,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_backend: AttentionBackend::Auto,
        };

        let attn = MultiHeadAttention::new(config.clone());
        let q = vec![1.0f32; seq_len * num_heads * head_dim];
        let k = vec![1.0f32; seq_len * num_kv_heads * head_dim];
        let v = vec![1.0f32; seq_len * num_kv_heads * head_dim];

        let ratio = num_heads / num_kv_heads;
        group.bench_with_input(
            BenchmarkId::new("ratio", format!("{}:1", ratio)),
            &ratio,
            |bencher, _| {
                bencher.iter(|| {
                    attn.compute_attention(black_box(&q), black_box(&k), black_box(&v), seq_len, 0)
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_attention_dot_product(criterion: &mut Criterion) {
    let head_dim = 64; // Standard head dimension

    // Benchmark just the dot product (building block)
    let a = vec![1.0f32; head_dim];
    let b = vec![1.0f32; head_dim];

    criterion.bench_function("dot_product_64", |bencher| {
        bencher.iter(|| {
            let mut sum = 0.0;
            let chunks = head_dim / 4;
            for i in 0..chunks {
                let idx = i * 4;
                sum += a[idx] * b[idx];
                sum += a[idx + 1] * b[idx + 1];
                sum += a[idx + 2] * b[idx + 2];
                sum += a[idx + 3] * b[idx + 3];
            }
            black_box(sum)
        });
    });
}

criterion_group!(
    benches,
    bench_attention_computation,
    bench_attention_gqa_ratios,
    bench_attention_dot_product
);
criterion_main!(benches);
