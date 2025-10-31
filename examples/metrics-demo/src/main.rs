//! Metrics System Demo
//!
//! This example demonstrates how to use the Realm.ai metrics system
//! to track inference performance, quality, and resource utilization.

use realm_metrics::{
    latency::GenerationTimer, types::CommonLabels, LatencyMetrics, MetricsCollector,
    QualityMetrics, ResourceMetrics,
};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("🚀 Realm.ai Metrics System Demo\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // ========== Create Metrics Collector ==========
    println!("📊 Creating metrics collector...");
    let mut collector = MetricsCollector::new();

    // Set common labels
    collector.set_common_labels(
        CommonLabels::new()
            .with_model("llama-7b")
            .with_backend("cpu"),
    );
    println!("   ✓ Collector created with common labels\n");

    // ========== Simulate Inference Requests ==========
    println!("⚡ Simulating inference requests...\n");

    for request_id in 1..=5 {
        println!("  Request #{}", request_id);

        // Start tracking this request
        collector.enqueue_request();
        std::thread::sleep(Duration::from_millis(5)); // Queue wait

        collector.dequeue_request();
        collector.start_request();

        // Track generation latency
        let mut timer = GenerationTimer::new();

        // Simulate token generation
        for token in 1..=20 {
            std::thread::sleep(Duration::from_millis(10)); // Simulate token generation
            timer.add_token();

            if token == 1 {
                let ttft = timer.ttft().unwrap();
                println!("    ⚡ TTFT: {:.2}ms", ttft.as_secs_f64() * 1000.0);
            }
        }

        let latency_metrics = timer.finish();
        collector.record_latency(latency_metrics.clone());

        println!(
            "    📈 Generated {} tokens in {:.2}ms ({:.1} tok/s)",
            latency_metrics.total_tokens,
            latency_metrics.total_time.as_secs_f64() * 1000.0,
            latency_metrics.tokens_per_sec
        );

        // Record quality metrics (simulated)
        let quality_metrics = QualityMetrics::new(
            8.5 + (request_id as f64 * 0.3),   // perplexity
            0.85 - (request_id as f64 * 0.02), // mean token probability
            0.65,                              // min token probability
            2.1,                               // entropy
            20,                                // num_tokens
        );
        collector.record_quality(quality_metrics);

        println!("    🎯 Perplexity: {:.2}", 8.5 + (request_id as f64 * 0.3));

        // Record resource metrics (simulated)
        let resource_metrics = ResourceMetrics::new(
            1_500_000_000, // 1.5 GB total
            900_000_000,   // 900 MB model
            400_000_000,   // 400 MB KV cache
            200_000_000,   // 200 MB activations
        )
        .with_kv_cache_stats(15, 20); // 15 cached tokens out of 20

        collector.record_resource(resource_metrics);

        // Complete request
        collector.record_request(20, Duration::from_millis(5));
        collector.finish_request();

        println!();
    }

    // ========== Display Summary ==========
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("📈 Metrics Summary\n");

    let summary = collector.summary();

    // Latency summary
    if let Some(latency) = summary.latency {
        println!("⚡ Latency Metrics:");
        println!("   • Mean TTFT: {:.2}ms", latency.mean_ttft_ms);
        println!("   • P99 TTFT: {:.2}ms", latency.p99_ttft_ms);
        println!(
            "   • Mean throughput: {:.1} tokens/sec",
            latency.mean_tokens_per_sec
        );
        println!("   • Total requests: {}", latency.total_requests);
        println!("   • Total tokens: {}", latency.total_tokens);
        println!();
    }

    // Quality summary
    if let Some(quality) = summary.quality {
        println!("🎯 Quality Metrics:");
        println!("   • Mean perplexity: {:.2}", quality.mean_perplexity);
        println!("   • P99 perplexity: {:.2}", quality.p99_perplexity);
        println!(
            "   • Mean token probability: {:.2}",
            quality.mean_token_probability
        );
        println!();
    }

    // Resource summary
    if let Some(resource) = summary.resource {
        println!("💾 Resource Metrics:");
        println!("   • Mean memory: {:.2} GB", resource.mean_memory_gb);
        println!("   • Peak memory: {:.2} GB", resource.peak_memory_gb);
        println!(
            "   • Cache hit rate: {:.1}%",
            resource.cache_hit_rate * 100.0
        );
        println!();
    }

    // Throughput summary
    if let Some(throughput) = summary.throughput {
        println!("🚀 Throughput Metrics:");
        println!("   • Requests/sec: {:.2}", throughput.requests_per_sec);
        println!("   • Tokens/sec: {:.1}", throughput.tokens_per_sec);
        println!(
            "   • Concurrent requests: {}",
            throughput.concurrent_requests
        );
        println!("   • Queue depth: {}", throughput.queue_depth);
        println!();
    }

    // ========== Export Metrics ==========
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("📤 Exporting Metrics\n");

    // Export as JSON
    let json = collector.export_json()?;
    println!("✓ JSON export ({} bytes)", json.len());
    println!();

    // Export as Prometheus
    let prometheus = collector.export_prometheus();
    println!("✓ Prometheus export ({} bytes)", prometheus.len());
    println!();

    // ========== Per-Tenant Metrics ==========
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("👥 Per-Tenant Metrics\n");

    // Simulate tenant-specific requests
    let tenants = vec!["user_alice", "user_bob", "user_charlie"];

    for tenant_id in &tenants {
        let tenant_metrics = collector.tenant_metrics(*tenant_id).unwrap();

        // Record some requests for this tenant
        tenant_metrics.record_latency(LatencyMetrics::new(
            Duration::from_millis(100),
            Duration::from_secs(1),
            30,
        ));

        tenant_metrics.record_request(30, Duration::from_millis(5));

        println!("   ✓ Tracked metrics for {}", tenant_id);
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("✅ Demo complete! Metrics system is working correctly.\n");
    println!("💡 Next steps:");
    println!("   • Integrate with your inference pipeline");
    println!("   • Export to Prometheus/OpenTelemetry");
    println!("   • Set up dashboards for visualization");
    println!();

    Ok(())
}
