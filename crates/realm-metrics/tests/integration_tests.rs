//! Integration tests for realm-metrics
//!
//! These tests verify that all components work together correctly
//! in realistic inference scenarios.

use realm_metrics::{
    latency::GenerationTimer, types::CommonLabels, LatencyMetrics, MetricsCollector,
    QualityMetrics, ResourceMetrics,
};
use std::time::Duration;

#[test]
fn test_end_to_end_inference_tracking() {
    // Simulate a complete inference request from start to finish
    let mut collector = MetricsCollector::new();

    // Set up common labels
    collector.set_common_labels(
        CommonLabels::new()
            .with_model("test-model")
            .with_backend("cpu"),
    );

    // Simulate queue phase
    collector.enqueue_request();
    std::thread::sleep(Duration::from_millis(5));

    // Start processing
    collector.dequeue_request();
    collector.start_request();

    // Track generation
    let mut timer = GenerationTimer::new();

    // Generate tokens
    for _ in 0..20 {
        std::thread::sleep(Duration::from_millis(2));
        timer.add_token();
    }

    let latency = timer.finish();
    collector.record_latency(latency.clone());

    // Record quality
    let quality = QualityMetrics::new(10.5, 0.85, 0.6, 2.1, 20);
    collector.record_quality(quality);

    // Record resource usage
    let resource = ResourceMetrics::new(1_000_000_000, 600_000_000, 300_000_000, 100_000_000)
        .with_kv_cache_stats(15, 20);
    collector.record_resource(resource);

    // Complete request
    collector.record_request(20, Duration::from_millis(5));
    collector.finish_request();

    // Verify metrics
    let summary = collector.summary();

    assert!(summary.latency.is_some());
    assert!(summary.quality.is_some());
    assert!(summary.resource.is_some());
    assert!(summary.throughput.is_some());

    let latency_summary = summary.latency.unwrap();
    assert_eq!(latency_summary.total_requests, 1);
    assert_eq!(latency_summary.total_tokens, 20);
    assert!(latency_summary.mean_ttft_ms > 0.0);
}

#[test]
fn test_concurrent_requests() {
    let collector = MetricsCollector::new();

    // Start multiple requests
    collector.start_request();
    collector.start_request();
    collector.start_request();

    // Record some latencies
    for i in 1..=3 {
        collector.record_latency(LatencyMetrics::new(
            Duration::from_millis(100 + i * 10),
            Duration::from_secs(1),
            20,
        ));
    }

    // Finish requests
    collector.finish_request();
    collector.finish_request();
    collector.finish_request();

    let summary = collector.summary();
    let latency = summary.latency.unwrap();

    assert_eq!(latency.total_requests, 3);
    assert_eq!(latency.total_tokens, 60);
}

#[test]
fn test_multi_tenant_isolation() {
    let collector = MetricsCollector::new();

    // Create tenant-specific metrics
    let tenant1 = collector.tenant_metrics("tenant1").unwrap();
    let tenant2 = collector.tenant_metrics("tenant2").unwrap();

    // Record different metrics for each tenant
    tenant1.record_latency(LatencyMetrics::new(
        Duration::from_millis(100),
        Duration::from_secs(1),
        25,
    ));

    tenant2.record_latency(LatencyMetrics::new(
        Duration::from_millis(200),
        Duration::from_secs(2),
        50,
    ));

    // Both tenants should maintain separate metrics
    assert_eq!(tenant1.tenant_id(), "tenant1");
    assert_eq!(tenant2.tenant_id(), "tenant2");
}

#[test]
fn test_export_json_format() {
    let collector = MetricsCollector::new();

    collector.record_latency(LatencyMetrics::new(
        Duration::from_millis(100),
        Duration::from_secs(1),
        20,
    ));

    let json = collector.export_json();
    assert!(json.is_ok());

    let json_str = json.unwrap();
    assert!(!json_str.is_empty());
    assert!(json_str.contains("\"timestamp\""));
}

#[test]
fn test_export_prometheus_format() {
    let collector = MetricsCollector::new();

    collector.record_latency(LatencyMetrics::new(
        Duration::from_millis(100),
        Duration::from_secs(1),
        20,
    ));

    let prometheus = collector.export_prometheus();
    assert!(!prometheus.is_empty());
}

#[test]
fn test_cache_hit_tracking() {
    let collector = MetricsCollector::new();

    // Record cache hits and misses
    for _ in 0..8 {
        collector.record_cache_hit();
    }
    for _ in 0..2 {
        collector.record_cache_miss();
    }

    // Total: 8 hits out of 10 requests = 80% hit rate
    // Note: We can't directly verify this without accessing internal state
    // but we can verify it doesn't panic
}

#[test]
fn test_queue_management() {
    let collector = MetricsCollector::new();

    // Simulate requests entering and leaving queue
    collector.enqueue_request();
    collector.enqueue_request();
    collector.enqueue_request();

    collector.dequeue_request();
    collector.dequeue_request();

    // Should have 1 request in queue
    // Verify this doesn't panic
}

#[test]
fn test_empty_collector() {
    let collector = MetricsCollector::new();

    let summary = collector.summary();

    // All metrics should be None or have zero values
    assert!(summary.latency.is_none() || summary.latency.unwrap().total_requests == 0);
}

#[test]
fn test_large_volume_tracking() {
    let collector = MetricsCollector::new();

    // Simulate 1000 requests
    for i in 1..=1000 {
        collector.start_request();
        collector.record_latency(LatencyMetrics::new(
            Duration::from_millis(50 + (i % 100)),
            Duration::from_secs(1),
            10 + (i % 20),
        ));
        collector.record_request(10 + (i % 20), Duration::from_millis(5));
        collector.finish_request();
    }

    let summary = collector.summary();
    let latency = summary.latency.unwrap();

    assert_eq!(latency.total_requests, 1000);
    assert!(latency.mean_ttft_ms > 0.0);
    assert!(latency.p99_ttft_ms >= latency.mean_ttft_ms);
}

#[test]
fn test_zero_duration_handling() {
    let collector = MetricsCollector::new();

    // Edge case: zero duration
    collector.record_latency(LatencyMetrics::new(Duration::ZERO, Duration::ZERO, 10));

    let summary = collector.summary();
    assert!(summary.latency.is_some());
}

#[test]
fn test_quality_metrics_integration() {
    let collector = MetricsCollector::new();

    // Record quality metrics with various values
    collector.record_quality(QualityMetrics::new(5.5, 0.95, 0.85, 1.5, 100));
    collector.record_quality(QualityMetrics::new(12.3, 0.75, 0.55, 2.8, 100));

    let summary = collector.summary();
    let quality = summary.quality.unwrap();

    assert!(quality.mean_perplexity > 0.0);
    assert!(quality.mean_token_probability > 0.0);
    assert!(quality.mean_token_probability <= 1.0);
}

#[test]
fn test_resource_metrics_integration() {
    let collector = MetricsCollector::new();

    // Record increasing memory usage
    for i in 1..=5 {
        let resource = ResourceMetrics::new(
            (i as u64) * 500_000_000,
            300_000_000,
            (i as u64) * 100_000_000,
            50_000_000,
        );
        collector.record_resource(resource);
    }

    let summary = collector.summary();
    let resource = summary.resource.unwrap();

    assert!(resource.peak_memory_gb > resource.mean_memory_gb);
    assert!(resource.peak_memory_gb > 0.0);
}

#[test]
fn test_labels_propagation() {
    let mut collector = MetricsCollector::new();

    collector.add_label("model", "llama-7b");
    collector.add_label("gpu", "a100");

    collector.record_latency(LatencyMetrics::new(
        Duration::from_millis(100),
        Duration::from_secs(1),
        20,
    ));

    let samples = collector.export_all();

    // Verify labels are present in exported samples
    assert!(samples.iter().any(|s| !s.labels.is_empty()));
}

#[test]
fn test_realistic_inference_scenario() {
    let mut collector = MetricsCollector::new();
    collector.set_common_labels(
        CommonLabels::new()
            .with_model("llama-13b")
            .with_backend("cuda"),
    );

    // Simulate 10 inference requests with realistic metrics
    for request_id in 1..=10 {
        collector.start_request();

        // TTFT varies 50-150ms
        let ttft = Duration::from_millis(50 + (request_id * 10));
        // Generation takes 1-2 seconds
        let total_time = Duration::from_millis(1000 + (request_id * 100));
        // Generate 20-30 tokens
        let tokens = 20u64 + (request_id % 10);

        collector.record_latency(LatencyMetrics::new(ttft, total_time, tokens));

        // Quality varies slightly
        let perplexity = 8.0 + (request_id as f64 * 0.2);
        collector.record_quality(QualityMetrics::new(
            perplexity,
            0.85 - (request_id as f64 * 0.01),
            0.65,
            2.0,
            tokens,
        ));

        // Memory grows with KV cache
        let kv_cache_size = 200_000_000 + (request_id * 10_000_000);
        collector.record_resource(ResourceMetrics::new(
            1_500_000_000,
            900_000_000,
            kv_cache_size,
            100_000_000,
        ));

        collector.record_request(tokens, Duration::from_millis(5));
        collector.finish_request();
    }

    // Verify comprehensive summary
    let summary = collector.summary();

    assert!(summary.latency.is_some());
    assert!(summary.quality.is_some());
    assert!(summary.resource.is_some());
    assert!(summary.throughput.is_some());

    let latency = summary.latency.unwrap();
    assert_eq!(latency.total_requests, 10);
    assert!(latency.mean_ttft_ms >= 50.0);
    assert!(latency.p99_ttft_ms > latency.mean_ttft_ms);

    let quality = summary.quality.unwrap();
    assert!(quality.mean_perplexity >= 8.0);
    assert!(quality.p99_perplexity >= quality.mean_perplexity);
}

#[test]
fn test_collector_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let collector = Arc::new(MetricsCollector::new());
    let mut handles = vec![];

    // Spawn 10 threads that all record metrics
    for i in 0..10 {
        let collector_clone = Arc::clone(&collector);
        let handle = thread::spawn(move || {
            for _ in 0..10 {
                collector_clone.record_latency(LatencyMetrics::new(
                    Duration::from_millis(100 + i * 10),
                    Duration::from_secs(1),
                    20,
                ));
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify we got all metrics
    let summary = collector.summary();
    let latency = summary.latency.unwrap();
    assert_eq!(latency.total_requests, 100); // 10 threads * 10 requests each
}
