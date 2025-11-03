//! Central metrics collector
//!
//! This module provides the main `MetricsCollector` that aggregates all metric types
//! and provides a unified interface for recording and exporting metrics.

use crate::business::BusinessMetricsTracker;
use crate::latency::{LatencyMetrics, LatencyTracker};
use crate::quality::{QualityMetrics, QualityTracker};
use crate::resource::{ResourceMetrics, ResourceTracker};
use crate::throughput::ThroughputTracker;
use crate::types::{CommonLabels, MetricSample};
use crate::usage::{CostConfig, TokenUsage, UsageTracker};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Central metrics collector
///
/// This is the main entry point for recording and exporting metrics.
/// It aggregates all metric types (latency, quality, resource, throughput)
/// and provides thread-safe access for multi-tenant scenarios.
pub struct MetricsCollector {
    /// Latency metrics tracker
    latency: Arc<Mutex<LatencyTracker>>,
    /// Quality metrics tracker
    quality: Arc<Mutex<QualityTracker>>,
    /// Resource metrics tracker
    resource: Arc<Mutex<ResourceTracker>>,
    /// Throughput metrics tracker
    throughput: Arc<Mutex<ThroughputTracker>>,
    /// Usage tracker for billing and cost metrics
    usage: Arc<Mutex<UsageTracker>>,
    /// Business metrics tracker for error rates, success tracking, client attribution
    business: Arc<Mutex<BusinessMetricsTracker>>,
    /// Common labels applied to all metrics
    common_labels: CommonLabels,
    /// Per-tenant metrics (optional)
    tenant_metrics: Arc<Mutex<HashMap<String, TenantMetrics>>>,
}

impl MetricsCollector {
    /// Create a new metrics collector with default window size
    pub fn new() -> Self {
        Self::with_window_size(100)
    }

    /// Create a new metrics collector with specified window size
    pub fn with_window_size(window_size: usize) -> Self {
        Self {
            latency: Arc::new(Mutex::new(LatencyTracker::new(window_size))),
            quality: Arc::new(Mutex::new(QualityTracker::new(window_size))),
            resource: Arc::new(Mutex::new(ResourceTracker::new(window_size))),
            throughput: Arc::new(Mutex::new(ThroughputTracker::new(window_size))),
            usage: Arc::new(Mutex::new(UsageTracker::new(CostConfig::simple(1.0, 2.0)))),
            business: Arc::new(Mutex::new(BusinessMetricsTracker::new())),
            common_labels: CommonLabels::new(),
            tenant_metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new metrics collector with usage cost configuration
    pub fn with_cost_config(cost_config: CostConfig) -> Self {
        let mut collector = Self::new();
        collector.usage = Arc::new(Mutex::new(UsageTracker::new(cost_config)));
        collector
    }

    /// Set common labels that will be applied to all metrics
    pub fn set_common_labels(&mut self, labels: CommonLabels) {
        self.common_labels = labels;
    }

    /// Add a common label
    pub fn add_label(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.common_labels.add(key, value);
    }

    // ========== Latency Metrics ==========

    /// Record latency metrics
    pub fn record_latency(&self, metrics: LatencyMetrics) {
        if let Ok(mut tracker) = self.latency.lock() {
            tracker.record(metrics);
        }
    }

    /// Get latency tracker for advanced usage
    pub fn latency_tracker(&self) -> Arc<Mutex<LatencyTracker>> {
        Arc::clone(&self.latency)
    }

    // ========== Quality Metrics ==========

    /// Record quality metrics
    pub fn record_quality(&self, metrics: QualityMetrics) {
        if let Ok(mut tracker) = self.quality.lock() {
            tracker.record(metrics);
        }
    }

    /// Get quality tracker for advanced usage
    pub fn quality_tracker(&self) -> Arc<Mutex<QualityTracker>> {
        Arc::clone(&self.quality)
    }

    // ========== Resource Metrics ==========

    /// Record resource metrics
    pub fn record_resource(&self, metrics: ResourceMetrics) {
        if let Ok(mut tracker) = self.resource.lock() {
            tracker.record(metrics);
        }
    }

    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        if let Ok(mut tracker) = self.resource.lock() {
            tracker.record_cache_hit();
        }
    }

    /// Record a cache miss
    pub fn record_cache_miss(&self) {
        if let Ok(mut tracker) = self.resource.lock() {
            tracker.record_cache_miss();
        }
    }

    /// Get resource tracker for advanced usage
    pub fn resource_tracker(&self) -> Arc<Mutex<ResourceTracker>> {
        Arc::clone(&self.resource)
    }

    // ========== Throughput Metrics ==========

    /// Record a completed request
    pub fn record_request(&self, tokens_generated: u64, queue_wait: std::time::Duration) {
        if let Ok(mut tracker) = self.throughput.lock() {
            tracker.record_request(tokens_generated, queue_wait);
        }
    }

    /// Mark that a request has started
    pub fn start_request(&self) {
        if let Ok(mut tracker) = self.throughput.lock() {
            tracker.start_request();
        }
    }

    /// Mark that a request has finished
    pub fn finish_request(&self) {
        if let Ok(mut tracker) = self.throughput.lock() {
            tracker.finish_request();
        }
    }

    /// Add a request to the queue
    pub fn enqueue_request(&self) {
        if let Ok(mut tracker) = self.throughput.lock() {
            tracker.enqueue_request();
        }
    }

    /// Remove a request from the queue
    pub fn dequeue_request(&self) {
        if let Ok(mut tracker) = self.throughput.lock() {
            tracker.dequeue_request();
        }
    }

    /// Get throughput tracker for advanced usage
    pub fn throughput_tracker(&self) -> Arc<Mutex<ThroughputTracker>> {
        Arc::clone(&self.throughput)
    }

    // ========== Usage/Billing Metrics ==========

    /// Record token usage for billing and cost tracking
    pub fn record_usage(
        &self,
        usage: TokenUsage,
        tenant_id: Option<&str>,
        model_name: Option<&str>,
    ) {
        if let Ok(mut tracker) = self.usage.lock() {
            tracker.record_usage(usage, tenant_id, model_name);
        }
    }

    /// Get usage tracker for advanced usage
    pub fn usage_tracker(&self) -> Arc<Mutex<UsageTracker>> {
        Arc::clone(&self.usage)
    }

    // ========== Business Metrics ==========

    /// Record a request for business metrics (success/failure, client attribution)
    pub fn record_business_request(
        &self,
        status_code: u16,
        api_key: Option<&str>,
        client_id: Option<&str>,
        tokens: Option<u64>,
    ) {
        if let Ok(mut tracker) = self.business.lock() {
            tracker.record_request(status_code, api_key, client_id, tokens);
        }
    }

    /// Record revenue (if charging customers)
    pub fn record_revenue(&self, amount: f64, tenant_id: Option<&str>, client_id: Option<&str>) {
        if let Ok(mut tracker) = self.business.lock() {
            tracker.record_revenue(amount, tenant_id, client_id);
        }
    }

    /// Get business metrics tracker for advanced usage
    pub fn business_tracker(&self) -> Arc<Mutex<BusinessMetricsTracker>> {
        Arc::clone(&self.business)
    }

    // ========== Per-Tenant Metrics ==========

    /// Get or create metrics for a specific tenant
    pub fn tenant_metrics(&self, tenant_id: impl Into<String>) -> Option<TenantMetrics> {
        let tenant_id = tenant_id.into();
        if let Ok(mut metrics) = self.tenant_metrics.lock() {
            Some(
                metrics
                    .entry(tenant_id.clone())
                    .or_insert_with(|| TenantMetrics::new(tenant_id))
                    .clone(),
            )
        } else {
            None
        }
    }

    // ========== Export ==========

    /// Export all metrics as samples
    pub fn export_all(&self) -> Vec<MetricSample> {
        let mut samples = Vec::new();
        let labels = self.common_labels.to_vec();

        // Export latency metrics
        if let Ok(tracker) = self.latency.lock() {
            samples.extend(tracker.export_samples(labels.clone()));
        }

        // Export quality metrics
        if let Ok(tracker) = self.quality.lock() {
            samples.extend(tracker.export_samples(labels.clone()));
        }

        // Export resource metrics
        if let Ok(tracker) = self.resource.lock() {
            samples.extend(tracker.export_samples(labels.clone()));
        }

        // Export throughput metrics
        if let Ok(tracker) = self.throughput.lock() {
            samples.extend(tracker.export_samples(labels.clone()));
        }

        // Export usage/billing metrics
        if let Ok(tracker) = self.usage.lock() {
            samples.extend(tracker.export_samples(labels.clone()));
            // Also export cache savings
            samples.extend(tracker.export_cache_savings(labels.clone()));
        }

        // Export business metrics (error rates, client attribution)
        if let Ok(tracker) = self.business.lock() {
            samples.extend(tracker.export_samples(labels));
        }

        samples
    }

    /// Export metrics in Prometheus text format
    pub fn export_prometheus(&self) -> String {
        let samples = self.export_all();
        format_prometheus(&samples)
    }

    /// Export metrics as JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        let samples = self.export_all();
        serde_json::to_string_pretty(&samples)
    }

    /// Get a summary of current metrics
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            latency: self.latency.lock().ok().map(|t| LatencySummary {
                mean_ttft_ms: t.mean_ttft().as_millis() as f64,
                p99_ttft_ms: t.p99_ttft().as_millis() as f64,
                mean_tokens_per_sec: t.mean_tokens_per_sec(),
                total_requests: t.total_requests(),
                total_tokens: t.total_tokens(),
            }),
            quality: self.quality.lock().ok().map(|t| QualitySummary {
                mean_perplexity: t.mean_perplexity(),
                p99_perplexity: t.p99_perplexity(),
                mean_token_probability: t.mean_token_probability(),
            }),
            resource: self.resource.lock().ok().map(|t| ResourceSummary {
                mean_memory_gb: t.mean_memory_gb(),
                peak_memory_gb: t.peak_memory_gb(),
                cache_hit_rate: t.overall_cache_hit_rate(),
            }),
            throughput: self.throughput.lock().ok().map(|t| ThroughputSummary {
                requests_per_sec: t.requests_per_sec(),
                tokens_per_sec: t.tokens_per_sec(),
                concurrent_requests: t.concurrent_requests(),
                queue_depth: t.queue_depth(),
            }),
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-tenant metrics tracking
#[derive(Clone)]
pub struct TenantMetrics {
    tenant_id: String,
    latency: Arc<Mutex<LatencyTracker>>,
    throughput: Arc<Mutex<ThroughputTracker>>,
}

impl TenantMetrics {
    fn new(tenant_id: String) -> Self {
        Self {
            tenant_id,
            latency: Arc::new(Mutex::new(LatencyTracker::new(100))),
            throughput: Arc::new(Mutex::new(ThroughputTracker::new(100))),
        }
    }

    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    pub fn record_latency(&self, metrics: LatencyMetrics) {
        if let Ok(mut tracker) = self.latency.lock() {
            tracker.record(metrics);
        }
    }

    pub fn record_request(&self, tokens: u64, queue_wait: std::time::Duration) {
        if let Ok(mut tracker) = self.throughput.lock() {
            tracker.record_request(tokens, queue_wait);
        }
    }
}

/// Summary of all metrics
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub latency: Option<LatencySummary>,
    pub quality: Option<QualitySummary>,
    pub resource: Option<ResourceSummary>,
    pub throughput: Option<ThroughputSummary>,
}

#[derive(Debug, Clone)]
pub struct LatencySummary {
    pub mean_ttft_ms: f64,
    pub p99_ttft_ms: f64,
    pub mean_tokens_per_sec: f64,
    pub total_requests: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct QualitySummary {
    pub mean_perplexity: f64,
    pub p99_perplexity: f64,
    pub mean_token_probability: f64,
}

#[derive(Debug, Clone)]
pub struct ResourceSummary {
    pub mean_memory_gb: f64,
    pub peak_memory_gb: f64,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Clone)]
pub struct ThroughputSummary {
    pub requests_per_sec: f64,
    pub tokens_per_sec: f64,
    pub concurrent_requests: u64,
    pub queue_depth: u64,
}

/// Format metrics as Prometheus text format
fn format_prometheus(samples: &[MetricSample]) -> String {
    use crate::types::MetricValue;

    let mut output = String::new();
    let mut metrics_by_name: HashMap<String, Vec<&MetricSample>> = HashMap::new();

    // Group samples by metric name (derived from labels or position)
    for (i, sample) in samples.iter().enumerate() {
        let metric_name = format!("realm_metric_{}", i);
        metrics_by_name.entry(metric_name).or_default().push(sample);
    }

    // Format each metric group
    for (name, samples) in metrics_by_name {
        for sample in samples {
            let labels_str = if sample.labels.is_empty() {
                String::new()
            } else {
                let labels: Vec<String> = sample
                    .labels
                    .iter()
                    .map(|l| format!("{}=\"{}\"", l.key, l.value))
                    .collect();
                format!("{{{}}}", labels.join(","))
            };

            let value = match &sample.value {
                MetricValue::Counter(v) => *v as f64,
                MetricValue::Gauge(v) => *v,
                MetricValue::Histogram { sum, .. } => *sum,
                MetricValue::Summary { sum, .. } => *sum,
            };

            output.push_str(&format!(
                "{}{} {} {}\n",
                name, labels_str, value, sample.timestamp
            ));
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();

        // Record some metrics
        collector.record_latency(LatencyMetrics::new(
            Duration::from_millis(100),
            Duration::from_secs(2),
            50,
        ));

        collector.record_quality(QualityMetrics::new(10.5, 0.85, 0.6, 2.1, 50));

        collector.record_resource(ResourceMetrics::new(
            1_000_000_000,
            600_000_000,
            300_000_000,
            100_000_000,
        ));

        collector.start_request();
        collector.record_request(50, Duration::from_millis(10));
        collector.finish_request();

        // Get summary
        let summary = collector.summary();
        assert!(summary.latency.is_some());
        assert!(summary.quality.is_some());
        assert!(summary.resource.is_some());
        assert!(summary.throughput.is_some());
    }

    #[test]
    fn test_export_json() {
        let collector = MetricsCollector::new();
        collector.record_latency(LatencyMetrics::new(
            Duration::from_millis(100),
            Duration::from_secs(1),
            10,
        ));

        let json = collector.export_json().unwrap();
        assert!(!json.is_empty());
    }

    #[test]
    fn test_common_labels() {
        let mut collector = MetricsCollector::new();
        collector.add_label("model", "llama-7b");
        collector.add_label("backend", "cuda");

        collector.record_latency(LatencyMetrics::new(
            Duration::from_millis(50),
            Duration::from_secs(1),
            20,
        ));

        let samples = collector.export_all();
        assert!(!samples.is_empty());
        // Check that labels are present
        assert!(samples.iter().any(|s| !s.labels.is_empty()));
    }

    #[test]
    fn test_tenant_metrics() {
        let collector = MetricsCollector::new();

        let tenant1 = collector.tenant_metrics("tenant_1").unwrap();
        tenant1.record_latency(LatencyMetrics::new(
            Duration::from_millis(100),
            Duration::from_secs(1),
            25,
        ));

        let tenant2 = collector.tenant_metrics("tenant_2").unwrap();
        tenant2.record_latency(LatencyMetrics::new(
            Duration::from_millis(150),
            Duration::from_secs(2),
            40,
        ));

        assert_eq!(tenant1.tenant_id(), "tenant_1");
        assert_eq!(tenant2.tenant_id(), "tenant_2");
    }
}
