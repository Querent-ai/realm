//! Resource utilization metrics
//!
//! This module tracks resource usage for AI inference:
//! - **Memory**: Model weights, KV cache, activation memory
//! - **Cache**: KV cache hit rates, prefill vs decode cache usage
//! - **Compute**: CPU/GPU utilization (when available)

use crate::types::{now_millis, MetricLabel, MetricSample, MetricValue, RollingWindow};

/// Resource metrics snapshot
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    /// Total memory allocated (bytes)
    pub memory_allocated_bytes: u64,
    /// Memory used by model weights (bytes)
    pub model_memory_bytes: u64,
    /// Memory used by KV cache (bytes)
    pub kv_cache_memory_bytes: u64,
    /// Memory used by activations/temporary tensors (bytes)
    pub activation_memory_bytes: u64,
    /// KV cache hit rate (0.0 to 1.0)
    pub kv_cache_hit_rate: f64,
    /// Number of cached tokens
    pub cached_tokens: u64,
    /// Total tokens in KV cache
    pub total_kv_tokens: u64,
}

impl ResourceMetrics {
    /// Create new resource metrics
    pub fn new(
        memory_allocated_bytes: u64,
        model_memory_bytes: u64,
        kv_cache_memory_bytes: u64,
        activation_memory_bytes: u64,
    ) -> Self {
        Self {
            memory_allocated_bytes,
            model_memory_bytes,
            kv_cache_memory_bytes,
            activation_memory_bytes,
            kv_cache_hit_rate: 0.0,
            cached_tokens: 0,
            total_kv_tokens: 0,
        }
    }

    /// Update KV cache statistics
    pub fn with_kv_cache_stats(mut self, cached_tokens: u64, total_tokens: u64) -> Self {
        self.cached_tokens = cached_tokens;
        self.total_kv_tokens = total_tokens;
        self.kv_cache_hit_rate = if total_tokens > 0 {
            cached_tokens as f64 / total_tokens as f64
        } else {
            0.0
        };
        self
    }

    /// Get total memory usage in GB
    pub fn total_memory_gb(&self) -> f64 {
        self.memory_allocated_bytes as f64 / 1_073_741_824.0 // 1024^3
    }

    /// Get model memory in GB
    pub fn model_memory_gb(&self) -> f64 {
        self.model_memory_bytes as f64 / 1_073_741_824.0
    }

    /// Get KV cache memory in GB
    pub fn kv_cache_memory_gb(&self) -> f64 {
        self.kv_cache_memory_bytes as f64 / 1_073_741_824.0
    }

    /// Convert to metric samples for export
    pub fn to_samples(&self, labels: Vec<MetricLabel>) -> Vec<MetricSample> {
        let timestamp = now_millis();
        vec![
            MetricSample {
                value: MetricValue::Gauge(self.memory_allocated_bytes as f64),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.model_memory_bytes as f64),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.kv_cache_memory_bytes as f64),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.activation_memory_bytes as f64),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.kv_cache_hit_rate),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Counter(self.cached_tokens),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Counter(self.total_kv_tokens),
                timestamp,
                labels,
            },
        ]
    }
}

/// Resource tracker with rolling window statistics
pub struct ResourceTracker {
    /// Rolling window for memory usage
    memory_window: RollingWindow,
    /// Rolling window for KV cache hit rate
    cache_hit_rate_window: RollingWindow,
    /// Current resource snapshot
    current: Option<ResourceMetrics>,
    /// Peak memory usage observed (bytes)
    peak_memory_bytes: u64,
    /// Total cache hits (cumulative)
    total_cache_hits: u64,
    /// Total cache requests (cumulative)
    total_cache_requests: u64,
}

impl ResourceTracker {
    /// Create a new resource tracker with the given window size
    pub fn new(window_size: usize) -> Self {
        Self {
            memory_window: RollingWindow::new(window_size),
            cache_hit_rate_window: RollingWindow::new(window_size),
            current: None,
            peak_memory_bytes: 0,
            total_cache_hits: 0,
            total_cache_requests: 0,
        }
    }

    /// Record a resource metrics snapshot
    pub fn record(&mut self, metrics: ResourceMetrics) {
        self.memory_window
            .add(metrics.memory_allocated_bytes as f64);
        self.cache_hit_rate_window.add(metrics.kv_cache_hit_rate);

        // Update peak memory
        if metrics.memory_allocated_bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = metrics.memory_allocated_bytes;
        }

        // Update cumulative cache stats
        self.total_cache_requests += metrics.total_kv_tokens;
        self.total_cache_hits += metrics.cached_tokens;

        self.current = Some(metrics);
    }

    /// Record a cache hit
    pub fn record_cache_hit(&mut self) {
        self.total_cache_hits += 1;
        self.total_cache_requests += 1;
    }

    /// Record a cache miss
    pub fn record_cache_miss(&mut self) {
        self.total_cache_requests += 1;
    }

    /// Get mean memory usage over the window
    pub fn mean_memory_bytes(&self) -> u64 {
        self.memory_window.mean() as u64
    }

    /// Get mean memory usage in GB
    pub fn mean_memory_gb(&self) -> f64 {
        self.mean_memory_bytes() as f64 / 1_073_741_824.0
    }

    /// Get peak memory usage observed
    pub fn peak_memory_bytes(&self) -> u64 {
        self.peak_memory_bytes
    }

    /// Get peak memory usage in GB
    pub fn peak_memory_gb(&self) -> f64 {
        self.peak_memory_bytes as f64 / 1_073_741_824.0
    }

    /// Get mean KV cache hit rate over the window
    pub fn mean_cache_hit_rate(&self) -> f64 {
        self.cache_hit_rate_window.mean()
    }

    /// Get overall cache hit rate (cumulative)
    pub fn overall_cache_hit_rate(&self) -> f64 {
        if self.total_cache_requests > 0 {
            self.total_cache_hits as f64 / self.total_cache_requests as f64
        } else {
            0.0
        }
    }

    /// Get current resource snapshot
    pub fn current(&self) -> Option<&ResourceMetrics> {
        self.current.as_ref()
    }

    /// Export current statistics as metric samples
    pub fn export_samples(&self, labels: Vec<MetricLabel>) -> Vec<MetricSample> {
        let timestamp = now_millis();
        let mut samples = vec![
            MetricSample {
                value: MetricValue::Gauge(self.mean_memory_bytes() as f64),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.peak_memory_bytes as f64),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.mean_cache_hit_rate()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Counter(self.total_cache_hits),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Counter(self.total_cache_requests),
                timestamp,
                labels: labels.clone(),
            },
        ];

        // Add current snapshot if available
        if let Some(current) = &self.current {
            samples.extend(current.to_samples(labels));
        }

        samples
    }
}

impl Default for ResourceTracker {
    fn default() -> Self {
        Self::new(100) // Default to 100-sample window
    }
}

/// Memory breakdown for detailed tracking
#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    /// Model weights memory
    pub weights: u64,
    /// KV cache memory
    pub kv_cache: u64,
    /// Attention activations
    pub attention: u64,
    /// FFN activations
    pub ffn: u64,
    /// Other/temporary allocations
    pub other: u64,
}

impl MemoryBreakdown {
    /// Create a new memory breakdown
    pub fn new() -> Self {
        Self {
            weights: 0,
            kv_cache: 0,
            attention: 0,
            ffn: 0,
            other: 0,
        }
    }

    /// Get total memory
    pub fn total(&self) -> u64 {
        self.weights + self.kv_cache + self.attention + self.ffn + self.other
    }

    /// Get total in GB
    pub fn total_gb(&self) -> f64 {
        self.total() as f64 / 1_073_741_824.0
    }

    /// Get percentage of each component
    pub fn percentages(&self) -> MemoryPercentages {
        let total = self.total() as f64;
        if total == 0.0 {
            return MemoryPercentages {
                weights: 0.0,
                kv_cache: 0.0,
                attention: 0.0,
                ffn: 0.0,
                other: 0.0,
            };
        }

        MemoryPercentages {
            weights: (self.weights as f64 / total) * 100.0,
            kv_cache: (self.kv_cache as f64 / total) * 100.0,
            attention: (self.attention as f64 / total) * 100.0,
            ffn: (self.ffn as f64 / total) * 100.0,
            other: (self.other as f64 / total) * 100.0,
        }
    }
}

impl Default for MemoryBreakdown {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory percentages for visualization
#[derive(Debug, Clone)]
pub struct MemoryPercentages {
    pub weights: f64,
    pub kv_cache: f64,
    pub attention: f64,
    pub ffn: f64,
    pub other: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_metrics() {
        let metrics = ResourceMetrics::new(
            1_000_000_000, // 1 GB total
            600_000_000,   // 600 MB model
            300_000_000,   // 300 MB KV cache
            100_000_000,   // 100 MB activations
        )
        .with_kv_cache_stats(800, 1000);

        assert_eq!(metrics.kv_cache_hit_rate, 0.8);
        assert!((metrics.total_memory_gb() - 0.93).abs() < 0.01); // ~0.93 GB
    }

    #[test]
    fn test_resource_tracker() {
        let mut tracker = ResourceTracker::new(3);

        tracker.record(ResourceMetrics::new(
            1_000_000_000,
            600_000_000,
            300_000_000,
            100_000_000,
        ));

        assert_eq!(tracker.peak_memory_bytes(), 1_000_000_000);
        assert!(tracker.peak_memory_gb() > 0.9);
    }

    #[test]
    fn test_cache_tracking() {
        let mut tracker = ResourceTracker::new(10);

        // Record some cache hits and misses
        tracker.record_cache_hit();
        tracker.record_cache_hit();
        tracker.record_cache_miss();

        assert_eq!(tracker.total_cache_hits, 2);
        assert_eq!(tracker.total_cache_requests, 3);
        assert!((tracker.overall_cache_hit_rate() - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_memory_breakdown() {
        let mut breakdown = MemoryBreakdown::new();
        breakdown.weights = 1_000_000_000; // 1 GB
        breakdown.kv_cache = 500_000_000; // 500 MB
        breakdown.attention = 300_000_000; // 300 MB
        breakdown.ffn = 100_000_000; // 100 MB
        breakdown.other = 100_000_000; // 100 MB

        assert_eq!(breakdown.total(), 2_000_000_000);

        let percentages = breakdown.percentages();
        assert!((percentages.weights - 50.0).abs() < 0.1);
        assert!((percentages.kv_cache - 25.0).abs() < 0.1);
    }
}
