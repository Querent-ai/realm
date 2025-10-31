//! Core types and traits for metrics collection
//!
//! This module defines the foundational types used throughout the metrics system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A single metric value with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSample {
    /// The metric value
    pub value: MetricValue,
    /// When the metric was recorded
    pub timestamp: u64, // Unix timestamp in milliseconds
    /// Labels for this metric (e.g., tenant_id, model_name)
    pub labels: Vec<MetricLabel>,
}

/// The value of a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    /// A counter that only increases (e.g., total_requests)
    Counter(u64),
    /// A gauge that can go up or down (e.g., memory_usage_bytes)
    Gauge(f64),
    /// A histogram bucket for latency tracking
    Histogram {
        count: u64,
        sum: f64,
        buckets: Vec<HistogramBucket>,
    },
    /// A summary with quantiles
    Summary {
        count: u64,
        sum: f64,
        quantiles: Vec<Quantile>,
    },
}

/// A histogram bucket for latency distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    /// Upper bound for this bucket (e.g., 0.1 for "≤100ms")
    pub le: f64,
    /// Cumulative count of samples in this bucket
    pub count: u64,
}

/// A quantile for summary metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quantile {
    /// Quantile value (e.g., 0.5 for median, 0.99 for p99)
    pub quantile: f64,
    /// Value at this quantile
    pub value: f64,
}

/// A label for a metric (key-value pair)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MetricLabel {
    pub key: String,
    pub value: String,
}

impl MetricLabel {
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// A timer for measuring duration
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    /// Start a new timer
    pub fn start(name: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            name: name.into(),
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and return elapsed time
    pub fn stop(self) -> Duration {
        self.elapsed()
    }

    /// Get the timer name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// A rolling window for computing statistics over recent samples
#[derive(Debug, Clone)]
pub struct RollingWindow {
    samples: Vec<f64>,
    capacity: usize,
    sum: f64,
}

impl RollingWindow {
    /// Create a new rolling window with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
            sum: 0.0,
        }
    }

    /// Add a sample to the window
    pub fn add(&mut self, value: f64) {
        if self.samples.len() >= self.capacity {
            // Remove oldest sample
            let oldest = self.samples.remove(0);
            self.sum -= oldest;
        }
        self.samples.push(value);
        self.sum += value;
    }

    /// Get the mean of samples in the window
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.sum / self.samples.len() as f64
        }
    }

    /// Get the median of samples in the window
    pub fn median(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Get the p99 (99th percentile) of samples in the window
    pub fn p99(&self) -> f64 {
        self.percentile(0.99)
    }

    /// Get a specific percentile
    pub fn percentile(&self, p: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((sorted.len() as f64 - 1.0) * p).ceil() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Get the number of samples in the window
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if the window is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Get current Unix timestamp in milliseconds
pub fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Common labels used across metrics
#[derive(Debug, Clone)]
pub struct CommonLabels {
    labels: HashMap<String, String>,
}

impl CommonLabels {
    /// Create new common labels
    pub fn new() -> Self {
        Self {
            labels: HashMap::new(),
        }
    }

    /// Add a label
    pub fn add(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.labels.insert(key.into(), value.into());
    }

    /// Get labels as a Vec
    pub fn to_vec(&self) -> Vec<MetricLabel> {
        self.labels
            .iter()
            .map(|(k, v)| MetricLabel::new(k.clone(), v.clone()))
            .collect()
    }

    /// Add model name label
    pub fn with_model(mut self, model_name: impl Into<String>) -> Self {
        self.add("model", model_name);
        self
    }

    /// Add tenant ID label
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.add("tenant_id", tenant_id);
        self
    }

    /// Add backend label (cpu, cuda, metal, webgpu)
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.add("backend", backend);
        self
    }
}

impl Default for CommonLabels {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_window() {
        let mut window = RollingWindow::new(3);
        window.add(1.0);
        window.add(2.0);
        window.add(3.0);
        assert_eq!(window.mean(), 2.0);
        assert_eq!(window.median(), 2.0);

        // Add one more, should evict oldest (1.0)
        window.add(4.0);
        assert_eq!(window.mean(), 3.0); // (2 + 3 + 4) / 3
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start("test");
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    fn test_common_labels() {
        let labels = CommonLabels::new()
            .with_model("llama-7b")
            .with_tenant("user123")
            .with_backend("cuda");

        let vec = labels.to_vec();
        assert_eq!(vec.len(), 3);
        assert!(vec
            .iter()
            .any(|l| l.key == "model" && l.value == "llama-7b"));
        assert!(vec
            .iter()
            .any(|l| l.key == "tenant_id" && l.value == "user123"));
        assert!(vec.iter().any(|l| l.key == "backend" && l.value == "cuda"));
    }

    #[test]
    fn test_rolling_window_empty() {
        let window = RollingWindow::new(5);
        assert_eq!(window.len(), 0);
        assert!(window.is_empty());
        assert_eq!(window.mean(), 0.0);
        assert_eq!(window.median(), 0.0);
        assert_eq!(window.p99(), 0.0);
    }

    #[test]
    fn test_rolling_window_single_value() {
        let mut window = RollingWindow::new(5);
        window.add(42.0);
        assert_eq!(window.len(), 1);
        assert!(!window.is_empty());
        assert_eq!(window.mean(), 42.0);
        assert_eq!(window.median(), 42.0);
        assert_eq!(window.p99(), 42.0);
    }

    #[test]
    fn test_rolling_window_percentiles() {
        let mut window = RollingWindow::new(100);
        for i in 1..=100 {
            window.add(i as f64);
        }

        // Test different percentiles
        // Note: percentile uses ceil, so median is slightly higher
        let median = window.percentile(0.5);
        assert!((50.0..=51.0).contains(&median)); // Median
        assert_eq!(window.percentile(0.0), 1.0); // Min
        assert_eq!(window.percentile(1.0), 100.0); // Max

        let p95 = window.percentile(0.95);
        assert!((95.0..=96.0).contains(&p95));
    }

    #[test]
    fn test_rolling_window_median_even_odd() {
        let mut window = RollingWindow::new(10);

        // Odd number of elements
        window.add(1.0);
        window.add(2.0);
        window.add(3.0);
        assert_eq!(window.median(), 2.0);

        // Even number of elements
        window.add(4.0);
        assert_eq!(window.median(), 2.5); // (2 + 3) / 2
    }

    #[test]
    fn test_metric_label_creation() {
        let label = MetricLabel::new("key", "value");
        assert_eq!(label.key, "key");
        assert_eq!(label.value, "value");
    }

    #[test]
    fn test_metric_label_equality() {
        let label1 = MetricLabel::new("key", "value");
        let label2 = MetricLabel::new("key", "value");
        let label3 = MetricLabel::new("key", "different");

        assert_eq!(label1, label2);
        assert_ne!(label1, label3);
    }

    #[test]
    fn test_metric_value_types() {
        let counter = MetricValue::Counter(100);
        let gauge = MetricValue::Gauge(42.5);
        let histogram = MetricValue::Histogram {
            count: 10,
            sum: 100.0,
            buckets: vec![],
        };
        let summary = MetricValue::Summary {
            count: 5,
            sum: 25.0,
            quantiles: vec![],
        };

        // Verify types are created correctly
        match counter {
            MetricValue::Counter(v) => assert_eq!(v, 100),
            _ => panic!("Expected Counter"),
        }

        match gauge {
            MetricValue::Gauge(v) => assert_eq!(v, 42.5),
            _ => panic!("Expected Gauge"),
        }

        match histogram {
            MetricValue::Histogram { count, sum, .. } => {
                assert_eq!(count, 10);
                assert_eq!(sum, 100.0);
            }
            _ => panic!("Expected Histogram"),
        }

        match summary {
            MetricValue::Summary { count, sum, .. } => {
                assert_eq!(count, 5);
                assert_eq!(sum, 25.0);
            }
            _ => panic!("Expected Summary"),
        }
    }

    #[test]
    fn test_histogram_bucket() {
        let bucket = HistogramBucket { le: 0.1, count: 50 };

        assert_eq!(bucket.le, 0.1);
        assert_eq!(bucket.count, 50);
    }

    #[test]
    fn test_quantile() {
        let quantile = Quantile {
            quantile: 0.99,
            value: 150.0,
        };

        assert_eq!(quantile.quantile, 0.99);
        assert_eq!(quantile.value, 150.0);
    }

    #[test]
    fn test_metric_sample_creation() {
        let labels = vec![MetricLabel::new("test", "label")];
        let sample = MetricSample {
            value: MetricValue::Counter(42),
            timestamp: now_millis(),
            labels,
        };

        assert_eq!(sample.labels.len(), 1);
        assert!(sample.timestamp > 0);
    }

    #[test]
    fn test_now_millis() {
        let t1 = now_millis();
        std::thread::sleep(Duration::from_millis(1));
        let t2 = now_millis();

        assert!(t2 > t1);
    }

    #[test]
    fn test_timer_name() {
        let timer = Timer::start("my_operation");
        assert_eq!(timer.name(), "my_operation");
    }

    #[test]
    fn test_common_labels_empty() {
        let labels = CommonLabels::new();
        let vec = labels.to_vec();
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_common_labels_add() {
        let mut labels = CommonLabels::new();
        labels.add("key1", "value1");
        labels.add("key2", "value2");

        let vec = labels.to_vec();
        assert_eq!(vec.len(), 2);
        assert!(vec.iter().any(|l| l.key == "key1" && l.value == "value1"));
        assert!(vec.iter().any(|l| l.key == "key2" && l.value == "value2"));
    }

    #[test]
    fn test_common_labels_default() {
        let labels = CommonLabels::default();
        assert_eq!(labels.to_vec().len(), 0);
    }
}
