//! Throughput and concurrency metrics
//!
//! This module tracks system-level throughput metrics:
//! - **Requests/sec**: Number of inference requests processed per second
//! - **Tokens/sec**: Total token generation throughput across all requests
//! - **Concurrent users**: Number of simultaneous active requests
//! - **Queue depth**: Number of requests waiting to be processed

use crate::types::{now_millis, MetricLabel, MetricSample, MetricValue, RollingWindow};
use std::time::{Duration, Instant};

/// Throughput metrics snapshot
#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    /// Requests processed per second
    pub requests_per_sec: f64,
    /// Total tokens generated per second (across all requests)
    pub tokens_per_sec: f64,
    /// Number of concurrent active requests
    pub concurrent_requests: u64,
    /// Number of requests in queue
    pub queue_depth: u64,
    /// Average queue wait time
    pub avg_queue_wait: Duration,
}

impl ThroughputMetrics {
    /// Create new throughput metrics
    pub fn new(
        requests_per_sec: f64,
        tokens_per_sec: f64,
        concurrent_requests: u64,
        queue_depth: u64,
        avg_queue_wait: Duration,
    ) -> Self {
        Self {
            requests_per_sec,
            tokens_per_sec,
            concurrent_requests,
            queue_depth,
            avg_queue_wait,
        }
    }

    /// Convert to metric samples for export
    pub fn to_samples(&self, labels: Vec<MetricLabel>) -> Vec<MetricSample> {
        let timestamp = now_millis();
        vec![
            MetricSample {
                value: MetricValue::Gauge(self.requests_per_sec),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.tokens_per_sec),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.concurrent_requests as f64),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.queue_depth as f64),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.avg_queue_wait.as_secs_f64()),
                timestamp,
                labels,
            },
        ]
    }
}

/// Throughput tracker with rolling window statistics
pub struct ThroughputTracker {
    /// Rolling window for requests/sec
    rps_window: RollingWindow,
    /// Rolling window for tokens/sec
    tps_window: RollingWindow,
    /// Rolling window for queue wait time
    queue_wait_window: RollingWindow,
    /// Start time for rate calculation
    start_time: Instant,
    /// Total requests processed (cumulative)
    total_requests: u64,
    /// Total tokens generated (cumulative)
    total_tokens: u64,
    /// Current concurrent requests
    concurrent_requests: u64,
    /// Current queue depth
    queue_depth: u64,
    /// Peak concurrent requests observed
    peak_concurrent_requests: u64,
    /// Peak queue depth observed
    peak_queue_depth: u64,
}

impl ThroughputTracker {
    /// Create a new throughput tracker with the given window size
    pub fn new(window_size: usize) -> Self {
        Self {
            rps_window: RollingWindow::new(window_size),
            tps_window: RollingWindow::new(window_size),
            queue_wait_window: RollingWindow::new(window_size),
            start_time: Instant::now(),
            total_requests: 0,
            total_tokens: 0,
            concurrent_requests: 0,
            queue_depth: 0,
            peak_concurrent_requests: 0,
            peak_queue_depth: 0,
        }
    }

    /// Record a completed request
    pub fn record_request(&mut self, tokens_generated: u64, queue_wait: Duration) {
        self.total_requests += 1;
        self.total_tokens += tokens_generated;
        self.queue_wait_window.add(queue_wait.as_secs_f64());

        // Calculate current rates
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let rps = self.total_requests as f64 / elapsed;
            let tps = self.total_tokens as f64 / elapsed;
            self.rps_window.add(rps);
            self.tps_window.add(tps);
        }
    }

    /// Increment concurrent requests (called when a request starts)
    pub fn start_request(&mut self) {
        self.concurrent_requests += 1;
        if self.concurrent_requests > self.peak_concurrent_requests {
            self.peak_concurrent_requests = self.concurrent_requests;
        }
    }

    /// Decrement concurrent requests (called when a request completes)
    pub fn finish_request(&mut self) {
        if self.concurrent_requests > 0 {
            self.concurrent_requests -= 1;
        }
    }

    /// Add a request to the queue
    pub fn enqueue_request(&mut self) {
        self.queue_depth += 1;
        if self.queue_depth > self.peak_queue_depth {
            self.peak_queue_depth = self.queue_depth;
        }
    }

    /// Remove a request from the queue
    pub fn dequeue_request(&mut self) {
        if self.queue_depth > 0 {
            self.queue_depth -= 1;
        }
    }

    /// Get current requests per second
    pub fn requests_per_sec(&self) -> f64 {
        self.rps_window.mean()
    }

    /// Get current tokens per second (system-wide)
    pub fn tokens_per_sec(&self) -> f64 {
        self.tps_window.mean()
    }

    /// Get overall requests per second (cumulative)
    pub fn overall_requests_per_sec(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_requests as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get overall tokens per second (cumulative)
    pub fn overall_tokens_per_sec(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_tokens as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get current concurrent requests
    pub fn concurrent_requests(&self) -> u64 {
        self.concurrent_requests
    }

    /// Get peak concurrent requests
    pub fn peak_concurrent_requests(&self) -> u64 {
        self.peak_concurrent_requests
    }

    /// Get current queue depth
    pub fn queue_depth(&self) -> u64 {
        self.queue_depth
    }

    /// Get peak queue depth
    pub fn peak_queue_depth(&self) -> u64 {
        self.peak_queue_depth
    }

    /// Get average queue wait time
    pub fn avg_queue_wait(&self) -> Duration {
        Duration::from_secs_f64(self.queue_wait_window.mean())
    }

    /// Get total requests processed
    pub fn total_requests(&self) -> u64 {
        self.total_requests
    }

    /// Get total tokens generated
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Export current statistics as metric samples
    pub fn export_samples(&self, labels: Vec<MetricLabel>) -> Vec<MetricSample> {
        let timestamp = now_millis();
        vec![
            MetricSample {
                value: MetricValue::Gauge(self.requests_per_sec()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.overall_requests_per_sec()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.tokens_per_sec()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.overall_tokens_per_sec()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.concurrent_requests as f64),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.peak_concurrent_requests as f64),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.queue_depth as f64),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.peak_queue_depth as f64),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Counter(self.total_requests),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Counter(self.total_tokens),
                timestamp,
                labels,
            },
        ]
    }

    /// Get a snapshot of current throughput metrics
    pub fn snapshot(&self) -> ThroughputMetrics {
        ThroughputMetrics::new(
            self.requests_per_sec(),
            self.tokens_per_sec(),
            self.concurrent_requests,
            self.queue_depth,
            self.avg_queue_wait(),
        )
    }
}

impl Default for ThroughputTracker {
    fn default() -> Self {
        Self::new(100) // Default to 100-sample window
    }
}

/// Request tracking helper
pub struct RequestTracker {
    /// When the request was enqueued
    enqueue_time: Instant,
    /// When the request started processing
    start_time: Option<Instant>,
    /// Number of tokens to generate
    tokens_generated: u64,
}

impl RequestTracker {
    /// Create a new request tracker
    pub fn new() -> Self {
        Self {
            enqueue_time: Instant::now(),
            start_time: None,
            tokens_generated: 0,
        }
    }

    /// Mark that processing has started
    pub fn start_processing(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Get queue wait time
    pub fn queue_wait(&self) -> Duration {
        if let Some(start) = self.start_time {
            start - self.enqueue_time
        } else {
            self.enqueue_time.elapsed()
        }
    }

    /// Get processing time
    pub fn processing_time(&self) -> Option<Duration> {
        self.start_time.map(|t| t.elapsed())
    }

    /// Add generated tokens
    pub fn add_tokens(&mut self, count: u64) {
        self.tokens_generated += count;
    }

    /// Get total tokens generated
    pub fn tokens(&self) -> u64 {
        self.tokens_generated
    }
}

impl Default for RequestTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_throughput_metrics() {
        let metrics = ThroughputMetrics::new(
            10.5,                       // 10.5 requests/sec
            250.0,                      // 250 tokens/sec
            5,                          // 5 concurrent requests
            3,                          // 3 requests in queue
            Duration::from_millis(100), // 100ms avg queue wait
        );

        assert_eq!(metrics.requests_per_sec, 10.5);
        assert_eq!(metrics.concurrent_requests, 5);
    }

    #[test]
    fn test_throughput_tracker() {
        let mut tracker = ThroughputTracker::new(10);

        // Simulate some requests
        tracker.start_request();
        tracker.record_request(50, Duration::from_millis(10));
        tracker.finish_request();

        tracker.start_request();
        tracker.record_request(75, Duration::from_millis(20));
        tracker.finish_request();

        assert_eq!(tracker.total_requests(), 2);
        assert_eq!(tracker.total_tokens(), 125);
        assert_eq!(tracker.concurrent_requests(), 0);
    }

    #[test]
    fn test_queue_tracking() {
        let mut tracker = ThroughputTracker::new(10);

        tracker.enqueue_request();
        tracker.enqueue_request();
        assert_eq!(tracker.queue_depth(), 2);
        assert_eq!(tracker.peak_queue_depth(), 2);

        tracker.dequeue_request();
        assert_eq!(tracker.queue_depth(), 1);
        assert_eq!(tracker.peak_queue_depth(), 2); // Peak stays at 2
    }

    #[test]
    fn test_request_tracker() {
        let mut req = RequestTracker::new();

        std::thread::sleep(Duration::from_millis(10));
        req.start_processing();

        let queue_wait = req.queue_wait();
        assert!(queue_wait.as_millis() >= 10);

        req.add_tokens(25);
        assert_eq!(req.tokens(), 25);
    }
}
