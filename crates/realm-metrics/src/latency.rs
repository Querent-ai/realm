//! Latency metrics for AI inference
//!
//! This module tracks critical latency metrics for LLM inference:
//! - **TTFT (Time-to-First-Token)**: How long until the first token is generated
//! - **Tokens/sec**: Generation throughput (tokens per second)
//! - **Total generation time**: End-to-end latency
//! - **Per-token latency**: Average time per token

use crate::types::{now_millis, MetricLabel, MetricSample, MetricValue, RollingWindow};
use std::time::Duration;

/// Latency metrics for a single generation request
#[derive(Debug, Clone)]
pub struct LatencyMetrics {
    /// Time to first token (critical for streaming UX)
    pub ttft: Duration,
    /// Tokens generated per second
    pub tokens_per_sec: f64,
    /// Total number of tokens generated
    pub total_tokens: u64,
    /// Total generation time (end-to-end)
    pub total_time: Duration,
    /// Average time per token
    pub per_token_latency: Duration,
}

impl LatencyMetrics {
    /// Create latency metrics from generation stats
    pub fn new(ttft: Duration, total_time: Duration, total_tokens: u64) -> Self {
        let tokens_per_sec = if total_time.as_secs_f64() > 0.0 {
            total_tokens as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        let per_token_latency = if total_tokens > 0 {
            total_time / total_tokens as u32
        } else {
            Duration::ZERO
        };

        Self {
            ttft,
            tokens_per_sec,
            total_tokens,
            total_time,
            per_token_latency,
        }
    }

    /// Convert to metric samples for export
    pub fn to_samples(&self, labels: Vec<MetricLabel>) -> Vec<MetricSample> {
        let timestamp = now_millis();
        vec![
            MetricSample {
                value: MetricValue::Gauge(self.ttft.as_secs_f64()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.tokens_per_sec),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Counter(self.total_tokens),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.total_time.as_secs_f64()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.per_token_latency.as_secs_f64()),
                timestamp,
                labels,
            },
        ]
    }
}

/// Latency tracker with rolling window statistics
pub struct LatencyTracker {
    /// Rolling window for TTFT
    ttft_window: RollingWindow,
    /// Rolling window for tokens/sec
    tps_window: RollingWindow,
    /// Rolling window for total time
    total_time_window: RollingWindow,
    /// Total tokens generated (cumulative counter)
    total_tokens_generated: u64,
    /// Total requests processed (cumulative counter)
    total_requests: u64,
    /// Current active timers (for in-flight requests)
    active_timers: Vec<GenerationTimer>,
}

impl LatencyTracker {
    /// Create a new latency tracker with the given window size
    pub fn new(window_size: usize) -> Self {
        Self {
            ttft_window: RollingWindow::new(window_size),
            tps_window: RollingWindow::new(window_size),
            total_time_window: RollingWindow::new(window_size),
            total_tokens_generated: 0,
            total_requests: 0,
            active_timers: Vec::new(),
        }
    }

    /// Start tracking a new generation
    pub fn start_generation(&mut self) -> GenerationTimer {
        GenerationTimer::new()
    }

    /// Record a completed generation
    pub fn record(&mut self, metrics: LatencyMetrics) {
        self.ttft_window.add(metrics.ttft.as_secs_f64());
        self.tps_window.add(metrics.tokens_per_sec);
        self.total_time_window.add(metrics.total_time.as_secs_f64());
        self.total_tokens_generated += metrics.total_tokens;
        self.total_requests += 1;
    }

    /// Get mean TTFT over the window
    pub fn mean_ttft(&self) -> Duration {
        Duration::from_secs_f64(self.ttft_window.mean())
    }

    /// Get p99 TTFT over the window
    pub fn p99_ttft(&self) -> Duration {
        Duration::from_secs_f64(self.ttft_window.p99())
    }

    /// Get mean tokens/sec over the window
    pub fn mean_tokens_per_sec(&self) -> f64 {
        self.tps_window.mean()
    }

    /// Get p99 tokens/sec over the window
    pub fn p99_tokens_per_sec(&self) -> f64 {
        self.tps_window.p99()
    }

    /// Get mean total time over the window
    pub fn mean_total_time(&self) -> Duration {
        Duration::from_secs_f64(self.total_time_window.mean())
    }

    /// Get total tokens generated (cumulative)
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens_generated
    }

    /// Get total requests processed (cumulative)
    pub fn total_requests(&self) -> u64 {
        self.total_requests
    }

    /// Get current number of in-flight requests
    pub fn active_requests(&self) -> usize {
        self.active_timers.len()
    }

    /// Export current statistics as metric samples
    pub fn export_samples(&self, labels: Vec<MetricLabel>) -> Vec<MetricSample> {
        let timestamp = now_millis();
        vec![
            MetricSample {
                value: MetricValue::Gauge(self.mean_ttft().as_secs_f64()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.p99_ttft().as_secs_f64()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.mean_tokens_per_sec()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Counter(self.total_tokens_generated),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Counter(self.total_requests),
                timestamp,
                labels,
            },
        ]
    }
}

/// Timer for tracking a single generation
pub struct GenerationTimer {
    start: std::time::Instant,
    first_token_time: Option<std::time::Instant>,
    tokens_generated: u64,
}

impl Default for GenerationTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationTimer {
    /// Create a new generation timer
    pub fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
            first_token_time: None,
            tokens_generated: 0,
        }
    }

    /// Mark that the first token has been generated
    pub fn mark_first_token(&mut self) {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(std::time::Instant::now());
        }
    }

    /// Increment the token count
    pub fn add_token(&mut self) {
        self.tokens_generated += 1;
        // Mark first token if this is the first one
        if self.tokens_generated == 1 {
            self.mark_first_token();
        }
    }

    /// Add multiple tokens
    pub fn add_tokens(&mut self, count: u64) {
        let was_zero = self.tokens_generated == 0;
        self.tokens_generated += count;
        // Mark first token if we just went from 0 to non-zero
        if was_zero && self.tokens_generated > 0 {
            self.mark_first_token();
        }
    }

    /// Get the current TTFT (if first token has been generated)
    pub fn ttft(&self) -> Option<Duration> {
        self.first_token_time.map(|t| t - self.start)
    }

    /// Get the current total elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get the number of tokens generated so far
    pub fn tokens(&self) -> u64 {
        self.tokens_generated
    }

    /// Finish the generation and return metrics
    pub fn finish(self) -> LatencyMetrics {
        let total_time = self.start.elapsed();
        let ttft = self.ttft().unwrap_or(total_time);

        LatencyMetrics::new(ttft, total_time, self.tokens_generated)
    }

    /// Finish with an explicit token count
    pub fn finish_with_tokens(mut self, tokens: u64) -> LatencyMetrics {
        self.tokens_generated = tokens;
        self.finish()
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new(100) // Default to 100-sample window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_metrics() {
        let metrics = LatencyMetrics::new(Duration::from_millis(100), Duration::from_secs(2), 50);

        assert_eq!(metrics.ttft.as_millis(), 100);
        assert_eq!(metrics.total_tokens, 50);
        assert_eq!(metrics.tokens_per_sec, 25.0); // 50 tokens / 2 seconds
    }

    #[test]
    fn test_generation_timer() {
        let mut timer = GenerationTimer::new();

        // Simulate first token
        timer.mark_first_token();
        std::thread::sleep(Duration::from_millis(10));

        // Add more tokens
        timer.add_tokens(9); // Total: 10 tokens

        let metrics = timer.finish_with_tokens(10);
        assert_eq!(metrics.total_tokens, 10);
        assert!(metrics.ttft < metrics.total_time);
    }

    #[test]
    fn test_latency_tracker() {
        let mut tracker = LatencyTracker::new(3);

        // Record some metrics
        tracker.record(LatencyMetrics::new(
            Duration::from_millis(100),
            Duration::from_secs(1),
            20,
        ));
        tracker.record(LatencyMetrics::new(
            Duration::from_millis(150),
            Duration::from_secs(2),
            40,
        ));

        assert_eq!(tracker.total_requests(), 2);
        assert_eq!(tracker.total_tokens(), 60);
        assert!(tracker.mean_ttft().as_millis() > 0);
    }

    #[test]
    fn test_latency_metrics_zero_tokens() {
        let metrics = LatencyMetrics::new(Duration::from_millis(100), Duration::from_secs(1), 0);

        assert_eq!(metrics.total_tokens, 0);
        assert_eq!(metrics.tokens_per_sec, 0.0);
        assert_eq!(metrics.per_token_latency, Duration::ZERO);
    }

    #[test]
    fn test_latency_metrics_zero_time() {
        let metrics = LatencyMetrics::new(Duration::ZERO, Duration::ZERO, 10);

        assert_eq!(metrics.tokens_per_sec, 0.0);
    }

    #[test]
    fn test_generation_timer_auto_first_token() {
        let mut timer = GenerationTimer::new();

        // Adding a token should automatically mark first token
        timer.add_token();

        let ttft = timer.ttft();
        assert!(ttft.is_some());
        assert_eq!(timer.tokens(), 1);
    }

    #[test]
    fn test_generation_timer_multiple_first_token_marks() {
        let mut timer = GenerationTimer::new();

        timer.mark_first_token();
        let first_ttft = timer.ttft().unwrap();

        // Marking again shouldn't change the TTFT
        std::thread::sleep(Duration::from_millis(5));
        timer.mark_first_token();

        let second_ttft = timer.ttft().unwrap();
        assert_eq!(first_ttft, second_ttft);
    }

    #[test]
    fn test_generation_timer_elapsed() {
        let timer = GenerationTimer::new();
        std::thread::sleep(Duration::from_millis(10));

        let elapsed = timer.elapsed();
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    fn test_generation_timer_no_first_token() {
        let timer = GenerationTimer::new();

        // If no first token marked, ttft should be None
        assert!(timer.ttft().is_none());

        // finish() should use total time as ttft
        let metrics = timer.finish();
        assert_eq!(metrics.ttft, metrics.total_time);
    }

    #[test]
    fn test_latency_tracker_default() {
        let tracker = LatencyTracker::default();
        assert_eq!(tracker.total_requests(), 0);
        assert_eq!(tracker.total_tokens(), 0);
    }

    #[test]
    fn test_latency_tracker_p99() {
        let mut tracker = LatencyTracker::new(100);

        // Add varying latencies
        for i in 1..=100 {
            tracker.record(LatencyMetrics::new(
                Duration::from_millis(i),
                Duration::from_secs(1),
                10,
            ));
        }

        let p99 = tracker.p99_ttft();
        // P99 should be >= 99ms
        assert!(p99.as_millis() >= 99);
    }

    #[test]
    fn test_latency_tracker_tokens_per_sec_statistics() {
        let mut tracker = LatencyTracker::new(10);

        tracker.record(LatencyMetrics::new(
            Duration::from_millis(50),
            Duration::from_secs(1),
            50,
        ));
        tracker.record(LatencyMetrics::new(
            Duration::from_millis(60),
            Duration::from_secs(1),
            60,
        ));

        let mean_tps = tracker.mean_tokens_per_sec();
        assert_eq!(mean_tps, 55.0); // (50 + 60) / 2

        let p99_tps = tracker.p99_tokens_per_sec();
        assert!(p99_tps > 0.0);
    }

    #[test]
    fn test_latency_metrics_to_samples() {
        let metrics = LatencyMetrics::new(Duration::from_millis(100), Duration::from_secs(2), 50);

        let labels = vec![];
        let samples = metrics.to_samples(labels);

        // Should have 5 samples: ttft, tps, total_tokens, total_time, per_token_latency
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_latency_tracker_export_samples() {
        let mut tracker = LatencyTracker::new(10);

        tracker.record(LatencyMetrics::new(
            Duration::from_millis(100),
            Duration::from_secs(1),
            20,
        ));

        let labels = vec![];
        let samples = tracker.export_samples(labels);

        // Should have multiple metric samples
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_generation_timer_add_tokens_from_zero() {
        let mut timer = GenerationTimer::new();

        // Adding multiple tokens when starting from 0 should mark first token
        timer.add_tokens(5);

        assert_eq!(timer.tokens(), 5);
        assert!(timer.ttft().is_some());
    }

    #[test]
    fn test_generation_timer_add_tokens_incremental() {
        let mut timer = GenerationTimer::new();

        timer.add_tokens(3);
        timer.add_tokens(2);
        timer.add_tokens(5);

        assert_eq!(timer.tokens(), 10);
    }
}
