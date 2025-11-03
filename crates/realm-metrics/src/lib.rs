//! Comprehensive metrics collection and export for Realm.ai inference
//!
//! This crate provides a robust metrics system for tracking AI inference performance,
//! quality, and resource utilization. It's designed to be:
//!
//! - **Production-ready**: Low overhead, thread-safe, zero-allocation hot paths
//! - **Dashboard-ready**: Prometheus and OpenTelemetry export support
//! - **AI-focused**: Specialized metrics for LLM inference (TTFT, tokens/sec, perplexity, etc.)
//! - **Multi-tenant aware**: Per-tenant isolation and resource tracking
//!
//! # Architecture
//!
//! The metrics system is organized into several categories:
//!
//! - **Latency**: Time-to-first-token (TTFT), tokens/sec, generation latency
//! - **Quality**: Perplexity, token probability distributions, confidence scores
//! - **Resource**: Memory usage, CPU utilization, cache hit rates
//! - **Throughput**: Requests/sec, concurrent users, queue depth
//! - **Errors**: Failed generations, OOM events, timeout rates
//!
//! # Example
//!
//! ```rust,ignore
//! use realm_metrics::{MetricsCollector, LatencyMetrics};
//!
//! // Create metrics collector
//! let mut collector = MetricsCollector::new();
//!
//! // Record inference
//! let start = std::time::Instant::now();
//! // ... perform inference ...
//! collector.record_latency(LatencyMetrics {
//!     ttft: start.elapsed(),
//!     tokens_per_sec: 42.5,
//!     total_tokens: 100,
//! });
//!
//! // Export to Prometheus
//! let metrics_text = collector.export_prometheus();
//! ```

pub mod business;
pub mod collector;
pub mod export;
pub mod latency;
pub mod quality;
pub mod resource;
pub mod throughput;
pub mod types;
pub mod usage;

// Re-export core types
pub use business::{BusinessMetricsTracker, ErrorType};
pub use collector::MetricsCollector;
pub use latency::{LatencyMetrics, LatencyTracker};
pub use quality::{QualityMetrics, QualityTracker};
pub use resource::{ResourceMetrics, ResourceTracker};
pub use throughput::{ThroughputMetrics, ThroughputTracker};
pub use types::{MetricLabel, MetricSample, MetricValue};
pub use usage::{CostConfig, TokenUsage, UsageMetrics, UsageTracker};

#[cfg(feature = "prometheus")]
pub use export::prometheus::PrometheusExporter;

#[cfg(feature = "opentelemetry")]
pub use export::opentelemetry::OpenTelemetryExporter;
