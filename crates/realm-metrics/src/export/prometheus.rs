//! Prometheus export support
//!
//! This module provides Prometheus text format export for metrics.
//! Implements the Prometheus exposition format:
//! https://prometheus.io/docs/instrumenting/exposition_formats/

use super::{ExportError, ExportFormat, MetricExporter};
use crate::types::{MetricLabel, MetricSample, MetricValue};
use std::collections::HashMap;

/// Prometheus text format exporter
pub struct PrometheusExporter {
    /// Namespace prefix for all metrics (e.g., "realm")
    namespace: String,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter with default namespace
    pub fn new() -> Self {
        Self {
            namespace: "realm".to_string(),
        }
    }

    /// Create a new Prometheus exporter with custom namespace
    pub fn with_namespace(namespace: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
        }
    }

    /// Register default collectors (no-op for text format export)
    pub fn register_defaults(&self) -> Result<(), ExportError> {
        // Text format doesn't require registration
        Ok(())
    }

    /// Gather metrics in Prometheus text format
    pub fn gather(&self) -> Result<Vec<u8>, ExportError> {
        // This method is for compatibility - text format is generated in export()
        Ok(Vec::new())
    }

    /// Format labels for Prometheus format
    fn format_labels(&self, labels: &[MetricLabel]) -> String {
        if labels.is_empty() {
            return String::new();
        }

        let label_pairs: Vec<String> = labels
            .iter()
            .map(|label| format!("{}=\"{}\"", label.key, escape_label_value(&label.value)))
            .collect();

        format!("{{{}}}", label_pairs.join(","))
    }

    /// Format a metric name with namespace
    fn metric_name(&self, name: &str) -> String {
        if self.namespace.is_empty() {
            name.to_string()
        } else {
            format!("{}_{}", self.namespace, name)
        }
    }
}

impl Default for PrometheusExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricExporter for PrometheusExporter {
    fn export(&self, samples: &[MetricSample]) -> Result<String, ExportError> {
        let mut output = String::new();

        // Group samples by metric name (from first label if present)
        let mut grouped: HashMap<String, Vec<&MetricSample>> = HashMap::new();
        for sample in samples {
            // Extract metric name from labels (e.g., "requests_total", "latency_seconds")
            let metric_name = sample
                .labels
                .iter()
                .find(|l| l.key == "metric" || l.key == "name")
                .map(|l| l.value.clone())
                .unwrap_or_else(|| "unknown".to_string());

            grouped.entry(metric_name).or_default().push(sample);
        }

        // Export each metric group
        for (base_name, metric_samples) in grouped.iter() {
            let metric_name = self.metric_name(base_name);

            for sample in metric_samples {
                // Filter out the "metric" or "name" label from output
                let filtered_labels: Vec<MetricLabel> = sample
                    .labels
                    .iter()
                    .filter(|l| l.key != "metric" && l.key != "name")
                    .cloned()
                    .collect();

                let labels_str = self.format_labels(&filtered_labels);

                match &sample.value {
                    MetricValue::Counter(value) => {
                        output.push_str(&format!("# TYPE {} counter\n", metric_name));
                        output.push_str(&format!(
                            "{}{} {} {}\n",
                            metric_name, labels_str, value, sample.timestamp
                        ));
                    }
                    MetricValue::Gauge(value) => {
                        output.push_str(&format!("# TYPE {} gauge\n", metric_name));
                        output.push_str(&format!(
                            "{}{} {} {}\n",
                            metric_name, labels_str, value, sample.timestamp
                        ));
                    }
                    MetricValue::Histogram {
                        count,
                        sum,
                        buckets,
                    } => {
                        output.push_str(&format!("# TYPE {} histogram\n", metric_name));

                        // Export histogram buckets
                        for bucket in buckets {
                            let mut bucket_labels = filtered_labels.clone();
                            bucket_labels.push(MetricLabel::new("le", format!("{}", bucket.le)));
                            let bucket_labels_str = self.format_labels(&bucket_labels);

                            output.push_str(&format!(
                                "{}_bucket{} {} {}\n",
                                metric_name, bucket_labels_str, bucket.count, sample.timestamp
                            ));
                        }

                        // Export +Inf bucket
                        let mut inf_labels = filtered_labels.clone();
                        inf_labels.push(MetricLabel::new("le", "+Inf"));
                        let inf_labels_str = self.format_labels(&inf_labels);
                        output.push_str(&format!(
                            "{}_bucket{} {} {}\n",
                            metric_name, inf_labels_str, count, sample.timestamp
                        ));

                        // Export sum and count
                        output.push_str(&format!(
                            "{}_sum{} {} {}\n",
                            metric_name, labels_str, sum, sample.timestamp
                        ));
                        output.push_str(&format!(
                            "{}_count{} {} {}\n",
                            metric_name, labels_str, count, sample.timestamp
                        ));
                    }
                    MetricValue::Summary {
                        count,
                        sum,
                        quantiles,
                    } => {
                        output.push_str(&format!("# TYPE {} summary\n", metric_name));

                        // Export quantiles
                        for q in quantiles {
                            let mut q_labels = filtered_labels.clone();
                            q_labels.push(MetricLabel::new("quantile", format!("{}", q.quantile)));
                            let q_labels_str = self.format_labels(&q_labels);

                            output.push_str(&format!(
                                "{}{} {} {}\n",
                                metric_name, q_labels_str, q.value, sample.timestamp
                            ));
                        }

                        // Export sum and count
                        output.push_str(&format!(
                            "{}_sum{} {} {}\n",
                            metric_name, labels_str, sum, sample.timestamp
                        ));
                        output.push_str(&format!(
                            "{}_count{} {} {}\n",
                            metric_name, labels_str, count, sample.timestamp
                        ));
                    }
                }

                output.push('\n');
            }
        }

        Ok(output)
    }

    fn format(&self) -> ExportFormat {
        ExportFormat::Prometheus
    }
}

/// Escape label values according to Prometheus format
/// Backslash and double quote need to be escaped
fn escape_label_value(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HistogramBucket, Quantile};

    #[test]
    fn test_prometheus_exporter_creation() {
        let exporter = PrometheusExporter::new();
        assert_eq!(exporter.format(), ExportFormat::Prometheus);
        assert_eq!(exporter.namespace, "realm");
    }

    #[test]
    fn test_custom_namespace() {
        let exporter = PrometheusExporter::with_namespace("custom");
        assert_eq!(exporter.namespace, "custom");
    }

    #[test]
    fn test_export_counter() {
        let exporter = PrometheusExporter::new();
        let samples = vec![MetricSample {
            value: MetricValue::Counter(42),
            timestamp: 1234567890,
            labels: vec![
                MetricLabel::new("name", "requests_total"),
                MetricLabel::new("tenant", "tenant1"),
            ],
        }];

        let output = exporter.export(&samples).unwrap();
        assert!(output.contains("# TYPE realm_requests_total counter"));
        assert!(output.contains("realm_requests_total{tenant=\"tenant1\"} 42 1234567890"));
    }

    #[test]
    fn test_export_gauge() {
        let exporter = PrometheusExporter::new();
        let samples = vec![MetricSample {
            value: MetricValue::Gauge(123.45),
            timestamp: 1234567890,
            labels: vec![
                MetricLabel::new("metric", "memory_bytes"),
                MetricLabel::new("type", "used"),
            ],
        }];

        let output = exporter.export(&samples).unwrap();
        assert!(output.contains("# TYPE realm_memory_bytes gauge"));
        assert!(output.contains("realm_memory_bytes{type=\"used\"} 123.45 1234567890"));
    }

    #[test]
    fn test_export_histogram() {
        let exporter = PrometheusExporter::new();
        let samples = vec![MetricSample {
            value: MetricValue::Histogram {
                count: 100,
                sum: 45.5,
                buckets: vec![
                    HistogramBucket { le: 0.1, count: 50 },
                    HistogramBucket { le: 0.5, count: 90 },
                    HistogramBucket {
                        le: 1.0,
                        count: 100,
                    },
                ],
            },
            timestamp: 1234567890,
            labels: vec![
                MetricLabel::new("name", "latency_seconds"),
                MetricLabel::new("endpoint", "/api/v1/completions"),
            ],
        }];

        let output = exporter.export(&samples).unwrap();
        assert!(output.contains("# TYPE realm_latency_seconds histogram"));
        assert!(output.contains(
            "realm_latency_seconds_bucket{endpoint=\"/api/v1/completions\",le=\"0.1\"} 50"
        ));
        assert!(output.contains(
            "realm_latency_seconds_bucket{endpoint=\"/api/v1/completions\",le=\"0.5\"} 90"
        ));
        assert!(output.contains(
            "realm_latency_seconds_bucket{endpoint=\"/api/v1/completions\",le=\"1\"} 100"
        ));
        assert!(output.contains(
            "realm_latency_seconds_bucket{endpoint=\"/api/v1/completions\",le=\"+Inf\"} 100"
        ));
        assert!(output.contains("realm_latency_seconds_sum{endpoint=\"/api/v1/completions\"} 45.5"));
        assert!(
            output.contains("realm_latency_seconds_count{endpoint=\"/api/v1/completions\"} 100")
        );
    }

    #[test]
    fn test_export_summary() {
        let exporter = PrometheusExporter::new();
        let samples = vec![MetricSample {
            value: MetricValue::Summary {
                count: 1000,
                sum: 567.8,
                quantiles: vec![
                    Quantile {
                        quantile: 0.5,
                        value: 0.45,
                    },
                    Quantile {
                        quantile: 0.9,
                        value: 0.89,
                    },
                    Quantile {
                        quantile: 0.99,
                        value: 1.23,
                    },
                ],
            },
            timestamp: 1234567890,
            labels: vec![
                MetricLabel::new("metric", "request_duration"),
                MetricLabel::new("method", "POST"),
            ],
        }];

        let output = exporter.export(&samples).unwrap();
        assert!(output.contains("# TYPE realm_request_duration summary"));
        assert!(output.contains("realm_request_duration{method=\"POST\",quantile=\"0.5\"} 0.45"));
        assert!(output.contains("realm_request_duration{method=\"POST\",quantile=\"0.9\"} 0.89"));
        assert!(output.contains("realm_request_duration{method=\"POST\",quantile=\"0.99\"} 1.23"));
        assert!(output.contains("realm_request_duration_sum{method=\"POST\"} 567.8"));
        assert!(output.contains("realm_request_duration_count{method=\"POST\"} 1000"));
    }

    #[test]
    fn test_label_escaping() {
        assert_eq!(escape_label_value("simple"), "simple");
        assert_eq!(escape_label_value("with\\backslash"), "with\\\\backslash");
        assert_eq!(escape_label_value("with\"quote"), "with\\\"quote");
        assert_eq!(escape_label_value("with\nnewline"), "with\\nnewline");
    }

    #[test]
    fn test_empty_labels() {
        let exporter = PrometheusExporter::new();
        let samples = vec![MetricSample {
            value: MetricValue::Counter(5),
            timestamp: 1234567890,
            labels: vec![MetricLabel::new("name", "simple_counter")],
        }];

        let output = exporter.export(&samples).unwrap();
        // Should have no label braces since only the "name" label is present (which is filtered out)
        assert!(output.contains("realm_simple_counter 5 1234567890"));
    }

    #[test]
    fn test_multiple_metrics() {
        let exporter = PrometheusExporter::new();
        let samples = vec![
            MetricSample {
                value: MetricValue::Counter(10),
                timestamp: 1000,
                labels: vec![
                    MetricLabel::new("name", "requests"),
                    MetricLabel::new("status", "200"),
                ],
            },
            MetricSample {
                value: MetricValue::Gauge(42.0),
                timestamp: 1000,
                labels: vec![
                    MetricLabel::new("metric", "memory"),
                    MetricLabel::new("type", "heap"),
                ],
            },
        ];

        let output = exporter.export(&samples).unwrap();
        assert!(output.contains("realm_requests"));
        assert!(output.contains("realm_memory"));
    }
}
