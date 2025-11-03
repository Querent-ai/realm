//! Metrics export interfaces
//!
//! This module provides exporters for different metrics backends:
//! - Prometheus: Text format for Prometheus scraping
//! - OpenTelemetry: OTLP export for OpenTelemetry collectors
//! - JSON: Simple JSON export for custom dashboards

#[cfg(feature = "prometheus")]
pub mod prometheus;

#[cfg(feature = "opentelemetry")]
pub mod opentelemetry;

use crate::types::{MetricSample, MetricValue};
use std::collections::HashMap;

/// Export format for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Prometheus text format
    Prometheus,
    /// JSON format
    Json,
    /// OpenTelemetry OTLP
    OpenTelemetry,
}

/// Generic exporter trait
pub trait MetricExporter {
    /// Export samples in the format specific to this exporter
    fn export(&self, samples: &[MetricSample]) -> Result<String, ExportError>;

    /// Get the format this exporter produces
    fn format(&self) -> ExportFormat;
}

/// Export error
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Export failed: {0}")]
    ExportFailed(String),

    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
}

/// Simple JSON exporter
pub struct JsonExporter;

impl JsonExporter {
    pub fn new() -> Self {
        Self
    }
}

impl MetricExporter for JsonExporter {
    fn export(&self, samples: &[MetricSample]) -> Result<String, ExportError> {
        serde_json::to_string_pretty(samples).map_err(|e| ExportError::Serialization(e.to_string()))
    }

    fn format(&self) -> ExportFormat {
        ExportFormat::Json
    }
}

impl Default for JsonExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Prometheus text format exporter
pub struct PrometheusTextExporter {
    /// Metric name prefix
    prefix: String,
}

impl PrometheusTextExporter {
    /// Create a new Prometheus text exporter with the given prefix
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    /// Format a metric name with prefix
    fn format_name(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}_{}", self.prefix, name)
        }
    }
}

impl Default for PrometheusTextExporter {
    fn default() -> Self {
        Self::new("realm")
    }
}

impl MetricExporter for PrometheusTextExporter {
    fn export(&self, samples: &[MetricSample]) -> Result<String, ExportError> {
        let mut output = String::new();
        let mut grouped: HashMap<String, Vec<&MetricSample>> = HashMap::new();

        // Group samples by metric type
        for sample in samples {
            let key = match &sample.value {
                MetricValue::Counter(_) => "counter",
                MetricValue::Gauge(_) => "gauge",
                MetricValue::Histogram { .. } => "histogram",
                MetricValue::Summary { .. } => "summary",
            };
            grouped.entry(key.to_string()).or_default().push(sample);
        }

        // Export each group
        for (metric_type, samples) in grouped {
            for (i, sample) in samples.iter().enumerate() {
                let metric_name = self.format_name(&format!("{}_{}", metric_type, i));

                // Format labels
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

                // Get value
                let value = match &sample.value {
                    MetricValue::Counter(v) => *v as f64,
                    MetricValue::Gauge(v) => *v,
                    MetricValue::Histogram { sum, .. } => *sum,
                    MetricValue::Summary { sum, .. } => *sum,
                };

                output.push_str(&format!(
                    "{}{} {} {}\n",
                    metric_name, labels_str, value, sample.timestamp
                ));
            }
        }

        Ok(output)
    }

    fn format(&self) -> ExportFormat {
        ExportFormat::Prometheus
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{now_millis, MetricLabel};

    #[test]
    fn test_json_exporter() {
        let exporter = JsonExporter::new();
        let samples = vec![MetricSample {
            value: MetricValue::Counter(42),
            timestamp: now_millis(),
            labels: vec![MetricLabel::new("test", "value")],
        }];

        let result = exporter.export(&samples);
        assert!(result.is_ok());
        let json = result.unwrap();
        assert!(json.contains("\"test\""));
        assert!(json.contains("\"value\""));
    }

    #[test]
    fn test_prometheus_text_exporter() {
        let exporter = PrometheusTextExporter::new("test");
        let samples = vec![
            MetricSample {
                value: MetricValue::Counter(100),
                timestamp: now_millis(),
                labels: vec![MetricLabel::new("model", "llama")],
            },
            MetricSample {
                value: MetricValue::Gauge(42.5),
                timestamp: now_millis(),
                labels: vec![],
            },
        ];

        let result = exporter.export(&samples);
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("test_"));
        assert!(text.contains("model=\"llama\""));
    }
}
