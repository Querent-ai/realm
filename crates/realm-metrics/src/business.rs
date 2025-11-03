//! Business metrics for sales, accounting, and decision-making
//!
//! This module provides metrics specifically designed for business stakeholders:
//! - **Success/Failure tracking**: Request success rates, error types
//! - **API Key/Client attribution**: Track usage by API key, client ID
//! - **Error categorization**: Rate limits, timeouts, validation errors, etc.
//! - **Revenue metrics**: If charging customers, track revenue per tenant/client

use crate::types::{now_millis, MetricLabel, MetricSample, MetricValue};
use std::collections::HashMap;

/// Error types for business metrics tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorType {
    /// Request was successful (2xx status)
    Success,
    /// Rate limit exceeded (429)
    RateLimit,
    /// Request timeout (408)
    Timeout,
    /// Invalid request (400)
    ValidationError,
    /// Authentication error (401)
    AuthenticationError,
    /// Authorization error (403)
    AuthorizationError,
    /// Not found (404)
    NotFound,
    /// Internal server error (500)
    InternalError,
    /// Service unavailable (503)
    ServiceUnavailable,
    /// Out of memory error
    OutOfMemory,
    /// Model not available
    ModelUnavailable,
    /// Other/unclassified error
    Other,
}

impl ErrorType {
    /// Get error type from HTTP status code
    pub fn from_status_code(code: u16) -> Self {
        match code {
            200..=299 => ErrorType::Success,
            400 => ErrorType::ValidationError,
            401 => ErrorType::AuthenticationError,
            403 => ErrorType::AuthorizationError,
            404 => ErrorType::NotFound,
            408 => ErrorType::Timeout,
            429 => ErrorType::RateLimit,
            500 => ErrorType::InternalError,
            503 => ErrorType::ServiceUnavailable,
            _ => ErrorType::Other,
        }
    }

    /// Get error type string for labels
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorType::Success => "success",
            ErrorType::RateLimit => "rate_limit",
            ErrorType::Timeout => "timeout",
            ErrorType::ValidationError => "validation_error",
            ErrorType::AuthenticationError => "authentication_error",
            ErrorType::AuthorizationError => "authorization_error",
            ErrorType::NotFound => "not_found",
            ErrorType::InternalError => "internal_error",
            ErrorType::ServiceUnavailable => "service_unavailable",
            ErrorType::OutOfMemory => "out_of_memory",
            ErrorType::ModelUnavailable => "model_unavailable",
            ErrorType::Other => "other",
        }
    }
}

/// Business metrics tracker for error rates and client attribution
pub struct BusinessMetricsTracker {
    /// Total requests (success + failure)
    total_requests: u64,
    /// Successful requests
    successful_requests: u64,
    /// Failed requests by error type
    errors_by_type: HashMap<ErrorType, u64>,
    /// Requests by status code
    requests_by_status: HashMap<u16, u64>,
    /// Requests by API key (for attribution)
    requests_by_api_key: HashMap<String, u64>,
    /// Requests by client ID (for customer attribution)
    requests_by_client: HashMap<String, u64>,
    /// Tokens by API key (for usage attribution)
    tokens_by_api_key: HashMap<String, u64>,
    /// Tokens by client ID
    tokens_by_client: HashMap<String, u64>,
    /// Revenue by tenant (if applicable)
    revenue_by_tenant: HashMap<String, f64>,
    /// Revenue by client
    revenue_by_client: HashMap<String, f64>,
}

impl BusinessMetricsTracker {
    /// Create a new business metrics tracker
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            errors_by_type: HashMap::new(),
            requests_by_status: HashMap::new(),
            requests_by_api_key: HashMap::new(),
            requests_by_client: HashMap::new(),
            tokens_by_api_key: HashMap::new(),
            tokens_by_client: HashMap::new(),
            revenue_by_tenant: HashMap::new(),
            revenue_by_client: HashMap::new(),
        }
    }

    /// Record a request with status code
    pub fn record_request(
        &mut self,
        status_code: u16,
        api_key: Option<&str>,
        client_id: Option<&str>,
        tokens: Option<u64>,
    ) {
        self.total_requests += 1;
        *self.requests_by_status.entry(status_code).or_insert(0) += 1;

        let error_type = ErrorType::from_status_code(status_code);
        if error_type == ErrorType::Success {
            self.successful_requests += 1;
        } else {
            *self.errors_by_type.entry(error_type).or_insert(0) += 1;
        }

        // Track by API key
        if let Some(key) = api_key {
            *self.requests_by_api_key.entry(key.to_string()).or_insert(0) += 1;
            if let Some(token_count) = tokens {
                *self.tokens_by_api_key.entry(key.to_string()).or_insert(0) += token_count;
            }
        }

        // Track by client ID
        if let Some(client) = client_id {
            *self
                .requests_by_client
                .entry(client.to_string())
                .or_insert(0) += 1;
            if let Some(token_count) = tokens {
                *self.tokens_by_client.entry(client.to_string()).or_insert(0) += token_count;
            }
        }
    }

    /// Record revenue (if charging customers)
    pub fn record_revenue(
        &mut self,
        amount: f64,
        tenant_id: Option<&str>,
        client_id: Option<&str>,
    ) {
        if let Some(tenant) = tenant_id {
            *self
                .revenue_by_tenant
                .entry(tenant.to_string())
                .or_insert(0.0) += amount;
        }
        if let Some(client) = client_id {
            *self
                .revenue_by_client
                .entry(client.to_string())
                .or_insert(0.0) += amount;
        }
    }

    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.successful_requests as f64 / self.total_requests as f64
        }
    }

    /// Get error rate (0.0 to 1.0)
    pub fn error_rate(&self) -> f64 {
        1.0 - self.success_rate()
    }

    /// Get total requests
    pub fn total_requests(&self) -> u64 {
        self.total_requests
    }

    /// Get successful requests
    pub fn successful_requests(&self) -> u64 {
        self.successful_requests
    }

    /// Export metrics as samples for Prometheus export
    pub fn export_samples(&self, base_labels: Vec<MetricLabel>) -> Vec<MetricSample> {
        let timestamp = now_millis();
        let mut samples = Vec::new();

        // Total requests
        let mut labels = base_labels.clone();
        labels.push(MetricLabel::new("name", "requests_total"));
        samples.push(MetricSample {
            value: MetricValue::Counter(self.total_requests),
            timestamp,
            labels: labels.clone(),
        });

        // Successful requests
        labels.pop();
        labels.push(MetricLabel::new("name", "requests_success_total"));
        samples.push(MetricSample {
            value: MetricValue::Counter(self.successful_requests),
            timestamp,
            labels: labels.clone(),
        });

        // Error rate
        labels.pop();
        labels.push(MetricLabel::new("name", "error_rate"));
        samples.push(MetricSample {
            value: MetricValue::Gauge(self.error_rate()),
            timestamp,
            labels: labels.clone(),
        });

        // Success rate
        labels.pop();
        labels.push(MetricLabel::new("name", "success_rate"));
        samples.push(MetricSample {
            value: MetricValue::Gauge(self.success_rate()),
            timestamp,
            labels: labels.clone(),
        });

        // Errors by type
        for (error_type, count) in &self.errors_by_type {
            labels.pop();
            labels.push(MetricLabel::new("name", "requests_errors_total"));
            let mut error_labels = labels.clone();
            error_labels.push(MetricLabel::new("error_type", error_type.as_str()));
            samples.push(MetricSample {
                value: MetricValue::Counter(*count),
                timestamp,
                labels: error_labels,
            });
        }

        // Requests by status code
        for (status, count) in &self.requests_by_status {
            labels.pop();
            labels.push(MetricLabel::new("name", "requests_by_status_total"));
            let mut status_labels = labels.clone();
            status_labels.push(MetricLabel::new("status_code", status.to_string()));
            samples.push(MetricSample {
                value: MetricValue::Counter(*count),
                timestamp,
                labels: status_labels,
            });
        }

        // Requests by API key (anonymized for privacy - hash or mask)
        for (api_key_hash, count) in &self.requests_by_api_key {
            labels.pop();
            labels.push(MetricLabel::new("name", "requests_by_api_key_total"));
            let mut key_labels = base_labels.clone();
            key_labels.push(MetricLabel::new("api_key_hash", api_key_hash.clone()));
            samples.push(MetricSample {
                value: MetricValue::Counter(*count),
                timestamp,
                labels: key_labels.clone(),
            });

            // Tokens by API key
            if let Some(&tokens) = self.tokens_by_api_key.get(api_key_hash) {
                key_labels.pop();
                key_labels.push(MetricLabel::new("name", "tokens_by_api_key_total"));
                key_labels.push(MetricLabel::new("api_key_hash", api_key_hash.clone()));
                samples.push(MetricSample {
                    value: MetricValue::Counter(tokens),
                    timestamp,
                    labels: key_labels,
                });
            }
        }

        // Requests by client ID
        for (client_id, count) in &self.requests_by_client {
            labels.pop();
            labels.push(MetricLabel::new("name", "requests_by_client_total"));
            let mut client_labels = base_labels.clone();
            client_labels.push(MetricLabel::new("client_id", client_id.clone()));
            samples.push(MetricSample {
                value: MetricValue::Counter(*count),
                timestamp,
                labels: client_labels.clone(),
            });

            // Tokens by client
            if let Some(&tokens) = self.tokens_by_client.get(client_id) {
                client_labels.pop();
                client_labels.push(MetricLabel::new("name", "tokens_by_client_total"));
                client_labels.push(MetricLabel::new("client_id", client_id.clone()));
                samples.push(MetricSample {
                    value: MetricValue::Counter(tokens),
                    timestamp,
                    labels: client_labels,
                });
            }
        }

        // Revenue by tenant
        for (tenant_id, revenue) in &self.revenue_by_tenant {
            labels.pop();
            labels.push(MetricLabel::new("name", "revenue_usd"));
            let mut revenue_labels = base_labels.clone();
            revenue_labels.push(MetricLabel::new("tenant", tenant_id.clone()));
            samples.push(MetricSample {
                value: MetricValue::Gauge(*revenue),
                timestamp,
                labels: revenue_labels,
            });
        }

        // Revenue by client
        for (client_id, revenue) in &self.revenue_by_client {
            labels.pop();
            labels.push(MetricLabel::new("name", "revenue_usd"));
            let mut revenue_labels = base_labels.clone();
            revenue_labels.push(MetricLabel::new("client_id", client_id.clone()));
            samples.push(MetricSample {
                value: MetricValue::Gauge(*revenue),
                timestamp,
                labels: revenue_labels,
            });
        }

        samples
    }

    /// Reset all metrics (e.g., for new period)
    pub fn reset(&mut self) {
        self.total_requests = 0;
        self.successful_requests = 0;
        self.errors_by_type.clear();
        self.requests_by_status.clear();
        self.requests_by_api_key.clear();
        self.requests_by_client.clear();
        self.tokens_by_api_key.clear();
        self.tokens_by_client.clear();
        self.revenue_by_tenant.clear();
        self.revenue_by_client.clear();
    }
}

impl Default for BusinessMetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_type_from_status() {
        assert_eq!(ErrorType::from_status_code(200), ErrorType::Success);
        assert_eq!(ErrorType::from_status_code(400), ErrorType::ValidationError);
        assert_eq!(ErrorType::from_status_code(429), ErrorType::RateLimit);
        assert_eq!(ErrorType::from_status_code(500), ErrorType::InternalError);
    }

    #[test]
    fn test_business_metrics_tracker() {
        let mut tracker = BusinessMetricsTracker::new();

        // Record some requests
        tracker.record_request(200, Some("key1"), Some("client1"), Some(1000));
        tracker.record_request(200, Some("key1"), Some("client1"), Some(500));
        tracker.record_request(429, Some("key2"), Some("client2"), Some(100));

        assert_eq!(tracker.total_requests(), 3);
        assert_eq!(tracker.successful_requests(), 2);
        assert!((tracker.success_rate() - 0.6667).abs() < 0.01);
        assert!((tracker.error_rate() - 0.3333).abs() < 0.01);

        // Check API key tracking
        assert_eq!(tracker.requests_by_api_key.get("key1"), Some(&2));
        assert_eq!(tracker.tokens_by_api_key.get("key1"), Some(&1500));
    }

    #[test]
    fn test_revenue_tracking() {
        let mut tracker = BusinessMetricsTracker::new();

        tracker.record_revenue(100.50, Some("tenant1"), Some("client1"));
        tracker.record_revenue(50.25, Some("tenant1"), Some("client2"));

        assert_eq!(tracker.revenue_by_tenant.get("tenant1"), Some(&150.75));
    }

    #[test]
    fn test_export_samples() {
        let mut tracker = BusinessMetricsTracker::new();
        tracker.record_request(200, Some("key1"), Some("client1"), Some(1000));
        tracker.record_request(429, None, None, None);

        let samples = tracker.export_samples(vec![]);
        assert!(!samples.is_empty());
        assert!(samples.iter().any(|s| {
            s.labels
                .iter()
                .any(|l| l.key == "name" && l.value == "requests_total")
        }));
    }
}
