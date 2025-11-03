//! Usage and billing metrics for cost tracking and justification
//!
//! This module provides metrics specifically designed for:
//! - **Cost attribution**: Track token usage per tenant/model for billing
//! - **Usage analytics**: Understand consumption patterns for capacity planning
//! - **Executive reporting**: Justify AI infrastructure costs with hard data
//!
//! # Key Metrics
//!
//! - **Token consumption**: Input/output tokens per request, tenant, model
//! - **Cost calculation**: Estimated costs based on token usage
//! - **Usage patterns**: Peak times, average tokens, model distribution
//! - **Billing periods**: Daily/monthly aggregations for invoicing

use crate::types::{now_millis, MetricLabel, MetricSample, MetricValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Token usage for a single request (inspired by Claude API response format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of input tokens (prompt)
    pub input_tokens: u64,
    /// Number of output tokens (generated)
    pub output_tokens: u64,
    /// Total tokens (input + output)
    pub total_tokens: u64,
    /// Cache creation tokens (for prompt caching, à la Claude)
    pub cache_creation_input_tokens: u64,
    /// Cache read tokens (tokens served from cache - cost savings!)
    pub cache_read_input_tokens: u64,
    /// Timestamp of this usage
    pub timestamp: u64,
    /// Model used (e.g., "claude-3-sonnet", "gpt-4-turbo")
    pub model: String,
    /// Stop reason (normal, max_tokens, stop_sequence, etc.)
    pub stop_reason: Option<String>,
}

impl TokenUsage {
    /// Create new token usage record
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            timestamp: now_millis(),
            model: "unknown".to_string(),
            stop_reason: None,
        }
    }

    /// Create with cache metrics (like Claude API)
    pub fn with_cache(
        input_tokens: u64,
        output_tokens: u64,
        cache_creation_tokens: u64,
        cache_read_tokens: u64,
    ) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cache_creation_input_tokens: cache_creation_tokens,
            cache_read_input_tokens: cache_read_tokens,
            timestamp: now_millis(),
            model: "unknown".to_string(),
            stop_reason: None,
        }
    }

    /// Set model name
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set stop reason
    pub fn with_stop_reason(mut self, reason: impl Into<String>) -> Self {
        self.stop_reason = Some(reason.into());
        self
    }

    /// Calculate cache savings percentage
    pub fn cache_hit_rate(&self) -> f64 {
        let total_input = self.input_tokens + self.cache_read_input_tokens;
        if total_input == 0 {
            0.0
        } else {
            self.cache_read_input_tokens as f64 / total_input as f64
        }
    }
}

/// Cost configuration for different models (based on real provider pricing)
#[derive(Debug, Clone)]
pub struct CostConfig {
    /// Cost per 1M input tokens (in USD)
    pub input_cost_per_million: f64,
    /// Cost per 1M output tokens (in USD)
    pub output_cost_per_million: f64,
    /// Cost per 1M cache write tokens (usually same as input, à la Claude)
    pub cache_creation_cost_per_million: f64,
    /// Cost per 1M cache read tokens (usually 10% of input, big savings!)
    pub cache_read_cost_per_million: f64,
}

impl CostConfig {
    /// Create cost config with cache pricing
    pub fn new(
        input_cost_per_million: f64,
        output_cost_per_million: f64,
        cache_creation_cost_per_million: f64,
        cache_read_cost_per_million: f64,
    ) -> Self {
        Self {
            input_cost_per_million,
            output_cost_per_million,
            cache_creation_cost_per_million,
            cache_read_cost_per_million,
        }
    }

    /// Simple config without cache pricing
    pub fn simple(input_cost_per_million: f64, output_cost_per_million: f64) -> Self {
        Self::new(
            input_cost_per_million,
            output_cost_per_million,
            input_cost_per_million,       // Cache write = input cost
            input_cost_per_million * 0.1, // Cache read = 10% of input (industry standard)
        )
    }

    /// Calculate cost for token usage (accounting for cache hits)
    pub fn calculate_cost(&self, usage: &TokenUsage) -> f64 {
        let input_cost = (usage.input_tokens as f64 / 1_000_000.0) * self.input_cost_per_million;
        let output_cost = (usage.output_tokens as f64 / 1_000_000.0) * self.output_cost_per_million;
        let cache_write_cost = (usage.cache_creation_input_tokens as f64 / 1_000_000.0)
            * self.cache_creation_cost_per_million;
        let cache_read_cost =
            (usage.cache_read_input_tokens as f64 / 1_000_000.0) * self.cache_read_cost_per_million;

        input_cost + output_cost + cache_write_cost + cache_read_cost
    }

    /// Calculate cost savings from caching
    pub fn calculate_cache_savings(&self, usage: &TokenUsage) -> f64 {
        // Savings = (what we would have paid) - (what we actually paid for cache reads)
        let full_price =
            (usage.cache_read_input_tokens as f64 / 1_000_000.0) * self.input_cost_per_million;
        let cache_price =
            (usage.cache_read_input_tokens as f64 / 1_000_000.0) * self.cache_read_cost_per_million;
        full_price - cache_price
    }

    /// Example costs for common providers (as of 2025, based on Claude API)
    pub fn claude_opus() -> Self {
        // Claude 3 Opus pricing with prompt caching
        Self::new(15.0, 75.0, 18.75, 1.50) // Cache writes +25%, reads 90% off
    }

    pub fn claude_sonnet() -> Self {
        // Claude 3.5 Sonnet pricing with prompt caching
        Self::new(3.0, 15.0, 3.75, 0.30) // Cache writes +25%, reads 90% off
    }

    pub fn claude_haiku() -> Self {
        // Claude 3 Haiku pricing with prompt caching
        Self::new(0.25, 1.25, 0.30, 0.03) // Cache writes +20%, reads 90% off
    }

    pub fn gpt4_turbo() -> Self {
        Self::simple(10.0, 30.0) // $10/1M input, $30/1M output
    }

    pub fn gpt35_turbo() -> Self {
        Self::simple(0.5, 1.5) // $0.50/1M input, $1.50/1M output
    }

    pub fn llama_hosted() -> Self {
        // Self-hosted costs (compute only, estimated)
        Self::simple(0.1, 0.1) // Much lower for self-hosted
    }
}

/// Usage metrics for a billing period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetrics {
    /// Total input tokens
    pub total_input_tokens: u64,
    /// Total output tokens
    pub total_output_tokens: u64,
    /// Total tokens (input + output)
    pub total_tokens: u64,
    /// Number of requests
    pub request_count: u64,
    /// Estimated cost (USD)
    pub estimated_cost: f64,
    /// Average tokens per request
    pub avg_tokens_per_request: f64,
    /// Period start timestamp
    pub period_start: u64,
    /// Period end timestamp
    pub period_end: u64,
}

impl UsageMetrics {
    /// Create new usage metrics
    pub fn new() -> Self {
        Self {
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_tokens: 0,
            request_count: 0,
            estimated_cost: 0.0,
            avg_tokens_per_request: 0.0,
            period_start: now_millis(),
            period_end: now_millis(),
        }
    }

    /// Convert to metric samples for export
    pub fn to_samples(&self, labels: Vec<MetricLabel>) -> Vec<MetricSample> {
        let timestamp = now_millis();
        vec![
            MetricSample {
                value: MetricValue::Counter(self.total_input_tokens),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Counter(self.total_output_tokens),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Counter(self.total_tokens),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Counter(self.request_count),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.estimated_cost),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.avg_tokens_per_request),
                timestamp,
                labels,
            },
        ]
    }
}

impl Default for UsageMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Usage tracker with cost calculation and per-tenant breakdown
pub struct UsageTracker {
    /// Cost configuration for the model
    cost_config: CostConfig,
    /// Aggregated usage metrics
    total_usage: UsageMetrics,
    /// Per-tenant usage breakdown
    tenant_usage: HashMap<String, UsageMetrics>,
    /// Per-model usage breakdown
    model_usage: HashMap<String, UsageMetrics>,
    /// Recent usage records (for time-series analysis)
    recent_usage: Vec<(u64, TokenUsage)>,
    /// Maximum records to keep
    max_recent_records: usize,
}

impl UsageTracker {
    /// Create new usage tracker
    pub fn new(cost_config: CostConfig) -> Self {
        Self {
            cost_config,
            total_usage: UsageMetrics::new(),
            tenant_usage: HashMap::new(),
            model_usage: HashMap::new(),
            recent_usage: Vec::new(),
            max_recent_records: 10000,
        }
    }

    /// Record token usage for a request
    pub fn record_usage(
        &mut self,
        usage: TokenUsage,
        tenant_id: Option<&str>,
        model_name: Option<&str>,
    ) {
        let cost = self.cost_config.calculate_cost(&usage);

        // Update total usage
        Self::update_metrics(&mut self.total_usage, &usage, cost);

        // Update tenant usage
        if let Some(tenant) = tenant_id {
            let tenant_metrics = self.tenant_usage.entry(tenant.to_string()).or_default();
            Self::update_metrics(tenant_metrics, &usage, cost);
        }

        // Update model usage
        if let Some(model) = model_name {
            let model_metrics = self.model_usage.entry(model.to_string()).or_default();
            Self::update_metrics(model_metrics, &usage, cost);
        }

        // Store recent usage
        self.recent_usage.push((usage.timestamp, usage));
        if self.recent_usage.len() > self.max_recent_records {
            self.recent_usage.remove(0);
        }
    }

    /// Helper to update metrics
    fn update_metrics(metrics: &mut UsageMetrics, usage: &TokenUsage, cost: f64) {
        metrics.total_input_tokens += usage.input_tokens;
        metrics.total_output_tokens += usage.output_tokens;
        metrics.total_tokens += usage.total_tokens;
        metrics.request_count += 1;
        metrics.estimated_cost += cost;
        metrics.avg_tokens_per_request = metrics.total_tokens as f64 / metrics.request_count as f64;
        metrics.period_end = usage.timestamp;
    }

    /// Get total usage
    pub fn total(&self) -> &UsageMetrics {
        &self.total_usage
    }

    /// Get usage for a specific tenant
    pub fn tenant(&self, tenant_id: &str) -> Option<&UsageMetrics> {
        self.tenant_usage.get(tenant_id)
    }

    /// Get usage for a specific model
    pub fn model(&self, model_name: &str) -> Option<&UsageMetrics> {
        self.model_usage.get(model_name)
    }

    /// Get all tenant usage (sorted by cost descending)
    pub fn top_tenants(&self, limit: usize) -> Vec<(String, UsageMetrics)> {
        let mut tenants: Vec<_> = self
            .tenant_usage
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        tenants.sort_by(|a, b| b.1.estimated_cost.partial_cmp(&a.1.estimated_cost).unwrap());
        tenants.into_iter().take(limit).collect()
    }

    /// Get usage for a time period (last N hours)
    pub fn usage_last_hours(&self, hours: u64) -> UsageMetrics {
        let cutoff = now_millis() - (hours * 3600 * 1000);
        let mut metrics = UsageMetrics::new();

        for (timestamp, usage) in &self.recent_usage {
            if *timestamp >= cutoff {
                let cost = self.cost_config.calculate_cost(usage);
                Self::update_metrics(&mut metrics, usage, cost);
            }
        }

        metrics
    }

    /// Generate executive summary
    pub fn executive_summary(&self) -> String {
        let total = &self.total_usage;
        let period_hours = if total.period_end > total.period_start {
            (total.period_end - total.period_start) as f64 / 1000.0 / 3600.0
        } else {
            0.0
        };

        let cost_per_hour = if period_hours > 0.0 {
            total.estimated_cost / period_hours
        } else {
            0.0
        };

        format!(
            "AI Usage Summary\n\
             ================\n\
             Period: {:.1} hours\n\
             Total Requests: {}\n\
             Total Tokens: {} ({} input, {} output)\n\
             Avg Tokens/Request: {:.1}\n\
             Estimated Cost: ${:.2}\n\
             Cost/Hour: ${:.2}\n\
             Cost/Request: ${:.4}\n\
             \n\
             Top 5 Tenants by Cost:\n{}",
            period_hours,
            total.request_count,
            total.total_tokens,
            total.total_input_tokens,
            total.total_output_tokens,
            total.avg_tokens_per_request,
            total.estimated_cost,
            cost_per_hour,
            total.estimated_cost / total.request_count as f64,
            self.top_tenants(5)
                .iter()
                .enumerate()
                .map(|(i, (tenant, usage))| format!(
                    "  {}. {} - ${:.2} ({} requests)",
                    i + 1,
                    tenant,
                    usage.estimated_cost,
                    usage.request_count
                ))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// Export usage data for billing system (CSV format)
    pub fn export_billing_csv(&self) -> String {
        let mut csv = String::from(
            "tenant_id,requests,input_tokens,output_tokens,total_tokens,estimated_cost_usd\n",
        );

        for (tenant, usage) in &self.tenant_usage {
            csv.push_str(&format!(
                "{},{},{},{},{},{:.4}\n",
                tenant,
                usage.request_count,
                usage.total_input_tokens,
                usage.total_output_tokens,
                usage.total_tokens,
                usage.estimated_cost
            ));
        }

        csv
    }

    /// Reset all metrics (e.g., for new billing period)
    pub fn reset(&mut self) {
        self.total_usage = UsageMetrics::new();
        self.tenant_usage.clear();
        self.model_usage.clear();
        self.recent_usage.clear();
    }

    /// Export usage metrics as samples for Prometheus export
    pub fn export_samples(&self, base_labels: Vec<MetricLabel>) -> Vec<MetricSample> {
        use crate::types::MetricValue;
        let mut samples = Vec::new();

        // Export total usage metrics
        let mut total_labels = base_labels.clone();
        total_labels.push(MetricLabel::new("name", "usage_input_tokens_total"));
        samples.push(MetricSample {
            value: MetricValue::Counter(self.total_usage.total_input_tokens),
            timestamp: self.total_usage.period_end,
            labels: total_labels.clone(),
        });

        total_labels.pop();
        total_labels.push(MetricLabel::new("name", "usage_output_tokens_total"));
        samples.push(MetricSample {
            value: MetricValue::Counter(self.total_usage.total_output_tokens),
            timestamp: self.total_usage.period_end,
            labels: total_labels.clone(),
        });

        total_labels.pop();
        total_labels.push(MetricLabel::new("name", "usage_tokens_total"));
        samples.push(MetricSample {
            value: MetricValue::Counter(self.total_usage.total_tokens),
            timestamp: self.total_usage.period_end,
            labels: total_labels.clone(),
        });

        total_labels.pop();
        total_labels.push(MetricLabel::new("name", "usage_requests_total"));
        samples.push(MetricSample {
            value: MetricValue::Counter(self.total_usage.request_count),
            timestamp: self.total_usage.period_end,
            labels: total_labels.clone(),
        });

        total_labels.pop();
        total_labels.push(MetricLabel::new("name", "usage_cost_usd"));
        samples.push(MetricSample {
            value: MetricValue::Gauge(self.total_usage.estimated_cost),
            timestamp: self.total_usage.period_end,
            labels: total_labels.clone(),
        });

        total_labels.pop();
        total_labels.push(MetricLabel::new("name", "usage_avg_tokens_per_request"));
        samples.push(MetricSample {
            value: MetricValue::Gauge(self.total_usage.avg_tokens_per_request),
            timestamp: self.total_usage.period_end,
            labels: total_labels,
        });

        // Export per-tenant usage metrics
        for (tenant_id, usage) in &self.tenant_usage {
            let mut tenant_labels = base_labels.clone();
            tenant_labels.push(MetricLabel::new("tenant", tenant_id.clone()));

            tenant_labels.push(MetricLabel::new("name", "usage_input_tokens_total"));
            samples.push(MetricSample {
                value: MetricValue::Counter(usage.total_input_tokens),
                timestamp: usage.period_end,
                labels: tenant_labels.clone(),
            });

            tenant_labels.pop();
            tenant_labels.push(MetricLabel::new("name", "usage_output_tokens_total"));
            samples.push(MetricSample {
                value: MetricValue::Counter(usage.total_output_tokens),
                timestamp: usage.period_end,
                labels: tenant_labels.clone(),
            });

            tenant_labels.pop();
            tenant_labels.push(MetricLabel::new("name", "usage_cost_usd"));
            samples.push(MetricSample {
                value: MetricValue::Gauge(usage.estimated_cost),
                timestamp: usage.period_end,
                labels: tenant_labels.clone(),
            });

            tenant_labels.pop();
            tenant_labels.push(MetricLabel::new("name", "usage_requests_total"));
            samples.push(MetricSample {
                value: MetricValue::Counter(usage.request_count),
                timestamp: usage.period_end,
                labels: tenant_labels,
            });
        }

        // Export per-model usage metrics
        for (model_name, usage) in &self.model_usage {
            let mut model_labels = base_labels.clone();
            model_labels.push(MetricLabel::new("model", model_name.clone()));

            model_labels.push(MetricLabel::new("name", "usage_tokens_total"));
            samples.push(MetricSample {
                value: MetricValue::Counter(usage.total_tokens),
                timestamp: usage.period_end,
                labels: model_labels.clone(),
            });

            model_labels.pop();
            model_labels.push(MetricLabel::new("name", "usage_cost_usd"));
            samples.push(MetricSample {
                value: MetricValue::Gauge(usage.estimated_cost),
                timestamp: usage.period_end,
                labels: model_labels,
            });
        }

        samples
    }

    /// Get cache savings metrics for export
    pub fn export_cache_savings(&self, base_labels: Vec<MetricLabel>) -> Vec<MetricSample> {
        use crate::types::{now_millis, MetricValue};
        let mut samples = Vec::new();
        let timestamp = now_millis();

        let mut total_cache_savings = 0.0;
        let mut total_cache_read_tokens = 0u64;
        let mut total_cache_creation_tokens = 0u64;

        for (_, usage) in &self.recent_usage {
            total_cache_read_tokens += usage.cache_read_input_tokens;
            total_cache_creation_tokens += usage.cache_creation_input_tokens;
            total_cache_savings += self.cost_config.calculate_cache_savings(usage);
        }

        let mut labels = base_labels.clone();
        labels.push(MetricLabel::new("name", "cache_savings_usd"));
        samples.push(MetricSample {
            value: MetricValue::Gauge(total_cache_savings),
            timestamp,
            labels: labels.clone(),
        });

        labels.pop();
        labels.push(MetricLabel::new("name", "cache_read_tokens_total"));
        samples.push(MetricSample {
            value: MetricValue::Counter(total_cache_read_tokens),
            timestamp,
            labels: labels.clone(),
        });

        labels.pop();
        labels.push(MetricLabel::new("name", "cache_creation_tokens_total"));
        samples.push(MetricSample {
            value: MetricValue::Counter(total_cache_creation_tokens),
            timestamp,
            labels,
        });

        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_usage() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_cost_calculation() {
        let config = CostConfig::simple(10.0, 30.0);
        let usage = TokenUsage::new(1_000_000, 1_000_000);
        let cost = config.calculate_cost(&usage);
        assert_eq!(cost, 40.0); // $10 + $30
    }

    #[test]
    fn test_cache_cost_savings() {
        let config = CostConfig::claude_sonnet(); // $3 input, $0.30 cache read (90% off)

        // 1M tokens from cache = $0.30 instead of $3.00
        let usage = TokenUsage::with_cache(0, 0, 0, 1_000_000);
        let cost = config.calculate_cost(&usage);
        let savings = config.calculate_cache_savings(&usage);

        assert_eq!(cost, 0.30); // Paid cache read price
        assert_eq!(savings, 2.70); // Saved $2.70 (90% off)
    }

    #[test]
    fn test_usage_tracker() {
        let mut tracker = UsageTracker::new(CostConfig::simple(1.0, 2.0));

        tracker.record_usage(TokenUsage::new(1000, 500), Some("tenant1"), Some("gpt-4"));
        tracker.record_usage(TokenUsage::new(2000, 1000), Some("tenant1"), Some("gpt-4"));
        tracker.record_usage(TokenUsage::new(500, 250), Some("tenant2"), Some("gpt-3.5"));

        let total = tracker.total();
        assert_eq!(total.request_count, 3);
        assert_eq!(total.total_input_tokens, 3500);
        assert_eq!(total.total_output_tokens, 1750);

        let tenant1 = tracker.tenant("tenant1").unwrap();
        assert_eq!(tenant1.request_count, 2);
        assert_eq!(tenant1.total_input_tokens, 3000);
    }

    #[test]
    fn test_top_tenants() {
        let mut tracker = UsageTracker::new(CostConfig::simple(1.0, 2.0));

        // Tenant1: expensive
        tracker.record_usage(TokenUsage::new(1_000_000, 500_000), Some("tenant1"), None);

        // Tenant2: cheap
        tracker.record_usage(TokenUsage::new(10_000, 5_000), Some("tenant2"), None);

        let top = tracker.top_tenants(5);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "tenant1"); // Most expensive first
        assert!(top[0].1.estimated_cost > top[1].1.estimated_cost);
    }

    #[test]
    fn test_executive_summary() {
        let mut tracker = UsageTracker::new(CostConfig::gpt4_turbo());

        tracker.record_usage(TokenUsage::new(100_000, 50_000), Some("acme_corp"), None);
        tracker.record_usage(TokenUsage::new(50_000, 25_000), Some("widgets_inc"), None);

        let summary = tracker.executive_summary();
        assert!(summary.contains("AI Usage Summary"));
        assert!(summary.contains("Total Requests: 2"));
        assert!(summary.contains("acme_corp"));
    }

    #[test]
    fn test_billing_csv_export() {
        let mut tracker = UsageTracker::new(CostConfig::simple(1.0, 2.0));

        tracker.record_usage(TokenUsage::new(1000, 500), Some("tenant1"), None);
        tracker.record_usage(TokenUsage::new(2000, 1000), Some("tenant2"), None);

        let csv = tracker.export_billing_csv();
        assert!(csv.contains("tenant_id,requests,input_tokens,output_tokens"));
        assert!(csv.contains("tenant1"));
        assert!(csv.contains("tenant2"));
    }
}
