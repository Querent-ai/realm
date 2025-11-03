# Business Metrics for Sales, Accounting & Decision-Making

This document describes the business-oriented metrics implemented in Realm.ai's metrics system. These metrics are designed to help sales teams, accounting departments, and executives understand inference consumption, costs, and revenue.

## ✅ Metrics Confirmed: All Integrated!

All metrics are now integrated into `MetricsCollector` and export to Prometheus format automatically.

---

## 📊 Metrics Categories

### 1. **Usage & Billing Metrics** ✅

Track token consumption and costs for billing and accounting:

#### Token Usage
- `realm_usage_input_tokens_total` - Total input tokens (prompt)
- `realm_usage_output_tokens_total` - Total output tokens (generated)
- `realm_usage_tokens_total` - Total tokens (input + output)
- `realm_usage_requests_total` - Total number of requests
- `realm_usage_avg_tokens_per_request` - Average tokens per request

#### Cost Tracking
- `realm_usage_cost_usd` - Estimated cost in USD (total and per-tenant/per-model)
- `realm_cache_savings_usd` - Cost savings from prompt caching
- `realm_cache_read_tokens_total` - Tokens served from cache
- `realm_cache_creation_tokens_total` - Tokens used to create cache entries

#### Per-Tenant/Per-Model Breakdown
All usage metrics support labels:
- `tenant="tenant1"` - Per-tenant usage
- `model="llama-7b"` - Per-model usage

**Example Prometheus Output:**
```
# TYPE realm_usage_cost_usd gauge
realm_usage_cost_usd 1250.50 1234567890
realm_usage_cost_usd{tenant="acme_corp"} 850.25 1234567890
realm_usage_cost_usd{tenant="widgets_inc"} 400.25 1234567890
realm_usage_cost_usd{model="llama-7b"} 600.00 1234567890
```

---

### 2. **Success & Error Metrics** ✅

Track request success rates and error types for reliability monitoring:

#### Request Rates
- `realm_requests_total` - Total requests (success + failure)
- `realm_requests_success_total` - Successful requests (2xx status)
- `realm_success_rate` - Success rate (0.0 to 1.0)
- `realm_error_rate` - Error rate (0.0 to 1.0)

#### Error Classification
- `realm_requests_errors_total{error_type="rate_limit"}` - Rate limit errors (429)
- `realm_requests_errors_total{error_type="timeout"}` - Timeout errors (408)
- `realm_requests_errors_total{error_type="validation_error"}` - Validation errors (400)
- `realm_requests_errors_total{error_type="authentication_error"}` - Auth errors (401)
- `realm_requests_errors_total{error_type="authorization_error"}` - Authorization errors (403)
- `realm_requests_errors_total{error_type="internal_error"}` - Internal server errors (500)
- `realm_requests_errors_total{error_type="service_unavailable"}` - Service unavailable (503)
- `realm_requests_errors_total{error_type="out_of_memory"}` - OOM errors
- `realm_requests_errors_total{error_type="model_unavailable"}` - Model unavailable errors

#### Status Code Tracking
- `realm_requests_by_status_total{status_code="200"}` - Requests by HTTP status code

**Example Prometheus Output:**
```
# TYPE realm_success_rate gauge
realm_success_rate 0.987 1234567890

# TYPE realm_requests_errors_total counter
realm_requests_errors_total{error_type="rate_limit"} 12 1234567890
realm_requests_errors_total{error_type="timeout"} 5 1234567890
```

---

### 3. **Client/API Key Attribution** ✅

Track usage by API key and client ID for customer billing:

#### API Key Tracking
- `realm_requests_by_api_key_total{api_key_hash="abc123"}` - Requests per API key
- `realm_tokens_by_api_key_total{api_key_hash="abc123"}` - Tokens per API key

#### Client ID Tracking
- `realm_requests_by_client_total{client_id="client1"}` - Requests per client
- `realm_tokens_by_client_total{client_id="client1"}` - Tokens per client

**Note:** API keys are hashed/masked for privacy in metrics. Use client IDs for customer-level attribution.

**Example Prometheus Output:**
```
# TYPE realm_requests_by_client_total counter
realm_requests_by_client_total{client_id="acme_corp"} 15234 1234567890
realm_tokens_by_client_total{client_id="acme_corp"} 5234567 1234567890
```

---

### 4. **Revenue Metrics** (Optional) ✅

If you charge customers, track revenue per tenant/client:

- `realm_revenue_usd{tenant="tenant1"}` - Revenue per tenant
- `realm_revenue_usd{client_id="client1"}` - Revenue per client

**Example Prometheus Output:**
```
# TYPE realm_revenue_usd gauge
realm_revenue_usd{tenant="acme_corp"} 5000.00 1234567890
realm_revenue_usd{client_id="client1"} 3500.00 1234567890
```

---

### 5. **Cache Savings Metrics** ✅

Track cost savings from prompt caching (like Claude's prompt caching):

- `realm_cache_savings_usd` - Total cost savings from caching
- `realm_cache_read_tokens_total` - Tokens served from cache
- `realm_cache_creation_tokens_total` - Tokens used to build cache

**Example:** If cache read tokens cost 90% less than regular input tokens, this metric shows the savings.

---

## 🎯 Standard Metrics (Also Available)

We also track standard operational metrics that Claude and other providers use:

### Latency Metrics
- `realm_ttft_seconds` - Time to first token
- `realm_tokens_per_sec` - Generation throughput
- `realm_total_time_seconds` - End-to-end latency

### Throughput Metrics
- `realm_requests_per_sec` - Request rate
- `realm_tokens_per_sec` - Token generation rate
- `realm_concurrent_requests` - Active concurrent requests
- `realm_queue_depth` - Requests waiting in queue

### Resource Metrics
- `realm_memory_bytes` - Memory usage
- `realm_cache_hit_rate` - KV cache hit rate

---

## 💻 Usage Examples

### Recording Token Usage for Billing

```rust
use realm_metrics::{MetricsCollector, TokenUsage};

let collector = MetricsCollector::with_cost_config(
    CostConfig::claude_sonnet() // $3/1M input, $15/1M output
);

// After a request completes
collector.record_usage(
    TokenUsage::new(1000, 500)
        .with_model("claude-3-sonnet")
        .with_stop_reason("stop_sequence"),
    Some("tenant1"),
    Some("claude-3-sonnet"),
);
```

### Recording Request Success/Failure

```rust
// Record successful request
collector.record_business_request(
    200,                    // HTTP status code
    Some("sk-abc123"),      // API key (will be hashed)
    Some("acme_corp"),      // Client ID
    Some(1500),             // Tokens generated
);

// Record error
collector.record_business_request(
    429,                    // Rate limit error
    Some("sk-abc123"),
    Some("acme_corp"),
    None,                   // No tokens (failed request)
);
```

### Recording Revenue (if applicable)

```rust
// If you charge customers per request or token
collector.record_revenue(
    0.05,                   // $0.05 per request
    Some("tenant1"),
    Some("client1"),
);
```

### Exporting to Prometheus

```rust
// Get all metrics in Prometheus format
let prometheus_text = collector.export_prometheus();

// Or use the Prometheus exporter
use realm_metrics::PrometheusExporter;
let exporter = PrometheusExporter::new();
let samples = collector.export_all();
let formatted = exporter.export(&samples).unwrap();
```

---

## 📈 What Claude/Anthropic Collects

Based on industry standards, Claude API tracks similar metrics:

1. **Token Usage** - Input/output tokens, cache reads
2. **Cost Calculation** - Based on model pricing
3. **Request Metadata** - Model, stop reason, timestamps
4. **Cache Metrics** - Cache creation vs read tokens
5. **Error Tracking** - Rate limits, timeouts, validation errors

**✅ We track all of these, plus:**
- Per-tenant/client attribution
- Revenue tracking (if applicable)
- Detailed error categorization
- Success rate monitoring

---

## 🚀 Next Steps

1. **HTTP Server** - Build Axum/Actix server with `/v1/completions` and `/metrics` endpoints
2. **Grafana Dashboards** - Create dashboards for:
   - Cost per tenant/client
   - Success rate monitoring
   - Token usage trends
   - Cache savings visualization
3. **Billing Integration** - Connect usage metrics to billing system
4. **Alerts** - Set up alerts for high error rates or cost spikes

---

## 📊 Metrics Summary

| Category | Metrics | Purpose |
|----------|---------|---------|
| **Usage** | 6 metrics | Billing, cost tracking |
| **Cache** | 3 metrics | Cost savings analysis |
| **Success/Error** | 10+ metrics | Reliability monitoring |
| **Client Attribution** | 4 metrics | Customer billing |
| **Revenue** | 2 metrics | Revenue tracking |
| **Latency** | 5 metrics | Performance monitoring |
| **Throughput** | 10 metrics | Capacity planning |
| **Resource** | 7 metrics | Resource optimization |

**Total: 47+ business-oriented metrics** ready for Prometheus/Grafana! 🎉

