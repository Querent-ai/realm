//! Rate Limiting for Multi-Tenant Inference
//!
//! Token bucket algorithm for per-tenant rate limiting.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, error, warn};

/// Rate limiter configuration
#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
    /// Default requests per minute
    pub requests_per_minute: usize,

    /// Default tokens per minute (for token-based limiting)
    pub tokens_per_minute: Option<usize>,

    /// Burst allowance (max tokens in bucket)
    pub burst_size: usize,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            tokens_per_minute: None,
            burst_size: 100,
        }
    }
}

/// Token bucket for rate limiting
#[derive(Debug)]
struct TokenBucket {
    /// Number of tokens available
    tokens: f64,

    /// Maximum tokens (burst size)
    capacity: f64,

    /// Tokens added per second
    refill_rate: f64,

    /// Last refill time
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a new token bucket
    fn new(requests_per_minute: usize, burst_size: usize) -> Self {
        let refill_rate = requests_per_minute as f64 / 60.0; // tokens per second

        Self {
            tokens: burst_size as f64,
            capacity: burst_size as f64,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();

        // Add tokens based on elapsed time
        let new_tokens = elapsed * self.refill_rate;
        self.tokens = (self.tokens + new_tokens).min(self.capacity);

        self.last_refill = now;
    }

    /// Try to consume tokens
    fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();

        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    /// Get available tokens
    fn available(&mut self) -> f64 {
        self.refill();
        self.tokens
    }

    /// Get time until tokens available
    fn time_until_available(&mut self, tokens: f64) -> Duration {
        self.refill();

        if self.tokens >= tokens {
            Duration::ZERO
        } else {
            let needed = tokens - self.tokens;
            let seconds = needed / self.refill_rate;
            Duration::from_secs_f64(seconds)
        }
    }
}

/// Per-tenant rate limiting statistics
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    /// Total requests made
    pub total_requests: usize,

    /// Total requests blocked
    pub blocked_requests: usize,

    /// Average tokens consumed per request
    pub avg_tokens_per_request: f64,

    /// Last request time
    pub last_request: Option<Instant>,
}

impl Default for RateLimitStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            blocked_requests: 0,
            avg_tokens_per_request: 1.0,
            last_request: None,
        }
    }
}

/// Tenant rate limiter entry
#[derive(Debug)]
struct TenantRateLimiter {
    /// Token bucket for rate limiting
    bucket: TokenBucket,

    /// Statistics
    stats: RateLimitStats,

    /// Custom rate limit (if set)
    custom_limit: Option<usize>,
}

impl TenantRateLimiter {
    fn new(requests_per_minute: usize, burst_size: usize) -> Self {
        Self {
            bucket: TokenBucket::new(requests_per_minute, burst_size),
            stats: RateLimitStats::default(),
            custom_limit: None,
        }
    }

    fn with_custom_limit(mut self, limit: usize, burst_size: usize) -> Self {
        self.custom_limit = Some(limit);
        self.bucket = TokenBucket::new(limit, burst_size);
        self
    }
}

/// Multi-tenant rate limiter
pub struct RateLimiter {
    /// Configuration
    config: RateLimiterConfig,

    /// Per-tenant limiters
    limiters: Arc<Mutex<HashMap<String, TenantRateLimiter>>>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(config: RateLimiterConfig) -> Self {
        Self {
            config,
            limiters: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Set custom rate limit for a tenant
    pub fn set_tenant_limit(&self, tenant_id: impl Into<String>, requests_per_minute: usize) {
        let tenant_id = tenant_id.into();
        let mut limiters = match self.limiters.lock() {
            Ok(guard) => guard,
            Err(e) => {
                error!("Failed to acquire limiters lock: {}", e);
                return;
            }
        };

        if let Some(limiter) = limiters.get_mut(&tenant_id) {
            limiter.custom_limit = Some(requests_per_minute);
            limiter.bucket = TokenBucket::new(requests_per_minute, self.config.burst_size);
        } else {
            let limiter =
                TenantRateLimiter::new(self.config.requests_per_minute, self.config.burst_size)
                    .with_custom_limit(requests_per_minute, self.config.burst_size);

            limiters.insert(tenant_id, limiter);
        }
    }

    /// Check if a request is allowed for a tenant
    pub fn check_rate_limit(&self, tenant_id: impl AsRef<str>) -> Result<()> {
        self.check_rate_limit_with_cost(tenant_id, 1.0)
    }

    /// Check rate limit with custom token cost
    pub fn check_rate_limit_with_cost(&self, tenant_id: impl AsRef<str>, cost: f64) -> Result<()> {
        let tenant_id = tenant_id.as_ref();
        let mut limiters = self
            .limiters
            .lock()
            .map_err(|e| anyhow!("Failed to acquire limiters lock: {}", e))?;

        let limiter = limiters.entry(tenant_id.to_string()).or_insert_with(|| {
            TenantRateLimiter::new(self.config.requests_per_minute, self.config.burst_size)
        });

        limiter.stats.total_requests += 1;
        limiter.stats.last_request = Some(Instant::now());

        if limiter.bucket.try_consume(cost) {
            debug!(
                "Rate limit check passed for tenant {}: {} tokens available",
                tenant_id,
                limiter.bucket.available()
            );
            Ok(())
        } else {
            limiter.stats.blocked_requests += 1;

            let wait_time = limiter.bucket.time_until_available(cost);

            warn!(
                "Rate limit exceeded for tenant {}: need to wait {:?}",
                tenant_id, wait_time
            );

            Err(anyhow!(
                "Rate limit exceeded. Try again in {} seconds.",
                wait_time.as_secs()
            ))
        }
    }

    /// Get rate limit statistics for a tenant
    pub fn get_stats(&self, tenant_id: impl AsRef<str>) -> Option<RateLimitStats> {
        self.limiters
            .lock()
            .ok()
            .and_then(|limiters| limiters.get(tenant_id.as_ref()).map(|l| l.stats.clone()))
    }

    /// Get available tokens for a tenant
    pub fn get_available_tokens(&self, tenant_id: impl AsRef<str>) -> Option<f64> {
        self.limiters.lock().ok().and_then(|mut limiters| {
            limiters
                .get_mut(tenant_id.as_ref())
                .map(|l| l.bucket.available())
        })
    }

    /// Reset rate limit for a tenant
    pub fn reset_tenant(&self, tenant_id: impl AsRef<str>) {
        if let Ok(mut limiters) = self.limiters.lock() {
            if let Some(limiter) = limiters.get_mut(tenant_id.as_ref()) {
                limiter.bucket.tokens = limiter.bucket.capacity;
                limiter.bucket.last_refill = Instant::now();
                limiter.stats = RateLimitStats::default();
            }
        } else {
            error!("Failed to acquire limiters lock for reset_tenant");
        }
    }

    /// Remove tenant rate limiter
    pub fn remove_tenant(&self, tenant_id: impl AsRef<str>) {
        if let Ok(mut limiters) = self.limiters.lock() {
            limiters.remove(tenant_id.as_ref());
        } else {
            error!("Failed to acquire limiters lock for remove_tenant");
        }
    }

    /// Get all tenant IDs
    pub fn list_tenants(&self) -> Vec<String> {
        self.limiters
            .lock()
            .ok()
            .map(|limiters| limiters.keys().cloned().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_token_bucket_basic() {
        let mut bucket = TokenBucket::new(60, 100); // 60 req/min = 1 req/sec

        // Should allow immediate requests
        assert!(bucket.try_consume(1.0));
        assert!(bucket.try_consume(1.0));
    }

    #[test]
    fn test_token_bucket_refill() {
        let mut bucket = TokenBucket::new(60, 100); // 1 token per second

        // Consume all tokens
        for _ in 0..100 {
            bucket.try_consume(1.0);
        }

        // Should be empty
        assert!(!bucket.try_consume(1.0));

        // Wait for refill
        thread::sleep(Duration::from_secs(2));

        // Should have ~2 tokens now
        assert!(bucket.try_consume(1.0));
        assert!(bucket.try_consume(1.0));
    }

    #[test]
    fn test_rate_limiter() {
        let config = RateLimiterConfig {
            requests_per_minute: 60,
            tokens_per_minute: None,
            burst_size: 10,
        };

        let limiter = RateLimiter::new(config);

        // Should allow 10 requests (burst)
        for i in 0..10 {
            assert!(
                limiter.check_rate_limit("tenant1").is_ok(),
                "Request {} failed",
                i
            );
        }

        // 11th request should fail
        assert!(limiter.check_rate_limit("tenant1").is_err());
    }

    #[test]
    fn test_custom_tenant_limit() {
        let config = RateLimiterConfig::default();
        let limiter = RateLimiter::new(config);

        // Set custom limit for tenant
        limiter.set_tenant_limit("premium", 120); // 2x default

        // Should have 100 tokens available (burst_size)
        let available = limiter.get_available_tokens("premium").unwrap();
        assert!(available > 90.0);
    }

    #[test]
    fn test_stats_tracking() {
        let config = RateLimiterConfig {
            requests_per_minute: 60,
            tokens_per_minute: None,
            burst_size: 5,
        };

        let limiter = RateLimiter::new(config);

        // Make requests
        for _ in 0..3 {
            let _ = limiter.check_rate_limit("tenant1");
        }

        let stats = limiter.get_stats("tenant1").unwrap();
        assert_eq!(stats.total_requests, 3);
        assert!(stats.last_request.is_some());
    }
}
