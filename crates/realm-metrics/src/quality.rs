//! Quality metrics for AI inference
//!
//! This module tracks generation quality metrics:
//! - **Perplexity**: How "surprised" the model is by the generated text (lower is better)
//! - **Token probabilities**: Confidence scores for generated tokens
//! - **Entropy**: Uncertainty in the probability distribution
//! - **Top-k accuracy**: How often the selected token was in the top-k

use crate::types::{now_millis, MetricLabel, MetricSample, MetricValue, RollingWindow};

/// Quality metrics for a single generation
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Perplexity of the generated sequence (exp of average cross-entropy)
    pub perplexity: f64,
    /// Mean probability of generated tokens
    pub mean_token_probability: f64,
    /// Minimum probability among generated tokens (confidence floor)
    pub min_token_probability: f64,
    /// Mean entropy of the probability distribution
    pub mean_entropy: f64,
    /// Number of tokens generated
    pub num_tokens: u64,
}

impl QualityMetrics {
    /// Create quality metrics from generation statistics
    pub fn new(
        perplexity: f64,
        mean_token_probability: f64,
        min_token_probability: f64,
        mean_entropy: f64,
        num_tokens: u64,
    ) -> Self {
        Self {
            perplexity,
            mean_token_probability,
            min_token_probability,
            mean_entropy,
            num_tokens,
        }
    }

    /// Create quality metrics from a sequence of log probabilities
    pub fn from_log_probs(log_probs: &[f64]) -> Self {
        if log_probs.is_empty() {
            return Self {
                perplexity: 0.0,
                mean_token_probability: 0.0,
                min_token_probability: 0.0,
                mean_entropy: 0.0,
                num_tokens: 0,
            };
        }

        // Calculate average log probability
        let sum_log_probs: f64 = log_probs.iter().sum();
        let mean_log_prob = sum_log_probs / log_probs.len() as f64;

        // Perplexity = exp(-mean_log_prob)
        let perplexity = (-mean_log_prob).exp();

        // Convert to probabilities
        let probs: Vec<f64> = log_probs.iter().map(|lp| lp.exp()).collect();
        let mean_prob = probs.iter().sum::<f64>() / probs.len() as f64;
        let min_prob = probs.iter().copied().fold(f64::INFINITY, f64::min);

        // Entropy (assuming these are the selected token probabilities)
        // For full distribution entropy, we'd need all token probabilities
        let mean_entropy = -mean_log_prob; // Approximation

        Self {
            perplexity,
            mean_token_probability: mean_prob,
            min_token_probability: min_prob,
            mean_entropy,
            num_tokens: log_probs.len() as u64,
        }
    }

    /// Calculate entropy from a probability distribution
    pub fn calculate_entropy(probs: &[f64]) -> f64 {
        probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum()
    }

    /// Convert to metric samples for export
    pub fn to_samples(&self, labels: Vec<MetricLabel>) -> Vec<MetricSample> {
        let timestamp = now_millis();
        vec![
            MetricSample {
                value: MetricValue::Gauge(self.perplexity),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.mean_token_probability),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.min_token_probability),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.mean_entropy),
                timestamp,
                labels,
            },
        ]
    }
}

/// Quality tracker with rolling window statistics
pub struct QualityTracker {
    /// Rolling window for perplexity
    perplexity_window: RollingWindow,
    /// Rolling window for mean token probability
    mean_prob_window: RollingWindow,
    /// Rolling window for min token probability
    min_prob_window: RollingWindow,
    /// Rolling window for entropy
    entropy_window: RollingWindow,
    /// Total tokens evaluated
    total_tokens: u64,
    /// Total generations tracked
    total_generations: u64,
}

impl QualityTracker {
    /// Create a new quality tracker with the given window size
    pub fn new(window_size: usize) -> Self {
        Self {
            perplexity_window: RollingWindow::new(window_size),
            mean_prob_window: RollingWindow::new(window_size),
            min_prob_window: RollingWindow::new(window_size),
            entropy_window: RollingWindow::new(window_size),
            total_tokens: 0,
            total_generations: 0,
        }
    }

    /// Record quality metrics from a generation
    pub fn record(&mut self, metrics: QualityMetrics) {
        self.perplexity_window.add(metrics.perplexity);
        self.mean_prob_window.add(metrics.mean_token_probability);
        self.min_prob_window.add(metrics.min_token_probability);
        self.entropy_window.add(metrics.mean_entropy);
        self.total_tokens += metrics.num_tokens;
        self.total_generations += 1;
    }

    /// Get mean perplexity over the window
    pub fn mean_perplexity(&self) -> f64 {
        self.perplexity_window.mean()
    }

    /// Get p99 perplexity over the window
    pub fn p99_perplexity(&self) -> f64 {
        self.perplexity_window.p99()
    }

    /// Get mean token probability over the window
    pub fn mean_token_probability(&self) -> f64 {
        self.mean_prob_window.mean()
    }

    /// Get mean entropy over the window
    pub fn mean_entropy(&self) -> f64 {
        self.entropy_window.mean()
    }

    /// Get total tokens evaluated
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Get total generations tracked
    pub fn total_generations(&self) -> u64 {
        self.total_generations
    }

    /// Export current statistics as metric samples
    pub fn export_samples(&self, labels: Vec<MetricLabel>) -> Vec<MetricSample> {
        let timestamp = now_millis();
        vec![
            MetricSample {
                value: MetricValue::Gauge(self.mean_perplexity()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.p99_perplexity()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.mean_token_probability()),
                timestamp,
                labels: labels.clone(),
            },
            MetricSample {
                value: MetricValue::Gauge(self.mean_entropy()),
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
}

impl Default for QualityTracker {
    fn default() -> Self {
        Self::new(100) // Default to 100-sample window
    }
}

/// Token-level quality information
#[derive(Debug, Clone)]
pub struct TokenQuality {
    /// The token ID
    pub token_id: u32,
    /// Probability of this token
    pub probability: f64,
    /// Log probability
    pub log_probability: f64,
    /// Entropy of the distribution at this step
    pub entropy: f64,
    /// Was this token in the top-k?
    pub in_top_k: bool,
    /// Rank of this token in the distribution (1 = most likely)
    pub rank: usize,
}

impl TokenQuality {
    /// Create token quality info
    pub fn new(
        token_id: u32,
        probability: f64,
        log_probability: f64,
        entropy: f64,
        rank: usize,
        k: usize,
    ) -> Self {
        Self {
            token_id,
            probability,
            log_probability,
            entropy,
            in_top_k: rank <= k,
            rank,
        }
    }
}

/// Aggregated quality statistics for a sequence
#[derive(Debug, Clone)]
pub struct SequenceQuality {
    /// Per-token quality information
    pub tokens: Vec<TokenQuality>,
    /// Overall sequence perplexity
    pub perplexity: f64,
    /// Mean token probability
    pub mean_probability: f64,
    /// Fraction of tokens in top-5
    pub top5_rate: f64,
    /// Fraction of tokens in top-10
    pub top10_rate: f64,
}

impl SequenceQuality {
    /// Create sequence quality from token-level info
    pub fn from_tokens(tokens: Vec<TokenQuality>) -> Self {
        if tokens.is_empty() {
            return Self {
                tokens,
                perplexity: 0.0,
                mean_probability: 0.0,
                top5_rate: 0.0,
                top10_rate: 0.0,
            };
        }

        // Calculate perplexity
        let mean_log_prob =
            tokens.iter().map(|t| t.log_probability).sum::<f64>() / tokens.len() as f64;
        let perplexity = (-mean_log_prob).exp();

        // Calculate mean probability
        let mean_probability =
            tokens.iter().map(|t| t.probability).sum::<f64>() / tokens.len() as f64;

        // Calculate top-k rates
        let top5_count = tokens.iter().filter(|t| t.rank <= 5).count();
        let top10_count = tokens.iter().filter(|t| t.rank <= 10).count();
        let top5_rate = top5_count as f64 / tokens.len() as f64;
        let top10_rate = top10_count as f64 / tokens.len() as f64;

        Self {
            tokens,
            perplexity,
            mean_probability,
            top5_rate,
            top10_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_metrics_from_log_probs() {
        // Test with some realistic log probabilities
        let log_probs = vec![-0.5, -1.0, -0.3, -0.8]; // High-confidence generation
        let metrics = QualityMetrics::from_log_probs(&log_probs);

        assert!(metrics.perplexity > 0.0);
        assert!(metrics.mean_token_probability > 0.0);
        assert_eq!(metrics.num_tokens, 4);
    }

    #[test]
    fn test_entropy_calculation() {
        // Uniform distribution over 4 outcomes
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let entropy = QualityMetrics::calculate_entropy(&probs);
        assert!((entropy - 2.0).abs() < 0.01); // Should be exactly 2 bits

        // Deterministic distribution
        let probs = vec![1.0, 0.0, 0.0, 0.0];
        let entropy = QualityMetrics::calculate_entropy(&probs);
        assert!(entropy.abs() < 0.01); // Should be ~0 bits
    }

    #[test]
    fn test_quality_tracker() {
        let mut tracker = QualityTracker::new(3);

        tracker.record(QualityMetrics::new(10.0, 0.8, 0.5, 2.0, 20));
        tracker.record(QualityMetrics::new(12.0, 0.75, 0.4, 2.2, 25));

        assert_eq!(tracker.total_tokens(), 45);
        assert_eq!(tracker.total_generations(), 2);
        assert!(tracker.mean_perplexity() > 0.0);
    }

    #[test]
    fn test_sequence_quality() {
        let tokens = vec![
            TokenQuality::new(123, 0.8, -0.22, 1.5, 1, 10),
            TokenQuality::new(456, 0.6, -0.51, 2.0, 3, 10),
            TokenQuality::new(789, 0.9, -0.11, 1.2, 1, 10),
        ];

        let seq_quality = SequenceQuality::from_tokens(tokens);
        assert_eq!(seq_quality.top5_rate, 1.0); // All tokens in top-5
        assert!(seq_quality.perplexity > 0.0);
    }
}
