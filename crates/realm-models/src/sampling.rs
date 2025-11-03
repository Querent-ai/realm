//! Logit Processing and Sampling
//!
//! Functionality for modeling sampling strategies and logits processing in text generation
//! with support for temperature-based sampling, top-k filtering, nucleus sampling (top-p),
//! and combinations thereof.
//!
//! Inspired by Candle's logits processor implementation.

use rand::distributions::weighted::WeightedIndex;
use rand::distributions::Distribution;
use rand::SeedableRng;

/// Sampling strategy for text generation
#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    /// Always select the token with highest probability
    ArgMax,
    /// Sample from all tokens with temperature scaling
    All { temperature: f64 },
    /// Sample from top-k tokens with temperature scaling
    TopK { k: usize, temperature: f64 },
    /// Nucleus sampling (top-p) with temperature scaling
    TopP { p: f64, temperature: f64 },
    /// Combined top-k then top-p sampling
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

/// Processes logits and samples next tokens according to various strategies
pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
    repetition_penalty: f32,
    previous_tokens: Vec<u32>,
}

impl LogitsProcessor {
    /// Create a new logits processor with a specific sampling strategy
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self {
            rng,
            sampling,
            repetition_penalty: 1.0,
            previous_tokens: Vec::new(),
        }
    }

    /// Create a new logits processor with temperature and top-p
    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    /// Create a new logits processor with all parameters
    pub fn with_params(
        seed: u64,
        temperature: f64,
        top_p: f64,
        top_k: usize,
        repetition_penalty: f32,
    ) -> Self {
        let sampling = if temperature < 1e-7 {
            Sampling::ArgMax
        } else if top_k > 0 && top_p > 0.0 && top_p < 1.0 {
            Sampling::TopKThenTopP {
                k: top_k,
                p: top_p,
                temperature,
            }
        } else if top_k > 0 {
            Sampling::TopK {
                k: top_k,
                temperature,
            }
        } else if top_p > 0.0 && top_p < 1.0 {
            Sampling::TopP {
                p: top_p,
                temperature,
            }
        } else {
            Sampling::All { temperature }
        };

        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self {
            rng,
            sampling,
            repetition_penalty,
            previous_tokens: Vec::new(),
        }
    }

    /// Set repetition penalty
    pub fn set_repetition_penalty(&mut self, penalty: f32) {
        self.repetition_penalty = penalty;
    }

    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(&self, logits: &mut [f32]) {
        if self.repetition_penalty == 1.0 {
            return;
        }

        for &token_id in &self.previous_tokens {
            let idx = token_id as usize;
            if idx < logits.len() {
                // If logit is positive, divide by penalty
                // If logit is negative, multiply by penalty
                if logits[idx] > 0.0 {
                    logits[idx] /= self.repetition_penalty;
                } else {
                    logits[idx] *= self.repetition_penalty;
                }
            }
        }
    }

    /// Apply temperature scaling and softmax to logits
    fn compute_probabilities(&self, logits: &[f32], temperature: f64) -> Vec<f32> {
        // Apply temperature scaling
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature as f32).collect();

        // Compute softmax
        let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|&x| (x - max_logit).exp()).sum();

        scaled
            .iter()
            .map(|&x| (x - max_logit).exp() / exp_sum)
            .collect()
    }

    /// Sample using argmax (greedy decoding)
    fn sample_argmax(&mut self, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// Sample from a multinomial distribution
    fn sample_multinomial(&mut self, probs: &[f32]) -> Result<u32, String> {
        let dist = WeightedIndex::new(probs)
            .map_err(|e| format!("Failed to create weighted index: {}", e))?;
        Ok(dist.sample(&mut self.rng) as u32)
    }

    /// Top-p sampling (nucleus sampling)
    ///
    /// Samples from the smallest set of tokens that exceed probability top_p.
    /// This way we never sample tokens that have very low probabilities.
    fn sample_topp(&mut self, probs: &mut [f32], top_p: f32) -> Result<u32, String> {
        let mut indices: Vec<usize> = (0..probs.len()).collect();

        // Sort by descending probability
        indices.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

        // Clamp smaller probabilities to zero
        let mut cumsum = 0.0;
        for &index in &indices {
            if cumsum >= top_p {
                probs[index] = 0.0;
            } else {
                cumsum += probs[index];
            }
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        self.sample_multinomial(probs)
    }

    /// Top-k sampling
    ///
    /// Samples from the k tokens with the largest probabilities.
    fn sample_topk(&mut self, probs: &mut [f32], top_k: usize) -> Result<u32, String> {
        if top_k >= probs.len() {
            return self.sample_multinomial(probs);
        }

        let mut indices: Vec<usize> = (0..probs.len()).collect();

        // Partially sort to find top-k
        indices.select_nth_unstable_by(top_k, |&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

        // Zero out probabilities outside top-k
        for &i in &indices[top_k..] {
            probs[i] = 0.0;
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        self.sample_multinomial(probs)
    }

    /// Combined top-k then top-p sampling
    fn sample_topk_topp(
        &mut self,
        probs: &mut [f32],
        top_k: usize,
        top_p: f32,
    ) -> Result<u32, String> {
        if top_k >= probs.len() {
            return self.sample_topp(probs, top_p);
        }

        let mut indices: Vec<usize> = (0..probs.len()).collect();

        // First apply top-k
        indices.select_nth_unstable_by(top_k, |&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

        // Zero out probabilities outside top-k
        for &i in &indices[top_k..] {
            probs[i] = 0.0;
        }

        // Renormalize after top-k
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        // Then apply top-p
        if top_p <= 0.0 || top_p >= 1.0 {
            self.sample_multinomial(probs)
        } else {
            self.sample_topp(probs, top_p)
        }
    }

    /// Sample next token from logits
    pub fn sample(&mut self, logits: &mut [f32]) -> Result<u32, String> {
        // Apply repetition penalty
        self.apply_repetition_penalty(logits);

        let next_token = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax(logits),
            Sampling::All { temperature } => {
                let probs = self.compute_probabilities(logits, *temperature);
                self.sample_multinomial(&probs)?
            }
            Sampling::TopP { p, temperature } => {
                let mut probs = self.compute_probabilities(logits, *temperature);
                if *p <= 0.0 || *p >= 1.0 {
                    self.sample_multinomial(&probs)?
                } else {
                    self.sample_topp(&mut probs, *p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let mut probs = self.compute_probabilities(logits, *temperature);
                self.sample_topk(&mut probs, *k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut probs = self.compute_probabilities(logits, *temperature);
                self.sample_topk_topp(&mut probs, *k, *p as f32)?
            }
        };

        // Track token for repetition penalty
        self.previous_tokens.push(next_token);

        Ok(next_token)
    }

    /// Clear the history of previous tokens
    pub fn clear_history(&mut self) {
        self.previous_tokens.clear();
    }

    /// Get the current sampling strategy
    pub fn sampling_strategy(&self) -> &Sampling {
        &self.sampling
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_sampling() {
        let mut processor = LogitsProcessor::from_sampling(42, Sampling::ArgMax);
        let mut logits = vec![1.0, 3.0, 2.0, 5.0, 1.5];
        let token = processor.sample(&mut logits).unwrap();
        assert_eq!(token, 3); // Index of max value (5.0)
    }

    #[test]
    fn test_temperature_sampling() {
        let mut processor = LogitsProcessor::from_sampling(42, Sampling::All { temperature: 1.0 });
        let mut logits = vec![1.0, 2.0, 3.0];
        let token = processor.sample(&mut logits).unwrap();
        assert!(token < 3);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut processor = LogitsProcessor::from_sampling(42, Sampling::ArgMax);
        processor.set_repetition_penalty(1.5);

        let mut logits = vec![5.0, 3.0, 2.0];
        let token1 = processor.sample(&mut logits).unwrap();
        assert_eq!(token1, 0);

        // Second sample should penalize token 0
        let mut logits = vec![5.0, 3.0, 2.0];
        processor.apply_repetition_penalty(&mut logits);
        assert!(logits[0] < 5.0); // Should be penalized
    }

    #[test]
    fn test_topk_sampling() {
        let mut processor = LogitsProcessor::from_sampling(
            42,
            Sampling::TopK {
                k: 2,
                temperature: 1.0,
            },
        );
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let token = processor.sample(&mut logits).unwrap();
        // Should only sample from indices 1 (5.0) or 4 (4.0)
        assert!(token == 1 || token == 4);
    }

    #[test]
    fn test_topp_sampling() {
        let mut processor = LogitsProcessor::from_sampling(
            42,
            Sampling::TopP {
                p: 0.9,
                temperature: 1.0,
            },
        );
        let mut logits = vec![1.0, 10.0, 2.0, 1.0, 1.0];
        let token = processor.sample(&mut logits).unwrap();
        assert!(token < 5);
    }
}
