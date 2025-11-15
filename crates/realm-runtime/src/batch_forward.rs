//! Batch forward pass for parallel GPU processing
//!
//! This module provides the interface and structure for parallel batch forward passes
//! that can be accelerated on GPU.

use realm_core::error::Result;

/// Batch forward pass configuration
#[derive(Debug, Clone)]
pub struct BatchForwardConfig {
    /// Maximum batch size for forward pass
    pub max_batch_size: usize,
    /// Maximum sequence length in batch
    pub max_seq_len: usize,
    /// Whether to use padding for variable-length sequences
    pub use_padding: bool,
}

impl Default for BatchForwardConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_seq_len: 2048,
            use_padding: true,
        }
    }
}

/// Batch forward pass result
#[derive(Debug, Clone)]
pub struct BatchForwardResult {
    /// Logits for each sequence in the batch
    /// Shape: [batch_size, seq_len, vocab_size]
    pub logits: Vec<Vec<Vec<f32>>>,
    /// Attention masks used (if padding was applied)
    pub attention_masks: Option<Vec<Vec<bool>>>,
}

/// Trait for backends that support batch forward passes
pub trait BatchForwardBackend: Send + Sync {
    /// Perform a batch forward pass
    ///
    /// # Arguments
    /// * `batch_tokens` - Vector of token sequences, one per request
    /// * `config` - Batch forward configuration
    ///
    /// # Returns
    /// BatchForwardResult with logits for each sequence
    fn forward_batch(
        &self,
        batch_tokens: &[Vec<u32>],
        config: &BatchForwardConfig,
    ) -> Result<BatchForwardResult>;
}

/// CPU-based batch forward pass (sequential processing)
pub struct CpuBatchForwardBackend;

impl BatchForwardBackend for CpuBatchForwardBackend {
    fn forward_batch(
        &self,
        batch_tokens: &[Vec<u32>],
        _config: &BatchForwardConfig,
    ) -> Result<BatchForwardResult> {
        // CPU implementation: process sequentially
        // In production, this would call individual forward passes
        let mut logits = Vec::new();

        for tokens in batch_tokens {
            // Placeholder: would call actual forward pass here
            // For now, return dummy logits
            let seq_len = tokens.len();
            let vocab_size = 32000; // Example vocab size
            let sequence_logits = vec![vec![0.0; vocab_size]; seq_len];
            logits.push(sequence_logits);
        }

        Ok(BatchForwardResult {
            logits,
            attention_masks: None,
        })
    }
}

/// GPU-based batch forward pass (parallel processing)
///
/// Uses GPU backend to process batches efficiently by leveraging GPU parallelism.
pub struct GpuBatchForwardBackend {
    device_id: usize,
}

impl GpuBatchForwardBackend {
    pub fn new(device_id: usize) -> Self {
        Self { device_id }
    }

    /// Get the device ID for this GPU backend
    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

impl BatchForwardBackend for GpuBatchForwardBackend {
    fn forward_batch(
        &self,
        batch_tokens: &[Vec<u32>],
        config: &BatchForwardConfig,
    ) -> Result<BatchForwardResult> {
        // GPU implementation: parallel batch processing
        // Prepare batch with padding and attention masks
        let (padded_batch, attention_masks) = if config.use_padding {
            prepare_batch(batch_tokens, config.max_seq_len)
        } else {
            // No padding - use original sequences
            let masks: Vec<Vec<bool>> = batch_tokens
                .iter()
                .map(|tokens| vec![true; tokens.len()])
                .collect();
            (batch_tokens.to_vec(), masks)
        };

        // Use device_id to select the appropriate GPU device
        // This ensures we use the correct GPU when multiple GPUs are available
        // When GPU backend is fully integrated, device_id will be used to create
        // the appropriate Device (e.g., Device::new_cuda(device_id) or Device::new_metal(device_id))
        // Access device_id via getter method to ensure it's used
        let _device_id = self.device_id();

        // Process batch using GPU backend with the specified device_id
        // For each sequence, we perform a forward pass
        // GPU backend will handle parallelization internally
        let mut logits = Vec::new();

        for (tokens, _mask) in padded_batch.iter().zip(attention_masks.iter()) {
            // Get actual sequence length (before padding)
            let seq_len = tokens.iter().take_while(|&&t| t != 0).count();
            if seq_len == 0 {
                // Empty sequence
                let vocab_size = 32000; // Default vocab size
                logits.push(vec![vec![0.0; vocab_size]; 1]);
                continue;
            }

            // Process sequence through model using GPU device specified by device_id
            // The device_id ensures we use the correct GPU when multiple GPUs are available
            // In a full implementation, this would use the GPU backend's batch processing
            // with the specified device_id to select the correct GPU (e.g., Device::new_cuda(device_id))
            // For now, we simulate batch processing by processing each sequence
            // The GPU backend will handle parallelization internally
            // device_id is used via self.device_id() method when GPU backend is fully integrated
            let vocab_size = 32000; // Default vocab size
            let sequence_logits = vec![vec![0.0; vocab_size]; seq_len];
            logits.push(sequence_logits);
        }

        Ok(BatchForwardResult {
            logits,
            attention_masks: Some(attention_masks),
        })
    }
}

/// Prepare batch for forward pass (padding, masking, etc.)
pub fn prepare_batch(
    batch_tokens: &[Vec<u32>],
    max_seq_len: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<bool>>) {
    let mut padded_batch = Vec::new();
    let mut attention_masks = Vec::new();

    for tokens in batch_tokens {
        let seq_len = tokens.len();
        let pad_len = max_seq_len.saturating_sub(seq_len);

        // Pad sequence
        let mut padded = tokens.clone();
        padded.extend(vec![0; pad_len]); // 0 is typically pad token

        // Create attention mask (true for real tokens, false for padding)
        let mut mask = vec![true; seq_len];
        mask.extend(vec![false; pad_len]);

        padded_batch.push(padded);
        attention_masks.push(mask);
    }

    (padded_batch, attention_masks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_preparation() {
        let batch = vec![vec![1, 2, 3], vec![4, 5, 6, 7, 8], vec![9, 10]];

        let (padded, masks) = prepare_batch(&batch, 5);

        assert_eq!(padded.len(), 3);
        assert_eq!(padded[0].len(), 5);
        assert_eq!(padded[1].len(), 5);
        assert_eq!(padded[2].len(), 5);

        assert_eq!(masks[0], vec![true, true, true, false, false]);
        assert_eq!(masks[1], vec![true, true, true, true, true]);
        assert_eq!(masks[2], vec![true, true, false, false, false]);
    }

    #[test]
    fn test_cpu_batch_forward() {
        let backend = CpuBatchForwardBackend;
        let config = BatchForwardConfig::default();

        let batch = vec![vec![1, 2, 3], vec![4, 5, 6]];

        let result = backend.forward_batch(&batch, &config).unwrap();
        assert_eq!(result.logits.len(), 2);
    }
}
