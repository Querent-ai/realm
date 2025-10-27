//! KV cache implementation for efficient attention computation

use realm_core::error::Result;

/// KV cache for a single layer - Microsoft AICI inspired implementation
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Cached keys [max_seq_len, num_kv_heads, head_dim]
    pub keys: Vec<f32>,
    /// Cached values [max_seq_len, num_kv_heads, head_dim]
    pub values: Vec<f32>,
    /// Current sequence length (number of tokens cached)
    pub current_seq_len: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

#[allow(dead_code)]
impl KVCache {
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let max_size = max_seq_len * num_kv_heads * head_dim;
        Self {
            keys: vec![0.0; max_size],
            values: vec![0.0; max_size],
            current_seq_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
        }
    }

    pub fn clear(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.current_seq_len = 0;
    }

    /// Append new K/V vectors to the cache
    /// Returns the current cached K/V tensors for attention computation
    pub fn append(&mut self, keys: &[f32], values: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let seq_len = keys.len() / (self.num_kv_heads * self.head_dim);

        // Check if we have space
        if self.current_seq_len + seq_len > self.max_seq_len {
            return Err(realm_core::error::Error::ParseError(format!(
                "KV cache overflow: {} + {} > {}",
                self.current_seq_len, seq_len, self.max_seq_len
            )));
        }

        // Append to cache
        let start_idx = self.current_seq_len * self.num_kv_heads * self.head_dim;
        let end_idx = start_idx + keys.len();

        self.keys[start_idx..end_idx].copy_from_slice(keys);
        self.values[start_idx..end_idx].copy_from_slice(values);

        self.current_seq_len += seq_len;

        // Return current cached data for attention computation
        let cached_len = self.current_seq_len * self.num_kv_heads * self.head_dim;
        Ok((
            self.keys[..cached_len].to_vec(),
            self.values[..cached_len].to_vec(),
        ))
    }

    /// Get current cached data without appending
    pub fn current_data(&self) -> (Vec<f32>, Vec<f32>) {
        let cached_len = self.current_seq_len * self.num_kv_heads * self.head_dim;
        (
            self.keys[..cached_len].to_vec(),
            self.values[..cached_len].to_vec(),
        )
    }
}
