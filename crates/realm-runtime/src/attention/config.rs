// Attention configuration
//
// Controls block sizes, precision, and other parameters for attention computation

/// Configuration for Flash Attention
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size for Q dimension (typically 64-256)
    pub block_size_q: usize,

    /// Block size for K/V dimension (typically 64-256)
    pub block_size_kv: usize,

    /// Number of parallel splits for computation
    pub num_splits: usize,

    /// Use FP16 (half precision) for computation
    pub use_fp16: bool,

    /// Softmax scale factor (typically 1/sqrt(head_dim))
    pub softmax_scale: Option<f32>,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size_q: 128,   // Good balance for most GPUs
            block_size_kv: 128,  // Same as Q for simplicity
            num_splits: 1,       // No splitting by default
            use_fp16: true,      // FP16 is faster on modern GPUs
            softmax_scale: None, // Will compute as 1/sqrt(head_dim)
        }
    }
}

impl FlashAttentionConfig {
    /// Create config optimized for a specific GPU
    pub fn for_gpu(gpu_name: &str) -> Self {
        match gpu_name {
            // NVIDIA GPUs
            "A100" | "H100" => Self {
                block_size_q: 256,
                block_size_kv: 256,
                num_splits: 1,
                use_fp16: true,
                softmax_scale: None,
            },
            "RTX 4090" | "RTX 3090" => Self {
                block_size_q: 128,
                block_size_kv: 128,
                num_splits: 1,
                use_fp16: true,
                softmax_scale: None,
            },

            // Apple Silicon
            "M1" | "M2" | "M3" => Self {
                block_size_q: 128,
                block_size_kv: 128,
                num_splits: 1,
                use_fp16: true,
                softmax_scale: None,
            },

            // Default for unknown GPUs
            _ => Self::default(),
        }
    }

    /// Create config optimized for CPU
    pub fn for_cpu() -> Self {
        Self {
            block_size_q: 64, // Smaller blocks for CPU cache
            block_size_kv: 64,
            num_splits: 4,   // More parallelism on CPU
            use_fp16: false, // CPU typically better with FP32
            softmax_scale: None,
        }
    }

    /// Calculate SRAM usage for this configuration
    pub fn sram_usage_bytes(&self, head_dim: usize) -> usize {
        let element_size = if self.use_fp16 { 2 } else { 4 };

        // Q block + K block + V block + S block (scores) + temp buffers
        let q_size = self.block_size_q * head_dim * element_size;
        let k_size = self.block_size_kv * head_dim * element_size;
        let v_size = self.block_size_kv * head_dim * element_size;
        let s_size = self.block_size_q * self.block_size_kv * element_size;
        let temp_size = self.block_size_q * head_dim * 4; // Accumulators in FP32

        q_size + k_size + v_size + s_size + temp_size
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.block_size_q == 0 || self.block_size_kv == 0 {
            return Err("Block sizes must be non-zero".to_string());
        }

        if self.block_size_q > 512 || self.block_size_kv > 512 {
            return Err("Block sizes should not exceed 512 for typical GPUs".to_string());
        }

        if self.num_splits == 0 {
            return Err("num_splits must be at least 1".to_string());
        }

        Ok(())
    }

    /// Get softmax scale factor
    pub fn get_softmax_scale(&self, head_dim: usize) -> f32 {
        self.softmax_scale
            .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt())
    }
}

/// Configuration for standard attention
#[derive(Debug, Clone, Default)]
pub struct StandardAttentionConfig {
    /// Use FP16 for computation
    #[allow(dead_code)]
    pub use_fp16: bool,

    /// Softmax scale factor (typically 1/sqrt(head_dim))
    pub softmax_scale: Option<f32>,
}

impl StandardAttentionConfig {
    /// Get softmax scale factor
    pub fn get_softmax_scale(&self, head_dim: usize) -> f32 {
        self.softmax_scale
            .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attention_config_default() {
        let config = FlashAttentionConfig::default();
        assert_eq!(config.block_size_q, 128);
        assert_eq!(config.block_size_kv, 128);
        assert!(config.use_fp16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_flash_attention_config_for_gpu() {
        let config = FlashAttentionConfig::for_gpu("A100");
        assert_eq!(config.block_size_q, 256);
        assert!(config.validate().is_ok());

        let config = FlashAttentionConfig::for_gpu("RTX 3090");
        assert_eq!(config.block_size_q, 128);
    }

    #[test]
    fn test_sram_usage() {
        let config = FlashAttentionConfig::default();
        let head_dim = 128;
        let usage = config.sram_usage_bytes(head_dim);

        // Should fit in typical GPU SRAM (192 KB per SM on A100)
        assert!(usage < 200_000, "SRAM usage too high: {} bytes", usage);
    }

    #[test]
    fn test_softmax_scale() {
        let config = FlashAttentionConfig::default();
        let scale = config.get_softmax_scale(64);
        assert!((scale - 0.125).abs() < 1e-6); // 1/sqrt(64) = 1/8 = 0.125
    }
}
