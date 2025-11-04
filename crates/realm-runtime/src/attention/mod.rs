// Attention module for wasm-chord
//
// Provides different attention implementations:
// - Standard: Traditional O(N²) attention
// - Flash: IO-aware Flash Attention (3-4x faster, O(N) memory)

pub mod config;
pub mod flash;
pub mod standard;

#[cfg(feature = "cuda")]
pub mod cuda_wrapper;
#[cfg(feature = "metal")]
pub mod metal_wrapper;

use realm_core::error::Result;

/// Attention mechanism trait
///
/// All attention implementations must implement this trait to ensure
/// compatibility across different backends (CPU, CUDA, Metal, WebGPU).
pub trait Attention: Send + Sync {
    /// Compute attention output
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, num_heads, seq_len_q, head_dim]
    /// * `k` - Key tensor [batch, num_heads, seq_len_k, head_dim]
    /// * `v` - Value tensor [batch, num_heads, seq_len_k, head_dim]
    /// * `mask` - Optional attention mask [batch, 1, seq_len_q, seq_len_k]
    ///
    /// # Returns
    /// Output tensor [batch, num_heads, seq_len_q, head_dim]
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
        batch_size: usize,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>>;

    /// Get the name of this attention implementation
    fn name(&self) -> &str;

    /// Check if this implementation is available on the current hardware
    fn is_available(&self) -> bool;

    /// Get estimated memory usage in bytes
    fn estimated_memory(&self, seq_len: usize, head_dim: usize, num_heads: usize) -> usize;
}

/// Attention backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionBackend {
    /// Standard O(N²) attention (always available)
    Standard,

    /// Flash Attention (requires GPU or optimized CPU)
    Flash,

    /// Automatically select best available backend
    Auto,
}

/// Factory function to create the best available attention implementation
pub fn create_attention(backend: AttentionBackend) -> Box<dyn Attention> {
    match backend {
        AttentionBackend::Standard => Box::new(standard::StandardAttention::new()),
        AttentionBackend::Flash => {
            // Try to create Flash Attention, fall back to standard if not available
            if let Some(flash) = flash::FlashAttention::try_new() {
                Box::new(flash)
            } else {
                // eprintln!("⚠️  Flash Attention not available, using standard attention");
                Box::new(standard::StandardAttention::new())
            }
        }
        AttentionBackend::Auto => {
            // Prefer Flash Attention if available
            if let Some(flash) = flash::FlashAttention::try_new() {
                Box::new(flash)
            } else {
                Box::new(standard::StandardAttention::new())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_factory() {
        // Should always be able to create standard attention
        let standard = create_attention(AttentionBackend::Standard);
        assert_eq!(standard.name(), "StandardAttention");
        assert!(standard.is_available());
    }

    #[test]
    fn test_auto_backend_selection() {
        // Should select the best available backend
        let auto = create_attention(AttentionBackend::Auto);
        assert!(auto.is_available());

        // Name should be either Standard or Flash
        let name = auto.name();
        assert!(name == "StandardAttention" || name == "FlashAttention");
    }
}
