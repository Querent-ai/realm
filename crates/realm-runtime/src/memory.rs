//! Memory management with Memory64 support for large models

use wasm_chord_core::error::{Error, Result};

/// Memory allocator configuration
pub struct MemoryConfig {
    /// Use 64-bit memory addressing (for models > 4GB)
    pub use_memory64: bool,
    /// Maximum memory size in bytes (for 32-bit: 4GB, for 64-bit: much larger)
    pub max_memory_bytes: usize,
    /// Initial memory allocation in bytes
    pub initial_memory_bytes: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "memory64")]
            use_memory64: true,
            #[cfg(not(feature = "memory64"))]
            use_memory64: false,

            // 4GB limit for 32-bit, 16GB for 64-bit
            #[cfg(feature = "memory64")]
            max_memory_bytes: (16_u64 * 1024 * 1024 * 1024) as usize, // 16GB
            #[cfg(not(feature = "memory64"))]
            max_memory_bytes: (4_u64 * 1024 * 1024 * 1024) as usize, // 4GB

            initial_memory_bytes: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Memory allocator for model weights and activations
pub struct MemoryAllocator {
    config: MemoryConfig,
    allocated_bytes: usize,
}

impl MemoryAllocator {
    /// Create a new memory allocator
    pub fn new(config: MemoryConfig) -> Self {
        Self { config, allocated_bytes: 0 }
    }

    /// Check if allocation would exceed memory limit
    pub fn can_allocate(&self, size: usize) -> bool {
        self.allocated_bytes + size <= self.config.max_memory_bytes
    }

    /// Allocate memory for a buffer
    pub fn allocate<T>(&mut self, count: usize) -> Result<Vec<T>>
    where
        T: Default + Clone,
    {
        let size_bytes = count * std::mem::size_of::<T>();

        if !self.can_allocate(size_bytes) {
            return Err(Error::AllocationFailed(format!(
                "Cannot allocate {} bytes (current: {}, max: {}, memory64: {})",
                size_bytes,
                self.allocated_bytes,
                self.config.max_memory_bytes,
                self.config.use_memory64
            )));
        }

        let buffer = vec![T::default(); count];
        self.allocated_bytes += size_bytes;

        Ok(buffer)
    }

    /// Deallocate memory
    pub fn deallocate(&mut self, size_bytes: usize) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(size_bytes);
    }

    /// Get current allocated memory in bytes
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Get maximum memory in bytes
    pub fn max_bytes(&self) -> usize {
        self.config.max_memory_bytes
    }

    /// Check if Memory64 is enabled
    pub fn is_memory64_enabled(&self) -> bool {
        self.config.use_memory64
    }

    /// Get memory usage percentage
    pub fn usage_percent(&self) -> f32 {
        (self.allocated_bytes as f32 / self.config.max_memory_bytes as f32) * 100.0
    }

    /// Reset allocator (for testing)
    pub fn reset(&mut self) {
        self.allocated_bytes = 0;
    }
}

/// Estimate memory required for a model
pub fn estimate_model_memory(
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    intermediate_size: usize,
) -> usize {
    let bytes_per_param = std::mem::size_of::<f32>();

    // Token embeddings
    let embedding_bytes = vocab_size * hidden_size * bytes_per_param;

    // Per-layer weights
    let attention_bytes = 4 * hidden_size * hidden_size * bytes_per_param; // Q, K, V, O
    let ffn_bytes =
        (hidden_size * intermediate_size + intermediate_size * hidden_size) * bytes_per_param;
    let layer_bytes = (attention_bytes + ffn_bytes) * num_layers;

    // Output head (often shared with embeddings, but count separately for safety)
    let output_bytes = vocab_size * hidden_size * bytes_per_param;

    embedding_bytes + layer_bytes + output_bytes
}

/// Check if a model requires Memory64 based on size
pub fn requires_memory64(model_size_bytes: usize) -> bool {
    // If model is larger than 3GB (leaving some headroom), recommend Memory64
    model_size_bytes > 3 * 1024 * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_allocator_basic() -> Result<()> {
        let mut allocator = MemoryAllocator::new(MemoryConfig::default());

        // Allocate 1000 f32 values
        let buffer = allocator.allocate::<f32>(1000)?;
        assert_eq!(buffer.len(), 1000);
        assert_eq!(allocator.allocated_bytes(), 1000 * 4);

        Ok(())
    }

    #[test]
    fn test_memory_allocator_limit() {
        let config =
            MemoryConfig { use_memory64: false, max_memory_bytes: 1000, initial_memory_bytes: 100 };
        let mut allocator = MemoryAllocator::new(config);

        // Try to allocate more than limit
        let result = allocator.allocate::<f32>(1000); // 1000 * 4 = 4000 bytes > 1000
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_usage_percent() -> Result<()> {
        let config =
            MemoryConfig { use_memory64: false, max_memory_bytes: 1000, initial_memory_bytes: 100 };
        let mut allocator = MemoryAllocator::new(config);

        // Allocate 25% of memory
        allocator.allocate::<u8>(250)?;
        assert!((allocator.usage_percent() - 25.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_model_memory_estimation() {
        // Small model (TinyLlama-like)
        let mem = estimate_model_memory(
            32000, // vocab_size
            2048,  // hidden_size
            22,    // num_layers
            5632,  // intermediate_size
        );

        // TinyLlama 1.1B has ~1.1B parameters, so ~4.4GB in FP32
        assert!(mem > 1_000_000_000); // > 1GB
        assert!(mem < 5_000_000_000); // < 5GB
    }

    #[test]
    fn test_requires_memory64() {
        // Small model doesn't need Memory64
        assert!(!requires_memory64(1024 * 1024 * 1024)); // 1GB

        // Large model needs Memory64
        assert!(requires_memory64(4 * 1024 * 1024 * 1024)); // 4GB
    }

    #[test]
    #[cfg(feature = "memory64")]
    fn test_memory64_enabled() {
        let allocator = MemoryAllocator::new(MemoryConfig::default());
        assert!(allocator.is_memory64_enabled());
        assert_eq!(allocator.max_bytes(), 16 * 1024 * 1024 * 1024);
    }

    #[test]
    #[cfg(not(feature = "memory64"))]
    fn test_memory64_disabled() {
        let allocator = MemoryAllocator::new(MemoryConfig::default());
        assert!(!allocator.is_memory64_enabled());
        assert_eq!(allocator.max_bytes(), 4 * 1024 * 1024 * 1024);
    }
}
