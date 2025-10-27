//! Memory64 Support for Large Models (>4GB)
//!
//! This module provides utilities and re-exports for Memory64 functionality.
//!
//! ## Architecture Overview
//!
//! Memory64 support is split into two parts:
//!
//! 1. **Host-side Runtime** (`memory64_host.rs`):
//!    - Manages Memory64 instances using Wasmtime API
//!    - Stores model weights (can be >4GB)
//!    - Provides host functions for WASM to access data
//!    - Production-hardened with overflow checks and pointer validation
//!
//! 2. **WASM-side FFI** (`memory64_ffi.rs`):
//!    - Safe Rust wrappers for host functions
//!    - Called from WASM module (wasm32)
//!    - Loads layer data from host into WASM memory
//!
//! ## When to Use Memory64
//!
//! Use Memory64 when your model size exceeds 3-4GB:
//! - 7B models (Q4_K_M): ~4GB → Need Memory64
//! - 13B models (Q4_K_M): ~8GB → Need Memory64
//! - 30B+ models: Multi-memory layout required
//!
//! ## Usage
//!
//! ### Host-side (Rust application using Wasmtime):
//! ```rust,ignore
//! use realm_runtime::memory64_host::{Memory64Runtime, MemoryLayout};
//!
//! // Create 8GB memory for 7B model
//! let layout = MemoryLayout::single(8, "model_storage")?;
//! let runtime = Memory64Runtime::new(layout, true);
//!
//! // Add host functions to linker
//! runtime.add_to_linker(&mut linker)?;
//!
//! // Initialize and load model
//! runtime.initialize(&mut store)?;
//! runtime.write_model_data(&mut store, 0, &model_weights)?;
//! ```
//!
//! ### WASM-side (your inference code):
//! ```rust,ignore
//! use realm_runtime::memory64_ffi::Memory64LayerLoader;
//!
//! let loader = Memory64LayerLoader::new();
//! if loader.is_enabled() {
//!     let mut layer_weights = vec![0u8; 200_000_000];
//!     loader.load_layer(15, &mut layer_weights)?;
//!     // Process layer...
//! }
//! ```

use realm_core::error::{Error, Result};

// Re-export the production-hardened host runtime
#[cfg(feature = "memory64")]
pub use crate::memory64_host::{
    LayerInfo, Memory64Runtime, Memory64State, MemoryLayout, MemoryRegion, MemoryStats,
};

// Re-export WASM FFI bindings
#[cfg(all(feature = "memory64", target_arch = "wasm32"))]
pub use crate::memory64_ffi::{
    get_memory64_stats, is_memory64_enabled, load_layer, read_memory64, Memory64LayerLoader,
};

/// Check if the current environment supports Memory64
///
/// This checks if the `memory64` feature flags are enabled at compile time.
/// At runtime, the host can additionally check browser/engine support.
pub fn supports_memory64() -> bool {
    cfg!(feature = "memory64")
}

/// Get the maximum memory size based on Memory64 support
///
/// Returns:
/// - 16GB if Memory64 is enabled
/// - 4GB if Memory64 is not enabled
pub fn get_max_memory_size() -> u64 {
    if supports_memory64() {
        16 * 1024 * 1024 * 1024 // 16GB with Memory64
    } else {
        4 * 1024 * 1024 * 1024 // 4GB without Memory64
    }
}

/// Simple allocator for tracking Memory64 usage (WASM-side)
///
/// This is a lightweight tracker for WASM-side code to estimate memory usage.
/// The actual Memory64 storage is managed by the host runtime.
pub struct WasmMemory64Allocator {
    max_bytes: u64,
    allocated_bytes: u64,
}

impl WasmMemory64Allocator {
    /// Create a new Memory64 allocator tracker
    pub fn new(_initial_bytes: u64, max_bytes: u64) -> Result<Self> {
        #[cfg(feature = "memory64")]
        {
            Ok(Self {
                max_bytes,
                allocated_bytes: _initial_bytes,
            })
        }

        #[cfg(not(feature = "memory64"))]
        {
            let _ = (_initial_bytes, max_bytes);
            Err(Error::AllocationFailed(
                "Memory64 feature not enabled. Enable with --features memory64".to_string(),
            ))
        }
    }

    /// Check if we can allocate the specified number of bytes
    pub fn can_allocate(&self, bytes: u64) -> bool {
        self.allocated_bytes
            .checked_add(bytes)
            .map(|total| total <= self.max_bytes)
            .unwrap_or(false)
    }

    /// Allocate memory for a buffer (tracking only)
    pub fn allocate<T>(&mut self, count: usize) -> Result<Vec<T>>
    where
        T: Default + Clone,
    {
        let size_bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::AllocationFailed("Size calculation overflow".to_string()))?;

        if !self.can_allocate(size_bytes as u64) {
            return Err(Error::AllocationFailed(format!(
                "Cannot allocate {} bytes (current: {}, max: {})",
                size_bytes, self.allocated_bytes, self.max_bytes
            )));
        }

        let buffer = vec![T::default(); count];
        self.allocated_bytes += size_bytes as u64;
        Ok(buffer)
    }

    /// Get current allocated memory in bytes
    pub fn size_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Get maximum memory in bytes
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Get memory usage percentage
    pub fn usage_percent(&self) -> f32 {
        (self.allocated_bytes as f32 / self.max_bytes as f32) * 100.0
    }

    /// Reset allocator (for testing)
    pub fn reset(&mut self) {
        self.allocated_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory64_support() {
        let supports = supports_memory64();
        let max_size = get_max_memory_size();

        if supports {
            assert_eq!(max_size, 16 * 1024 * 1024 * 1024);
        } else {
            assert_eq!(max_size, 4 * 1024 * 1024 * 1024);
        }
    }

    #[test]
    #[cfg(feature = "memory64")]
    fn test_allocator_basic() {
        let allocator = WasmMemory64Allocator::new(0, 1024 * 1024).unwrap();
        assert!(allocator.can_allocate(1000));
        assert_eq!(allocator.size_bytes(), 0);
    }

    #[test]
    #[cfg(feature = "memory64")]
    fn test_allocator_overflow_check() {
        let mut allocator = WasmMemory64Allocator::new(0, 1000).unwrap();

        // This should fail - too large
        let result = allocator.allocate::<u8>(2000);
        assert!(result.is_err());
    }
}
