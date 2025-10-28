//! WASM-side FFI bindings for Memory64 host functions
//!
//! This module provides safe Rust wrappers around the host functions
//! that enable WASM modules to access Memory64 storage managed by the host.
//!
//! # Architecture
//!
//! The Memory64 bridge works as follows:
//! 1. Host manages Memory64 instances (>4GB)
//! 2. Host provides these functions for WASM to call
//! 3. WASM calls these functions to load data into its own memory (<4GB)
//! 4. WASM processes data in its standard memory
//!
//! # Feature Flag
//!
//! These bindings are only available when the `memory64` feature is enabled.
//! When disabled, the safe wrappers return errors indicating unavailability.

use anyhow::{anyhow, Result};

/// Import Memory64 host functions
///
/// These are provided by the host runtime (Wasmtime/Wasmer) and allow
/// WASM modules to access data stored in Memory64 instances.
#[cfg(feature = "memory64-wasm")]
#[link(wasm_import_module = "env")]
extern "C" {
    /// Load a layer's weights from Memory64 into WASM memory
    ///
    /// # Parameters
    /// - `layer_id`: The layer ID to load
    /// - `wasm_ptr`: Pointer in WASM memory to write to
    /// - `max_size`: Maximum size of the buffer
    ///
    /// # Returns
    /// - Positive: Number of bytes written
    /// - Negative: Error code
    ///   - -1: Memory64 not enabled or lock failed
    ///   - -2: Layer not found
    ///   - -3: Buffer too small
    ///   - -4: Failed to read from Memory64
    ///   - -5: No WASM memory export
    ///   - -6: Failed to write to WASM memory
    fn memory64_load_layer(layer_id: u32, wasm_ptr: u32, max_size: u32) -> i32;

    /// Read data from Memory64 at a specific offset
    ///
    /// # Parameters
    /// - `offset`: Offset in Memory64 to read from (can be >4GB)
    /// - `wasm_ptr`: Pointer in WASM memory to write to
    /// - `size`: Number of bytes to read
    ///
    /// # Returns
    /// - Positive: Number of bytes read
    /// - Negative: Error code
    ///   - -1: Memory64 not enabled or lock failed
    ///   - -2: Failed to read from Memory64
    ///   - -3: No WASM memory export
    ///   - -4: Failed to write to WASM memory
    fn memory64_read(offset: u64, wasm_ptr: u32, size: u32) -> i32;

    /// Check if Memory64 is enabled
    ///
    /// # Returns
    /// - 1: Memory64 is enabled
    /// - 0: Memory64 is disabled or unavailable
    fn memory64_is_enabled() -> i32;

    /// Get Memory64 statistics
    ///
    /// # Returns
    /// - Number of reads performed (for monitoring)
    /// - -1: Error accessing statistics
    fn memory64_stats() -> i64;
}

/// Safe wrapper for loading layer weights from Memory64
///
/// # Arguments
/// - `layer_id`: The layer ID to load
/// - `buffer`: Buffer to write the layer weights into
///
/// # Returns
/// - `Ok(usize)`: Number of bytes written to buffer
/// - `Err`: Error with description
///
/// # Example
/// ```ignore
/// let mut weights = vec![0u8; 200_000_000]; // 200MB layer
/// let bytes_read = load_layer(15, &mut weights)?;
/// println!("Loaded layer 15: {} bytes", bytes_read);
/// ```
pub fn load_layer(layer_id: u32, buffer: &mut [u8]) -> Result<usize> {
    #[cfg(feature = "memory64-wasm")]
    {
        unsafe {
            let result =
                memory64_load_layer(layer_id, buffer.as_mut_ptr() as u32, buffer.len() as u32);

            if result < 0 {
                return Err(match result {
                    -1 => anyhow!("Memory64 not enabled or lock failed"),
                    -2 => anyhow!("Layer {} not found", layer_id),
                    -3 => anyhow!("Buffer too small for layer {}", layer_id),
                    -4 => anyhow!("Failed to read layer {} from Memory64", layer_id),
                    -5 => anyhow!("No WASM memory export available"),
                    -6 => anyhow!("Failed to write layer {} to WASM memory", layer_id),
                    _ => anyhow!("Unknown error loading layer {}: code {}", layer_id, result),
                });
            }

            Ok(result as usize)
        }
    }

    #[cfg(not(feature = "memory64-wasm"))]
    {
        let _ = (layer_id, buffer);
        Err(anyhow!(
            "Memory64 feature not enabled. Rebuild with --features memory64-wasm"
        ))
    }
}

/// Safe wrapper for reading data from Memory64 at a specific offset
///
/// # Arguments
/// - `offset`: Offset in Memory64 to read from (can exceed 4GB)
/// - `buffer`: Buffer to read data into
///
/// # Returns
/// - `Ok(usize)`: Number of bytes read
/// - `Err`: Error with description
///
/// # Example
/// ```ignore
/// let mut embeddings = vec![0u8; 1_000_000]; // 1MB embeddings
/// let bytes_read = read_memory64(0, &mut embeddings)?;
/// println!("Read embeddings: {} bytes", bytes_read);
/// ```
pub fn read_memory64(offset: u64, buffer: &mut [u8]) -> Result<usize> {
    #[cfg(feature = "memory64-wasm")]
    {
        unsafe {
            let result = memory64_read(offset, buffer.as_mut_ptr() as u32, buffer.len() as u32);

            if result < 0 {
                return Err(match result {
                    -1 => anyhow!("Memory64 not enabled or lock failed"),
                    -2 => anyhow!("Failed to read from Memory64 at offset {}", offset),
                    -3 => anyhow!("No WASM memory export available"),
                    -4 => anyhow!("Failed to write to WASM memory"),
                    _ => anyhow!(
                        "Unknown error reading from offset {}: code {}",
                        offset,
                        result
                    ),
                });
            }

            Ok(result as usize)
        }
    }

    #[cfg(not(feature = "memory64-wasm"))]
    {
        let _ = (offset, buffer);
        Err(anyhow!(
            "Memory64 feature not enabled. Rebuild with --features memory64-wasm"
        ))
    }
}

/// Check if Memory64 is enabled and available
///
/// # Returns
/// - `true`: Memory64 is enabled
/// - `false`: Memory64 is disabled or unavailable
///
/// # Example
/// ```ignore
/// if is_memory64_enabled() {
///     println!("Using Memory64 for large model support");
///     load_large_model()?;
/// } else {
///     println!("Falling back to standard memory");
///     load_small_model()?;
/// }
/// ```
pub fn is_memory64_enabled() -> bool {
    #[cfg(feature = "memory64-wasm")]
    {
        unsafe { memory64_is_enabled() == 1 }
    }

    #[cfg(not(feature = "memory64-wasm"))]
    {
        false
    }
}

/// Get Memory64 statistics for monitoring
///
/// # Returns
/// - `Ok(u64)`: Number of read operations performed
/// - `Err`: Error accessing statistics
///
/// # Example
/// ```ignore
/// let reads = get_memory64_stats()?;
/// println!("Memory64 read operations: {}", reads);
/// ```
pub fn get_memory64_stats() -> Result<u64> {
    #[cfg(feature = "memory64-wasm")]
    {
        unsafe {
            let result = memory64_stats();
            if result < 0 {
                return Err(anyhow!("Failed to get Memory64 statistics"));
            }
            Ok(result as u64)
        }
    }

    #[cfg(not(feature = "memory64-wasm"))]
    {
        Err(anyhow!(
            "Memory64 feature not enabled. Rebuild with --features memory64-wasm"
        ))
    }
}

/// Memory64 layer loader for model inference
///
/// This is a higher-level abstraction that handles loading layers
/// with proper error handling and buffer management.
pub struct Memory64LayerLoader {
    /// Whether Memory64 is actually available at runtime
    enabled: bool,
}

impl Memory64LayerLoader {
    /// Create a new layer loader
    ///
    /// Checks at runtime whether Memory64 is available.
    pub fn new() -> Self {
        Self {
            enabled: is_memory64_enabled(),
        }
    }

    /// Check if Memory64 is available
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Load a layer by ID
    ///
    /// # Arguments
    /// - `layer_id`: The layer to load (0-based)
    /// - `buffer`: Pre-allocated buffer for layer weights
    ///
    /// # Returns
    /// - `Ok(usize)`: Bytes loaded
    /// - `Err`: If Memory64 not available or load failed
    pub fn load_layer(&self, layer_id: u32, buffer: &mut [u8]) -> Result<usize> {
        if !self.enabled {
            return Err(anyhow!(
                "Memory64 not available - cannot load layer {}",
                layer_id
            ));
        }

        load_layer(layer_id, buffer)
    }

    /// Read data from a specific offset
    ///
    /// Useful for loading embeddings, lm_head, or other non-layer data.
    ///
    /// # Arguments
    /// - `offset`: Offset in Memory64 (can be >4GB)
    /// - `buffer`: Buffer to read into
    pub fn read_at_offset(&self, offset: u64, buffer: &mut [u8]) -> Result<usize> {
        if !self.enabled {
            return Err(anyhow!(
                "Memory64 not available - cannot read at offset {}",
                offset
            ));
        }

        read_memory64(offset, buffer)
    }

    /// Get statistics
    pub fn stats(&self) -> Result<u64> {
        if !self.enabled {
            return Ok(0);
        }

        get_memory64_stats()
    }
}

impl Default for Memory64LayerLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_creation() {
        let loader = Memory64LayerLoader::new();
        // Will return false in tests since host functions not available
        assert!(!loader.is_enabled());
    }

    #[test]
    fn test_loader_without_memory64() {
        let loader = Memory64LayerLoader::new();
        let mut buffer = vec![0u8; 1024];

        // Should fail gracefully when Memory64 not available
        let result = loader.load_layer(0, &mut buffer);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(not(feature = "memory64-wasm"))]
    fn test_feature_flag_disabled() {
        // When feature is disabled, functions should return errors
        assert!(!is_memory64_enabled());

        let mut buffer = vec![0u8; 1024];
        assert!(load_layer(0, &mut buffer).is_err());
        assert!(read_memory64(0, &mut buffer).is_err());
        assert!(get_memory64_stats().is_err());
    }
}
