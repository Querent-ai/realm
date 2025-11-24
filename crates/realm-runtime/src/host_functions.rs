//! Host functions exported to WASM
//!
//! These functions are called from WASM to perform GPU operations,
//! Memory64 access, and other native functionality.
//!
//! This module provides a higher-level API wrapper around the production-grade
//! Memory64Runtime implementation in memory64_host.rs.

use anyhow::Result;
use std::sync::Arc;
use tracing::debug;
use wasmtime::{Linker, Store};

use crate::memory64_host::{Memory64Runtime, MemoryLayout};

/// Host function context shared across all WASM instances
///
/// This wraps the Memory64Runtime and provides a simplified API for
/// creating and managing host functions in multi-tenant deployments.
pub struct HostContext {
    /// Memory64 runtime (contains CPU/GPU backends)
    runtime: Arc<Memory64Runtime>,
}

impl HostContext {
    /// Create a new host context with default 8GB memory layout
    pub fn new() -> Self {
        let layout = MemoryLayout::single(8, "default_model_storage")
            .expect("Failed to create default memory layout");
        let runtime = Memory64Runtime::new(layout, true);

        Self {
            runtime: Arc::new(runtime),
        }
    }

    /// Create with custom memory layout
    pub fn with_layout(layout: MemoryLayout) -> Self {
        let runtime = Memory64Runtime::new(layout, true);
        Self {
            runtime: Arc::new(runtime),
        }
    }

    /// Initialize the runtime (must be called before use)
    pub fn initialize(&self, store: &mut Store<()>) -> Result<()> {
        debug!("Initializing host context");
        self.runtime.initialize(store)
    }

    /// Add all host functions to a Wasmtime linker
    pub fn add_to_linker(&self, linker: &mut Linker<()>) -> Result<()> {
        debug!("Adding host functions to linker");
        self.runtime.add_to_linker(linker)
    }

    /// Get the underlying Memory64Runtime
    pub fn runtime(&self) -> &Memory64Runtime {
        &self.runtime
    }

    /// Set streaming callback for token generation
    /// This enables real token-by-token streaming via realm_stream_token host function
    /// Uses blocking channel since host functions are called from blocking context
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_stream_callback(&self, sender: std::sync::mpsc::Sender<String>) {
        self.runtime.set_stream_callback(sender);
    }

    /// Clear streaming callback
    #[cfg(not(target_arch = "wasm32"))]
    pub fn clear_stream_callback(&self) {
        self.runtime.clear_stream_callback();
    }
}

impl Default for HostContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Host Functions Available from WASM
///
/// The following host functions are automatically added to the Wasmtime linker
/// when you call `HostContext::add_to_linker()`. These functions are implemented
/// in `memory64_host.rs` and provide production-grade functionality with full
/// error handling, bounds checking, and performance optimizations.
///
/// # Available Host Functions
///
/// ## Memory64 Operations
///
/// - `memory64_load_layer(layer_id: u32, wasm_ptr: u32, max_size: u32) -> i32`
///   Load a model layer from Memory64 storage into WASM linear memory.
///   Returns the number of bytes loaded, or negative error code.
///
/// - `memory64_read(offset: u64, wasm_ptr: u32, size: u32) -> i32`
///   Read arbitrary data from Memory64 into WASM memory.
///   Returns the number of bytes read, or negative error code.
///
/// - `memory64_is_enabled() -> i32`
///   Check if Memory64 is enabled. Returns 1 if enabled, 0 otherwise.
///
/// - `memory64_stats() -> i64`
///   Get memory statistics (currently returns read count).
///
/// ## Candle GPU Acceleration (CPU backend only for now)
///
/// - `candle_matmul(a_ptr: u32, b_ptr: u32, result_ptr: u32, m: u32, k: u32, n: u32) -> i32`
///   Matrix multiplication: C = A @ B where A is [M, K] and B is [K, N].
///   Returns the number of result elements, or negative error code.
///
/// - `candle_matmul_transposed(a_ptr: u32, b_ptr: u32, result_ptr: u32, m: u32, k: u32, n: u32) -> i32`
///   Matrix multiplication with transposed B: C = A @ B^T.
///   Returns the number of result elements, or negative error code.
///
/// # Usage Example
///
/// ```rust,ignore
/// use realm_runtime::HostContext;
/// use wasmtime::{Config, Engine, Linker, Module, Store};
///
/// // Create host context
/// let host_context = HostContext::new();
///
/// // Create Wasmtime engine and linker
/// let mut config = Config::new();
/// config.wasm_bulk_memory(true);
/// let engine = Engine::new(&config)?;
/// let mut linker = Linker::new(&engine);
///
/// // Add all host functions
/// host_context.add_to_linker(&mut linker)?;
///
/// // Initialize Memory64
/// let mut store = Store::new(&engine, ());
/// host_context.initialize(&mut store)?;
///
/// // Load WASM module and instantiate
/// let module = Module::from_file(&engine, "model.wasm")?;
/// let instance = linker.instantiate(&mut store, &module)?;
/// ```
///
/// # Error Codes
///
/// All host functions return negative error codes on failure:
/// - `-1`: Feature not available/enabled
/// - `-2`: Layer/resource not found
/// - `-3`: Buffer too small / validation failed
/// - `-4`: Read/write operation failed
/// - `-5`: Memory export not available
/// - `-6`: Pointer overflow
/// - `-7`: Pointer out of bounds
/// - `-8`: Write to WASM memory failed
///
/// # Performance
///
/// - Memory64 operations use zero-copy when possible
/// - Matrix multiplications use Candle's optimized CPU backend (BLAS/MKL)
/// - GPU backend support coming soon (CUDA/Metal)
/// - All operations include comprehensive bounds checking
///
/// # Thread Safety
///
/// All host functions are thread-safe and can be called from multiple WASM
/// instances concurrently. Memory64 state uses `parking_lot::Mutex` for
/// fast, poison-free locking.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_context_creation() {
        let context = HostContext::new();
        // Context should be created successfully with default 8GB layout
        assert!(context.runtime().stats().is_ok());
    }

    #[test]
    #[cfg_attr(target_os = "windows", ignore)] // Skip on Windows CI due to memory limits (16GB allocation)
    fn test_host_context_with_custom_layout() {
        // Create custom 16GB layout
        let layout = MemoryLayout::single(16, "custom_storage").expect("Failed to create layout");
        let context = HostContext::with_layout(layout);
        assert!(context.runtime().stats().is_ok());
    }

    #[test]
    #[cfg_attr(target_os = "windows", ignore)] // Skip on Windows CI due to memory limits (8GB allocation via HostContext::new)
    fn test_host_context_initialize() {
        use wasmtime::{Config, Engine, Store};

        let context = HostContext::new();

        // Create Wasmtime store
        let mut config = Config::new();
        config.wasm_bulk_memory(true);
        let engine = Engine::new(&config).expect("Failed to create engine");
        let mut store = Store::new(&engine, ());

        // Initialize should succeed
        assert!(context.initialize(&mut store).is_ok());
    }

    #[test]
    fn test_host_context_streaming_callback() {
        let context = HostContext::new();

        // Create a test channel
        let (tx, rx) = std::sync::mpsc::channel();

        // Set callback
        context.set_stream_callback(tx);

        // Send a test token
        let callback_guard = context.runtime().stream_callback().lock();
        if let Some(ref sender) = *callback_guard {
            sender.send("test".to_string()).unwrap();
        }
        drop(callback_guard);

        // Receive token
        let token = rx.recv().unwrap();
        assert_eq!(token, "test");

        // Clear callback
        context.clear_stream_callback();

        // Verify callback is cleared
        let callback_guard = context.runtime().stream_callback().lock();
        assert!(callback_guard.is_none());
    }
}
