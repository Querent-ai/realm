//! Realm Runtime - Host runtime with Memory64 and Candle acceleration
//!
//! This crate provides the host-side runtime that:
//! - Manages Memory64 for large models (>4GB)
//! - Provides Candle GPU/CPU acceleration
//! - Exports host functions for WASM to call
//! - Handles model loading and caching

pub mod host_functions;
pub mod memory64;
pub mod memory64_host;

pub use host_functions::*;
pub use memory64_host::*;

/// Runtime version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
