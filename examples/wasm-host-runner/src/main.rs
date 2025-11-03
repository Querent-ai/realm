//! WASM Host Runner - Test harness for WASM module with host-side storage
//!
//! This program:
//! 1. Loads the realm-wasm WASM module
//! 2. Sets up Memory64Runtime to provide host functions
//! 3. Calls loadModel() and generate() to test end-to-end
//!
//! Architecture:
//! ```
//! ┌──────────────────┐
//! │  WASM Module     │  loadModel() → realm_store_model()
//! │  (realm-wasm)    │                         ↓
//! └────────┬─────────┘                         ↓
//!          │                          ┌────────▼────────┐
//!          │  Host Functions          │  HOST Storage   │
//!          └─────────────────────────►│  (637MB in RAM) │
//!                                     └─────────────────┘
//! ```

use anyhow::{Context, Result};
use realm_runtime::memory64_host::Memory64Runtime;
use realm_runtime::memory64_host::MemoryLayout;
use std::path::PathBuf;
use tracing::{info, warn};
use wasmtime::*;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🚀 WASM Host Runner - Testing Host-Side Storage");

    // Get model path from args or use default
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf".to_string());

    let model_path = PathBuf::from(model_path);
    if !model_path.exists() {
        anyhow::bail!("Model file not found: {}", model_path.display());
    }

    info!("📁 Loading model from: {}", model_path.display());

    // Create Wasmtime engine and store
    let mut config = Config::new();
    config.wasm_memory64(true); // Enable Memory64 support
    config.wasm_bulk_memory(true);
    config.wasm_simd(true);

    let engine = Engine::new(&config)?;
    let mut store = Store::new(&engine, ());

    info!("🔧 Created Wasmtime engine");

    // Create Memory64 runtime with host-side storage
    let layout =
        MemoryLayout::single(8, "model_storage").context("Failed to create memory layout")?;
    let memory64_runtime = Memory64Runtime::new(layout, true);

    info!("💾 Created Memory64 runtime for host-side storage");

    // Create linker and add host functions
    let mut linker = Linker::new(&engine);

    // Add Memory64 host functions (including model storage functions)
    memory64_runtime
        .add_to_linker(&mut linker)
        .context("Failed to add host functions to linker")?;

    info!("🔗 Added host functions to linker");

    // Load WASM module
    let wasm_path = "crates/realm-wasm/pkg/realm_wasm_bg.wasm";
    if !PathBuf::from(wasm_path).exists() {
        anyhow::bail!(
            "WASM module not found at {}. Run 'cd crates/realm-wasm && wasm-pack build --target nodejs --release'",
            wasm_path
        );
    }

    info!("📦 Loading WASM module from: {}", wasm_path);

    let module = Module::from_file(&engine, wasm_path).context("Failed to load WASM module")?;

    info!("✅ WASM module loaded");

    // Instantiate the module
    let instance = linker
        .instantiate(&mut store, &module)
        .context("Failed to instantiate WASM module")?;

    info!("🎯 WASM module instantiated with host functions");

    // Get exports (wasm-bindgen generates specific function names)
    let memory = instance
        .get_memory(&mut store, "memory")
        .context("Failed to get WASM memory export")?;

    info!("📊 WASM memory size: {} bytes", memory.data_size(&store));

    // Get the __wbindgen_malloc function (used by wasm-bindgen)
    let wbindgen_malloc = instance
        .get_typed_func::<u32, u32>(&mut store, "__wbindgen_malloc")
        .context("Failed to get __wbindgen_malloc export")?;

    info!("✅ Found __wbindgen_malloc function");

    // Read model file
    let model_bytes = std::fs::read(&model_path)
        .with_context(|| format!("Failed to read model file: {}", model_path.display()))?;

    info!("📁 Read model file: {} bytes", model_bytes.len());

    // Allocate memory in WASM for model bytes
    info!(
        "💾 Allocating {} bytes in WASM memory...",
        model_bytes.len()
    );
    let model_ptr = wbindgen_malloc
        .call(&mut store, model_bytes.len() as u32)
        .context("Failed to allocate WASM memory")?;

    info!("✅ Allocated at WASM ptr: 0x{:x}", model_ptr);

    // Write model bytes to WASM memory
    memory
        .write(&mut store, model_ptr as usize, &model_bytes)
        .context("Failed to write model bytes to WASM memory")?;

    info!("✅ Wrote model bytes to WASM memory");

    // Get loadModel function (wasm-bindgen generates: realm_Realm_load_model)
    let load_model_func = instance
        .get_func(&mut store, "realm_Realm_load_model")
        .or_else(|| instance.get_func(&mut store, "loadModel"))
        .or_else(|| {
            warn!("Listing all exports:");
            for export in instance.exports(&mut store) {
                warn!("  - {}", export.name());
            }
            None
        })
        .context("Failed to find loadModel function in WASM exports")?;

    info!("✅ Found loadModel function");

    // Call loadModel with the model bytes
    info!("🚀 Calling loadModel()...");

    let load_model_typed = load_model_func
        .typed::<(u32, u32, u32), u32>(&store)
        .context("Failed to type loadModel function")?;

    let result = load_model_typed
        .call(
            &mut store,
            (0, model_ptr, model_bytes.len() as u32), // (this_ptr, model_ptr, model_len)
        )
        .context("Failed to call loadModel")?;

    if result == 0 {
        info!("✅ loadModel() succeeded!");
    } else {
        anyhow::bail!("loadModel() failed with code: {}", result);
    }

    info!("🎉 SUCCESS! Model loaded with host-side storage");
    info!("📊 WASM memory usage: {} bytes", memory.data_size(&store));

    Ok(())
}
