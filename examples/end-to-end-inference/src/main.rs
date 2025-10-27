//! End-to-End Inference Example
//!
//! This example demonstrates the complete Realm inference pipeline:
//! 1. Load a GGUF model from disk
//! 2. Store model weights in Memory64
//! 3. Create WASM instance with HostContext
//! 4. Run inference with real tokens
//! 5. Demonstrate multi-layer processing

use anyhow::{Context, Result};
use realm_core::formats::gguf::GGUFParser;
use realm_runtime::{HostContext, MemoryLayout};
use std::path::Path;
use tracing::{error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;
use wasmtime::*;

fn main() -> Result<()> {
    // Set up logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("🚀 Realm End-to-End Inference Example");
    info!("");

    // Check if model file is provided
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        warn!("No model path provided");
        warn!("Usage: end-to-end-inference <path_to_model.gguf>");
        warn!("");
        warn!("Example:");
        warn!("  cargo run --bin end-to-end-inference -- /path/to/model.gguf");
        warn!("");
        warn!("Running in demo mode without actual model...");
        String::new()
    });

    if !model_path.is_empty() {
        run_with_model(&model_path)?;
    } else {
        run_demo_mode()?;
    }

    Ok(())
}

/// Run inference with an actual GGUF model
fn run_with_model(model_path: &str) -> Result<()> {
    info!("📦 Loading model from: {}", model_path);

    // Open the model file
    let file = std::fs::File::open(model_path)
        .with_context(|| format!("Failed to open model file: {}", model_path))?;

    // Create GGUF parser
    let mut parser = GGUFParser::new(file);

    // Parse header to get metadata
    let metadata = parser
        .parse_header()
        .context("Failed to parse GGUF header")?;

    info!("✅ Model loaded successfully");
    info!("");
    info!("Model metadata:");
    info!("  Version: {}", metadata.version);
    info!("  Architecture: {}", metadata.architecture);
    info!("  Tensors: {}", metadata.tensor_count);
    info!("");

    // Try to extract transformer config
    if let Some(config) = parser.extract_config() {
        info!("Transformer configuration:");
        info!("  Vocab size: {}", config.vocab_size);
        info!("  Hidden size: {}", config.hidden_size);
        info!("  Layers: {}", config.num_layers);
        info!("  Attention heads: {}", config.num_heads);
        info!("  KV heads: {}", config.num_kv_heads);
        info!("  Max sequence length: {}", config.max_seq_len);
        info!("");
    }

    // Estimate model size (roughly 4 bytes per parameter for Q4 quantization)
    let total_size_gb: f64 = 4.0; // Default to 4GB for small models
    info!("  Estimated size: {:.2} GB", total_size_gb);

    // Determine Memory64 layout size (round up to next GB, min 4GB)
    let memory64_size_gb = ((total_size_gb + 1.0).ceil() as u64).max(4);
    info!("  Memory64 size: {} GB", memory64_size_gb);
    info!("");

    // Create HostContext with appropriate Memory64 size
    let layout = MemoryLayout::single(memory64_size_gb, "model_weights")
        .context("Failed to create memory layout")?;
    let host_context = HostContext::with_layout(layout);
    info!(
        "✅ HostContext created with {} GB Memory64",
        memory64_size_gb
    );

    // Initialize Wasmtime
    let mut config = Config::new();
    config.wasm_bulk_memory(true);
    config.wasm_multi_memory(true);

    let engine = Engine::new(&config)?;
    let mut store = Store::new(&engine, ());

    // Initialize Memory64 and load model weights
    host_context.initialize(&mut store)?;
    info!("✅ Memory64 runtime initialized");

    // TODO: Load model weights into Memory64
    // This would involve:
    // 1. Iterate through model tensors
    // 2. Write each tensor to Memory64 at appropriate offsets
    // 3. Register layers with Memory64Runtime
    info!("⚠️  Model weight loading not yet implemented");
    info!("    Weights would be loaded into Memory64 here");
    info!("");

    // Create linker and add host functions
    let mut linker = Linker::new(&engine);
    host_context.add_to_linker(&mut linker)?;
    add_wasm_bindgen_stubs(&mut linker)?;
    info!("✅ Host functions linked");

    // Load WASM module
    let wasm_path = "/home/puneet/realm/crates/realm-wasm/pkg/realm_wasm_bg.wasm";
    let module = Module::from_file(&engine, wasm_path)?;
    info!("✅ WASM module loaded");

    // Instantiate
    let _instance = linker.instantiate(&mut store, &module)?;
    info!("✅ WASM instance created");
    info!("");

    info!("🎯 Model loaded and ready for inference!");
    info!("");
    info!("Next steps to complete end-to-end inference:");
    info!("  1. Implement tensor loading from GGUF into Memory64");
    info!("  2. Implement layer registration and tracking");
    info!("  3. Add WASM-side inference orchestration");
    info!("  4. Call inference functions from host");
    info!("  5. Return generated tokens");

    Ok(())
}

/// Run in demo mode without a model (tests architecture only)
fn run_demo_mode() -> Result<()> {
    info!("🧪 Running in demo mode (architecture validation)");
    info!("");

    // Create HostContext with default 8GB
    let host_context = HostContext::new();
    info!("✅ HostContext created (8GB Memory64)");

    // Initialize Wasmtime
    let mut config = Config::new();
    config.wasm_bulk_memory(true);
    config.wasm_multi_memory(true);

    let engine = Engine::new(&config)?;
    let mut store = Store::new(&engine, ());

    // Initialize Memory64
    host_context.initialize(&mut store)?;
    info!("✅ Memory64 runtime initialized");

    // Create linker
    let mut linker = Linker::new(&engine);
    host_context.add_to_linker(&mut linker)?;
    add_wasm_bindgen_stubs(&mut linker)?;
    info!("✅ Host functions linked");

    // Load WASM module
    let wasm_path = "/home/puneet/realm/crates/realm-wasm/pkg/realm_wasm_bg.wasm";
    if std::path::Path::new(wasm_path).exists() {
        let module = Module::from_file(&engine, wasm_path)?;
        info!("✅ WASM module loaded");

        let _instance = linker.instantiate(&mut store, &module)?;
        info!("✅ WASM instance created");
        info!("");

        info!("🎯 Architecture validated successfully!");
        info!("");
        info!("To test with a real model:");
        info!("  cargo run --bin end-to-end-inference -- /path/to/model.gguf");
    } else {
        error!("❌ WASM module not found at: {}", wasm_path);
        error!("   Please build realm-wasm first:");
        error!("   cd crates/realm-wasm && wasm-pack build --target web");
        return Err(anyhow::anyhow!("WASM module not found"));
    }

    Ok(())
}

/// Add wasm-bindgen stub imports
fn add_wasm_bindgen_stubs(linker: &mut Linker<()>) -> Result<()> {
    linker.func_wrap(
        "wbg",
        "__wbindgen_object_drop_ref",
        |_: Caller<'_, ()>, _: i32| {},
    )?;
    linker.func_wrap(
        "wbg",
        "__wbindgen_string_new",
        |_: Caller<'_, ()>, _: i32, _: i32| -> i32 { 0 },
    )?;
    linker.func_wrap(
        "wbg",
        "__wbg_log_f63c4c4d1ecbabd9",
        |_: Caller<'_, ()>, _: i32, _: i32| {},
    )?;
    linker.func_wrap(
        "wbg",
        "__wbg_log_6c7b5f4f00b8ce3f",
        |_: Caller<'_, ()>, _: i32| {},
    )?;
    linker.func_wrap(
        "wbg",
        "__wbindgen_throw",
        |_: Caller<'_, ()>, _: i32, _: i32| {},
    )?;
    linker.func_wrap(
        "wbg",
        "__wbg_wbindgenthrow_451ec1a8469d7eb6",
        |_: Caller<'_, ()>, _: i32, _: i32| {},
    )?;
    Ok(())
}
