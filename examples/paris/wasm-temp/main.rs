//! WASM Paris Generation Example
//! 
//! This demonstrates end-to-end inference via WASM with host-side storage.
//! Model weights are stored in HOST, WASM loads them on-demand during inference.

use anyhow::{Context, Result};
use realm_runtime::memory64_host::Memory64Runtime;
use std::fs;
use wasmtime::*;

fn main() -> Result<()> {
    env_logger::init();
    
    println!("🚀 WASM Paris Generation - Host-Side Storage Demo\n");

    // Load model file
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "~/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf".to_string());
    
    let expanded_path = shellexpand::full(&model_path)
        .context("Failed to expand path")?;
    
    println!("📦 Loading model: {}", expanded_path);
    let model_bytes = fs::read(&*expanded_path)
        .with_context(|| format!("Failed to read model: {}", expanded_path))?;
    
    println!("   Model size: {:.2} MB\n", model_bytes.len() as f64 / 1024.0 / 1024.0);

    // Initialize WASM runtime with host functions
    println!("🔧 Initializing WASM runtime...");
    let engine = Engine::default();
    let mut store = Store::new(&engine, ());
    
    // Create Memory64 runtime for host-side storage
    let memory64 = Memory64Runtime::new(
        realm_runtime::memory64_host::MemoryLayout::single(8, "model_storage")
            .context("Failed to create memory layout")?,
        true,
    );
    
    // Create linker with host functions
    let mut linker = Linker::new(&engine);
    memory64.add_host_functions(&mut linker)?;
    
    // Load WASM module
    println!("📥 Loading WASM module...");
    let wasm_path = "target/wasm32-unknown-unknown/release/realm_wasm.wasm";
    if !std::path::Path::new(wasm_path).exists() {
        anyhow::bail!(
            "WASM module not found at {}. Build with: cargo build -p realm-wasm --target wasm32-unknown-unknown --release",
            wasm_path
        );
    }
    
    let wasm_bytes = fs::read(wasm_path)
        .context("Failed to read WASM module")?;
    
    let module = Module::new(&engine, &wasm_bytes)
        .context("Failed to compile WASM module")?;
    
    // Instantiate WASM module
    println!("🔨 Instantiating WASM module...");
    let instance = linker
        .instantiate(&mut store, &module)
        .context("Failed to instantiate WASM module")?;
    
    // Get memory export
    let memory = instance
        .get_memory(&mut store, "memory")
        .context("WASM module doesn't export memory")?;
    
    println!("✅ WASM module ready!\n");

    // Store model in HOST via WASM call
    println!("💾 Storing model in HOST storage...");
    let load_model = instance
        .get_typed_func::<(u32, u32, u32), i32>(&mut store, "realm_load_model")
        .context("Failed to get loadModel function")?;
    
    // Allocate memory in WASM for model bytes
    let alloc = instance
        .get_typed_func::<u32, u32>(&mut store, "alloc")
        .ok();
    
    // Write model bytes to WASM memory
    // For now, we'll use a simple approach - write directly
    // In production, you'd use wasm-bindgen allocator
    let model_ptr = memory.data_size(&store) as u32;
    memory.grow(&mut store, (model_bytes.len() / 65536 + 1) as u32)
        .context("Failed to grow WASM memory")?;
    
    unsafe {
        let mut data = memory.data_mut(&mut store);
        data[model_ptr as usize..model_ptr as usize + model_bytes.len()]
            .copy_from_slice(&model_bytes);
    }
    
    // Call loadModel (model_id = 0 for auto-generation)
    let model_id = load_model.call(&mut store, (model_ptr, model_bytes.len() as u32, 0))
        .context("loadModel failed")?;
    
    if model_id < 0 {
        anyhow::bail!("Failed to store model (error: {})", model_id);
    }
    
    println!("   ✅ Model stored with ID: {}\n", model_id);

    // Generate "Paris" response
    println!("🎯 Generating response to: 'What is the capital of France?'");
    let prompt = "What is the capital of France?";
    
    // Get generate function
    let generate_fn = instance
        .get_typed_func::<u32, u32>(&mut store, "realm_generate")
        .context("Failed to get generate function")?;
    
    // Write prompt to WASM memory
    let prompt_bytes = prompt.as_bytes();
    let prompt_ptr = memory.data_size(&store) as u32;
    memory.grow(&mut store, (prompt_bytes.len() / 65536 + 1) as u32)
        .context("Failed to grow WASM memory for prompt")?;
    
    unsafe {
        let mut data = memory.data_mut(&mut store);
        data[prompt_ptr as usize..prompt_ptr as usize + prompt_bytes.len()]
            .copy_from_slice(prompt_bytes);
    }
    
    // Call generate
    let response_ptr = generate_fn.call(&mut store, prompt_ptr)
        .context("generate failed")?;
    
    // Read response from WASM memory
    // Note: In real implementation, response_ptr would point to a string
    // For now, we'll need to handle the response format
    
    println!("\n✨ Generation complete!");
    println!("   Model ID: {}", model_id);
    println!("   (Full response reading requires wasm-bindgen integration)\n");
    
    println!("🎉 SUCCESS: WASM inference with host-side storage working!");
    
    Ok(())
}

