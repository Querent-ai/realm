//! Memory64 Demo - Loading Large Models (>4GB)
//!
//! This example demonstrates how Realm handles large models using Memory64:
//! 1. Automatic detection of model size
//! 2. Memory64 initialization for models >3GB
//! 3. Lazy layer loading for efficient memory usage
//! 4. On-demand weight loading during inference

use anyhow::Result;
use realm_core::{GGUFParser, TensorLoader, Tokenizer};
use realm_runtime::memory64_model::Memory64ModelLoader;
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🧠 Realm Memory64 Demo");
    println!("======================\n");

    // Get model path from args or use default
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/tinyllama-1.1b.Q4_K_M.gguf".to_string());

    println!("📦 Loading model: {}", model_path);
    println!();

    // Open and parse GGUF file
    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    println!("✅ Model header parsed");
    println!("   Version: {}", meta.version);
    println!("   Tensors: {}", meta.tensor_count);
    println!();

    // Extract configuration
    let config_data = parser
        .extract_config()
        .ok_or_else(|| anyhow::anyhow!("Failed to extract config"))?;
    let config: realm_models::TransformerConfig = config_data.into();

    println!("⚙️  Model Configuration");
    println!("   Layers: {}", config.num_layers);
    println!("   Hidden size: {}", config.hidden_size);
    println!("   Vocab size: {}", config.vocab_size);
    println!();

    // Calculate total model size
    let total_size: u64 = meta
        .tensors
        .iter()
        .map(|tensor| tensor.size_bytes as u64)
        .sum();

    let size_gb = total_size as f64 / 1_000_000_000.0;
    println!("📊 Model Size Analysis");
    println!("   Total size: {:.2} GB", size_gb);

    // Check if Memory64 will be used
    let will_use_memory64 = total_size > 3_000_000_000;
    if will_use_memory64 {
        println!("   ✅ Model >3GB - Memory64 will be used");
        println!("   📦 Layers will be loaded on-demand");
        println!("   💾 Memory efficiency: High");
    } else {
        println!("   ℹ️  Model ≤3GB - Standard loading will be used");
        println!("   📦 All weights loaded into RAM");
        println!("   ⚡ Performance: Fast (all in memory)");
    }
    println!();

    // Create Memory64-aware model loader
    let mut memory64_loader = Memory64ModelLoader::new(config, total_size);

    // Initialize Memory64 if needed
    if will_use_memory64 {
        println!("🔧 Initializing Memory64 runtime...");
        memory64_loader.initialize_memory64()?;

        if let Some(runtime) = memory64_loader.runtime() {
            let state_arc = runtime.state();
            let state_guard = state_arc.lock();
            let layout = state_guard.layout();
            println!("   Layout: {:.2} GB total capacity", layout.total_gb());
            println!("   Regions: {}", layout.regions.len());
            drop(state_guard); // Explicitly drop the lock guard
            println!();
        }
    }

    // Load tokenizer
    let _tokenizer = Tokenizer::from_gguf(&meta)?;
    println!(
        "✅ Tokenizer loaded: {} tokens\n",
        meta.vocab_size.unwrap_or(0)
    );

    // Load model weights
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor in meta.tensors.iter() {
        tensor_loader.register_tensor(tensor.name.clone(), tensor.clone(), tensor.offset);
    }

    // Reopen file for tensor loading
    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    println!("🚀 Loading model weights...");
    let _model = memory64_loader.load_model(&mut tensor_loader, &mut parser)?;

    println!();
    println!("✅ SUCCESS! Model loaded and ready for inference");
    println!();

    // Print summary
    println!("📝 Summary");
    println!("   Model: {}", model_path);
    println!("   Size: {:.2} GB", size_gb);
    println!(
        "   Memory64: {}",
        if will_use_memory64 {
            "✅ Enabled"
        } else {
            "❌ Not needed"
        }
    );
    println!("   Status: Ready for inference");
    println!();

    if will_use_memory64 {
        println!("💡 Memory64 Features:");
        println!("   • On-demand layer loading");
        println!("   • Efficient memory usage for large models");
        println!("   • Support for models >4GB (WASM memory limit)");
        println!("   • Lazy loading reduces startup time");
        println!("   • Cache management for optimal performance");
    } else {
        println!("💡 Standard Loading:");
        println!("   • All weights loaded into RAM");
        println!("   • Fastest inference performance");
        println!("   • Suitable for models <3GB");
    }
    println!();

    println!("🎯 Next Steps:");
    println!("   1. Run inference with this model");
    println!("   2. Try with a larger model (>3GB) to see Memory64 in action");
    println!("   3. Monitor memory usage during inference");

    Ok(())
}
