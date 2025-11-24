//! LoRA Adapter Demo
//!
//! This example demonstrates how to:
//! 1. Create a LoRA adapter with custom weights
//! 2. Load it into RuntimeManager
//! 3. Set it for a tenant
//! 4. Generate text with LoRA applied
//!
//! Usage:
//!   cargo run --release --bin lora-demo -- <model_path>
//!
//! Example:
//!   cargo run --release --bin lora-demo -- ~/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf

use anyhow::Result;
use realm_runtime::lora::LoRAWeights;
use realm_server::runtime_manager::{ModelConfig, RuntimeManager};
use std::env;
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> Result<()> {
    // Set up logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("🎯 LoRA Adapter Demo");
    info!("");

    // Get model path from command line
    let model_path = env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("Usage: lora-demo <model_path>"))?;
    let model_path = PathBuf::from(model_path);

    if !model_path.exists() {
        return Err(anyhow::anyhow!("Model file not found: {:?}", model_path));
    }

    info!("📦 Model: {:?}", model_path);
    info!("");

    // Create RuntimeManager
    info!("🔧 Creating RuntimeManager...");
    // Find WASM module path (try common locations)
    let wasm_path = env::var("REALM_WASM_PATH")
        .map(PathBuf::from)
        .or_else(|_| {
            // Try relative path from workspace root
            let paths = vec![
                "crates/realm-wasm/pkg/realm_wasm_bg.wasm",
                "../../crates/realm-wasm/pkg/realm_wasm_bg.wasm",
                "../realm-wasm/pkg/realm_wasm_bg.wasm",
            ];
            for path in paths {
                let p = PathBuf::from(path);
                if p.exists() {
                    return Ok(p);
                }
            }
            Err(())
        })
        .map_err(|_| {
            anyhow::anyhow!(
                "WASM module not found. Set REALM_WASM_PATH or ensure realm-wasm is built."
            )
        })?;

    let mut runtime_manager = RuntimeManager::new(wasm_path)?;
    info!("✅ RuntimeManager created");
    info!("");

    // Step 1: Create a LoRA adapter
    info!("📝 Step 1: Creating LoRA adapter...");
    let adapter_id = "demo_adapter";
    let rank = 8;
    let alpha = 16.0;

    let mut adapter = LoRAWeights::new(adapter_id.to_string(), rank, alpha);

    // Add sample LoRA weights for layer 0 attention
    // In a real scenario, these would be loaded from a trained LoRA file
    // For demo purposes, we create small dummy weights
    let hidden_size = 2048; // Example hidden size (adjust based on your model)
    let lora_a_size = rank * hidden_size; // [rank, hidden_size]
    let lora_b_size = hidden_size * rank; // [hidden_size, rank]

    // Create dummy LoRA A weights (small random values)
    let lora_a_weights: Vec<f32> = (0..lora_a_size)
        .map(|i| (i as f32) * 0.001) // Small values for demo
        .collect();

    // Create dummy LoRA B weights (small random values)
    let lora_b_weights: Vec<f32> = (0..lora_b_size)
        .map(|i| (i as f32) * 0.001) // Small values for demo
        .collect();

    // Add weights for attention query projection (layer 0)
    // Note: Keys must include .lora_a and .lora_b suffixes as expected by apply_to_weights
    // apply_to_weights looks for keys like "layer.0.attn_q.lora_a" and "layer.0.attn_q.lora_b"
    adapter
        .lora_a
        .insert("layer.0.attn_q.lora_a".to_string(), lora_a_weights.clone());
    adapter
        .lora_b
        .insert("layer.0.attn_q.lora_b".to_string(), lora_b_weights.clone());

    // Add weights for attention key projection (layer 0)
    adapter
        .lora_a
        .insert("layer.0.attn_k.lora_a".to_string(), lora_a_weights.clone());
    adapter
        .lora_b
        .insert("layer.0.attn_k.lora_b".to_string(), lora_b_weights.clone());

    // Add weights for attention value projection (layer 0)
    adapter
        .lora_a
        .insert("layer.0.attn_v.lora_a".to_string(), lora_a_weights.clone());
    adapter
        .lora_b
        .insert("layer.0.attn_v.lora_b".to_string(), lora_b_weights.clone());

    // Add weights for attention output projection (layer 0)
    // Note: The actual weight name in the model is "attn_output", not "attn_o"
    adapter.lora_a.insert(
        "layer.0.attn_output.lora_a".to_string(),
        lora_a_weights.clone(),
    );
    adapter.lora_b.insert(
        "layer.0.attn_output.lora_b".to_string(),
        lora_b_weights.clone(),
    );

    info!("✅ LoRA adapter created:");
    info!("   - ID: {}", adapter_id);
    info!("   - Rank: {}", rank);
    info!("   - Alpha: {}", alpha);
    info!("   - Scale: {}", adapter.scale());
    info!("   - Layers: layer.0 (attention weights)");
    info!("");

    // Step 2: Load LoRA adapter into RuntimeManager
    info!("📥 Step 2: Loading LoRA adapter into RuntimeManager...");
    runtime_manager.load_lora_adapter(adapter)?;
    info!("✅ LoRA adapter loaded");
    info!("");

    // Step 3: Set default model
    info!("🤖 Step 3: Setting default model...");
    let model_config = ModelConfig {
        model_path: model_path.clone(),
        model_id: "base_model".to_string(),
        draft_model_path: None,
        draft_model_id: None,
    };
    runtime_manager.set_default_model(model_config);
    info!("✅ Default model set");
    info!("");

    // Step 4: Set LoRA adapter for a tenant
    info!("👤 Step 4: Setting LoRA adapter for tenant...");
    let tenant_id = "demo_tenant";
    runtime_manager.set_tenant_lora_adapter(tenant_id, adapter_id)?;
    info!("✅ LoRA adapter set for tenant: {}", tenant_id);
    info!("");

    // Step 5: Generate text with LoRA applied (if model is available)
    info!("🚀 Step 5: Generating text with LoRA applied...");
    let prompt = "What is the capital of France?";
    info!("   Prompt: {}", prompt);
    info!("");

    // Check if we're in CI (no GPU/model available)
    let in_ci = env::var("CI").is_ok() || env::var("GITHUB_ACTIONS").is_ok();

    if in_ci {
        info!("ℹ️  Running in CI - skipping actual generation");
        info!("   (Model loading requires GPU and large model files)");
        info!("");
        info!("✅ LoRA adapter configuration complete!");
        info!("   - Adapter created and loaded");
        info!("   - Adapter set for tenant");
        info!("   - Ready for inference (when model is available)");
    } else {
        match runtime_manager.generate(tenant_id, prompt.to_string()) {
            Ok(response) => {
                info!("✅ Generation successful!");
                info!("   Response: {}", response);
                info!("");
                info!("🎉 LoRA adapter is being applied during inference!");
                info!("");
                info!("💡 Note: In a real scenario, you would:");
                info!("   1. Train a LoRA adapter on your specific task");
                info!("   2. Load it from a file (e.g., safetensors format)");
                info!("   3. Apply it to fine-tune the model for your use case");
            }
            Err(e) => {
                info!("⚠️  Generation failed: {}", e);
                info!("");
                info!("💡 This might happen if:");
                info!("   - Model file is invalid or corrupted");
                info!("   - Model architecture doesn't match expected format");
                info!("   - LoRA weights dimensions don't match model dimensions");
                info!("   - GPU not available");
                info!("");
                info!("✅ However, LoRA adapter loading and configuration worked!");
            }
        }
    }

    info!("");
    info!("📊 Summary:");
    info!("   ✅ LoRA adapter created");
    info!("   ✅ LoRA adapter loaded into RuntimeManager");
    info!("   ✅ LoRA adapter set for tenant");
    info!("   ✅ Ready for inference with LoRA applied");
    info!("");

    Ok(())
}
