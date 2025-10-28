//! Paris Generation Example
//!
//! This example demonstrates complete end-to-end inference:
//! - Question: "What is the capital of France?"
//! - Expected Answer: "Paris"

use anyhow::Result;
use realm_core::{GGUFParser, TensorLoader, Tokenizer};
use std::fs::File;
use std::io::BufReader;

// Import from realm-models crate
use realm_models::GenerationConfig;
use realm_models::Model;
use realm_models::TransformerConfig;

fn main() -> Result<()> {
    // Initialize logger (RUST_LOG=info for INFO level, =debug for DEBUG level)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🗼 Realm Paris Generation Example");
    println!("===================================\n");

    // Use first arg or default model path
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../models/tinyllama-1.1b.Q4_K_M.gguf".to_string());

    println!("📦 Loading model: {}", model_path);

    // Open and parse GGUF file
    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    println!("✅ Model header parsed");
    println!("   Version: {}", meta.version);
    println!("   Tensors: {}", meta.tensor_count);

    // Extract configuration
    let config_data = parser
        .extract_config()
        .ok_or_else(|| anyhow::anyhow!("Failed to extract config"))?;
    let config: TransformerConfig = config_data.into();

    println!(
        "✅ Config loaded: {} layers, {} heads",
        config.num_layers, config.num_heads
    );

    // Create model
    let mut model = Model::new(config.clone());
    println!("🔧 Model created");

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded: {} tokens", tokenizer.vocab_size());

    // Load weights
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor in meta.tensors.iter() {
        tensor_loader.register_tensor(tensor.name.clone(), tensor.clone(), tensor.offset);
    }

    // Reopen file for tensor loading
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Weights loaded\n");

    // Use ChatML template like wasm-chord for correct behavior
    use realm_runtime::{ChatMessage, ChatTemplate};
    let template = ChatTemplate::ChatML;
    let conversation = vec![
        ChatMessage::system("You are a helpful AI assistant."),
        ChatMessage::user("What is the capital of France?"),
    ];
    let prompt = template.format(&conversation)?;
    println!("📝 Prompt: \"{}\"\n", prompt);

    // Generation config - match wasm-chord exactly
    let gen_config = GenerationConfig {
        max_tokens: 20,
        temperature: 0.0, // Greedy for deterministic output
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    println!("🤖 Generating response...\n");
    let response = model.generate(&prompt, &tokenizer, &gen_config)?;

    println!("✨ Response: {}\n", response);

    // Check if response contains "Paris"
    if response.to_lowercase().contains("paris") {
        println!("✅ SUCCESS: Model correctly identified Paris as the capital of France!");
    } else {
        println!(
            "⚠️  WARNING: Expected 'Paris' in response, got: {}",
            response
        );
    }

    Ok(())
}
