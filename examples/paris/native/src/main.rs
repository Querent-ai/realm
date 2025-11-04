//! Paris Generation Example - Native Rust
//!
//! This example demonstrates complete end-to-end inference using Realm's native Rust API:
//! - Question: "What is the capital of France?"
//! - Expected Answer: "Paris"
//!
//! This is the simplest way to use Realm - direct Rust API with no WASM.

use anyhow::Result;
use realm_core::{GGUFParser, TensorLoader, Tokenizer};
use realm_metrics::{CostConfig, UsageTracker};
use std::fs::File;
use std::io::BufReader;

use realm_models::{GenerationConfig, Model, TransformerConfig};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("Realm Paris Generation - Native Rust");
    log::info!("====================================\n");

    // Use first arg or default model path
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../../models/tinyllama-1.1b.Q4_K_M.gguf".to_string());

    log::info!("Loading model: {}", model_path);

    // Open and parse GGUF file
    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    log::info!("Model header parsed");
    log::debug!("Version: {}, Tensors: {}", meta.version, meta.tensor_count);

    // Extract configuration
    let config_data = parser
        .extract_config()
        .ok_or_else(|| anyhow::anyhow!("Failed to extract config"))?;
    let config: TransformerConfig = config_data.into();

    log::info!(
        "Config loaded: {} layers, {} heads",
        config.num_layers,
        config.num_heads
    );

    // Create model
    let mut model = Model::new(config.clone());
    log::info!("Model created");

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    log::info!("Tokenizer loaded: {} tokens", tokenizer.vocab_size());

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

    model.load_from_gguf(&mut tensor_loader, &mut parser, None, None)?;
    log::info!("Weights loaded\n");

    // Setup usage tracking (optional)
    let cost_config = CostConfig::simple(3.0, 15.0);
    let tracker = UsageTracker::new(cost_config);
    model.set_usage_tracker(tracker);
    model.set_model_name("tinyllama-1.1b-q4");
    model.set_tenant_id("demo-user");
    log::info!("Usage tracking enabled\n");

    // Use ChatML template
    use realm_runtime::{ChatMessage, ChatTemplate};
    let template = ChatTemplate::ChatML;
    let conversation = vec![
        ChatMessage::system("You are a helpful AI assistant."),
        ChatMessage::user("What is the capital of France?"),
    ];
    let prompt = template.format(&conversation)?;
    log::info!("Prompt: \"{}\"\n", prompt);

    // Generation config - deterministic for testing
    let gen_config = GenerationConfig {
        max_tokens: 20,
        temperature: 0.0, // Greedy for deterministic output
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    log::info!("Generating response...\n");
    let response = model.generate(&prompt, &tokenizer, &gen_config)?;

    log::info!("Response: {}\n", response);

    // Check if response contains "Paris"
    if response.to_lowercase().contains("paris") {
        log::info!("✅ SUCCESS: Model correctly identified Paris as the capital of France!");
    } else {
        log::warn!("⚠️  Expected 'Paris' in response, got: {}", response);
    }

    // Display usage metrics
    if let Some(tracker) = model.usage_tracker() {
        log::info!("\n=== Usage Metrics ===");
        let total = tracker.total();
        log::info!("Input tokens: {}", total.total_input_tokens);
        log::info!("Output tokens: {}", total.total_output_tokens);
        log::info!("Total tokens: {}", total.total_tokens);
        log::info!("Estimated cost: ${:.6}", total.estimated_cost);
    }

    Ok(())
}

