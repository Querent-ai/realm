use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "realm")]
#[command(author, version, about = "Realm - Multi-tenant LLM inference runtime", long_about = None)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference with a model
    Run {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Input prompt
        #[arg(short, long)]
        prompt: Option<String>,

        /// Maximum tokens to generate
        #[arg(short = 'n', long, default_value = "100")]
        max_tokens: usize,

        /// Sampling temperature
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Top-k sampling
        #[arg(short = 'k', long, default_value = "50")]
        top_k: usize,

        /// Top-p (nucleus) sampling
        #[arg(short = 'p', long, default_value = "0.9")]
        top_p: f32,

        /// Use GPU (CUDA/Metal)
        #[arg(long)]
        gpu: bool,
    },

    /// Download a model from Hugging Face
    Download {
        /// Model identifier (e.g., "TheBloke/Llama-2-7B-Chat-GGUF")
        model: String,

        /// Specific file to download (optional)
        #[arg(short, long)]
        file: Option<String>,

        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// List available models
    List {
        /// Show local models
        #[arg(long)]
        local: bool,
    },

    /// Start WebSocket inference server
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Path to WASM module (realm.wasm)
        #[arg(short, long)]
        wasm: PathBuf,

        /// Model to load (GGUF file)
        #[arg(short, long)]
        model: PathBuf,

        /// Metrics port
        #[arg(long, default_value = "9090")]
        metrics_port: u16,

        /// Enable authentication
        #[arg(long)]
        auth: bool,

        /// API keys file path (required when --auth is enabled)
        #[arg(long)]
        api_keys: Option<PathBuf>,

        /// Enable GPU
        #[arg(long)]
        gpu: bool,

        /// Maximum concurrent tenants
        #[arg(long, default_value = "8")]
        max_tenants: usize,
    },

    /// Show system information
    Info,

    /// Benchmark a model
    Bench {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Prompt length to test
        #[arg(long, default_value = "128")]
        prompt_len: usize,

        /// Number of tokens to generate
        #[arg(long, default_value = "100")]
        gen_len: usize,

        /// Use GPU
        #[arg(long)]
        gpu: bool,
    },

    /// Manage API keys
    ApiKey {
        #[command(subcommand)]
        command: ApiKeyCommands,
    },

    /// Manage models
    Models {
        #[command(subcommand)]
        command: ModelsCommands,
    },

    /// Manage pipelines
    Pipeline {
        #[command(subcommand)]
        command: PipelineCommands,
    },
}

#[derive(Subcommand)]
enum ApiKeyCommands {
    /// Generate a new API key
    Generate {
        /// Tenant ID for this key
        #[arg(short, long)]
        tenant: String,

        /// Optional name for this key
        #[arg(short, long)]
        name: Option<String>,

        /// API keys file path
        #[arg(short, long, default_value = "api-keys.json")]
        file: PathBuf,
    },

    /// List all API keys
    List {
        /// API keys file path
        #[arg(short, long, default_value = "api-keys.json")]
        file: PathBuf,

        /// Filter by tenant ID
        #[arg(short, long)]
        tenant: Option<String>,
    },

    /// Disable an API key
    Disable {
        /// API key to disable
        key: String,

        /// API keys file path
        #[arg(short, long, default_value = "api-keys.json")]
        file: PathBuf,
    },

    /// Enable an API key
    Enable {
        /// API key to enable
        key: String,

        /// API keys file path
        #[arg(short, long, default_value = "api-keys.json")]
        file: PathBuf,
    },
}

#[derive(Subcommand)]
enum ModelsCommands {
    /// List available models
    List {
        /// Filter by source (ollama, huggingface, all)
        #[arg(short, long)]
        source: Option<String>,

        /// Filter by capability (chat, completion, embedding, etc.)
        #[arg(short, long)]
        capability: Option<String>,
    },

    /// Search for models
    Search {
        /// Search query
        query: String,
    },

    /// Show model details
    Info {
        /// Model ID (e.g., "llama-2-7b-chat")
        model: String,
    },

    /// Check if model is cached
    Status {
        /// Model spec (e.g., "llama-2-7b-chat:Q4_K_M")
        model: String,
    },

    /// Download a model from its source
    Download {
        /// Model spec (e.g., "llama-2-7b-chat:Q4_K_M")
        model: String,

        /// Output directory (optional, defaults to ~/.realm/models)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum PipelineCommands {
    /// List available pipelines
    List {
        /// Pipeline directory (defaults to ./pipelines)
        #[arg(short, long)]
        dir: Option<PathBuf>,
    },

    /// Show pipeline details
    Info {
        /// Path to pipeline file (YAML or JSON)
        file: PathBuf,
    },

    /// Validate a pipeline definition
    Validate {
        /// Path to pipeline file (YAML or JSON)
        file: PathBuf,
    },

    /// Load a pipeline into the orchestrator
    Load {
        /// Path to pipeline file (YAML or JSON)
        file: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };

    let subscriber = FmtSubscriber::builder().with_max_level(level).finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Print banner
    print_banner();

    match cli.command {
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            gpu,
        } => {
            cmd_run(model, prompt, max_tokens, temperature, top_k, top_p, gpu)?;
        }
        Commands::Download {
            model,
            file,
            output,
        } => {
            // cmd_download is async, run it in tokio runtime
            tokio::runtime::Runtime::new()?.block_on(cmd_download_async(model, file, output))?;
        }
        Commands::List { local } => {
            cmd_list(local)?;
        }
        Commands::Serve {
            host,
            port,
            wasm,
            model,
            metrics_port,
            auth,
            api_keys,
            gpu,
            max_tenants,
        } => {
            cmd_serve(
                host,
                port,
                wasm,
                model,
                metrics_port,
                auth,
                api_keys,
                gpu,
                max_tenants,
            )?;
        }
        Commands::Info => {
            cmd_info()?;
        }
        Commands::Bench {
            model,
            prompt_len,
            gen_len,
            gpu,
        } => {
            cmd_bench(model, prompt_len, gen_len, gpu)?;
        }
        Commands::ApiKey { command } => {
            cmd_api_key(command)?;
        }
        Commands::Models { command } => {
            cmd_models(command)?;
        }
        Commands::Pipeline { command } => {
            cmd_pipeline(command)?;
        }
    }

    Ok(())
}

fn print_banner() {
    println!("{}", "╔═══════════════════════════════════════════╗".cyan());
    println!(
        "{}",
        "║           Realm Inference CLI            ║".cyan().bold()
    );
    println!("{}", "║   Multi-tenant LLM Runtime v0.1.0        ║".cyan());
    println!("{}", "╚═══════════════════════════════════════════╝".cyan());
    println!();
}

fn cmd_run(
    model: PathBuf,
    prompt: Option<String>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    gpu: bool,
) -> Result<()> {
    info!("🚀 {}", "Starting inference".green().bold());
    info!("📦 Model: {}", model.display());
    info!("🎯 Max tokens: {}", max_tokens);
    info!("🌡️  Temperature: {}", temperature);
    info!("🔝 Top-k: {}, Top-p: {}", top_k, top_p);
    info!(
        "💻 Backend: {}",
        if gpu { "GPU (CUDA/Metal)" } else { "CPU" }
    );
    println!();

    // Check if model exists
    if !model.exists() {
        eprintln!(
            "{} Model file not found: {}",
            "✗".red().bold(),
            model.display()
        );
        eprintln!("{} Try downloading a model first:", "ℹ".blue());
        eprintln!("  realm download TheBloke/Llama-2-7B-Chat-GGUF");
        return Ok(());
    }

    let prompt_text = prompt.unwrap_or_else(|| {
        println!("{} Enter your prompt (Ctrl+D to finish):", "→".cyan());
        let mut lines = Vec::new();
        use std::io::BufRead;
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            if let Ok(line) = line {
                lines.push(line);
            } else {
                break;
            }
        }
        lines.join("\n")
    });

    if prompt_text.trim().is_empty() {
        eprintln!("{} No prompt provided", "✗".red().bold());
        return Ok(());
    }

    println!();
    println!("{} {}", "Prompt:".yellow().bold(), prompt_text);
    println!();
    println!("{}", "Generating...".cyan().bold());
    println!();

    // TODO: Implement actual inference
    println!("{} Implementation coming soon!", "ℹ".blue());
    println!("This will:");
    println!("  1. Load model into Memory64");
    println!("  2. Initialize WASM runtime");
    println!("  3. Run inference with Candle backend");
    println!("  4. Stream tokens back");

    Ok(())
}

async fn cmd_download_async(
    model: String,
    file: Option<String>,
    output: Option<PathBuf>,
) -> Result<()> {
    use realm_models::registry::{ModelRegistry, RegistryConfig};

    println!("{}", "📥 Downloading Model".green().bold());
    println!();

    // Create registry
    let mut registry_config = RegistryConfig::default();
    if let Some(output_dir) = output {
        registry_config.cache_dir = output_dir;
    }
    let registry = ModelRegistry::new(registry_config)?;

    // Check if model is in registry
    let model_spec = if let Some(ref file) = file {
        // If file is specified, try to construct full spec
        format!("{}:{}", model, file)
    } else {
        model.clone()
    };

    // Check if already cached
    if registry.is_cached(&model_spec) {
        let cached_path = registry.resolve(&model_spec)?;
        println!("{} Model already cached", "✓".green().bold());
        println!("  Path: {}", cached_path.display());

        if let Ok(metadata) = std::fs::metadata(&cached_path) {
            let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
            println!("  Size: {:.2} MB", size_mb);
        }
        return Ok(());
    }

    // Try to download
    println!("{} Downloading model: {}", "→".cyan(), model_spec);
    println!();

    match registry.download_model(&model_spec).await {
        Ok(path) => {
            println!();
            println!("{} Download complete!", "✓".green().bold());
            println!("  Path: {}", path.display());

            if let Ok(metadata) = std::fs::metadata(&path) {
                let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                println!("  Size: {:.2} MB", size_mb);
            }

            println!();
            println!("{} You can now use this model:", "ℹ".blue());
            println!("  realm serve --wasm realm.wasm --model {}", model_spec);
        }
        Err(e) => {
            eprintln!("{} Download failed: {}", "✗".red().bold(), e);
            eprintln!();
            eprintln!("{} Available options:", "ℹ".blue());
            eprintln!("  1. Check if model exists in registry: realm models list");
            eprintln!("  2. Download manually from HuggingFace Hub");
            eprintln!("  3. Provide direct path to model file");
            return Err(e);
        }
    }

    Ok(())
}

fn cmd_list(local: bool) -> Result<()> {
    if local {
        info!("📚 {}", "Local models".green().bold());

        let model_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".realm")
            .join("models");

        if !model_dir.exists() {
            println!("{} No local models found", "ℹ".blue());
            println!("Download a model with: realm download <model>");
            return Ok(());
        }

        println!("Location: {}", model_dir.display());
        println!();

        // TODO: Scan directory for .gguf files
        println!("{} Scanning for models...", "→".cyan());
    } else {
        info!("📚 {}", "Popular models".green().bold());
        println!();
        println!("{}  TheBloke/Llama-2-7B-Chat-GGUF", "•".cyan());
        println!("{}  TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "•".cyan());
        println!("{}  TheBloke/CodeLlama-7B-Instruct-GGUF", "•".cyan());
        println!("{}  TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "•".cyan());
        println!();
        println!("Download with: realm download <model>");
    }

    Ok(())
}

#[tokio::main]
#[allow(clippy::too_many_arguments)]
async fn cmd_serve(
    host: String,
    port: u16,
    wasm: PathBuf,
    model: PathBuf,
    metrics_port: u16,
    auth: bool,
    api_keys: Option<PathBuf>,
    gpu: bool,
    max_tenants: usize,
) -> Result<()> {
    use realm_models::registry::{ModelRegistry, RegistryConfig};
    use realm_server::auth::ApiKeyStoreConfig;
    use realm_server::metrics_server::MetricsServerConfig;
    use realm_server::{
        runtime_manager::{ModelConfig, RuntimeManager},
        RealmServer, ServerConfig,
    };
    use std::sync::Arc;

    // Resolve model path (support both paths and registry names)
    let model_path = if model.exists() {
        // Direct path provided
        model
    } else {
        // Try to resolve from registry
        let model_spec = model.to_string_lossy().to_string();
        info!(
            "Model path not found, checking registry for: {}",
            model_spec
        );

        let registry_config = RegistryConfig::default();
        let registry = ModelRegistry::new(registry_config)?;

        match registry.resolve(&model_spec) {
            Ok(path) => {
                info!("✓ Model found in cache: {:?}", path);
                path
            }
            Err(e) => {
                eprintln!(
                    "{} Could not resolve model: {}",
                    "✗".red().bold(),
                    model_spec
                );
                eprintln!("{}", e);
                eprintln!();
                eprintln!("{} Available options:", "ℹ".blue());
                eprintln!(
                    "  1. Download the model manually to: {}",
                    registry.cache_dir().display()
                );
                eprintln!("  2. Use a different model: realm models list");
                eprintln!("  3. Provide a direct path to a GGUF file");
                return Ok(());
            }
        }
    };

    info!("🌐 {}", "Starting Realm WebSocket Server".green().bold());
    info!("📦 WASM Module: {}", wasm.display());
    info!("📦 Model: {}", model_path.display());
    info!("🔗 WebSocket: ws://{}:{}", host, port);
    info!("📊 Metrics: http://{}:{}/metrics", host, metrics_port);
    info!("👥 Max tenants: {}", max_tenants);
    info!(
        "🔐 Authentication: {}",
        if auth { "enabled" } else { "disabled" }
    );
    info!("💻 Backend: {}", if gpu { "GPU" } else { "CPU" });
    println!();

    // Check files exist
    if !wasm.exists() {
        eprintln!(
            "{} WASM module not found: {}",
            "✗".red().bold(),
            wasm.display()
        );
        eprintln!("{} Build the WASM module first:", "ℹ".blue());
        eprintln!("  cd crates/realm-wasm && wasm-pack build --target nodejs");
        return Ok(());
    }

    // Model existence already checked during resolution above

    // Validate authentication configuration
    if auth && api_keys.is_none() {
        eprintln!(
            "{} Authentication enabled but no API keys file specified",
            "✗".red().bold()
        );
        eprintln!("{} Use --api-keys to specify the API keys file", "ℹ".blue());
        eprintln!("Example:");
        eprintln!("  realm api-key generate --tenant tenant1 --file api-keys.json");
        eprintln!(
            "  realm serve --wasm realm.wasm --model model.gguf --auth --api-keys api-keys.json"
        );
        return Ok(());
    }

    if auth {
        if let Some(ref keys_file) = api_keys {
            if !keys_file.exists() {
                eprintln!(
                    "{} API keys file not found: {}",
                    "✗".red().bold(),
                    keys_file.display()
                );
                eprintln!("{} Generate API keys first:", "ℹ".blue());
                eprintln!(
                    "  realm api-key generate --tenant tenant1 --file {:?}",
                    keys_file
                );
                return Ok(());
            }
        }
    }

    println!("{}", "Starting server...".cyan().bold());
    println!();

    // Create runtime manager
    let mut runtime_manager = RuntimeManager::new(wasm)?;

    // Set default model
    let model_id = "default".to_string();
    runtime_manager.set_default_model(ModelConfig {
        model_path: model_path.clone(),
        model_id: model_id.clone(),
    });

    let runtime_manager = Arc::new(runtime_manager);

    // Create model orchestrator with runtime manager
    use realm_server::orchestrator::{ModelOrchestrator, ModelSpec, ModelType};
    let orchestrator = Arc::new(ModelOrchestrator::new(runtime_manager.clone()));

    // Register the default model with orchestrator
    // Extract model name from path for better identification
    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("default-model")
        .to_string();

    orchestrator
        .register_model(ModelSpec::new(
            model_id.clone(),
            ModelType::Completion, // Default to completion, can be overridden
            model_path.clone(),
            model_name.clone(),
        ))
        .context("Failed to register default model with orchestrator")?;

    info!("✓ Registered model '{}' with orchestrator", model_id);

    // Create dispatcher with both runtime manager and orchestrator
    let dispatcher = realm_server::dispatcher::FunctionDispatcher::with_runtime_and_orchestrator(
        runtime_manager,
        orchestrator,
    );

    // Build server config
    let config = ServerConfig {
        host: host.clone(),
        port,
        enable_auth: auth,
        max_connections: max_tenants,
        metrics_config: Some(MetricsServerConfig {
            host: host.clone(),
            port: metrics_port,
        }),
        api_key_store_config: api_keys.map(|keys_file| ApiKeyStoreConfig {
            keys_file: Some(keys_file),
            in_memory_only: false,
            ..Default::default()
        }),
        rate_limiter_config: Some(realm_server::rate_limiter::RateLimiterConfig::default()),
    };

    // Create server with runtime-enabled dispatcher
    let server = RealmServer::with_dispatcher(config, dispatcher)?;

    println!("{} Server ready!", "✓".green().bold());
    println!();
    println!("{}", "Available functions:".yellow().bold());
    println!("  • generate(prompt, options) - Generate text");
    println!("  • pipeline(pipeline, input) - Execute multi-model pipeline");
    println!("  • health() - Health check");
    println!("  • metadata() - List available functions");
    println!();
    println!("{}", "Connect with wscat:".yellow().bold());
    println!("  wscat -c ws://{}:{}", host, port);
    println!();
    println!("{}", "Example function call:".yellow().bold());
    println!(
        r#"  {{"id":"req_1","function":"generate","params":{{"prompt":"Hello","max_tokens":50}}}}"#
    );
    println!();
    println!("{} Press Ctrl+C to stop", "ℹ".blue());
    println!();

    // Run server
    server.run().await?;

    Ok(())
}

fn cmd_info() -> Result<()> {
    info!("ℹ️  {}", "System Information".green().bold());
    println!();

    println!("{}", "Realm Version:".yellow().bold());
    println!("  {}", env!("CARGO_PKG_VERSION"));
    println!();

    println!("{}", "Runtime:".yellow().bold());
    println!("  OS: {}", std::env::consts::OS);
    println!("  Arch: {}", std::env::consts::ARCH);
    println!("  Cores: {}", num_cpus());
    println!();

    println!("{}", "Features:".yellow().bold());
    println!("  {} WASM support", "✓".green());
    println!("  {} Memory64 support", "✓".green());
    println!("  {} CPU backend", "✓".green());

    #[cfg(feature = "cuda")]
    println!("  {} CUDA support", "✓".green());
    #[cfg(not(feature = "cuda"))]
    println!("  {} CUDA support", "✗".red());

    #[cfg(feature = "metal")]
    println!("  {} Metal support", "✓".green());
    #[cfg(not(feature = "metal"))]
    println!("  {} Metal support", "✗".red());

    println!();
    println!("{}", "Crates:".yellow().bold());
    println!("  realm-core");
    println!("  realm-models");
    println!("  realm-compute-cpu");
    println!("  realm-compute-gpu");
    println!("  realm-runtime");
    println!("  realm-wasm");

    Ok(())
}

fn cmd_bench(model: PathBuf, prompt_len: usize, gen_len: usize, gpu: bool) -> Result<()> {
    info!("⚡ {}", "Running benchmark".green().bold());
    info!("📦 Model: {}", model.display());
    info!("📏 Prompt length: {} tokens", prompt_len);
    info!("🎯 Generation length: {} tokens", gen_len);
    info!("💻 Backend: {}", if gpu { "GPU" } else { "CPU" });
    println!();

    if !model.exists() {
        eprintln!(
            "{} Model file not found: {}",
            "✗".red().bold(),
            model.display()
        );
        return Ok(());
    }

    println!("{} Implementation coming soon!", "ℹ".blue());
    println!("This will benchmark:");
    println!("  • Model loading time");
    println!("  • Prompt processing speed (tokens/sec)");
    println!("  • Generation speed (tokens/sec)");
    println!("  • Memory usage");
    println!("  • TTFT (Time To First Token)");

    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn cmd_api_key(command: ApiKeyCommands) -> Result<()> {
    use realm_server::auth::{generate_api_key, ApiKey, ApiKeyStore, ApiKeyStoreConfig};

    match command {
        ApiKeyCommands::Generate { tenant, name, file } => {
            info!("Generating API key for tenant: {}", tenant);

            // Generate key
            let key_value = generate_api_key("sk_live");
            let mut api_key = ApiKey::new(&key_value, &tenant);

            if let Some(ref key_name) = name {
                api_key = api_key.with_name(key_name);
            }

            // Load or create store
            let config = ApiKeyStoreConfig {
                keys_file: Some(file.clone()),
                in_memory_only: false,
                ..Default::default()
            };

            let store = ApiKeyStore::new(config)?;
            store.add_key(api_key)?;

            println!("{} API key generated successfully!", "✓".green().bold());
            println!();
            println!("{}", "Details:".yellow().bold());
            println!("  Tenant ID: {}", tenant);
            if let Some(n) = name {
                println!("  Name: {}", n);
            }
            println!("  Key: {}", key_value.cyan());
            println!("  File: {}", file.display());
            println!();
            println!(
                "{}",
                "To authenticate WebSocket connections:".yellow().bold()
            );
            println!(r#"  {{"type":"auth","api_key":"{}"}}"#, key_value);
            println!();
            println!("{} Keep this key secure!", "⚠".yellow());

            Ok(())
        }

        ApiKeyCommands::List { file, tenant } => {
            if !file.exists() {
                eprintln!(
                    "{} API keys file not found: {}",
                    "✗".red().bold(),
                    file.display()
                );
                return Ok(());
            }

            info!("Listing API keys from: {:?}", file);

            let config = ApiKeyStoreConfig {
                keys_file: Some(file.clone()),
                in_memory_only: false,
                ..Default::default()
            };

            let store = ApiKeyStore::new(config)?;

            let keys = if let Some(ref tenant_id) = tenant {
                store.list_tenant_keys(tenant_id)
            } else {
                store.list_all_keys()
            };

            if keys.is_empty() {
                println!("{} No API keys found", "ℹ".blue());
                return Ok(());
            }

            println!("{}", "API Keys:".green().bold());
            println!();

            for key in keys {
                let status = if key.enabled {
                    "enabled".green()
                } else {
                    "disabled".red()
                };

                println!("{} Tenant: {}", "•".cyan(), key.tenant_id);
                if let Some(ref name) = key.name {
                    println!("  Name: {}", name);
                }
                println!("  Key: {}", key.key);
                println!("  Status: {}", status);
                if let Some(last_used) = key.last_used {
                    use chrono::{DateTime, Utc};
                    let dt: DateTime<Utc> =
                        DateTime::from_timestamp(last_used, 0).unwrap_or_else(Utc::now);
                    println!("  Last used: {}", dt.format("%Y-%m-%d %H:%M:%S UTC"));
                }
                println!();
            }

            Ok(())
        }

        ApiKeyCommands::Disable { key, file } => {
            if !file.exists() {
                eprintln!(
                    "{} API keys file not found: {}",
                    "✗".red().bold(),
                    file.display()
                );
                return Ok(());
            }

            let config = ApiKeyStoreConfig {
                keys_file: Some(file.clone()),
                in_memory_only: false,
                ..Default::default()
            };

            let store = ApiKeyStore::new(config)?;
            store.disable_key(&key)?;

            println!("{} API key disabled: {}", "✓".green().bold(), key);

            Ok(())
        }

        ApiKeyCommands::Enable { key, file } => {
            if !file.exists() {
                eprintln!(
                    "{} API keys file not found: {}",
                    "✗".red().bold(),
                    file.display()
                );
                return Ok(());
            }

            let config = ApiKeyStoreConfig {
                keys_file: Some(file.clone()),
                in_memory_only: false,
                ..Default::default()
            };

            let store = ApiKeyStore::new(config)?;
            store.enable_key(&key)?;

            println!("{} API key enabled: {}", "✓".green().bold(), key);

            Ok(())
        }
    }
}

fn cmd_models(command: ModelsCommands) -> Result<()> {
    use realm_models::registry::{ModelCapability, ModelRegistry, RegistryConfig};

    let config = RegistryConfig::default();
    let registry = ModelRegistry::new(config)?;

    match command {
        ModelsCommands::List { source, capability } => {
            info!("Listing models...");

            let mut models = if let Some(cap_str) = capability {
                let cap = match cap_str.to_lowercase().as_str() {
                    "chat" => ModelCapability::Chat,
                    "completion" => ModelCapability::Completion,
                    "embedding" => ModelCapability::Embedding,
                    "code" | "code-completion" => ModelCapability::CodeCompletion,
                    "instruction" => ModelCapability::Instruction,
                    _ => {
                        eprintln!("{} Unknown capability: {}", "✗".red().bold(), cap_str);
                        eprintln!("Valid capabilities: chat, completion, embedding, code-completion, instruction");
                        return Ok(());
                    }
                };
                registry.filter_by_capability(cap)
            } else {
                registry.list()
            };

            // Filter by source if specified
            if let Some(ref src) = source {
                models.retain(|m| match src.to_lowercase().as_str() {
                    "ollama" => {
                        matches!(m.source, realm_models::registry::ModelSource::Ollama { .. })
                    }
                    "huggingface" | "hf" => matches!(
                        m.source,
                        realm_models::registry::ModelSource::HuggingFace { .. }
                    ),
                    "http" => matches!(m.source, realm_models::registry::ModelSource::Http { .. }),
                    "local" => {
                        matches!(m.source, realm_models::registry::ModelSource::Local { .. })
                    }
                    "all" => true,
                    _ => true,
                });
            }

            if models.is_empty() {
                println!("{} No models found", "ℹ".blue());
                return Ok(());
            }

            println!("{}", "Available Models:".green().bold());
            println!();

            for model in models {
                let cached = if registry.is_cached(&model.id) {
                    "✓".green()
                } else {
                    " ".normal()
                };

                println!(
                    "{} {} {}",
                    cached,
                    model.id.cyan().bold(),
                    format!("({}B)", model.parameters).dimmed()
                );
                println!("   {}", model.description);
                println!(
                    "   Capabilities: {}",
                    model
                        .capabilities
                        .iter()
                        .map(|c| format!("{:?}", c))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                println!(
                    "   Quantizations: {}",
                    model
                        .quantizations
                        .iter()
                        .map(|q| q.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                println!();
            }

            println!("{}", "Usage:".yellow().bold());
            println!("  realm serve --model llama-2-7b-chat:Q4_K_M");
            println!();
            println!("{} Models marked with ✓ are cached locally", "ℹ".blue());

            Ok(())
        }

        ModelsCommands::Search { query } => {
            info!("Searching for: {}", query);

            let models = registry.search(&query);

            if models.is_empty() {
                println!("{} No models found matching '{}'", "ℹ".blue(), query);
                return Ok(());
            }

            println!("{} {}", "Search results for:".green().bold(), query.cyan());
            println!();

            for model in models {
                let cached = if registry.is_cached(&model.id) {
                    "✓".green()
                } else {
                    " ".normal()
                };

                println!("{} {}", cached, model.id.cyan().bold());
                println!("   {}", model.description);
                println!();
            }

            Ok(())
        }

        ModelsCommands::Info { model } => {
            let entry = registry.get(&model);

            if let Some(entry) = entry {
                println!("{}", "Model Information:".green().bold());
                println!();
                println!("  ID: {}", entry.id.cyan());
                println!("  Name: {}", entry.name);
                println!("  Family: {}", entry.family);
                println!("  Parameters: {}B", entry.parameters);
                println!("  Context Length: {}", entry.context_length);
                println!("  License: {}", entry.license);
                println!();
                println!("  Description:");
                println!("    {}", entry.description);
                println!();
                println!("  Capabilities:");
                for cap in &entry.capabilities {
                    println!("    • {:?}", cap);
                }
                println!();
                println!("  Available Quantizations:");
                for quant in &entry.quantizations {
                    println!("    • {}", quant.as_str());
                }
                println!();
                println!("  Tags: {}", entry.tags.join(", "));
                println!();
                println!("{}", "Usage:".yellow().bold());
                println!("  realm serve --model {}:Q4_K_M", entry.id);
            } else {
                eprintln!("{} Model not found: {}", "✗".red().bold(), model);
                eprintln!();
                eprintln!("Try: realm models search {}", model);
            }

            Ok(())
        }

        ModelsCommands::Status { model } => {
            match registry.resolve(&model) {
                Ok(path) => {
                    println!("{} Model is cached", "✓".green().bold());
                    println!("  Path: {}", path.display());

                    if let Ok(metadata) = std::fs::metadata(&path) {
                        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                        println!("  Size: {:.2} MB", size_mb);
                    }
                }
                Err(e) => {
                    println!("{} Model is not cached", "✗".yellow());
                    println!("  {}", e);
                    println!();
                    println!("{}", "To use this model:".yellow().bold());
                    println!(
                        "  1. Download manually to: {}",
                        registry.cache_dir().display()
                    );
                    println!("  2. Or use: realm models download {}", model);
                }
            }

            Ok(())
        }

        ModelsCommands::Download { model, output } => {
            // Create registry with custom output if specified
            let mut registry_config = RegistryConfig::default();
            if let Some(output_dir) = output {
                registry_config.cache_dir = output_dir;
            }
            let registry = ModelRegistry::new(registry_config)?;

            // Check if already cached
            if registry.is_cached(&model) {
                let cached_path = registry.resolve(&model)?;
                println!("{} Model already cached", "✓".green().bold());
                println!("  Path: {}", cached_path.display());

                if let Ok(metadata) = std::fs::metadata(&cached_path) {
                    let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                    println!("  Size: {:.2} MB", size_mb);
                }
                return Ok(());
            }

            // Download
            println!("{} Downloading model: {}", "→".cyan(), model);
            println!();

            let rt = tokio::runtime::Runtime::new()?;
            match rt.block_on(registry.download_model(&model)) {
                Ok(path) => {
                    println!();
                    println!("{} Download complete!", "✓".green().bold());
                    println!("  Path: {}", path.display());

                    if let Ok(metadata) = std::fs::metadata(&path) {
                        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                        println!("  Size: {:.2} MB", size_mb);
                    }

                    println!();
                    println!("{} You can now use this model:", "ℹ".blue());
                    println!("  realm serve --wasm realm.wasm --model {}", model);
                }
                Err(e) => {
                    eprintln!("{} Download failed: {}", "✗".red().bold(), e);
                    eprintln!();
                    eprintln!("{} Available options:", "ℹ".blue());
                    eprintln!("  1. Check if model exists: realm models list");
                    eprintln!("  2. Download manually from HuggingFace Hub");
                    eprintln!("  3. Provide direct path to model file");
                    return Err(e);
                }
            }

            Ok(())
        }
    }
}

fn cmd_pipeline(command: PipelineCommands) -> Result<()> {
    use colored::Colorize;
    use realm_server::pipeline_dsl::PipelineDef;

    match command {
        PipelineCommands::List { dir } => {
            let pipelines_dir = dir.unwrap_or_else(|| PathBuf::from("./pipelines"));

            println!();
            println!("{}", "Available Pipelines:".cyan().bold());
            println!();

            if !pipelines_dir.exists() {
                println!(
                    "  {} No pipelines directory found at: {}",
                    "ℹ".blue(),
                    pipelines_dir.display()
                );
                println!();
                println!("{}", "Try creating a pipeline:".yellow().bold());
                println!("  mkdir -p pipelines");
                println!("  realm pipeline info examples/pipelines/simple-chat.yaml");
                return Ok(());
            }

            // Find all YAML and JSON files
            let entries = std::fs::read_dir(&pipelines_dir)?;
            let mut pipeline_files = Vec::new();

            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "yaml" || ext == "yml" || ext == "json" {
                            pipeline_files.push(path);
                        }
                    }
                }
            }

            if pipeline_files.is_empty() {
                println!(
                    "  {} No pipeline files found in: {}",
                    "ℹ".blue(),
                    pipelines_dir.display()
                );
                return Ok(());
            }

            // Display each pipeline
            for (i, path) in pipeline_files.iter().enumerate() {
                if i > 0 {
                    println!();
                }

                // Try to load and display pipeline info
                let pipeline = if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    PipelineDef::from_json(path)
                } else {
                    PipelineDef::from_yaml(path)
                };

                match pipeline {
                    Ok(p) => {
                        println!(
                            "  {} {}",
                            p.id.cyan().bold(),
                            path.file_name().unwrap().to_string_lossy().dimmed()
                        );
                        println!("     {}", p.description);
                        println!("     {} steps", p.steps.len());
                    }
                    Err(_) => {
                        println!(
                            "  {} {}",
                            "⚠".yellow(),
                            path.file_name().unwrap().to_string_lossy()
                        );
                        println!("     Invalid pipeline file");
                    }
                }
            }

            println!();
            println!("{}", "Usage:".yellow().bold());
            println!("  realm pipeline info <file>");
            println!("  realm pipeline validate <file>");

            Ok(())
        }

        PipelineCommands::Info { file } => {
            let pipeline = if file.extension().and_then(|s| s.to_str()) == Some("json") {
                PipelineDef::from_json(file)?
            } else {
                PipelineDef::from_yaml(file)?
            };

            println!();
            println!("{}", "Pipeline Information:".cyan().bold());
            println!();
            println!("  ID: {}", pipeline.id.cyan().bold());
            println!("  Name: {}", pipeline.name);
            println!("  Description: {}", pipeline.description);
            println!();
            println!("  {} Steps:", "Steps:".yellow().bold());
            for (i, step) in pipeline.steps.iter().enumerate() {
                println!();
                println!("    {}. {} ({})", i + 1, step.name.bold(), step.id.dimmed());
                match &step.model {
                    realm_server::pipeline_dsl::ModelSpec::Id { model } => {
                        println!("       Model: {}", model.cyan());
                    }
                    realm_server::pipeline_dsl::ModelSpec::Type { model_type } => {
                        println!("       Model Type: {}", model_type.yellow());
                    }
                }
                println!("       Output: {}", step.output.green());
            }
            println!();

            // Validate
            match pipeline.validate() {
                Ok(_) => {
                    println!("{} Pipeline is valid", "✓".green().bold());
                }
                Err(e) => {
                    println!("{} Validation error: {}", "✗".red().bold(), e);
                }
            }

            Ok(())
        }

        PipelineCommands::Validate { file } => {
            let pipeline = if file.extension().and_then(|s| s.to_str()) == Some("json") {
                PipelineDef::from_json(file)?
            } else {
                PipelineDef::from_yaml(file)?
            };

            match pipeline.validate() {
                Ok(_) => {
                    println!(
                        "{} Pipeline is valid: {}",
                        "✓".green().bold(),
                        pipeline.id.cyan().bold()
                    );
                    println!("  {} steps", pipeline.steps.len());
                }
                Err(e) => {
                    eprintln!("{} Validation failed: {}", "✗".red().bold(), e);
                    std::process::exit(1);
                }
            }

            Ok(())
        }

        PipelineCommands::Load { file } => {
            let pipeline = if file.extension().and_then(|s| s.to_str()) == Some("json") {
                PipelineDef::from_json(file)?
            } else {
                PipelineDef::from_yaml(file)?
            };

            // Validate before loading
            pipeline.validate()?;

            // Convert to orchestrator pipeline
            let _orch_pipeline = pipeline.to_pipeline()?;

            println!(
                "{} Pipeline loaded successfully: {}",
                "✓".green().bold(),
                pipeline.id.cyan().bold()
            );
            println!();
            println!("{}", "Note:".yellow().bold());
            println!("  Pipeline is validated but not registered with a running server.");
            println!("  To use this pipeline:");
            println!("    1. Start a server with: realm serve");
            println!("    2. Register the pipeline via WebSocket API");

            Ok(())
        }
    }
}
