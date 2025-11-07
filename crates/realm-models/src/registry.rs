//! Model Registry
//!
//! Automatic model discovery, downloading, and caching from various sources.

use anyhow::{anyhow, Context, Result};
#[cfg(feature = "download")]
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Model source type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ModelSource {
    /// Ollama model (uses Ollama's local storage)
    Ollama { model_name: String },

    /// HuggingFace model (downloads from HF Hub)
    HuggingFace { repo_id: String, filename: String },

    /// Direct HTTP URL
    Http { url: String },

    /// Local file path
    Local { path: PathBuf },
}

/// Quantization format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(non_camel_case_types)]
pub enum Quantization {
    F32,
    F16,
    Q8_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ4_NL,
    IQ4_XS,
}

impl Quantization {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "Q8_0" => Some(Self::Q8_0),
            "Q4_0" => Some(Self::Q4_0),
            "Q4_1" => Some(Self::Q4_1),
            "Q5_0" => Some(Self::Q5_0),
            "Q5_1" => Some(Self::Q5_1),
            "Q2_K" | "Q2_K_M" => Some(Self::Q2_K),
            "Q3_K" | "Q3_K_M" | "Q3_K_S" => Some(Self::Q3_K),
            "Q4_K" | "Q4_K_M" | "Q4_K_S" => Some(Self::Q4_K),
            "Q5_K" | "Q5_K_M" | "Q5_K_S" => Some(Self::Q5_K),
            "Q6_K" => Some(Self::Q6_K),
            "Q8_K" => Some(Self::Q8_K),
            "IQ4_NL" => Some(Self::IQ4_NL),
            "IQ4_XS" => Some(Self::IQ4_XS),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q8_0 => "Q8_0",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ4_XS => "IQ4_XS",
        }
    }
}

/// Model capability
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ModelCapability {
    Completion,
    Chat,
    Embedding,
    CodeCompletion,
    Instruction,
}

/// Model entry in the registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Unique model identifier (e.g., "llama-2-7b-chat")
    pub id: String,

    /// Display name
    pub name: String,

    /// Model description
    pub description: String,

    /// Model family (e.g., "llama", "mistral", "phi")
    pub family: String,

    /// Parameter count (in billions)
    pub parameters: f32,

    /// Context length
    pub context_length: usize,

    /// Available quantizations
    pub quantizations: Vec<Quantization>,

    /// Model capabilities
    pub capabilities: Vec<ModelCapability>,

    /// Model source
    pub source: ModelSource,

    /// License (e.g., "MIT", "Apache-2.0", "Llama-2")
    pub license: String,

    /// Tags for search
    pub tags: Vec<String>,
}

impl ModelEntry {
    /// Get the file path for a specific quantization
    pub fn get_filename(&self, quant: &Quantization) -> String {
        format!("{}.{}.gguf", self.id, quant.as_str())
    }
}

/// Model registry configuration
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Cache directory for downloaded models
    pub cache_dir: PathBuf,

    /// Registry file path (JSON)
    pub registry_file: Option<PathBuf>,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        let cache_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".realm")
            .join("models");

        Self {
            cache_dir,
            registry_file: None,
        }
    }
}

/// Model registry
pub struct ModelRegistry {
    /// Configuration
    config: RegistryConfig,

    /// Registered models (id -> entry)
    models: HashMap<String, ModelEntry>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new(config: RegistryConfig) -> Result<Self> {
        // Ensure cache directory exists
        std::fs::create_dir_all(&config.cache_dir)
            .context("Failed to create model cache directory")?;

        let mut registry = Self {
            config,
            models: HashMap::new(),
        };

        // Load built-in models
        registry.load_builtin_models();

        // Load from file if configured
        let registry_file_path = registry.config.registry_file.clone();
        if let Some(registry_file) = registry_file_path {
            if registry_file.exists() {
                registry.load_from_file(&registry_file)?;
            }
        }

        Ok(registry)
    }

    /// Load built-in model definitions
    fn load_builtin_models(&mut self) {
        // Llama 2 models
        self.register_model(ModelEntry {
            id: "llama-2-7b".to_string(),
            name: "Llama 2 7B".to_string(),
            description: "Meta's Llama 2 7B base model".to_string(),
            family: "llama".to_string(),
            parameters: 7.0,
            context_length: 4096,
            quantizations: vec![Quantization::Q4_K, Quantization::Q5_K, Quantization::Q8_0],
            capabilities: vec![ModelCapability::Completion],
            source: ModelSource::HuggingFace {
                repo_id: "TheBloke/Llama-2-7B-GGUF".to_string(),
                filename: "llama-2-7b.Q4_K_M.gguf".to_string(),
            },
            license: "Llama-2".to_string(),
            tags: vec!["llama".to_string(), "meta".to_string(), "7b".to_string()],
        });

        self.register_model(ModelEntry {
            id: "llama-2-7b-chat".to_string(),
            name: "Llama 2 7B Chat".to_string(),
            description: "Meta's Llama 2 7B chat-tuned model".to_string(),
            family: "llama".to_string(),
            parameters: 7.0,
            context_length: 4096,
            quantizations: vec![Quantization::Q4_K, Quantization::Q5_K, Quantization::Q8_0],
            capabilities: vec![ModelCapability::Chat, ModelCapability::Instruction],
            source: ModelSource::HuggingFace {
                repo_id: "TheBloke/Llama-2-7B-Chat-GGUF".to_string(),
                filename: "llama-2-7b-chat.Q4_K_M.gguf".to_string(),
            },
            license: "Llama-2".to_string(),
            tags: vec![
                "llama".to_string(),
                "meta".to_string(),
                "chat".to_string(),
                "7b".to_string(),
            ],
        });

        // Mistral models
        self.register_model(ModelEntry {
            id: "mistral-7b-instruct".to_string(),
            name: "Mistral 7B Instruct".to_string(),
            description: "Mistral AI's 7B instruction-following model".to_string(),
            family: "mistral".to_string(),
            parameters: 7.0,
            context_length: 8192,
            quantizations: vec![Quantization::Q4_K, Quantization::Q5_K, Quantization::Q8_0],
            capabilities: vec![ModelCapability::Chat, ModelCapability::Instruction],
            source: ModelSource::HuggingFace {
                repo_id: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF".to_string(),
                filename: "mistral-7b-instruct-v0.2.Q4_K_M.gguf".to_string(),
            },
            license: "Apache-2.0".to_string(),
            tags: vec![
                "mistral".to_string(),
                "instruct".to_string(),
                "7b".to_string(),
            ],
        });

        // TinyLlama
        self.register_model(ModelEntry {
            id: "tinyllama-1.1b".to_string(),
            name: "TinyLlama 1.1B".to_string(),
            description: "Small, fast Llama-based model for testing".to_string(),
            family: "llama".to_string(),
            parameters: 1.1,
            context_length: 2048,
            quantizations: vec![Quantization::Q4_K, Quantization::Q5_K, Quantization::Q8_0],
            capabilities: vec![ModelCapability::Chat, ModelCapability::Completion],
            source: ModelSource::HuggingFace {
                repo_id: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
                filename: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string(),
            },
            license: "Apache-2.0".to_string(),
            tags: vec![
                "tinyllama".to_string(),
                "small".to_string(),
                "1b".to_string(),
            ],
        });

        // Phi-2
        self.register_model(ModelEntry {
            id: "phi-2".to_string(),
            name: "Phi-2".to_string(),
            description: "Microsoft's 2.7B parameter efficient model".to_string(),
            family: "phi".to_string(),
            parameters: 2.7,
            context_length: 2048,
            quantizations: vec![Quantization::Q4_K, Quantization::Q5_K, Quantization::Q8_0],
            capabilities: vec![ModelCapability::Completion, ModelCapability::CodeCompletion],
            source: ModelSource::HuggingFace {
                repo_id: "TheBloke/phi-2-GGUF".to_string(),
                filename: "phi-2.Q4_K_M.gguf".to_string(),
            },
            license: "MIT".to_string(),
            tags: vec![
                "phi".to_string(),
                "microsoft".to_string(),
                "code".to_string(),
            ],
        });

        info!("Loaded {} built-in models", self.models.len());
    }

    /// Register a model
    fn register_model(&mut self, entry: ModelEntry) {
        debug!("Registering model: {} ({})", entry.id, entry.name);
        self.models.insert(entry.id.clone(), entry);
    }

    /// Load models from JSON file
    fn load_from_file(&mut self, path: &Path) -> Result<()> {
        let content = std::fs::read_to_string(path)?;
        let entries: Vec<ModelEntry> = serde_json::from_str(&content)?;

        for entry in entries {
            self.register_model(entry);
        }

        info!("Loaded models from {:?}", path);
        Ok(())
    }

    /// Get a model by ID
    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.models.get(id)
    }

    /// List all models
    pub fn list(&self) -> Vec<&ModelEntry> {
        self.models.values().collect()
    }

    /// Search models by query
    pub fn search(&self, query: &str) -> Vec<&ModelEntry> {
        let query = query.to_lowercase();
        self.models
            .values()
            .filter(|entry| {
                entry.id.to_lowercase().contains(&query)
                    || entry.name.to_lowercase().contains(&query)
                    || entry.description.to_lowercase().contains(&query)
                    || entry
                        .tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(&query))
            })
            .collect()
    }

    /// Filter models by capability
    pub fn filter_by_capability(&self, capability: ModelCapability) -> Vec<&ModelEntry> {
        self.models
            .values()
            .filter(|entry| entry.capabilities.contains(&capability))
            .collect()
    }

    /// Resolve a model specification to a file path
    /// Spec format: "model-id" or "model-id:quantization"
    pub fn resolve(&self, spec: &str) -> Result<PathBuf> {
        // Parse spec: "llama-2-7b-chat:Q4_K_M" or just "llama-2-7b-chat"
        let (model_id, quant) = if let Some((id, q)) = spec.split_once(':') {
            let quant =
                Quantization::parse(q).ok_or_else(|| anyhow!("Unknown quantization: {}", q))?;
            (id, Some(quant))
        } else {
            (spec, None)
        };

        // Get model entry
        let entry = self
            .get(model_id)
            .ok_or_else(|| anyhow!("Model not found in registry: {}", model_id))?;

        // Use specified quantization or default to first available
        let quantization = quant.unwrap_or_else(|| entry.quantizations[0].clone());

        // Check if model is already cached
        let filename = entry.get_filename(&quantization);
        let cached_path = self.config.cache_dir.join(&filename);

        if cached_path.exists() {
            debug!("Model found in cache: {:?}", cached_path);
            return Ok(cached_path);
        }

        // Model not cached - need to download
        Err(anyhow!(
            "Model not cached: {}. Run 'realm models download {}' to download it.",
            spec,
            spec
        ))
    }

    /// Download a model from its source to the cache directory
    #[cfg(feature = "download")]
    pub async fn download_model(&self, spec: &str) -> Result<PathBuf> {
        // Parse spec and get model entry
        let (model_id, quant) = if let Some((id, q)) = spec.split_once(':') {
            let quant =
                Quantization::parse(q).ok_or_else(|| anyhow!("Unknown quantization: {}", q))?;
            (id, Some(quant))
        } else {
            (spec, None)
        };

        let entry = self
            .get(model_id)
            .ok_or_else(|| anyhow!("Model not found in registry: {}", model_id))?;

        let quantization = quant.unwrap_or_else(|| entry.quantizations[0].clone());
        let filename = entry.get_filename(&quantization);
        let cached_path = self.config.cache_dir.join(&filename);

        // Check if already cached
        if cached_path.exists() {
            info!("Model already cached: {:?}", cached_path);
            return Ok(cached_path);
        }

        // Ensure cache directory exists
        std::fs::create_dir_all(&self.config.cache_dir)
            .context("Failed to create cache directory")?;

        info!(
            "Downloading model: {} ({})",
            model_id,
            quantization.as_str()
        );
        info!("Destination: {:?}", cached_path);

        // Download based on source type
        #[cfg(feature = "download")]
        {
            match &entry.source {
                ModelSource::HuggingFace {
                    repo_id,
                    filename: hf_filename,
                } => {
                    self.download_from_huggingface(repo_id, hf_filename, &cached_path)
                        .await?;
                }
                ModelSource::Http { url } => {
                    self.download_from_url(url, &cached_path).await?;
                }
                ModelSource::Local { path } => {
                    // Copy local file to cache
                    std::fs::copy(path, &cached_path)
                        .with_context(|| format!("Failed to copy local file: {:?}", path))?;
                    info!("Copied local model to cache");
                }
                ModelSource::Ollama { model_name } => {
                    return Err(anyhow!(
                        "Ollama models must be pulled manually: ollama pull {}",
                        model_name
                    ));
                }
            }
        }

        #[cfg(not(feature = "download"))]
        {
            return Err(anyhow!(
                "Download feature not enabled. Enable with 'download' feature flag."
            ));
        }

        info!("✓ Model downloaded successfully: {:?}", cached_path);
        Ok(cached_path)
    }

    /// Download a file from HuggingFace Hub
    #[cfg(feature = "download")]
    async fn download_from_huggingface(
        &self,
        repo_id: &str,
        filename: &str,
        output_path: &Path,
    ) -> Result<()> {
        // HuggingFace Hub URL format
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, filename
        );

        info!("Downloading from HuggingFace: {}", url);
        self.download_from_url(&url, output_path).await
    }

    /// Download a file from HTTP URL with progress
    #[cfg(feature = "download")]
    async fn download_from_url(&self, url: &str, output_path: &Path) -> Result<()> {
        use reqwest::Client;
        use std::io::Write;

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(3600)) // 1 hour timeout for large files
            .build()
            .context("Failed to create HTTP client")?;

        let response = client
            .get(url)
            .send()
            .await
            .context("Failed to send HTTP request")?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "HTTP request failed: {} {}",
                response.status(),
                response.status().canonical_reason().unwrap_or("Unknown")
            ));
        }

        let total_size = response.content_length().unwrap_or(0);

        info!(
            "Download size: {} bytes ({:.2} MB)",
            total_size,
            total_size as f64 / 1_048_576.0
        );

        let mut file = std::fs::File::create(output_path)
            .with_context(|| format!("Failed to create output file: {:?}", output_path))?;

        let mut stream = response.bytes_stream();
        let mut downloaded: u64 = 0;
        let mut last_progress = std::time::Instant::now();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.context("Failed to read chunk")?;

            file.write_all(&chunk)
                .context("Failed to write chunk to file")?;

            downloaded += chunk.len() as u64;

            // Print progress every 10MB or every 5 seconds
            if last_progress.elapsed().as_secs() >= 5 || downloaded.is_multiple_of(10_485_760) {
                if total_size > 0 {
                    let percent = (downloaded as f64 / total_size as f64) * 100.0;
                    info!(
                        "Download progress: {:.1}% ({:.2} MB / {:.2} MB)",
                        percent,
                        downloaded as f64 / 1_048_576.0,
                        total_size as f64 / 1_048_576.0
                    );
                } else {
                    info!("Downloaded: {:.2} MB", downloaded as f64 / 1_048_576.0);
                }
                last_progress = std::time::Instant::now();
            }
        }

        info!(
            "✓ Download complete: {:.2} MB",
            downloaded as f64 / 1_048_576.0
        );
        Ok(())
    }

    #[cfg(not(feature = "download"))]
    #[allow(dead_code)] // Stub implementation when download feature is disabled
    async fn download_from_url(&self, _url: &str, _output_path: &Path) -> Result<()> {
        Err(anyhow!(
            "Download feature not enabled. Enable with 'download' feature flag."
        ))
    }

    #[cfg(not(feature = "download"))]
    pub async fn download_model(&self, _spec: &str) -> Result<PathBuf> {
        Err(anyhow!(
            "Download feature not enabled. Enable with 'download' feature flag."
        ))
    }

    /// Check if a model is cached
    pub fn is_cached(&self, spec: &str) -> bool {
        self.resolve(spec).is_ok()
    }

    /// Get cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.config.cache_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let config = RegistryConfig::default();
        let registry = ModelRegistry::new(config).unwrap();

        assert!(!registry.list().is_empty());
    }

    #[test]
    fn test_model_lookup() {
        let config = RegistryConfig::default();
        let registry = ModelRegistry::new(config).unwrap();

        let model = registry.get("llama-2-7b-chat");
        assert!(model.is_some());
        assert_eq!(model.unwrap().parameters, 7.0);
    }

    #[test]
    fn test_model_search() {
        let config = RegistryConfig::default();
        let registry = ModelRegistry::new(config).unwrap();

        let results = registry.search("llama");
        assert!(!results.is_empty());

        let results = registry.search("chat");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_capability_filter() {
        let config = RegistryConfig::default();
        let registry = ModelRegistry::new(config).unwrap();

        let chat_models = registry.filter_by_capability(ModelCapability::Chat);
        assert!(!chat_models.is_empty());
    }

    #[test]
    fn test_quantization_parsing() {
        assert_eq!(Quantization::parse("Q4_K_M"), Some(Quantization::Q4_K));
        assert_eq!(Quantization::parse("Q5_K"), Some(Quantization::Q5_K));
        assert_eq!(Quantization::parse("invalid"), None);
    }
}
