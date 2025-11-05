//! Runtime Manager
//!
//! Manages per-tenant WASM runtime instances with model loading and inference.

use anyhow::{anyhow, Context, Result};
use realm_runtime::lora::LoRAManager;
use realm_runtime::HostContext;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};
use wasmtime::{Config, Engine, Instance, Linker, Module, Store};

/// Configuration for a model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to the GGUF model file
    pub model_path: PathBuf,

    /// Model ID for reference
    pub model_id: String,
}

/// WASM runtime instance for a tenant
pub struct TenantRuntime {
    /// Wasmtime store
    store: Store<()>,

    /// WASM instance
    instance: Instance,

    /// Host context for Memory64 and GPU operations
    host_context: Arc<HostContext>,

    /// Model configuration
    model_config: Option<ModelConfig>,

    /// Tenant ID
    tenant_id: String,

    /// Optional LoRA adapter ID for this tenant
    #[allow(dead_code)] // Used for future LoRA application
    lora_adapter_id: Option<String>,
}

impl TenantRuntime {
    /// Create a new tenant runtime
    pub fn new(
        engine: &Engine,
        wasm_module: &Module,
        tenant_id: impl Into<String>,
    ) -> Result<Self> {
        let tenant_id = tenant_id.into();
        debug!("Creating runtime for tenant: {}", tenant_id);

        // Create host context
        let host_context = Arc::new(HostContext::new());

        // Create linker and add host functions
        let mut linker = Linker::new(engine);
        host_context
            .add_to_linker(&mut linker)
            .context("Failed to add host functions to linker")?;

        // Create store
        let mut store = Store::new(engine, ());

        // Initialize host context
        host_context
            .initialize(&mut store)
            .context("Failed to initialize host context")?;

        // Instantiate WASM module
        let instance = linker
            .instantiate(&mut store, wasm_module)
            .context("Failed to instantiate WASM module")?;

        info!("Runtime created for tenant: {}", tenant_id);

        Ok(Self {
            store,
            instance,
            host_context,
            model_config: None,
            tenant_id,
            lora_adapter_id: None,
        })
    }

    /// Load a model into this runtime
    pub fn load_model(&mut self, config: ModelConfig) -> Result<()> {
        info!(
            "Loading model for tenant {}: {:?}",
            self.tenant_id, config.model_path
        );

        // Read model bytes
        let model_bytes = std::fs::read(&config.model_path)
            .with_context(|| format!("Failed to read model file: {:?}", config.model_path))?;

        debug!(
            "Read {} bytes from model file for tenant {}",
            model_bytes.len(),
            self.tenant_id
        );

        // Get the loadModel function from WASM
        let load_model = self
            .instance
            .get_typed_func::<(u32, u32), ()>(&mut self.store, "loadModel")
            .context("Failed to get loadModel function from WASM")?;

        // Allocate memory in WASM for model bytes
        let memory = self
            .instance
            .get_memory(&mut self.store, "memory")
            .context("Failed to get WASM memory")?;

        // Get current memory size
        let mem_size = memory.data_size(&self.store);
        debug!("Current WASM memory size: {} bytes", mem_size);

        // Grow memory if needed
        let needed_pages = (model_bytes.len() / 65536) + 1;
        let current_pages = memory.size(&self.store);

        if needed_pages > current_pages as usize {
            let pages_to_grow = needed_pages - current_pages as usize;
            memory
                .grow(&mut self.store, pages_to_grow as u64)
                .with_context(|| {
                    format!("Failed to grow WASM memory by {} pages", pages_to_grow)
                })?;
            debug!(
                "Grew WASM memory from {} to {} pages",
                current_pages,
                current_pages + pages_to_grow as u64
            );
        }

        // Write model bytes to WASM memory
        // We'll use a simple offset at the beginning of memory
        let model_ptr = 0u32;
        memory
            .write(&mut self.store, model_ptr as usize, &model_bytes)
            .context("Failed to write model bytes to WASM memory")?;

        debug!("Wrote model bytes to WASM memory at offset 0");

        // Call loadModel(ptr, len)
        load_model
            .call(&mut self.store, (model_ptr, model_bytes.len() as u32))
            .context("loadModel function call failed")?;

        info!("Model loaded successfully for tenant: {}", self.tenant_id);

        self.model_config = Some(config);
        Ok(())
    }

    /// Generate text from a prompt
    pub fn generate(&mut self, prompt: String) -> Result<String> {
        if self.model_config.is_none() {
            return Err(anyhow!("No model loaded. Call load_model() first."));
        }

        debug!("Generating for tenant {}: '{}'", self.tenant_id, prompt);

        // Get the generate function from WASM
        let generate = self
            .instance
            .get_typed_func::<(u32, u32), u32>(&mut self.store, "generate")
            .context("Failed to get generate function from WASM")?;

        // Allocate memory for prompt
        let memory = self
            .instance
            .get_memory(&mut self.store, "memory")
            .context("Failed to get WASM memory")?;

        // Write prompt to memory (offset after model data)
        let prompt_bytes = prompt.as_bytes();
        let prompt_ptr = 1024 * 1024; // 1MB offset
        memory
            .write(&mut self.store, prompt_ptr, prompt_bytes)
            .context("Failed to write prompt to WASM memory")?;

        // Call generate(prompt_ptr, prompt_len) -> result_ptr
        let result_ptr = generate
            .call(
                &mut self.store,
                (prompt_ptr as u32, prompt_bytes.len() as u32),
            )
            .context("generate function call failed")?;

        // Read result from memory
        // The result should be a null-terminated string or we need to know the length
        // For simplicity, we'll read up to 10KB and find the null terminator
        let mut result_bytes = vec![0u8; 10240];
        memory
            .read(&self.store, result_ptr as usize, &mut result_bytes)
            .context("Failed to read result from WASM memory")?;

        // Find null terminator
        let result_len = result_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(result_bytes.len());
        let result_str = String::from_utf8_lossy(&result_bytes[..result_len]).to_string();

        debug!(
            "Generated {} characters for tenant {}",
            result_str.len(),
            self.tenant_id
        );

        Ok(result_str)
    }

    /// Check if model is loaded
    pub fn is_model_loaded(&self) -> bool {
        self.model_config.is_some()
    }

    /// Get tenant ID
    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    /// Get host context
    pub fn host_context(&self) -> &HostContext {
        &self.host_context
    }
}

/// Runtime manager for multiple tenants
pub struct RuntimeManager {
    /// Wasmtime engine (shared across all tenants)
    engine: Engine,

    /// Pre-compiled WASM module
    wasm_module: Module,

    /// Active runtimes per tenant
    runtimes: Arc<Mutex<HashMap<String, TenantRuntime>>>,

    /// Default model configuration
    default_model: Option<ModelConfig>,

    /// LoRA manager for per-tenant adapters
    lora_manager: Arc<LoRAManager>,

    /// Per-tenant LoRA adapter mappings
    tenant_lora_adapters: Arc<Mutex<HashMap<String, String>>>,
}

impl RuntimeManager {
    /// Create a new runtime manager
    pub fn new(wasm_path: impl Into<PathBuf>) -> Result<Self> {
        let wasm_path = wasm_path.into();

        info!("Initializing runtime manager with WASM: {:?}", wasm_path);

        // Configure Wasmtime
        let mut config = Config::new();
        config.wasm_bulk_memory(true);
        config.wasm_multi_memory(true);

        // Create engine
        let engine = Engine::new(&config).context("Failed to create Wasmtime engine")?;

        // Load and compile WASM module
        let wasm_module = Module::from_file(&engine, &wasm_path)
            .with_context(|| format!("Failed to load WASM module from {:?}", wasm_path))?;

        info!("WASM module loaded and compiled");

        Ok(Self {
            engine,
            wasm_module,
            runtimes: Arc::new(Mutex::new(HashMap::new())),
            default_model: None,
            lora_manager: Arc::new(LoRAManager::new()),
            tenant_lora_adapters: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Set default model configuration
    pub fn set_default_model(&mut self, config: ModelConfig) {
        info!("Set default model: {:?}", config.model_path);
        self.default_model = Some(config);
    }

    /// Validate tenant ID format (alphanumeric, hyphens, underscores, 3-64 chars)
    pub fn validate_tenant_id(tenant_id: &str) -> Result<()> {
        if tenant_id.len() < 3 || tenant_id.len() > 64 {
            return Err(anyhow!(
                "Tenant ID must be between 3 and 64 characters, got {}",
                tenant_id.len()
            ));
        }

        if !tenant_id
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            return Err(anyhow!(
                "Tenant ID must contain only alphanumeric characters, hyphens, and underscores"
            ));
        }

        Ok(())
    }

    /// Check if tenant ID is already in use
    pub fn is_tenant_id_taken(&self, tenant_id: impl AsRef<str>) -> bool {
        let runtimes = self.runtimes.lock().unwrap();
        runtimes.contains_key(tenant_id.as_ref())
    }

    /// Get or create a runtime for a tenant with validation
    /// Uses atomic HashMap::entry() to prevent race conditions
    pub fn get_or_create_runtime(&self, tenant_id: impl Into<String>) -> Result<()> {
        let tenant_id = tenant_id.into();

        // Validate tenant ID format
        Self::validate_tenant_id(&tenant_id)?;

        let mut runtimes = self.runtimes.lock().unwrap();

        // Use entry() API for atomic check-and-insert (prevents race conditions)
        match runtimes.entry(tenant_id.clone()) {
            std::collections::hash_map::Entry::Occupied(_) => {
                debug!("Runtime already exists for tenant: {}", tenant_id);
                return Ok(());
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                debug!("Creating new runtime for tenant: {}", tenant_id);

                // Create new runtime
                let mut runtime = TenantRuntime::new(&self.engine, &self.wasm_module, &tenant_id)?;

                // Load default model if configured
                if let Some(ref model_config) = self.default_model {
                    runtime.load_model(model_config.clone())?;
                }

                // Insert atomically
                entry.insert(runtime);
                info!("Runtime created and initialized for tenant: {}", tenant_id);
            }
        }

        Ok(())
    }

    /// Get or create a runtime with a specific model
    /// Uses atomic HashMap::entry() to prevent race conditions
    pub fn get_or_create_runtime_with_model(
        &self,
        tenant_id: impl Into<String>,
        model: impl AsRef<str>,
    ) -> Result<()> {
        let tenant_id = tenant_id.into();
        let model = model.as_ref();

        // Validate tenant ID format
        Self::validate_tenant_id(&tenant_id)?;

        // Resolve model name/URL to path
        let model_path = Self::resolve_model(model)?;

        let mut runtimes = self.runtimes.lock().unwrap();

        // Use entry() API for atomic check-and-insert (prevents race conditions)
        match runtimes.entry(tenant_id.clone()) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                // Check if model matches
                let runtime = entry.get();
                if let Some(ref config) = runtime.model_config {
                    if config.model_path == model_path {
                        debug!(
                            "Runtime already exists with same model for tenant: {}",
                            tenant_id
                        );
                        return Ok(());
                    } else {
                        warn!(
                            "Tenant {} already has model {:?}, requested {:?}. Reusing existing runtime.",
                            tenant_id, config.model_path, model_path
                        );
                        // For now, allow reuse - could return error if strict model matching needed
                        return Ok(());
                    }
                } else {
                    // Runtime exists but no model loaded - this shouldn't happen, but handle gracefully
                    debug!(
                        "Runtime exists for tenant {} but no model loaded",
                        tenant_id
                    );
                    return Ok(());
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                debug!(
                    "Creating new runtime for tenant: {} with model: {:?}",
                    tenant_id, model_path
                );

                // Create new runtime
                let mut runtime = TenantRuntime::new(&self.engine, &self.wasm_module, &tenant_id)?;

                // Load the specified model
                let model_config = ModelConfig {
                    model_path: model_path.clone(),
                    model_id: model.to_string(),
                };
                runtime.load_model(model_config)?;

                // Set LoRA adapter if configured for this tenant
                let tenant_adapters = self.tenant_lora_adapters.lock().unwrap();
                if let Some(adapter_id) = tenant_adapters.get(&tenant_id) {
                    runtime.lora_adapter_id = Some(adapter_id.clone());
                    info!(
                        "LoRA adapter '{}' configured for tenant: {}",
                        adapter_id, tenant_id
                    );
                }
                drop(tenant_adapters);

                // Insert atomically
                entry.insert(runtime);
                info!(
                    "Runtime created and initialized for tenant: {} with model: {:?}",
                    tenant_id, model_path
                );
            }
        }

        Ok(())
    }

    /// Set LoRA adapter for a tenant
    pub fn set_tenant_lora_adapter(
        &self,
        tenant_id: impl Into<String>,
        adapter_id: impl Into<String>,
    ) -> Result<()> {
        let tenant_id = tenant_id.into();
        let adapter_id = adapter_id.into();

        // Validate adapter exists
        if self.lora_manager.get_adapter(&adapter_id).is_none() {
            return Err(anyhow!("LoRA adapter '{}' not found", adapter_id));
        }

        // Store mapping
        let mut adapters = self.tenant_lora_adapters.lock().unwrap();
        adapters.insert(tenant_id.clone(), adapter_id.clone());
        drop(adapters);

        // Update runtime if it exists
        let mut runtimes = self.runtimes.lock().unwrap();
        if let Some(runtime) = runtimes.get_mut(&tenant_id) {
            runtime.lora_adapter_id = Some(adapter_id.clone());
            info!(
                "Updated LoRA adapter for tenant: {} to '{}'",
                tenant_id, adapter_id
            );
        }

        Ok(())
    }

    /// Load a LoRA adapter
    pub fn load_lora_adapter(&self, adapter: realm_runtime::lora::LoRAWeights) -> Result<()> {
        let adapter_id = adapter.adapter_id.clone();
        self.lora_manager.load_adapter(adapter)?;
        info!("LoRA adapter loaded: {}", adapter_id);
        Ok(())
    }

    /// Get LoRA manager (for external access)
    pub fn lora_manager(&self) -> Arc<LoRAManager> {
        self.lora_manager.clone()
    }

    /// Get LoRA adapter ID for a tenant (if configured)
    pub fn get_tenant_lora_adapter(&self, tenant_id: impl AsRef<str>) -> Option<String> {
        let adapters = self.tenant_lora_adapters.lock().unwrap();
        adapters.get(tenant_id.as_ref()).cloned()
    }

    /// Resolve model name or URL to file path
    fn resolve_model(model: &str) -> Result<PathBuf> {
        // If it's a URL, download it (simplified - in production would use proper downloading)
        if model.starts_with("http://") || model.starts_with("https://") {
            // For now, return error - would need to implement download logic
            // In production, would download to cache and return path
            return Err(anyhow!(
                "Model URL download not yet implemented. Use local file path or model name."
            ));
        }

        // If it's an absolute or relative path, check if it exists
        let path = PathBuf::from(model);
        if path.exists() {
            return Ok(path);
        }

        // Get model directory from environment variable or use defaults
        let model_dirs = Self::get_model_directories();

        // Try each model directory
        for dir in &model_dirs {
            let candidate = dir.join(model);
            if candidate.exists() {
                debug!("Found model in directory: {:?}", dir);
                return Ok(candidate);
            }
        }

        // Provide helpful error message with searched directories
        let searched_dirs: Vec<String> = model_dirs
            .iter()
            .map(|d| d.to_string_lossy().to_string())
            .collect();

        Err(anyhow!(
            "Model not found: {}\nSearched in: {}\n\nHint: Set REALM_MODEL_DIR environment variable or provide absolute path.",
            model,
            searched_dirs.join(", ")
        ))
    }

    /// Get model directories from environment variable or defaults
    fn get_model_directories() -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        // Check environment variable first (highest priority)
        if let Ok(env_dir) = std::env::var("REALM_MODEL_DIR") {
            let env_path = PathBuf::from(env_dir);
            if env_path.exists() {
                dirs.push(env_path);
                info!("Using REALM_MODEL_DIR: {:?}", dirs.last().unwrap());
            } else {
                warn!(
                    "REALM_MODEL_DIR environment variable set but path does not exist: {:?}",
                    env_path
                );
            }
        }

        // Add default model directories
        dirs.push(PathBuf::from("./models"));
        dirs.push(PathBuf::from("../models"));

        // Try to expand ~ to home directory
        if let Ok(home) = std::env::var("HOME") {
            dirs.push(PathBuf::from(home).join("models"));
        }

        // Also try current directory
        dirs.push(PathBuf::from("."));

        dirs
    }

    /// Generate text for a tenant
    pub fn generate(&self, tenant_id: impl AsRef<str>, prompt: String) -> Result<String> {
        let tenant_id = tenant_id.as_ref();
        let mut runtimes = self.runtimes.lock().unwrap();

        let runtime = runtimes
            .get_mut(tenant_id)
            .ok_or_else(|| anyhow!("No runtime for tenant: {}", tenant_id))?;

        runtime.generate(prompt)
    }

    /// Remove a tenant runtime (cleanup)
    pub fn remove_runtime(&self, tenant_id: impl AsRef<str>) -> Result<()> {
        let tenant_id = tenant_id.as_ref();
        let mut runtimes = self.runtimes.lock().unwrap();

        runtimes
            .remove(tenant_id)
            .ok_or_else(|| anyhow!("No runtime for tenant: {}", tenant_id))?;

        info!("Runtime removed for tenant: {}", tenant_id);
        Ok(())
    }

    /// Get number of active runtimes
    pub fn active_runtime_count(&self) -> usize {
        self.runtimes.lock().unwrap().len()
    }

    /// List all tenant IDs
    pub fn list_tenants(&self) -> Vec<String> {
        self.runtimes.lock().unwrap().keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_creation() {
        let config = ModelConfig {
            model_path: PathBuf::from("/path/to/model.gguf"),
            model_id: "test-model".to_string(),
        };

        assert_eq!(config.model_id, "test-model");
        assert_eq!(config.model_path, PathBuf::from("/path/to/model.gguf"));
    }

    // Note: Full integration tests require a compiled WASM module
    // and are better suited for the examples directory
}
