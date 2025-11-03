//! Runtime Manager
//!
//! Manages per-tenant WASM runtime instances with model loading and inference.

use anyhow::{anyhow, Context, Result};
use realm_runtime::HostContext;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};
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
        })
    }

    /// Set default model configuration
    pub fn set_default_model(&mut self, config: ModelConfig) {
        info!("Set default model: {:?}", config.model_path);
        self.default_model = Some(config);
    }

    /// Get or create a runtime for a tenant
    pub fn get_or_create_runtime(&self, tenant_id: impl Into<String>) -> Result<()> {
        let tenant_id = tenant_id.into();
        let mut runtimes = self.runtimes.lock().unwrap();

        if runtimes.contains_key(&tenant_id) {
            debug!("Runtime already exists for tenant: {}", tenant_id);
            return Ok(());
        }

        debug!("Creating new runtime for tenant: {}", tenant_id);

        // Create new runtime
        let mut runtime = TenantRuntime::new(&self.engine, &self.wasm_module, &tenant_id)?;

        // Load default model if configured
        if let Some(ref model_config) = self.default_model {
            runtime.load_model(model_config.clone())?;
        }

        runtimes.insert(tenant_id.clone(), runtime);
        info!("Runtime created and initialized for tenant: {}", tenant_id);

        Ok(())
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
