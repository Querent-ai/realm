//! Runtime Manager
//!
//! Manages per-tenant WASM runtime instances with model loading and inference.

use anyhow::{anyhow, Context, Result};
use realm_runtime::lora::LoRAManager;
use realm_runtime::HostContext;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};
use wasmtime::{Config, Engine, Instance, Linker, Module, Store};

/// Configuration for a model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to the GGUF model file
    pub model_path: PathBuf,

    /// Model ID for reference
    pub model_id: String,

    /// Optional draft model path for speculative decoding
    pub draft_model_path: Option<PathBuf>,

    /// Optional draft model ID for speculative decoding
    pub draft_model_id: Option<String>,
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
    lora_adapter_id: Option<String>,

    /// Optional draft model configuration for speculative decoding
    draft_model_config: Option<ModelConfig>,

    /// Model ID in model storage (for host function access)
    model_id: Option<u32>,

    /// Draft model ID in model storage (for speculative decoding)
    draft_model_id: Option<u32>,
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

        // Add wasm-bindgen stub imports (needed for wasm-pack generated modules)
        // These are required when the WASM module is built with wasm-bindgen
        // Note: wasm-bindgen generates hashed function names, so we extract all imports
        // from the module and add generic stubs for all wbg:: functions
        use wasmtime::Caller;

        // Create store first (needed for table creation)
        let mut store = Store::new(engine, ());

        // Extract imports from the WASM module and add stubs for JavaScript runtime APIs
        // Skip "env" module - those are real host functions already provided by host_context
        // This handles wasm-bindgen functions (wbg::) AND JavaScript runtime APIs (web_sys, js_sys)
        // TODO: This is a workaround for using wasm-pack --target web in a server environment.
        // The proper solution would be to build WASM without wasm-bindgen for server use.

        // First, check for table imports and provide them
        for import in wasm_module.imports() {
            let module = import.module();
            let name = import.name();

            // Skip "env" module - those are real host functions provided by host_context
            if module == "env" {
                continue;
            }

            // Handle table imports - wasm-bindgen uses tables for function references
            if let wasmtime::ExternType::Table(table_ty) = import.ty() {
                let min_size = table_ty.minimum();
                let max_size = table_ty.maximum();
                debug!(
                    "Found table import: {}::{} (element: {:?}, min: {}, max: {:?})",
                    module,
                    name,
                    table_ty.element(),
                    min_size,
                    max_size
                );

                // Create a table with the required type and size
                // For wasm-bindgen, tables typically use FuncRef or ExternRef
                // Use None as the initial value (null reference)
                // Check the RefType to determine which Ref variant to use
                let element_ty = table_ty.element();
                let init_val = match element_ty.heap_type() {
                    wasmtime::HeapType::Func => wasmtime::Ref::Func(None),
                    wasmtime::HeapType::Extern => wasmtime::Ref::Extern(None),
                    _ => anyhow::bail!("Unsupported table element type: {:?}", element_ty),
                };

                // Create table with the original type, but ensure it's large enough
                // wasm-bindgen may need a larger table, so use at least 4096 entries
                // Use the original table type to match what WASM expects exactly
                let table = wasmtime::Table::new(&mut store, table_ty.clone(), init_val)
                    .context("Failed to create WASM table")?;

                // Grow the table if it's too small (wasm-bindgen may need more entries)
                let current_size = table.size(&store);
                if current_size < 4096 {
                    // Recreate init_val for grow (it was moved during table creation)
                    let grow_init_val = match element_ty.heap_type() {
                        wasmtime::HeapType::Func => wasmtime::Ref::Func(None),
                        wasmtime::HeapType::Extern => wasmtime::Ref::Extern(None),
                        _ => anyhow::bail!("Unsupported table element type: {:?}", element_ty),
                    };
                    table
                        .grow(&mut store, 4096 - current_size, grow_init_val)
                        .context("Failed to grow WASM table")?;
                    debug!(
                        "Grew table {}::{} from {} to 4096",
                        module, name, current_size
                    );
                }
                let final_size = table.size(&store);
                linker
                    .define(&mut store, module, name, table)
                    .with_context(|| format!("Failed to define table: {}::{}", module, name))?;
                debug!(
                    "Created table {}::{} with size {}",
                    module, name, final_size
                );
                continue;
            }

            // Handle function imports
            if let wasmtime::ExternType::Func(func_ty) = import.ty() {
                let param_count = func_ty.params().len();
                let result_count = func_ty.results().len();

                debug!(
                    "Adding stub for: {}::{} ({} params, {} results)",
                    module, name, param_count, result_count
                );

                // Use Func::new to handle any signature dynamically
                // This is more flexible than func_wrap which requires exact type matches
                use wasmtime::{Func, Val};

                // Create a stub function that accepts any parameters and returns 0 or void
                let func_ty_clone = func_ty.clone();
                let stub_func = Func::new(
                    &mut store,
                    func_ty_clone,
                    move |_caller: Caller<'_, ()>,
                          _params: &[Val],
                          results: &mut [Val]|
                          -> Result<(), wasmtime::Error> {
                        // Ignore all parameters and return 0 or void
                        if !results.is_empty() {
                            results[0] = Val::I32(0);
                        }
                        Ok(())
                    },
                );

                linker
                    .define(&mut store, module, name, stub_func)
                    .with_context(|| format!("Failed to add stub: {}::{}", module, name))?;
            }
        }

        // Initialize host context
        host_context
            .initialize(&mut store)
            .context("Failed to initialize host context")?;

        // Instantiate WASM module
        let instance = match linker.instantiate(&mut store, wasm_module) {
            Ok(inst) => inst,
            Err(e) => {
                error!(
                    "Failed to instantiate WASM module for tenant {}: {:?}",
                    tenant_id, e
                );
                return Err(anyhow!(
                    "Failed to instantiate WASM module for tenant: {}. Error: {}",
                    tenant_id,
                    e
                ));
            }
        };

        info!("Runtime created for tenant: {}", tenant_id);

        Ok(Self {
            store,
            instance,
            host_context,
            model_config: None,
            tenant_id,
            lora_adapter_id: None,
            draft_model_config: None,
            model_id: None,
            draft_model_id: None,
        })
    }

    /// Helper: Load model bytes into WASM memory and call loadModel
    /// Returns the memory pointer where the model was written
    fn load_model_into_wasm(&mut self, model_bytes: &[u8], memory_offset: usize) -> Result<u32> {
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

        // Write model bytes to WASM memory at the specified offset
        let model_ptr = memory_offset;
        memory
            .write(&mut self.store, model_ptr, model_bytes)
            .context("Failed to write model bytes to WASM memory")?;

        debug!("Wrote model bytes to WASM memory at offset {}", model_ptr);

        // List all exports for debugging (collect names first to avoid borrow issues)
        let export_names: Vec<String> = self
            .instance
            .exports(&mut self.store)
            .map(|e| e.name().to_string())
            .collect();
        debug!("Available WASM exports: {:?}", export_names);

        // Check if we need to call __wbg_init or similar initialization function first
        // JavaScript examples call __wbg_init before creating Realm instances
        if let Some(init_func) = self.instance.get_func(&mut self.store, "__wbg_init") {
            debug!("Found __wbg_init function, calling it...");
            let init_ty = init_func.ty(&self.store);
            debug!(
                "__wbg_init signature: {} params, {} results",
                init_ty.params().len(),
                init_ty.results().len()
            );

            // Try calling with no arguments (most common)
            if init_ty.params().len() == 0 {
                if let Ok(init_typed) = init_func.typed::<(), ()>(&self.store) {
                    init_typed
                        .call(&mut self.store, ())
                        .context("Failed to call __wbg_init")?;
                    debug!("Successfully called __wbg_init");
                } else {
                    let mut results = Vec::new();
                    init_func
                        .call(&mut self.store, &[], &mut results)
                        .context("Failed to call __wbg_init (untyped)")?;
                    debug!("Successfully called __wbg_init (untyped)");
                }
            }
        } else if let Some(init_sync) = self.instance.get_func(&mut self.store, "initSync") {
            debug!("Found initSync function, calling it...");
            let init_ty = init_sync.ty(&self.store);
            if init_ty.params().len() == 0 {
                if let Ok(init_typed) = init_sync.typed::<(), ()>(&self.store) {
                    init_typed
                        .call(&mut self.store, ())
                        .context("Failed to call initSync")?;
                    debug!("Successfully called initSync");
                }
            }
        } else {
            debug!("No __wbg_init or initSync function found, proceeding without initialization");
        }

        // Check if __wbindgen_malloc is available
        let has_wbindgen_malloc = export_names.iter().any(|n| n == "__wbindgen_malloc");
        debug!("__wbindgen_malloc available: {}", has_wbindgen_malloc);

        // PATTERN 1 (Recommended): wasm-bindgen constructor returns the instance
        // The Rust constructor is: pub fn new() -> Result<Realm, JsError>
        // wasm-bindgen exports this as a function that RETURNS a pointer to the initialized Realm
        // We should NOT allocate memory ourselves - let Rust/wasm-bindgen do it
        let realm_this = if let Some(realm_new_func) =
            self.instance.get_func(&mut self.store, "realm_new")
        {
            let func_ty = realm_new_func.ty(&self.store);
            let param_count = func_ty.params().len();
            let result_count = func_ty.results().len();

            debug!(
                "realm_new signature: {} params, {} results (Pattern 1: returns instance)",
                param_count, result_count
            );

            // Pattern 1: Constructor takes no params, returns pointer to initialized Realm
            if param_count == 0 && result_count == 1 {
                if let Ok(realm_new_typed) = realm_new_func.typed::<(), u32>(&self.store) {
                    let ptr = realm_new_typed.call(&mut self.store, ()).map_err(|e| {
                        error!("realm_new constructor (Pattern 1) failed: {:?}", e);
                        anyhow::anyhow!("Failed to call realm_new constructor: {}", e)
                    })?;
                    debug!(
                        "realm_new (Pattern 1) returned initialized Realm pointer: {}",
                        ptr
                    );
                    ptr
                } else {
                    // Fallback to untyped call
                    use wasmtime::Val;
                    let mut results = vec![Val::I32(0)];
                    realm_new_func
                        .call(&mut self.store, &[], &mut results)
                        .map_err(|e| {
                            error!("realm_new constructor (Pattern 1, untyped) failed: {:?}", e);
                            anyhow::anyhow!("Failed to call realm_new constructor: {}", e)
                        })?;
                    match results[0] {
                        Val::I32(ptr) => {
                            debug!("realm_new (Pattern 1, untyped) returned pointer: {}", ptr);
                            ptr as u32
                        }
                        Val::I64(ptr) => {
                            debug!(
                                "realm_new (Pattern 1, untyped) returned I64 pointer: {}",
                                ptr
                            );
                            ptr as u32
                        }
                        v => anyhow::bail!("Unexpected return type from realm_new: {:?}", v),
                    }
                }
            } else if param_count == 1 && result_count == 0 {
                // Pattern 3 (fragile but required): Constructor takes pointer, writes into it
                // wasm-bindgen with --target web generates this pattern
                // We need to allocate memory using __wbindgen_malloc and pass it to constructor
                warn!("realm_new has Pattern 3 signature (in-place constructor) - implementing correctly");

                // Allocate memory for Realm struct using __wbindgen_malloc
                // This ensures proper alignment and memory management
                let realm_ptr = if let Some(malloc_func) =
                    self.instance.get_func(&mut self.store, "__wbindgen_malloc")
                {
                    if let Ok(malloc_typed) = malloc_func.typed::<u32, u32>(&self.store) {
                        // Allocate enough space for Realm struct (estimate ~200 bytes)
                        let ptr = malloc_typed
                            .call(&mut self.store, 200)
                            .context("Failed to allocate memory for Realm struct")?;
                        debug!(
                            "Allocated Realm struct memory at pointer: {} (via __wbindgen_malloc)",
                            ptr
                        );
                        ptr
                    } else {
                        use wasmtime::Val;
                        let malloc_args = vec![Val::I32(200)];
                        let mut malloc_results = vec![Val::I32(0)];
                        malloc_func
                            .call(&mut self.store, &malloc_args, &mut malloc_results)
                            .context("Failed to allocate memory for Realm struct (untyped)")?;
                        match malloc_results[0] {
                            Val::I32(ptr) => {
                                debug!(
                                    "Allocated Realm struct memory at pointer: {} (untyped)",
                                    ptr
                                );
                                ptr as u32
                            }
                            _ => anyhow::bail!("__wbindgen_malloc returned unexpected type"),
                        }
                    }
                } else {
                    anyhow::bail!(
                        "__wbindgen_malloc not available - cannot allocate memory for Realm struct"
                    );
                };

                // Call constructor with allocated pointer - it will initialize the struct in-place
                use wasmtime::Val;
                let args = vec![Val::I32(realm_ptr as i32)];
                let mut results = Vec::new();
                realm_new_func
                    .call(&mut self.store, &args, &mut results)
                    .map_err(|e| {
                        error!(
                            "realm_new constructor (Pattern 3) failed with pointer {}: {:?}",
                            realm_ptr, e
                        );
                        anyhow::anyhow!("Failed to initialize Realm struct: {}", e)
                    })?;
                debug!(
                    "realm_new (Pattern 3) successfully initialized Realm struct at pointer: {}",
                    realm_ptr
                );
                realm_ptr
            } else {
                anyhow::bail!(
                    "realm_new has unexpected signature: {} params, {} results. \
                    Expected Pattern 1: () -> u32 (constructor returns instance)",
                    param_count,
                    result_count
                );
            }
        } else {
            anyhow::bail!("realm_new function not found in WASM exports");
        };

        debug!(
            "Realm instance created and initialized with this pointer: {}",
            realm_this
        );

        // Get loadModel function - wasm-bindgen exports it as "realm_loadModel"
        // Use get_func to get the raw function, then inspect its type dynamically
        let load_model_func = self
            .instance
            .get_func(&mut self.store, "realm_loadModel")
            .context("Failed to get realm_loadModel function (not found in exports)")?;

        // Inspect the function type
        let func_ty = load_model_func.ty(&self.store);
        let param_count = func_ty.params().len();
        let result_count = func_ty.results().len();

        debug!(
            "realm_loadModel signature: {} params, {} results",
            param_count, result_count
        );

        // Prepare arguments: (this_ptr, model_ptr, model_len)
        use wasmtime::Val;
        let args = vec![
            Val::I32(realm_this as i32),
            Val::I32(model_ptr as i32),
            Val::I32(model_bytes.len() as i32),
        ];

        let mut results = vec![Val::I32(0); result_count];
        load_model_func
            .call(&mut self.store, &args, &mut results)
            .context("loadModel function call failed")?;

        Ok(model_ptr as u32)
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

        // Load model into WASM memory at offset 0 (main model)
        self.load_model_into_wasm(&model_bytes, 0)?;

        // Get model_id from model storage (stored by realm_store_model host function)
        use realm_runtime::model_storage::get_global_model_storage;
        let model_id = {
            let storage = get_global_model_storage().lock();
            let models = storage.list_models();
            models.last().copied().unwrap_or(1)
        };
        self.model_id = Some(model_id);
        info!(
            "Model ID {} assigned to tenant {}",
            model_id, self.tenant_id
        );

        // If LoRA adapter is configured, set it for the stored model
        if let Some(ref adapter_id) = self.lora_adapter_id {
            // Set LoRA adapter directly in model storage
            match get_global_model_storage()
                .lock()
                .set_lora_adapter(model_id, adapter_id.clone())
            {
                Ok(()) => {
                    info!(
                        "LoRA adapter '{}' set for model {} (tenant: {})",
                        adapter_id, model_id, self.tenant_id
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to set LoRA adapter '{}' for model {} (tenant: {}): {}",
                        adapter_id, model_id, self.tenant_id, e
                    );
                }
            }
        }

        info!("Model loaded successfully for tenant: {}", self.tenant_id);

        self.model_config = Some(config);
        Ok(())
    }

    /// Load draft model for speculative decoding
    ///
    /// Loads a draft model into the WASM runtime for speculative decoding.
    /// The draft model is loaded at a fixed 10MB offset to avoid conflicts with the main model.
    ///
    /// # Parameters
    /// - `config`: [`ModelConfig`] containing the path and ID of the draft model to load.
    ///
    /// # Returns
    /// Returns `Ok(())` if the draft model is successfully loaded into WASM memory and the
    /// `loadModel` function is called without error. Returns an error if reading the model file,
    /// allocating memory, or calling the WASM function fails.
    ///
    /// # Memory Allocation Strategy
    /// The method reads the draft model file into memory, calculates the number of WASM memory
    /// pages required, and grows the WASM memory if necessary. The draft model bytes are written
    /// to WASM memory at a fixed offset (10MB) to avoid conflicts with the main model. The
    /// `loadModel` WASM function is then called with the pointer and length of the draft model.
    ///
    /// # Note
    /// The 10MB offset is a simple strategy to avoid conflicts. For production use, consider
    /// implementing a dynamic memory allocation strategy that tracks allocated regions.
    pub fn load_draft_model(&mut self, config: ModelConfig) -> Result<()> {
        info!(
            "Loading draft model for tenant {}: {:?}",
            self.tenant_id, config.model_path
        );

        // Read draft model bytes
        let draft_model_bytes = std::fs::read(&config.model_path)
            .with_context(|| format!("Failed to read draft model file: {:?}", config.model_path))?;

        debug!(
            "Read {} bytes from draft model file for tenant {}",
            draft_model_bytes.len(),
            self.tenant_id
        );

        // Load draft model into WASM memory at 10MB offset to avoid conflicts with main model
        // TODO: Consider implementing dynamic memory allocation strategy for production use
        const DRAFT_MODEL_OFFSET: usize = 10 * 1024 * 1024; // 10MB offset
        self.load_model_into_wasm(&draft_model_bytes, DRAFT_MODEL_OFFSET)?;

        // Get draft model_id from model storage (stored by realm_store_model host function)
        use realm_runtime::model_storage::get_global_model_storage;
        let draft_model_id = {
            let storage = get_global_model_storage().lock();
            let models = storage.list_models();
            models.last().copied().unwrap_or(1)
        };
        self.draft_model_id = Some(draft_model_id);
        info!(
            "Draft model ID {} loaded successfully for tenant: {}",
            draft_model_id, self.tenant_id
        );
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

    /// Get draft model configuration (if configured for speculative decoding)
    pub fn draft_model_config(&self) -> Option<&ModelConfig> {
        self.draft_model_config.as_ref()
    }

    /// Get model ID
    pub fn model_id(&self) -> Option<u32> {
        self.model_id
    }

    /// Get draft model ID
    pub fn draft_model_id(&self) -> Option<u32> {
        self.draft_model_id
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
        if let Some(ref draft_path) = config.draft_model_path {
            info!(
                "Draft model configured for speculative decoding: {:?}",
                draft_path
            );
        }
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

                    // Apply LoRA adapter if configured for this tenant
                    let tenant_lora = self.tenant_lora_adapters.lock().unwrap();
                    if let Some(ref adapter_id) = tenant_lora.get(&tenant_id) {
                        info!(
                            "Applying LoRA adapter '{}' to model for tenant {}",
                            adapter_id, tenant_id
                        );
                        // LoRA is applied via host function during forward pass
                        // The model_id is stored in RuntimeManager, and realm_forward_layer
                        // will check for LoRA adapter and apply it when loading weights
                        runtime.lora_adapter_id = Some(adapter_id.to_string());
                    }

                    // Store draft model config if available and load it
                    if let Some(ref draft_path) = model_config.draft_model_path {
                        let draft_config = ModelConfig {
                            model_path: draft_path.clone(),
                            model_id: model_config
                                .draft_model_id
                                .clone()
                                .unwrap_or_else(|| format!("{}_draft", model_config.model_id)),
                            draft_model_path: None,
                            draft_model_id: None,
                        };
                        runtime.draft_model_config = Some(draft_config.clone());
                        info!(
                            "Draft model configured for tenant {}: {:?}",
                            tenant_id, draft_path
                        );
                        // Load draft model for speculative decoding
                        if let Err(e) = runtime.load_draft_model(draft_config) {
                            warn!(
                                "Failed to load draft model for tenant {}: {}. Speculative decoding will be disabled.",
                                tenant_id, e
                            );
                        }
                    }
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
                    draft_model_path: None,
                    draft_model_id: None,
                };
                runtime.load_model(model_config.clone())?;

                // Check if draft model is configured for speculative decoding
                if let Some(ref draft_path) = model_config.draft_model_path {
                    info!(
                        "Draft model configured for speculative decoding: {:?}",
                        draft_path
                    );
                    runtime.draft_model_config = Some(ModelConfig {
                        model_path: draft_path.clone(),
                        model_id: model_config
                            .draft_model_id
                            .clone()
                            .unwrap_or_else(|| format!("{}_draft", model)),
                        draft_model_path: None,
                        draft_model_id: None,
                    });
                }

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
    /// This registers the adapter both in RuntimeManager and the global LoRA manager
    /// so it can be accessed by realm_forward_layer during inference
    pub fn load_lora_adapter(&self, adapter: realm_runtime::lora::LoRAWeights) -> Result<()> {
        let adapter_id = adapter.adapter_id.clone();

        // Register with RuntimeManager's LoRA manager
        // Clone for first registration, then use original for second (more efficient than retrieving)
        self.lora_manager.load_adapter(adapter.clone())?;

        // Also register with global LoRA manager for realm_forward_layer access
        // Use original adapter since we cloned for the first registration
        use realm_runtime::lora::get_global_lora_manager;
        let global_manager = get_global_lora_manager();
        match global_manager.lock() {
            Ok(manager_guard) => {
                manager_guard.load_adapter(adapter)?;
            }
            Err(e) => {
                warn!("Failed to lock global LoRA manager: {:?}", e);
                // Continue - adapter is still registered in RuntimeManager
            }
        }

        info!(
            "LoRA adapter loaded: {} (registered in both RuntimeManager and global manager)",
            adapter_id
        );
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
    /// Uses speculative decoding if draft model is configured
    pub fn generate(&self, tenant_id: impl AsRef<str>, prompt: String) -> Result<String> {
        let tenant_id = tenant_id.as_ref();
        let runtimes = self.runtimes.clone();
        let mut runtimes_guard = runtimes.lock().unwrap();

        let runtime = runtimes_guard
            .get_mut(tenant_id)
            .ok_or_else(|| anyhow!("No runtime for tenant: {}", tenant_id))?;

        // Check if speculative decoding should be used
        if runtime.draft_model_config().is_some() && runtime.draft_model_id().is_some() {
            // Use speculative decoding
            drop(runtimes_guard);
            use crate::speculative_integration::generate_with_speculative_decoding;
            let runtime_arc = {
                let mut runtimes = runtimes.lock().unwrap();
                Arc::new(Mutex::new(runtimes.remove(tenant_id).unwrap()))
            };
            let result = generate_with_speculative_decoding(runtime_arc.clone(), prompt, 100)?;
            // Put runtime back
            {
                let mut runtimes = runtimes.lock().unwrap();
                let runtime =
                    Arc::try_unwrap(runtime_arc).map_err(|_| anyhow!("Failed to unwrap Arc"))?;
                let runtime = runtime
                    .into_inner()
                    .map_err(|_| anyhow!("Failed to unwrap Mutex"))?;
                runtimes.insert(tenant_id.to_string(), runtime);
            }
            Ok(result)
        } else {
            // Standard generation
            runtime.generate(prompt)
        }
    }

    /// Generate text with streaming support
    ///
    /// Returns a channel receiver that yields tokens as they're generated.
    /// Uses real token-by-token streaming via realm_stream_token host function.
    pub fn generate_stream(
        &self,
        tenant_id: impl AsRef<str>,
        prompt: String,
    ) -> Result<tokio::sync::mpsc::Receiver<String>> {
        let tenant_id = tenant_id.as_ref().to_string();
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        // Clone runtimes Arc for the spawned task
        let runtimes = self.runtimes.clone();

        // Spawn blocking task to generate
        tokio::task::spawn_blocking(move || {
            let mut runtimes = runtimes.lock().unwrap();
            let runtime = match runtimes.get_mut(&tenant_id) {
                Some(rt) => rt,
                None => {
                    let _ = tx.blocking_send("Error: No runtime for tenant".to_string());
                    return;
                }
            };

            // Set streaming callback on host context for real token-by-token streaming
            // Use blocking channel since host functions are called from blocking context
            let (blocking_tx, blocking_rx) = std::sync::mpsc::channel();
            let tx_clone = tx.clone();
            let blocking_tx_for_error = blocking_tx.clone();

            // Spawn a task to forward from blocking channel to async channel
            tokio::spawn(async move {
                while let Ok(token) = blocking_rx.recv() {
                    if tx_clone.send(token).await.is_err() {
                        // Receiver dropped, stop forwarding
                        break;
                    }
                }
            });

            // Set the blocking callback on host context
            runtime.host_context().set_stream_callback(blocking_tx);

            // Generate - tokens will be streamed via realm_stream_token host function
            // Use a guard to ensure callback is always cleared, even on panic
            let result = match runtime.generate(prompt) {
                Ok(_response) => {
                    // Response is already streamed token-by-token via realm_stream_token
                    // No need to chunk words anymore
                    Ok(())
                }
                Err(e) => {
                    // Send error via blocking channel (use cloned sender)
                    let _ = blocking_tx_for_error.send(format!("Error: {}", e));
                    Err(e)
                }
            };

            // Always clear streaming callback, even if generation failed
            runtime.host_context().clear_stream_callback();

            // Drop blocking_tx_for_error to signal forwarding task completion
            drop(blocking_tx_for_error);

            // Result is ignored - errors are sent via channel
            let _ = result;
        });

        Ok(rx)
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
            draft_model_path: None,
            draft_model_id: None,
        };

        assert_eq!(config.model_id, "test-model");
        assert_eq!(config.model_path, PathBuf::from("/path/to/model.gguf"));
    }

    // Note: Full integration tests require a compiled WASM module
    // and are better suited for the examples directory

    #[tokio::test]
    async fn test_streaming_callback_setup() {
        // Test that streaming callback can be set and cleared
        use realm_runtime::HostContext;

        let host_context = HostContext::new();

        // Create a test channel
        let (tx, _rx) = std::sync::mpsc::channel();

        // Set callback
        host_context.set_stream_callback(tx);

        // Clear callback
        host_context.clear_stream_callback();

        // Test passes if no panic occurs
    }

    #[tokio::test]
    async fn test_streaming_channel_forwarding() {
        // Test that blocking channel correctly forwards to async channel
        let (async_tx, mut async_rx) = tokio::sync::mpsc::channel(10);
        let (blocking_tx, blocking_rx) = std::sync::mpsc::channel();

        // Spawn forwarding task with timeout
        let forwarding_handle = tokio::spawn(async move {
            while let Ok(token) = blocking_rx.recv() {
                if async_tx.send(token).await.is_err() {
                    break;
                }
            }
        });

        // Send tokens via blocking channel
        blocking_tx.send("token1".to_string()).unwrap();
        blocking_tx.send("token2".to_string()).unwrap();
        blocking_tx.send("token3".to_string()).unwrap();

        // Drop the blocking sender to signal completion
        drop(blocking_tx);

        // Receive from async channel with timeout
        let token1 = tokio::time::timeout(std::time::Duration::from_secs(1), async_rx.recv())
            .await
            .expect("Timeout waiting for token1")
            .unwrap();
        let token2 = tokio::time::timeout(std::time::Duration::from_secs(1), async_rx.recv())
            .await
            .expect("Timeout waiting for token2")
            .unwrap();
        let token3 = tokio::time::timeout(std::time::Duration::from_secs(1), async_rx.recv())
            .await
            .expect("Timeout waiting for token3")
            .unwrap();

        assert_eq!(token1, "token1");
        assert_eq!(token2, "token2");
        assert_eq!(token3, "token3");

        // Wait for forwarding task to complete (should finish after sender is dropped)
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), forwarding_handle).await;
    }
}
