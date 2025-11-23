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

    /// Realm instance pointer (from realm_new)
    realm_this: Option<u32>,

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
        // CRITICAL: We must skip "env" module imports - they are real host functions!
        let mut env_imports = std::collections::HashSet::new();
        for import in wasm_module.imports() {
            let (module, name) = (import.module(), import.name());
            if module == "env" {
                env_imports.insert(name.to_string());
                debug!(
                    "Found env::{} import - will NOT stub (real host function)",
                    name
                );
            }
        }

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
                // CRITICAL: Skip "env" module - these are real host functions already registered!
                if module == "env" {
                    debug!(
                        "Skipping stub for env::{} - already registered as real host function",
                        name
                    );
                    continue;
                }

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

        // Create Realm instance (realm_new) immediately after instantiation
        let realm_this = if let Some(realm_new_func) = instance.get_func(&mut store, "realm_new") {
            let func_ty = realm_new_func.ty(&store);
            let param_count = func_ty.params().len();
            let result_count = func_ty.results().len();

            debug!(
                "realm_new signature: {} params, {} results (Pattern 1: returns instance)",
                param_count, result_count
            );

            // Pattern 1: Constructor takes no params, returns pointer to initialized Realm
            if param_count == 0 && result_count == 1 {
                if let Ok(realm_new_typed) = realm_new_func.typed::<(), u32>(&store) {
                    match realm_new_typed.call(&mut store, ()) {
                        Ok(ptr) => {
                            debug!(
                                "realm_new (Pattern 1) returned initialized Realm pointer: {}",
                                ptr
                            );
                            Some(ptr)
                        }
                        Err(e) => {
                            error!("realm_new constructor (Pattern 1) failed: {:?}", e);
                            None
                        }
                    }
                } else {
                    // Fallback to untyped call
                    use wasmtime::Val;
                    let mut results = vec![Val::I32(0)];
                    match realm_new_func.call(&mut store, &[], &mut results) {
                        Ok(()) => match results[0] {
                            Val::I32(ptr) => {
                                debug!("realm_new (Pattern 1, untyped) returned pointer: {}", ptr);
                                Some(ptr as u32)
                            }
                            Val::I64(ptr) => {
                                debug!(
                                    "realm_new (Pattern 1, untyped) returned I64 pointer: {}",
                                    ptr
                                );
                                Some(ptr as u32)
                            }
                            v => {
                                error!("Unexpected return type from realm_new: {:?}", v);
                                None
                            }
                        },
                        Err(e) => {
                            error!("realm_new constructor (Pattern 1, untyped) failed: {:?}", e);
                            None
                        }
                    }
                }
            } else if param_count == 1 && result_count == 0 {
                // Pattern 3: Constructor takes pointer, writes into it
                warn!("realm_new has Pattern 3 signature (in-place constructor) - implementing correctly");

                // Allocate memory for Realm struct using __wbindgen_malloc
                let realm_ptr = if let Some(malloc_func) =
                    instance.get_func(&mut store, "__wbindgen_malloc")
                {
                    if let Ok(malloc_typed) = malloc_func.typed::<u32, u32>(&store) {
                        match malloc_typed.call(&mut store, 200) {
                            Ok(ptr) => {
                                debug!("Allocated Realm struct memory at pointer: {} (via __wbindgen_malloc)", ptr);
                                Some(ptr)
                            }
                            Err(e) => {
                                error!("Failed to allocate memory for Realm struct: {:?}", e);
                                None
                            }
                        }
                    } else {
                        use wasmtime::Val;
                        let malloc_args = vec![Val::I32(200)];
                        let mut malloc_results = vec![Val::I32(0)];
                        match malloc_func.call(&mut store, &malloc_args, &mut malloc_results) {
                            Ok(()) => {
                                match malloc_results[0] {
                                    Val::I32(ptr) => {
                                        debug!("Allocated Realm struct memory at pointer: {} (untyped)", ptr);
                                        Some(ptr as u32)
                                    }
                                    _ => {
                                        error!("__wbindgen_malloc returned unexpected type");
                                        None
                                    }
                                }
                            }
                            Err(e) => {
                                error!(
                                    "Failed to allocate memory for Realm struct (untyped): {:?}",
                                    e
                                );
                                None
                            }
                        }
                    }
                } else {
                    error!(
                        "__wbindgen_malloc not available - cannot allocate memory for Realm struct"
                    );
                    None
                };

                if let Some(realm_ptr) = realm_ptr {
                    // Call constructor with allocated pointer
                    use wasmtime::Val;
                    let args = vec![Val::I32(realm_ptr as i32)];
                    let mut results = Vec::new();
                    match realm_new_func.call(&mut store, &args, &mut results) {
                        Ok(()) => {
                            debug!("realm_new (Pattern 3) successfully initialized Realm struct at pointer: {}", realm_ptr);
                            Some(realm_ptr)
                        }
                        Err(e) => {
                            error!(
                                "realm_new constructor (Pattern 3) failed with pointer {}: {:?}",
                                realm_ptr, e
                            );
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                error!(
                    "realm_new has unexpected signature: {} params, {} results. Expected Pattern 1: () -> u32",
                    param_count, result_count
                );
                None
            }
        } else {
            error!("realm_new function not found in WASM exports");
            None
        };

        if realm_this.is_none() {
            warn!(
                "Failed to create Realm instance for tenant {} - model loading may fail",
                tenant_id
            );
        } else {
            debug!(
                "Realm instance created for tenant {} with pointer: {:?}",
                tenant_id, realm_this
            );
        }

        Ok(Self {
            store,
            instance,
            host_context,
            realm_this,
            model_config: None,
            tenant_id,
            lora_adapter_id: None,
            draft_model_config: None,
            model_id: None,
            draft_model_id: None,
        })
    }

    // DELETED: load_model_into_wasm() function (257 lines)
    // This function violated the HOST-side storage architecture by loading
    // entire model files (637MB+) into WASM memory, causing OOM errors.
    //
    // Models are now stored directly in HOST via ModelStorage::store_model()
    // and WASM gets only the model_id (4 bytes) via loadModelById().

    /// Load a model into this runtime
    ///
    /// Models are stored in HOST memory, not WASM memory.
    /// This function stores the model in HOST storage first, then tells WASM about it.
    pub fn load_model(&mut self, config: ModelConfig) -> Result<()> {
        error!(
            "🔍 load_model ENTRY: tenant={}, model_path={:?}",
            self.tenant_id, config.model_path
        );
        error!("🔍 realm_this before load: {:?}", self.realm_this);

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

        // Store model in HOST memory first (not WASM memory!)
        // This is the correct architecture: models live in HOST, WASM only gets model_id
        use realm_runtime::model_storage::get_global_model_storage;
        let model_id = {
            let mut storage = get_global_model_storage().lock();
            storage
                .store_model(&model_bytes, None) // Auto-generate ID from hash
                .context("Failed to store model in HOST storage")?
        };

        info!(
            "Model stored in HOST storage with ID {} (tenant: {})",
            model_id, self.tenant_id
        );

        self.model_id = Some(model_id);

        // CRITICAL FIX: Don't write entire model to WASM memory!
        // The model is already in HOST storage. Use loadModelById() to initialize WASM Realm
        // with config/tokenizer from HOST storage, without needing model bytes.

        error!(
            "🔍 After storing model_id={}, checking realm_this...",
            model_id
        );
        error!("🔍 realm_this value: {:?}", self.realm_this);

        // Get realm_this pointer (should already be created in TenantRuntime::new())
        let realm_this = self.realm_this.ok_or_else(|| {
            error!("❌ realm_this is None! Realm instance not initialized");
            anyhow::anyhow!(
                "Realm instance not initialized - realm_new failed during TenantRuntime creation"
            )
        })?;

        info!("✅ Got realm_this={} for model_id={}", realm_this, model_id);

        // SKIP calling load_model_by_id - we'll pass model_id directly to generate() instead
        // This avoids the WASM export issue with wasm-bindgen stripping no_mangle functions
        info!(
            "Model ID {} stored in HOST storage. Will pass model_id directly to generate()",
            model_id
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

        info!(
            "Read {} MB from draft model file for tenant {}",
            draft_model_bytes.len() / 1024 / 1024,
            self.tenant_id
        );

        // Store draft model directly in HOST storage (no WASM involvement!)
        // This is the correct architecture: models live in HOST, WASM only gets model_id
        use realm_runtime::model_storage::get_global_model_storage;
        let draft_model_id = {
            let mut storage = get_global_model_storage().lock();
            storage
                .store_model(&draft_model_bytes, None)
                .context("Failed to store draft model in HOST storage")?
        };

        self.draft_model_id = Some(draft_model_id);
        info!(
            "Draft model stored in HOST with ID {} for tenant {}",
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

        // Get model_id (should be set by load_model)
        let model_id = self
            .model_id
            .ok_or_else(|| anyhow!("No model_id set. Call load_model() first."))?;

        // Get the generate function from WASM (now takes model_id as third parameter)
        // Try "generate" first (C-ABI), fallback to "realm_generate" (wasm-bindgen)
        let generate_func = if let Some(func) = self.instance.get_func(&mut self.store, "generate")
        {
            debug!("Found 'generate' function (C-ABI)");
            func
        } else if let Some(func) = self.instance.get_func(&mut self.store, "realm_generate") {
            debug!("Found 'realm_generate' function (wasm-bindgen)");
            func
        } else {
            return Err(anyhow!(
                "Failed to find generate function (tried: 'generate', 'realm_generate')"
            ));
        };

        let func_ty = generate_func.ty(&self.store);
        let param_count = func_ty.params().len();

        debug!("generate function signature: {} params", param_count);

        // Create default GenOptions in WASM memory
        use realm_runtime::GenOptions;
        let default_options = GenOptions::default();

        // Write GenOptions to WASM memory (after prompt, before output buffer)
        let options_ptr = (prompt_ptr + prompt_bytes.len() + 1024) as u32; // Offset after prompt
        let options_bytes = unsafe {
            std::slice::from_raw_parts(
                &default_options as *const GenOptions as *const u8,
                std::mem::size_of::<GenOptions>(),
            )
        };
        memory
            .write(&mut self.store, options_ptr as usize, options_bytes)
            .context("Failed to write GenOptions to WASM memory")?;

        // Use untyped call (more reliable than typed call for C-ABI functions)
        // The typed call fails because WASM function signature might not match exactly
        use wasmtime::Val;
        let mut args = Vec::new();

        // Build args based on param count
        if param_count >= 1 {
            args.push(Val::I32(prompt_ptr as i32));
        }
        if param_count >= 2 {
            args.push(Val::I32(prompt_bytes.len() as i32));
        }
        if param_count >= 3 {
            args.push(Val::I32(model_id as i32));
        }
        if param_count >= 4 {
            args.push(Val::I32(options_ptr as i32));
        }

        eprintln!("[TRACE] Calling generate with {} args: prompt_ptr={}, prompt_len={}, model_id={}, options_ptr={}", 
               args.len(), prompt_ptr, prompt_bytes.len(), model_id, options_ptr);
        eprintln!(
            "[TRACE] Function signature: {} params, {} results",
            param_count,
            func_ty.results().len()
        );
        debug!("Calling generate with {} args: prompt_ptr={}, prompt_len={}, model_id={}, options_ptr={}", 
               args.len(), prompt_ptr, prompt_bytes.len(), model_id, options_ptr);
        debug!(
            "Function signature: {} params, {} results",
            param_count,
            func_ty.results().len()
        );

        let mut results = vec![Val::I32(0); func_ty.results().len().max(1)];
        eprintln!("[TRACE] About to call generate_func.call()");
        match generate_func.call(&mut self.store, &args, &mut results) {
            Ok(()) => {
                eprintln!(
                    "[TRACE] generate function call succeeded, result: {:?}",
                    results
                );
                debug!("generate function call succeeded, result: {:?}", results);
            }
            Err(e) => {
                eprintln!("[TRACE] generate function call FAILED: {}", e);
                eprintln!("[TRACE] Args passed: {:?}", args);
                error!("generate function call failed with error: {}", e);
                error!("Args passed: {:?}", args);
                error!(
                    "Function type: params={:?}, results={:?}",
                    func_ty.params().collect::<Vec<_>>(),
                    func_ty.results().collect::<Vec<_>>()
                );
                return Err(anyhow!("generate function call failed: {}", e));
            }
        }

        // Extract result
        let result_ptr = if let Some(Val::I32(ptr)) = results.first() {
            *ptr as u32
        } else {
            return Err(anyhow!(
                "generate returned unexpected result type: {:?}",
                results
            ));
        };

        // Check if result_ptr is 0 (error case)
        if result_ptr == 0 {
            return Err(anyhow!(
                "WASM generate() returned null pointer (error occurred)"
            ));
        }

        // Read result from memory
        // The result should be a null-terminated string
        let mut result_bytes = vec![0u8; 10240];
        memory
            .read(&self.store, result_ptr as usize, &mut result_bytes)
            .context("Failed to read result from WASM memory")?;

        // Find null terminator
        let result_len = result_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(result_bytes.len());

        // If result is empty or all zeros, it's an error
        if result_len == 0 {
            return Err(anyhow!("WASM generate() returned empty result"));
        }

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
        // Mutex lock failure is rare and indicates a serious issue
        // For this read-only check, we return false on lock failure
        self.runtimes
            .lock()
            .map(|runtimes| runtimes.contains_key(tenant_id.as_ref()))
            .unwrap_or(false)
    }

    /// Get or create a runtime for a tenant with validation
    /// Uses atomic HashMap::entry() to prevent race conditions
    pub fn get_or_create_runtime(&self, tenant_id: impl Into<String>) -> Result<()> {
        let tenant_id = tenant_id.into();

        // Validate tenant ID format
        Self::validate_tenant_id(&tenant_id)?;

        let mut runtimes = self
            .runtimes
            .lock()
            .map_err(|e| anyhow!("Failed to acquire runtimes lock: {}", e))?;

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
                    let tenant_lora = self.tenant_lora_adapters.lock().map_err(|e| {
                        anyhow!("Failed to acquire tenant_lora_adapters lock: {}", e)
                    })?;
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

        let mut runtimes = self
            .runtimes
            .lock()
            .map_err(|e| anyhow!("Failed to acquire runtimes lock: {}", e))?;

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
                let tenant_adapters = self
                    .tenant_lora_adapters
                    .lock()
                    .map_err(|e| anyhow!("Failed to acquire tenant_lora_adapters lock: {}", e))?;
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
        let mut adapters = self
            .tenant_lora_adapters
            .lock()
            .map_err(|e| anyhow!("Failed to acquire tenant_lora_adapters lock: {}", e))?;
        adapters.insert(tenant_id.clone(), adapter_id.clone());
        drop(adapters);

        // Update runtime if it exists
        let mut runtimes = self
            .runtimes
            .lock()
            .map_err(|e| anyhow!("Failed to acquire runtimes lock: {}", e))?;
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
        self.tenant_lora_adapters
            .lock()
            .ok()
            .and_then(|adapters| adapters.get(tenant_id.as_ref()).cloned())
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
                if let Some(last_dir) = dirs.last() {
                    info!("Using REALM_MODEL_DIR: {:?}", last_dir);
                }
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
        let mut runtimes_guard = runtimes
            .lock()
            .map_err(|e| anyhow!("Failed to acquire runtime lock: {}", e))?;

        let runtime = runtimes_guard
            .get_mut(tenant_id)
            .ok_or_else(|| anyhow!("No runtime for tenant: {}", tenant_id))?;

        // Check if speculative decoding should be used
        if runtime.draft_model_config().is_some() && runtime.draft_model_id().is_some() {
            // Use speculative decoding
            drop(runtimes_guard);
            use crate::speculative_integration::generate_with_speculative_decoding;
            let runtime_arc = {
                let mut runtimes = runtimes
                    .lock()
                    .map_err(|e| anyhow!("Failed to acquire runtime lock: {}", e))?;
                let runtime = runtimes.remove(tenant_id).ok_or_else(|| {
                    anyhow!("Runtime removed while processing for tenant: {}", tenant_id)
                })?;
                Arc::new(Mutex::new(runtime))
            };
            let result = generate_with_speculative_decoding(runtime_arc.clone(), prompt, 100)?;
            // Put runtime back
            {
                let mut runtimes = runtimes
                    .lock()
                    .map_err(|e| anyhow!("Failed to acquire runtime lock: {}", e))?;
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
            let mut runtimes = match runtimes.lock() {
                Ok(guard) => guard,
                Err(e) => {
                    let _ =
                        tx.blocking_send(format!("Error: Failed to acquire runtime lock: {}", e));
                    return;
                }
            };
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
        let mut runtimes = self
            .runtimes
            .lock()
            .map_err(|e| anyhow!("Failed to acquire runtimes lock: {}", e))?;

        runtimes
            .remove(tenant_id)
            .ok_or_else(|| anyhow!("No runtime for tenant: {}", tenant_id))?;

        info!("Runtime removed for tenant: {}", tenant_id);
        Ok(())
    }

    /// Get number of active runtimes
    pub fn active_runtime_count(&self) -> Result<usize> {
        Ok(self
            .runtimes
            .lock()
            .map_err(|e| anyhow!("Failed to acquire runtime lock: {}", e))?
            .len())
    }

    /// List all tenant IDs
    pub fn list_tenants(&self) -> Result<Vec<String>> {
        Ok(self
            .runtimes
            .lock()
            .map_err(|e| anyhow!("Failed to acquire runtime lock: {}", e))?
            .keys()
            .cloned()
            .collect())
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
        // In tests, channel send failures are expected to be handled
        let _ = blocking_tx.send("token1".to_string());
        let _ = blocking_tx.send("token2".to_string());
        let _ = blocking_tx.send("token3".to_string());

        // Drop the blocking sender to signal completion
        drop(blocking_tx);

        // Receive from async channel with timeout
        let token1 = tokio::time::timeout(std::time::Duration::from_secs(1), async_rx.recv())
            .await
            .expect("Timeout waiting for token1")
            .expect("Channel closed unexpectedly");
        let token2 = tokio::time::timeout(std::time::Duration::from_secs(1), async_rx.recv())
            .await
            .expect("Timeout waiting for token2")
            .expect("Channel closed unexpectedly");
        let token3 = tokio::time::timeout(std::time::Duration::from_secs(1), async_rx.recv())
            .await
            .expect("Timeout waiting for token3")
            .expect("Channel closed unexpectedly");

        assert_eq!(token1, "token1");
        assert_eq!(token2, "token2");
        assert_eq!(token3, "token3");

        // Wait for forwarding task to complete (should finish after sender is dropped)
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), forwarding_handle).await;
    }

    #[test]
    fn test_tenant_id_validation() {
        // Valid tenant IDs
        assert!(RuntimeManager::validate_tenant_id("tenant-123").is_ok());
        assert!(RuntimeManager::validate_tenant_id("user_abc").is_ok());
        assert!(RuntimeManager::validate_tenant_id("test123").is_ok());

        // Invalid tenant IDs
        assert!(RuntimeManager::validate_tenant_id("ab").is_err()); // Too short
        assert!(RuntimeManager::validate_tenant_id("").is_err()); // Empty
        assert!(RuntimeManager::validate_tenant_id(&"a".repeat(65)).is_err()); // Too long
    }

    #[test]
    fn test_model_config_with_draft() {
        let config = ModelConfig {
            model_path: PathBuf::from("/path/to/model.gguf"),
            model_id: "target-model".to_string(),
            draft_model_path: Some(PathBuf::from("/path/to/draft.gguf")),
            draft_model_id: Some("draft-model".to_string()),
        };

        assert_eq!(config.model_id, "target-model");
        assert!(config.draft_model_path.is_some());
        assert_eq!(config.draft_model_id, Some("draft-model".to_string()));
    }

    #[test]
    fn test_runtime_manager_creation() {
        // This test requires a WASM file, so we'll skip if not available
        let wasm_path = PathBuf::from("./target/wasm32-unknown-unknown/release/realm_wasm.wasm");
        if wasm_path.exists() {
            let rm = RuntimeManager::new(wasm_path);
            assert!(rm.is_ok(), "RuntimeManager should be created successfully");
        } else {
            eprintln!("⚠️  WASM file not found, skipping RuntimeManager creation test");
        }
    }

    #[test]
    fn test_is_tenant_id_taken() {
        let wasm_path = PathBuf::from("./target/wasm32-unknown-unknown/release/realm_wasm.wasm");
        if wasm_path.exists() {
            if let Ok(rm) = RuntimeManager::new(wasm_path) {
                // Initially no tenants
                assert!(!rm.is_tenant_id_taken("test-tenant"));
            }
        } else {
            eprintln!("⚠️  WASM file not found, skipping tenant ID test");
        }
    }

    #[test]
    fn test_active_runtime_count() {
        let wasm_path = PathBuf::from("./target/wasm32-unknown-unknown/release/realm_wasm.wasm");
        if wasm_path.exists() {
            if let Ok(rm) = RuntimeManager::new(wasm_path) {
                let count = rm.active_runtime_count();
                assert!(count.is_ok(), "active_runtime_count should succeed");
                assert_eq!(count.unwrap(), 0, "Initially should have 0 runtimes");
            }
        } else {
            eprintln!("⚠️  WASM file not found, skipping runtime count test");
        }
    }

    #[test]
    fn test_list_tenants() {
        let wasm_path = PathBuf::from("./target/wasm32-unknown-unknown/release/realm_wasm.wasm");
        if wasm_path.exists() {
            if let Ok(rm) = RuntimeManager::new(wasm_path) {
                let tenants = rm.list_tenants();
                assert!(tenants.is_ok(), "list_tenants should succeed");
                assert_eq!(
                    tenants.unwrap().len(),
                    0,
                    "Initially should have no tenants"
                );
            }
        } else {
            eprintln!("⚠️  WASM file not found, skipping list tenants test");
        }
    }
}
