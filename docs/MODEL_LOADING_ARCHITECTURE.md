# Model Loading Architecture - Best Practices

## Current Implementation

### How It Works Now

**Server-Side Loading (Current)**:
```
1. Server reads model file from disk
2. Server stores in HOST storage → gets model_id
3. Server passes model_id to WASM generate()
```

**Flow**:
```
HTTP Request → Server load_model() → HOST Storage → model_id → WASM generate()
```

**Issues**:
- ❌ Model loading is server-side only
- ❌ WASM can't dynamically load models
- ❌ No way for WASM to switch models at runtime
- ❌ Limited orchestration capabilities

## Recommended Approach

### Host Function for Model Loading by Name

**YES, we should have a host function `realm_host_load_model_by_name`**

This enables true WASM orchestration where WASM can:
- Dynamically choose which model to use
- Load models based on user input
- Switch models mid-pipeline
- Build multi-model applications

### Architecture

```
WASM (Orchestrator)
    ↓ calls realm_host_load_model_by_name("tinyllama")
    ↓
HOST (Computation)
    ↓ reads file, stores in HOST storage
    ↓ returns model_id
    ↓
WASM (Orchestrator)
    ↓ uses model_id for inference
```

### Implementation

**Host Function** (`realm-runtime/src/memory64_host.rs`):
```rust
linker.func_wrap(
    "env",
    "realm_host_load_model_by_name",
    move |mut caller: Caller<'_, ()>,
          model_name_ptr: u32,
          model_name_len: u32|
          -> i32 {
        // Read model name from WASM memory
        let model_name = read_string_from_wasm(&caller, model_name_ptr, model_name_len)?;
        
        // HOST handles file I/O
        let model_path = format!("models/{}.gguf", model_name);
        let model_bytes = std::fs::read(&model_path)?;
        
        // Store in HOST storage
        let storage = get_global_model_storage().lock();
        let model_id = storage.store_model(&model_bytes, None)?;
        
        // Return model_id to WASM
        model_id as i32
    }
)
```

**WASM Function** (`realm-wasm/src/lib.rs`):
```rust
extern "C" {
    fn realm_host_load_model_by_name(
        model_name_ptr: *const u8,
        model_name_len: u32,
    ) -> i32; // Returns model_id or negative on error
}

#[no_mangle]
pub extern "C" fn load_model_by_name(model_name_ptr: u32, model_name_len: u32) -> i32 {
    let model_name = unsafe {
        // Read string from WASM memory
        let slice = std::slice::from_raw_parts(
            model_name_ptr as *const u8,
            model_name_len as usize
        );
        std::str::from_utf8(slice).unwrap()
    };
    
    // Call HOST to load model
    let model_id = unsafe {
        realm_host_load_model_by_name(
            model_name_ptr as *const u8,
            model_name_len,
        )
    };
    
    if model_id < 0 {
        wasm_log!("Failed to load model: error code {}", model_id);
        return model_id;
    }
    
    wasm_log!("Model loaded with ID: {}", model_id);
    model_id
}
```

## Benefits

1. **True Orchestration**: WASM can decide which model to use
2. **Dynamic Loading**: Load models based on runtime conditions
3. **Multi-Model Pipelines**: Switch between models in a single request
4. **Better Isolation**: Each WASM instance can load its own models
5. **Standardized Interface**: Consistent API for AI apps

## Comparison

### Option A: Server-Side Loading (Current)
- ✅ Simple
- ❌ Limited orchestration
- ❌ WASM can't choose models
- ❌ Less flexible

### Option B: Host Function by Name (Recommended)
- ✅ True WASM orchestration
- ✅ Dynamic model selection
- ✅ Multi-model support
- ✅ Better for complex apps
- ⚠️ Requires model name → path mapping

### Option C: Host Function by Path
- ✅ Full flexibility
- ✅ No mapping needed
- ⚠️ Security concerns (path validation needed)
- ⚠️ Less user-friendly

## Recommendation

**Use Option B: `realm_host_load_model_by_name`**

This provides the best balance of:
- Orchestration capabilities
- Security (controlled model paths)
- User-friendliness (names vs paths)
- Flexibility for AI apps

## Implementation Priority

1. **High**: Add `realm_host_load_model_by_name` host function
2. **High**: Add WASM `load_model_by_name()` function
3. **Medium**: Add model registry/name mapping
4. **Low**: Add model caching by name

