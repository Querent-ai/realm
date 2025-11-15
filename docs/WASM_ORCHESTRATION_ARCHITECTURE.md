# WASM Orchestration Architecture

## Core Principle

**WASM = Lightweight Orchestrator | HOST = Heavy Computation**

WASM provides a standardized interface for AI apps to interact with models, but **never loads model weights**. All computation happens on the HOST side.

## Architecture Flow

```
realm-server (Rust)
    ↓
WASM generate() (orchestration only)
    ↓
HOST realm_host_generate() (actual inference)
    ↓
Model.forward() (weights in HOST memory)
    ↓
Result
```

## Key Components

### 1. WASM Layer (`realm-wasm/src/lib.rs`)

**Purpose**: Provides standardized API for AI apps

**What it does**:
- ✅ Lightweight orchestration
- ✅ Calls HOST functions
- ✅ Manages control flow
- ❌ **NEVER loads model weights**
- ❌ **NEVER does heavy computation**

**Example** (`generate()` function):
```rust
#[no_mangle]
pub extern "C" fn generate(prompt_ptr: u32, prompt_len: u32) -> u32 {
    // WASM orchestrates by calling HOST
    let bytes_written = unsafe {
        realm_host_generate(
            model_id,
            prompt_ptr,
            prompt_len,
            output_ptr,
            OUTPUT_BUFFER_SIZE,
        )
    };
    // Return result pointer
    output_ptr as u32
}
```

### 2. HOST Layer (`realm-runtime/src/memory64_host.rs`)

**Purpose**: Does all heavy computation

**What it does**:
- ✅ Stores model weights (quantized, in HOST memory)
- ✅ Loads Model instances for inference
- ✅ Performs forward passes
- ✅ GPU acceleration
- ✅ Returns only activations/results to WASM

**Example** (`realm_host_generate` function):
```rust
linker.func_wrap("env", "realm_host_generate", |...| {
    // Get Model instance (weights in HOST, not WASM)
    let mut model = stored_model.load_model_instance()?;
    
    // Create InferenceSession
    let mut session = InferenceSession::new(model_id, prompt_tokens, options);
    
    // Generate tokens (computation happens here)
    while !session.is_complete() {
        session.next_token_with_model(&mut model, None)?;
    }
    
    // Return result to WASM (only text, no weights)
    tokenizer.decode(&generated_tokens, true)
});
```

## Data Flow

### Weights (NEVER in WASM)
```
GGUF File → HOST Storage (quantized) → Model Instance (HOST) → Forward Pass (HOST)
```

### Activations (flow to/from WASM)
```
WASM: prompt (string)
    ↓
HOST: tokenize → prompt_tokens
    ↓
HOST: model.forward() → logits
    ↓
HOST: sample → token_id
    ↓
HOST: decode → text
    ↓
WASM: result (string)
```

## Benefits

1. **Multi-tenant Isolation**: Each tenant gets isolated WASM sandbox
2. **Memory Efficiency**: Model shared across all tenants (637MB → shared)
3. **Performance**: All heavy compute (GPU ops) stays in HOST
4. **Security**: WASM prevents tenant code from accessing others' data
5. **Scalability**: 100s of WASM instances per GB RAM
6. **Standardization**: WASM provides consistent API for AI apps

## Host Functions Available

WASM can call these HOST functions for orchestration:

1. **`realm_host_generate`** - Complete inference (current approach)
   - WASM: Just calls this
   - HOST: Does everything

2. **`realm_forward_layer`** - Single layer forward pass
   - WASM: Orchestrates loop over layers
   - HOST: Computes each layer

3. **`realm_encode_tokens`** - Tokenization
   - WASM: Calls for tokenization
   - HOST: Uses tokenizer from storage

4. **`realm_decode_tokens`** - Detokenization
   - WASM: Calls for decoding
   - HOST: Uses tokenizer from storage

## Current Implementation

✅ **Correct**: `realm_host_generate` is a HOST function that uses Model instance
✅ **Correct**: WASM `generate()` just calls HOST function (orchestration)
✅ **Correct**: Weights stay in HOST, never enter WASM

## Future Optimizations

1. **Granular Orchestration**: WASM could orchestrate by calling `realm_forward_layer` for each layer
2. **Model Caching**: Cache loaded Model instances in HOST to avoid reloading
3. **Streaming**: Add streaming host functions for token-by-token generation

## Key Takeaway

**WASM provides standardization and isolation, HOST provides computation and storage.**

This architecture enables:
- Building AI apps directly in WASM (standardized interface)
- Not needing to interact with APIs (direct model access)
- Multi-tenant isolation (each app in its own WASM sandbox)
- Efficient resource usage (shared models, isolated execution)

