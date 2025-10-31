# Host-Side Storage Architecture

## Overview

Realm implements an innovative **host-side quantized storage** system that solves WASM memory limitations by storing model weights in the HOST runtime instead of WASM memory.

### The Problem

```
Traditional WASM Inference:
┌────────────────────────────┐
│  WASM Memory (2GB limit)   │
│                            │
│  Model: TinyLlama 1.1B     │
│  - 637MB quantized (Q4_K)  │
│  - Dequantized → 2.5GB f32 │
│  - Result: OUT OF MEMORY ❌│
└────────────────────────────┘
```

### The Solution

```
Host-Side Storage:
┌──────────────────────┐          ┌─────────────────────────┐
│  WASM (Lightweight)  │          │  HOST (Heavy Storage)   │
│                      │          │                         │
│  model_id: 42  (4B)  │◄────────►│  Model 42:              │
│  config      (~1KB)  │   FFI    │  - 637MB quantized      │
│  activations (~10MB) │          │  - Shared multi-tenant  │
│  KV cache    (~20MB) │          │  - LRU cache (future)   │
│                      │          │                         │
│  Total: ~50MB ✅     │          │  Total: 637MB ✅        │
└──────────────────────┘          └─────────────────────────┘
```

### Memory Savings

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| WASM | 2.5GB (OOM) | ~50MB | **98%** |
| HOST | N/A | 637MB | Shared |
| Multi-tenant | N × 2.5GB | 637MB | **N×** |

---

## Architecture Components

### 1. Model Storage (`crates/realm-runtime/src/model_storage.rs`)

**Core Types:**

```rust
/// Quantized tensor in HOST memory
pub struct QuantizedTensor {
    pub data: Vec<u8>,      // Raw Q4_K_M/Q8_0 bytes
    pub dtype: DataType,     // Quantization format
    pub shape: Vec<u64>,     // Tensor dimensions
    pub name: String,        // e.g., "blk.0.attn_q.weight"
}

/// Complete model with all tensors
pub struct StoredModel {
    pub id: u32,                                    // Unique model ID
    pub metadata: ModelMeta,                        // GGUF metadata
    pub tensors: HashMap<String, QuantizedTensor>,  // All model weights
    pub total_size: usize,                          // Total bytes
}

/// Thread-safe global storage
pub struct ModelStorage {
    models: Arc<Mutex<HashMap<u32, StoredModel>>>,
}

lazy_static! {
    pub static ref GLOBAL_MODEL_STORAGE: ModelStorage = ModelStorage::new();
}
```

**Key Design:**
- Weights stay quantized (Q4_K_M format)
- Thread-safe with `Arc<Mutex<>>`
- Global singleton via `lazy_static`
- Atomic model ID generation

### 2. FFI Host Functions (`crates/realm-runtime/src/memory64_host.rs`)

**Available Functions:**

```rust
// Store GGUF model in HOST
realm_store_model(gguf_ptr: *const u8, gguf_len: u32) -> model_id

// Get tensor (auto-dequantizes to f32)
realm_get_tensor(
    model_id: u32,
    tensor_name_ptr: *const u8,
    tensor_name_len: u32,
    out_ptr: *mut u8,
    out_max_len: u32
) -> bytes_written

// Get model info
realm_get_model_info(
    model_id: u32,
    out_tensor_count: *mut u32,
    out_total_size: *mut u64
) -> status

// Cleanup
realm_remove_model(model_id: u32) -> status
```

**Implementation Details:**
- Full WASM pointer validation
- Overflow protection
- Automatic dequantization
- Comprehensive error handling

### 3. Dequantization (`crates/realm-core/src/quant.rs`)

**Main Entry Point:**

```rust
pub fn dequantize_tensor(
    data: &[u8],
    dtype: DataType,
    element_count: usize
) -> Result<Vec<f32>>
```

**Supported Formats:**
- Q4_K (TinyLlama default)
- Q5_K, Q6_K, Q8_K
- Q8_0
- F32, F16

**Performance:**
- Block-wise processing (256 elements)
- Zero-copy when possible
- Parallel-ready (future)

### 4. WASM Integration (`crates/realm-wasm/src/lib.rs`)

**Modified loadModel():**

```rust
#[wasm_bindgen(js_name = loadModel)]
pub fn load_model(&mut self, model_bytes: &[u8]) -> Result<(), JsError> {
    // Parse header (lightweight)
    let meta = parser.parse_header()?;
    let config = extract_config(&meta)?;
    let tokenizer = Tokenizer::from_gguf(&meta)?;

    #[cfg(target_arch = "wasm32")]
    {
        // Store in HOST
        let model_id = unsafe {
            realm_store_model(model_bytes.as_ptr(), model_bytes.len() as u32)
        };

        // Keep only handle + config in WASM
        self.model_id = Some(model_id as u32);
        self.transformer_config = Some(config);
        self.tokenizer = Some(tokenizer);
    }

    Ok(())
}
```

---

## Current Status

### ✅ Completed

1. **Model Storage Infrastructure**
   - `QuantizedTensor`, `StoredModel`, `ModelStorage`
   - Thread-safe global storage
   - GGUF parsing and tensor extraction

2. **FFI Host Functions**
   - `realm_store_model()` - Store model in HOST
   - `realm_get_tensor()` - Retrieve + dequantize tensors
   - `realm_get_model_info()` - Get metadata
   - `realm_remove_model()` - Cleanup

3. **Dequantization Support**
   - `dequantize_tensor()` - Universal dequantizer
   - All Q4/Q5/Q6/Q8 formats supported
   - Integrated with FFI

4. **WASM Module Updates**
   - Modified `loadModel()` to use HOST storage
   - Extern declarations for host functions
   - Lightweight model handles

### ⏳ In Progress

5. **Inference Path Integration**
   - Need: Connect `Model::forward()` to host functions
   - Need: On-demand tensor loading during inference
   - Status: HOST functions ready, WASM inference not yet connected

### 📋 TODO

6. **LRU Caching Layer** (See Section below)
7. **Neon Bridge for Node.js** (See Section below)
8. **End-to-End Testing**
9. **Performance Optimization**
10. **Memory Management Polish**

---

## Next Steps

### Step 1: Complete Inference Path (2-3 hours)

**Goal:** Enable WASM inference to load weights from HOST on-demand.

**Changes Needed:**

```rust
// In realm-wasm/src/lib.rs

pub fn generate(&mut self, prompt: String) -> Result<String, JsError> {
    let model_id = self.model_id.ok_or(JsError::new("Model not loaded"))?;
    let config = self.transformer_config.as_ref().unwrap();

    // Tokenize
    let tokens = self.tokenizer.as_ref().unwrap().encode(&prompt)?;

    // Get embeddings from HOST
    let embedding_size = config.vocab_size * config.hidden_size;
    let mut embedding_data = vec![0f32; embedding_size];

    #[cfg(target_arch = "wasm32")]
    {
        let bytes_written = unsafe {
            realm_get_tensor(
                model_id,
                "token_embd.weight".as_ptr(),
                "token_embd.weight".len() as u32,
                embedding_data.as_mut_ptr() as *mut u8,
                (embedding_size * 4) as u32,
            )
        };

        if bytes_written < 0 {
            return Err(JsError::new("Failed to load embeddings"));
        }
    }

    // For each layer, load weights on-demand
    for layer_id in 0..config.num_layers {
        // Load attention weights
        let wq = load_weight(model_id, &format!("blk.{}.attn_q.weight", layer_id))?;
        let wk = load_weight(model_id, &format!("blk.{}.attn_k.weight", layer_id))?;
        let wv = load_weight(model_id, &format!("blk.{}.attn_v.weight", layer_id))?;
        let wo = load_weight(model_id, &format!("blk.{}.attn_output.weight", layer_id))?;

        // Load FFN weights
        let w_gate = load_weight(model_id, &format!("blk.{}.ffn_gate.weight", layer_id))?;
        let w_up = load_weight(model_id, &format!("blk.{}.ffn_up.weight", layer_id))?;
        let w_down = load_weight(model_id, &format!("blk.{}.ffn_down.weight", layer_id))?;

        // Compute layer forward
        hidden = layer.forward(&hidden, &wq, &wk, &wv, &wo, &w_gate, &w_up, &w_down)?;
    }

    // Sample token
    // ...
}

fn load_weight(model_id: u32, name: &str) -> Result<Vec<f32>, JsError> {
    #[cfg(target_arch = "wasm32")]
    {
        // Calculate size from tensor name (parse from metadata)
        let size = calculate_tensor_size(name)?;
        let mut data = vec![0f32; size];

        let bytes_written = unsafe {
            realm_get_tensor(
                model_id,
                name.as_ptr(),
                name.len() as u32,
                data.as_mut_ptr() as *mut u8,
                (size * 4) as u32,
            )
        };

        if bytes_written < 0 {
            return Err(JsError::new(&format!("Failed to load {}", name)));
        }

        Ok(data)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Native path - weights already in memory
        unimplemented!()
    }
}
```

**Testing:**

```bash
# 1. Build WASM module
cd crates/realm-wasm
wasm-pack build --target nodejs --release

# 2. Test with Wasmtime runner (needs wasm-bindgen shim)
cargo build --release -p wasm-host-runner
./examples/wasm-host-runner/target/release/wasm-host-runner

# Expected output:
# ✅ Model stored in HOST with ID 1
# ✅ Generated: "The capital of France is Paris."
# ✅ WASM memory: ~50MB
```

---

## LRU Caching Layer Design

### Architecture

```
┌─────────────────────────────────────────────────────┐
│              Tensor Request Flow                    │
│                                                      │
│  1. Request: blk.5.attn_q.weight                    │
│                ↓                                     │
│  2. Check LRU Cache                                 │
│                ↓                                     │
│     ┌──────────┴──────────┐                         │
│     │                     │                         │
│  HIT│                     │MISS                     │
│     ↓                     ↓                         │
│  Return cached        3. Load from storage          │
│  (instant)               ↓                          │
│                     4. Dequantize (Q4_K → f32)      │
│                        ↓                            │
│                     5. Cache it                     │
│                        ↓                            │
│                     6. Return                       │
└─────────────────────────────────────────────────────┘
```

### Implementation

```rust
// In realm-runtime/src/layer_cache.rs (NEW FILE)

use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Cache key: (model_id, tensor_name)
type CacheKey = (u32, String);

/// Cached tensor data (dequantized f32)
#[derive(Clone)]
struct CachedTensor {
    data: Vec<f32>,
    size_bytes: usize,
}

/// LRU cache for dequantized tensors
pub struct LayerCache {
    cache: Arc<Mutex<LruCache<CacheKey, CachedTensor>>>,
    max_size_bytes: usize,
    current_size: Arc<parking_lot::Mutex<usize>>,

    // Metrics
    hits: Arc<std::sync::atomic::AtomicU64>,
    misses: Arc<std::sync::atomic::AtomicU64>,
}

impl LayerCache {
    /// Create new cache with max size in bytes
    pub fn new(max_size_mb: usize) -> Self {
        let max_size_bytes = max_size_mb * 1024 * 1024;
        let capacity = NonZeroUsize::new(1000).unwrap(); // Max 1000 tensors

        Self {
            cache: Arc::new(Mutex::new(LruCache::new(capacity))),
            max_size_bytes,
            current_size: Arc::new(parking_lot::Mutex::new(0)),
            hits: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            misses: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Get tensor from cache
    pub fn get(&self, model_id: u32, tensor_name: &str) -> Option<Vec<f32>> {
        let key = (model_id, tensor_name.to_string());
        let mut cache = self.cache.lock();

        if let Some(cached) = cache.get(&key) {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(cached.data.clone())
        } else {
            self.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }

    /// Put tensor in cache
    pub fn put(&self, model_id: u32, tensor_name: String, data: Vec<f32>) {
        let size_bytes = data.len() * 4; // f32 = 4 bytes
        let key = (model_id, tensor_name);
        let cached = CachedTensor { data, size_bytes };

        // Evict until we have space
        self.ensure_space(size_bytes);

        let mut cache = self.cache.lock();
        if let Some(old) = cache.push(key, cached) {
            // Evicted old tensor
            let mut current = self.current_size.lock();
            *current -= old.size_bytes;
        }

        let mut current = self.current_size.lock();
        *current += size_bytes;
    }

    /// Ensure we have space for new_size bytes
    fn ensure_space(&self, new_size: usize) {
        let mut cache = self.cache.lock();
        let mut current = self.current_size.lock();

        while *current + new_size > self.max_size_bytes && !cache.is_empty() {
            if let Some((_, evicted)) = cache.pop_lru() {
                *current -= evicted.size_bytes;
                tracing::debug!("Evicted tensor from cache: {} bytes", evicted.size_bytes);
            }
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.misses.load(std::sync::atomic::Ordering::Relaxed);
        let current_size = *self.current_size.lock();
        let cache = self.cache.lock();

        CacheStats {
            hits,
            misses,
            hit_rate: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
            current_size_mb: current_size as f64 / 1024.0 / 1024.0,
            max_size_mb: self.max_size_bytes as f64 / 1024.0 / 1024.0,
            entry_count: cache.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub current_size_mb: f64,
    pub max_size_mb: f64,
    pub entry_count: usize,
}

/// Global layer cache
lazy_static::lazy_static! {
    pub static ref GLOBAL_LAYER_CACHE: LayerCache = LayerCache::new(500); // 500MB cache
}
```

### Integration with FFI

```rust
// In memory64_host.rs, modify realm_get_tensor:

// Before dequantization, check cache
use crate::layer_cache::GLOBAL_LAYER_CACHE;

if let Some(cached_data) = GLOBAL_LAYER_CACHE.get(model_id, tensor_name) {
    // Cache hit!
    let f32_bytes: Vec<u8> = cached_data
        .iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();

    wasm_memory.write(&mut caller, out_ptr as usize, &f32_bytes)?;
    return f32_bytes.len() as i32;
}

// Cache miss - dequantize and cache
let dequantized = dequantize_tensor(&tensor.data, tensor.dtype, element_count)?;

// Put in cache for next time
GLOBAL_LAYER_CACHE.put(model_id, tensor_name.to_string(), dequantized.clone());

// Return to caller
// ...
```

### Dependencies

Add to `realm-runtime/Cargo.toml`:

```toml
[dependencies]
lru = "0.12"
```

### Expected Performance

**Without Cache:**
- Each layer forward: Load 7 tensors × 5ms dequant = 35ms
- 22 layers = 770ms overhead per token

**With Cache (after warmup):**
- Cache hit rate: 95%+ (same layers repeated)
- Each layer forward: 7 × 0.1ms = 0.7ms
- 22 layers = 15ms overhead per token
- **50× speedup after warmup**

---

## Neon Bridge Setup Guide

### Overview

Create a Node.js native addon that provides HOST functions to WASM module.

```
┌──────────────────────────────────────────────┐
│          JavaScript (Node.js)                │
│                                              │
│  const realm = require('@realm-ai/sdk');    │
│  realm.loadModel(modelBytes);               │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│     WASM Module (wasm-bindgen)               │
│                                              │
│  extern "C" {                                │
│    fn realm_store_model(...);                │
│  }                                           │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│    JS Shim (realm-bridge.js)                 │
│                                              │
│  export function realm_store_model(ptr, len) │
│    return nativeAddon.storeModel(ptr, len);  │
│  }                                           │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│    Native Addon (@realm-ai/native)           │
│    Built with neon-bindings                  │
│                                              │
│  Uses realm-runtime directly                 │
└──────────────────────────────────────────────┘
```

### Step 1: Create Neon Project

```bash
cd crates
npm init neon realm-native

cd realm-native
```

### Step 2: Add Rust Dependencies

Edit `Cargo.toml`:

```toml
[dependencies]
neon = "1.0"
realm-runtime = { path = "../realm-runtime", features = ["memory64-host"] }
realm-core = { path = "../realm-core" }
```

### Step 3: Implement Native Functions

Edit `src/lib.rs`:

```rust
use neon::prelude::*;
use realm_runtime::model_storage::GLOBAL_MODEL_STORAGE;
use realm_core::quant::dequantize_tensor;

/// Store model from GGUF bytes
fn store_model(mut cx: FunctionContext) -> JsResult<JsNumber> {
    // Get Uint8Array from JS
    let buffer = cx.argument::<JsBuffer>(0)?;
    let bytes = cx.borrow(&buffer, |data| data.as_slice::<u8>());

    // Store in global storage
    match GLOBAL_MODEL_STORAGE.store_model(bytes) {
        Ok(model_id) => Ok(cx.number(model_id as f64)),
        Err(e) => cx.throw_error(format!("Failed to store model: {}", e)),
    }
}

/// Get tensor and dequantize
fn get_tensor(mut cx: FunctionContext) -> JsResult<JsBuffer> {
    let model_id = cx.argument::<JsNumber>(0)?.value(&mut cx) as u32;
    let tensor_name = cx.argument::<JsString>(1)?.value(&mut cx);

    // Get from storage
    let tensor = GLOBAL_MODEL_STORAGE
        .get_tensor(model_id, &tensor_name)
        .or_else(|e| cx.throw_error(format!("Tensor not found: {}", e)))?;

    // Dequantize
    let element_count = tensor.element_count() as usize;
    let dequantized = dequantize_tensor(&tensor.data, tensor.dtype, element_count)
        .or_else(|e| cx.throw_error(format!("Dequantization failed: {}", e)))?;

    // Convert to bytes
    let bytes: Vec<u8> = dequantized
        .iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();

    // Return as Buffer
    let mut result = cx.buffer(bytes.len())?;
    cx.borrow_mut(&mut result, |data| {
        data.as_mut_slice().copy_from_slice(&bytes);
    });

    Ok(result)
}

/// Remove model
fn remove_model(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let model_id = cx.argument::<JsNumber>(0)?.value(&mut cx) as u32;

    GLOBAL_MODEL_STORAGE
        .remove_model(model_id)
        .or_else(|e| cx.throw_error(format!("Failed to remove model: {}", e)))?;

    Ok(cx.undefined())
}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("storeModel", store_model)?;
    cx.export_function("getTensor", get_tensor)?;
    cx.export_function("removeModel", remove_model)?;
    Ok(())
}
```

### Step 4: Build Native Addon

```bash
npm run build
```

### Step 5: Create JS Bridge

Create `js/bridge.js`:

```javascript
const native = require('../native');
const { instantiate } = require('../realm-wasm/pkg');

// WASM instance
let wasmInstance = null;

// Initialize WASM with native backend
async function initWasm(wasmBytes) {
    // Provide host functions to WASM
    const imports = {
        env: {
            realm_store_model: (ptr, len) => {
                // Read from WASM memory
                const bytes = new Uint8Array(
                    wasmInstance.exports.memory.buffer,
                    ptr,
                    len
                );

                // Call native addon
                return native.storeModel(Buffer.from(bytes));
            },

            realm_get_tensor: (modelId, namePtr, nameLen, outPtr, outMaxLen) => {
                // Read tensor name from WASM
                const nameBytes = new Uint8Array(
                    wasmInstance.exports.memory.buffer,
                    namePtr,
                    nameLen
                );
                const name = new TextDecoder().decode(nameBytes);

                // Get from native addon
                const tensorData = native.getTensor(modelId, name);

                // Write to WASM memory
                if (tensorData.length > outMaxLen) {
                    return -1; // Buffer too small
                }

                const wasmMemory = new Uint8Array(
                    wasmInstance.exports.memory.buffer,
                    outPtr,
                    tensorData.length
                );
                wasmMemory.set(tensorData);

                return tensorData.length;
            },

            realm_remove_model: (modelId) => {
                native.removeModel(modelId);
                return 0;
            },
        },
    };

    wasmInstance = await instantiate(wasmBytes, imports);
    return wasmInstance;
}

module.exports = {
    initWasm,
    native,
};
```

### Step 6: Create User-Facing API

Create `js/index.js`:

```javascript
const { initWasm, native } = require('./bridge');
const fs = require('fs');
const path = require('path');

class Realm {
    constructor() {
        this.wasm = null;
        this.realm = null;
    }

    async init() {
        // Load WASM module
        const wasmPath = path.join(__dirname, '../realm-wasm/pkg/realm_wasm_bg.wasm');
        const wasmBytes = fs.readFileSync(wasmPath);

        this.wasm = await initWasm(wasmBytes);
        this.realm = this.wasm.Realm.new();
    }

    loadModel(modelBytes) {
        return this.realm.loadModel(modelBytes);
    }

    generate(prompt, options = {}) {
        const config = new this.wasm.WasmGenerationConfig();
        config.max_tokens = options.maxTokens || 100;
        config.temperature = options.temperature || 0.7;

        this.realm.setConfig(config);
        return this.realm.generate(prompt);
    }
}

module.exports = { Realm };
```

### Step 7: Test End-to-End

Create `test.js`:

```javascript
const { Realm } = require('./js');
const fs = require('fs');

async function main() {
    const realm = new Realm();
    await realm.init();

    // Load model
    const modelPath = process.env.MODEL_PATH || 'tinyllama-1.1b.Q4_K_M.gguf';
    const modelBytes = fs.readFileSync(modelPath);

    console.log('Loading model...');
    realm.loadModel(modelBytes);

    console.log('Generating...');
    const response = realm.generate('What is the capital of France?', {
        maxTokens: 50,
        temperature: 0.0,
    });

    console.log('Response:', response);
    // Expected: "Paris"
}

main().catch(console.error);
```

```bash
node test.js
```

### Expected Output

```
Loading model...
✅ Model stored in HOST with ID 1 (637MB)
Generating...
✅ Cache miss: token_embd.weight (dequantizing...)
✅ Cache miss: blk.0.attn_q.weight (dequantizing...)
✅ Cache miss: blk.0.attn_k.weight (dequantizing...)
... (more cache misses on first run)
Response: The capital of France is Paris.

Memory usage:
- HOST: 637MB (quantized model)
- HOST cache: 150MB (dequantized tensors)
- WASM: 45MB (activations + KV cache)
```

### Publishing

```bash
# Package structure:
@realm-ai/
├── native/          # Neon addon
├── wasm/            # WASM module
└── sdk/             # User-facing API
    ├── index.js
    └── package.json

# Publish to npm
npm publish --access public
```

---

## Performance Optimization Roadmap

### Phase 1: Parallel Dequantization

```rust
use rayon::prelude::*;

pub fn dequantize_tensor_parallel(
    data: &[u8],
    dtype: DataType,
    element_count: usize
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; element_count];

    match dtype {
        DataType::Q4_K => {
            let block_size = 256;
            let bytes_per_block = 144;
            let num_blocks = data.len() / bytes_per_block;

            // Parallel dequantization
            output
                .par_chunks_mut(block_size)
                .zip(data.par_chunks(bytes_per_block))
                .for_each(|(out_chunk, data_chunk)| {
                    let block = unsafe { &*(data_chunk.as_ptr() as *const BlockQ4_K) };
                    dequantize_q4_k(block, out_chunk).unwrap();
                });
        }
        // ...
    }

    Ok(output)
}
```

### Phase 2: Prefetching Pipeline

```rust
use tokio::task;

async fn forward_with_prefetch(
    model_id: u32,
    hidden_states: Vec<f32>,
    layer_id: usize,
) -> Result<Vec<f32>> {
    // Spawn prefetch for next layer
    let next_layer_id = layer_id + 1;
    let prefetch_handle = task::spawn(async move {
        prefetch_layer(model_id, next_layer_id).await
    });

    // Compute current layer
    let output = compute_layer(model_id, hidden_states, layer_id).await?;

    // Wait for prefetch (likely already done)
    prefetch_handle.await?;

    Ok(output)
}
```

### Phase 3: GPU Acceleration

```rust
// Dequantize directly on GPU
#[cfg(feature = "cuda")]
fn dequantize_q4k_cuda(
    data: &[u8],
    output: &mut [f32],
    cuda_stream: &CudaStream,
) -> Result<()> {
    // Use cuBLAS/cutlass for fast dequantization
    // 10-100× faster than CPU
}
```

---

## Troubleshooting

### WASM OOM during loadModel

**Symptom:** `RuntimeError: unreachable` during model loading

**Cause:** Weights being loaded into WASM memory instead of HOST

**Solution:** Ensure `#[cfg(target_arch = "wasm32")]` path calls `realm_store_model()`

### Cache Not Working

**Symptom:** Slow inference, high dequantization time

**Cause:** Cache not integrated with FFI

**Solution:** Verify `GLOBAL_LAYER_CACHE.get()` is called before dequantization

### Neon Build Fails

**Symptom:** `error: failed to run custom build command for 'neon'`

**Cause:** Missing Node.js headers

**Solution:**
```bash
npm install -g node-gyp
node-gyp configure
npm run build
```

### WASM Import Errors

**Symptom:** `unknown import: realm_store_model has not been defined`

**Cause:** Host functions not provided to WASM

**Solution:** Ensure bridge provides all functions in `imports.env`

---

## Summary

**Completed:**
- ✅ Host-side storage infrastructure
- ✅ FFI functions with dequantization
- ✅ WASM loadModel integration

**In Progress:**
- 🔄 Inference path connection
- 🔄 LRU caching layer

**Next:**
- 📋 Complete inference loop
- 📋 Add caching
- 📋 Neon bridge
- 📋 End-to-end testing
- 📋 Performance optimization

**Architecture Benefits:**
- 98% memory reduction in WASM
- Multi-tenant model sharing
- Production-ready design
- Extensible for caching/prefetching
