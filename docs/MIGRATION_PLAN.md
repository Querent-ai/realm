# Realm.ai Migration Plan

This document outlines the detailed migration plan from the experimental `wasm-chord` repository to the production-ready `realm` architecture.

## Phase 1: Core Infrastructure (Current)

### ✅ Completed

1. **Repository Structure Created**
   - Created workspace with 6 core crates + CLI + server
   - Set up proper Cargo.toml with shared dependencies
   - Created placeholder files for all crates

2. **realm-core Extracted** ✅
   - Successfully copied from `wasm-chord-core`
   - Contains: GGUF parser, tokenizer, tensor loader, quantization, error types
   - **Status**: Builds successfully!

### 🔧 Next Steps

#### Step 1.1: Extract realm-models

**Source**: `/home/puneet/wasm-chord/crates/wasm-chord-runtime/src/transformer/`

**Files to copy**:
```
transformer/
├── attention.rs       → realm-models/src/attention.rs
├── ffn.rs            → realm-models/src/ffn.rs
├── layer.rs          → realm-models/src/layer.rs
├── model.rs          → realm-models/src/model.rs
├── mod.rs            → realm-models/src/transformer.rs
└── kv_cache.rs       → realm-models/src/kv_cache.rs
```

**Required changes**:
- Replace `use wasm_chord_core::` with `use realm_core::`
- Replace `use wasm_chord_cpu::` with `use realm_compute_cpu::`
- Replace `use wasm_chord_gpu::` with `use realm_compute_gpu::`
- Remove Memory64-specific code (will move to realm-runtime)
- Create `lib.rs` that re-exports all modules

**Create**:
```rust
// realm-models/src/lib.rs
pub mod attention;
pub mod ffn;
pub mod layer;
pub mod transformer;
pub mod kv_cache;

pub use attention::MultiHeadAttention;
pub use ffn::FeedForward;
pub use layer::TransformerLayer;
pub use transformer::{TransformerConfig, Model};
pub use kv_cache::KVCache;
```

#### Step 1.2: Extract realm-compute-cpu

**Source**: `/home/puneet/wasm-chord/crates/wasm-chord-cpu/src/`

**Files to copy**:
```
wasm-chord-cpu/src/
├── lib.rs                    → realm-compute-cpu/src/lib.rs
├── candle_backend.rs         → realm-compute-cpu/src/candle_backend.rs
├── candle_cpu_backend.rs     → realm-compute-cpu/src/candle_cpu_backend.rs
├── cpu_backend_trait.rs      → realm-compute-cpu/src/cpu_backend_trait.rs
├── naive_backend.rs          → realm-compute-cpu/src/naive_backend.rs
├── fused/                    → realm-compute-cpu/src/fused/
├── gemm/                     → realm-compute-cpu/src/gemm/
└── kernels/                  → realm-compute-cpu/src/kernels/
```

**Required changes**:
- Replace `wasm_chord_core` imports with `realm_core`
- Update lib.rs exports
- Keep Candle backends optional behind features

#### Step 1.3: Extract realm-compute-gpu

**Source**: `/home/puneet/wasm-chord/crates/wasm-chord-gpu/src/`

**Files to copy**:
```
wasm-chord-gpu/src/
├── lib.rs                    → realm-compute-gpu/src/lib.rs
├── candle_gpu_backend.rs     → realm-compute-gpu/src/candle_gpu_backend.rs
├── gpu_backend_trait.rs      → realm-compute-gpu/src/gpu_backend_trait.rs
└── webgpu/                   → realm-compute-gpu/src/webgpu/ (if exists)
```

**Required changes**:
- Replace `wasm_chord_core` imports with `realm_core`
- Update feature flags for CUDA/Metal
- Keep CUDA and Metal behind separate features

#### Step 1.4: Build realm-runtime

**Source**: `/home/puneet/wasm-chord/crates/wasm-chord-runtime/src/memory64_host.rs`

**Key Components to Extract**:

1. **Memory64 Manager**:
   ```
   memory64_host.rs → realm-runtime/src/memory64.rs
   ```

2. **Host Functions**:
   - `candle_matmul()`
   - `candle_matmul_transposed()`
   - `memory64_store_layer()`
   - `memory64_load_layer()`
   - `memory64_get_cache_stats()`

3. **Create New Files**:
   - `realm-runtime/src/lib.rs` - Main entry point
   - `realm-runtime/src/runtime.rs` - Runtime struct
   - `realm-runtime/src/memory64.rs` - Memory64 manager
   - `realm-runtime/src/host_functions.rs` - Host function exports
   - `realm-runtime/src/candle_integration.rs` - Candle backend wrappers

**Structure**:
```rust
// realm-runtime/src/lib.rs
pub mod runtime;
pub mod memory64;
pub mod host_functions;
pub mod candle_integration;

pub use runtime::RealmRuntime;
pub use memory64::Memory64Manager;
```

**Key Runtime API**:
```rust
pub struct RealmRuntime {
    engine: wasmtime::Engine,
    store: wasmtime::Store<RuntimeState>,
    memory64: Memory64Manager,
    cpu_backend: Option<Arc<dyn CpuBackendTrait>>,
    #[cfg(any(feature = "cuda", feature = "metal"))]
    gpu_backend: Option<Arc<dyn GpuBackendTrait>>,
}

impl RealmRuntime {
    pub fn new(config: RuntimeConfig) -> Result<Self>;
    pub fn load_wasm(&mut self, wasm_bytes: &[u8]) -> Result<()>;
    pub fn call_function(&mut self, name: &str, args: &[Val]) -> Result<Vec<Val>>;
}
```

#### Step 1.5: Build realm-wasm Orchestrator

**Create New Files**:

1. **`realm-wasm/src/lib.rs`** - WASM public API
2. **`realm-wasm/src/host_bindings.rs`** - Extern declarations
3. **`realm-wasm/src/inference.rs`** - Inference orchestration
4. **`realm-wasm/src/model_manager.rs`** - Model loading/unloading

**Key WASM API**:
```rust
#[wasm_bindgen]
pub struct Realm {
    models: HashMap<String, ModelHandle>,
    config: RealmConfig,
}

#[wasm_bindgen]
impl Realm {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<Realm, JsError>;

    pub async fn load_model(
        &mut self,
        model_id: String,
        options: JsValue
    ) -> Result<String, JsError>;

    pub async fn generate(
        &self,
        model_handle: String,
        prompt: String,
        config: JsValue
    ) -> Result<String, JsError>;

    pub fn unload_model(&mut self, model_handle: String) -> Result<(), JsError>;
}
```

**Host Function Bindings**:
```rust
// realm-wasm/src/host_bindings.rs
#[link(wasm_import_module = "realm_host")]
extern "C" {
    fn candle_matmul(
        a_ptr: *const f32, a_len: u32,
        b_ptr: *const f32, b_len: u32,
        m: u32, k: u32, n: u32,
        result_ptr: *mut f32
    ) -> i32;

    fn memory64_load_layer(
        model_id: u32,
        layer_id: u32,
        buffer_ptr: *mut u8,
        buffer_len: u32
    ) -> i32;
}
```

#### Step 1.6: End-to-End Integration Test

**Create**: `realm/examples/integration-test/`

**Test Flow**:
```rust
// examples/integration-test/src/main.rs

fn main() -> Result<()> {
    // 1. Initialize RealmRuntime with Memory64 + Candle
    let runtime = RealmRuntime::new(RuntimeConfig {
        memory64_enabled: true,
        cpu_backend: Some(CandleCpuBackend::new()?),
        gpu_backend: None, // or CandleGpuBackend for CUDA
    })?;

    // 2. Load realm-wasm module
    let wasm_bytes = include_bytes!("../../../target/wasm32-unknown-unknown/release/realm_wasm.wasm");
    runtime.load_wasm(wasm_bytes)?;

    // 3. Call WASM API to load model
    runtime.call_function("load_model", &[
        Val::String("tinyllama-1.1b-q4".into())
    ])?;

    // 4. Call WASM API to generate
    let result = runtime.call_function("generate", &[
        Val::String("tinyllama-1.1b-q4".into()),
        Val::String("What is the capital of France?".into())
    ])?;

    // 5. Verify result contains "Paris"
    assert!(result[0].as_string().contains("Paris"));

    println!("✅ End-to-end test passed!");
    Ok(())
}
```

---

## Phase 2: APIs & SDKs (Week 3-4)

### 2.1: JavaScript/TypeScript SDK

**Create**: `sdks/js/`

**Structure**:
```
sdks/js/
├── src/
│   ├── index.ts
│   ├── realm.ts
│   ├── types.ts
│   └── utils.ts
├── package.json
├── tsconfig.json
└── README.md
```

**Key API**:
```typescript
import { Realm } from '@realm-ai/sdk';

const realm = new Realm({ /* config */ });
await realm.loadModel('tinyllama-1.1b-q4');
const response = await realm.generate('What is AI?', { maxTokens: 100 });
```

### 2.2: Python Bindings

**Create**: `sdks/python/`

Use PyO3 to create Python bindings:
```python
from realm import Realm

realm = Realm()
realm.load_model('tinyllama-1.1b-q4')
response = realm.generate('What is AI?', max_tokens=100)
```

### 2.3: REST API Server

**Implement**: `server/src/main.rs`

**Endpoints**:
- `POST /v1/models/load` - Load a model
- `POST /v1/generate` - Generate text
- `POST /v1/stream` - Stream tokens
- `DELETE /v1/models/{id}` - Unload model
- `GET /v1/stats` - Get usage stats

---

## Phase 3: Multi-Tenancy (Week 5-6)

### 3.1: Multi-Realm Manager

**Add to**: `realm-runtime/src/multi_realm.rs`

```rust
pub struct MultiRealmManager {
    realms: HashMap<TenantId, RealmRuntime>,
    resource_limits: ResourceLimits,
}
```

### 3.2: Resource Limits

- Memory limits per Realm
- CPU/GPU quota
- Request rate limiting

### 3.3: Metrics & Observability

- OpenTelemetry integration
- Prometheus metrics
- Distributed tracing

---

## Phase 4: Production Hardening (Week 7-8)

### 4.1: Error Handling

- Comprehensive error types
- Graceful degradation
- Circuit breakers

### 4.2: Logging & Tracing

- Structured logging with `tracing`
- Log levels configuration
- Trace sampling

### 4.3: Performance Benchmarks

- Latency benchmarks
- Throughput benchmarks
- Memory usage profiling

### 4.4: Security Audit

- WASM sandbox verification
- Input validation
- Resource exhaustion prevention

---

## Phase 5: Scale & Optimize (Week 9+)

### 5.1: Distributed Deployment

- Kubernetes manifests
- Helm charts
- Horizontal pod autoscaling

### 5.2: Advanced Features

- Batching and scheduling
- Model quantization pipeline
- Multi-modal support

---

## Success Criteria

### Phase 1 (Week 1-2):
- ✅ realm-core builds
- ✅ realm-models builds
- ✅ realm-compute-cpu builds
- ✅ realm-compute-gpu builds
- ✅ realm-runtime builds
- ✅ realm-wasm builds
- ✅ End-to-end test passes: WASM → host → Candle → "Paris" generation

### Phase 2 (Week 3-4):
- ✅ JS SDK works in Node.js
- ✅ Python bindings work
- ✅ REST API server runs
- ✅ Documentation published

### Phase 3 (Week 5-6):
- ✅ Multi-Realm manager supports 8+ concurrent Realms
- ✅ Resource limits enforced
- ✅ Metrics exported to Prometheus

### Phase 4 (Week 7-8):
- ✅ Error handling comprehensive
- ✅ Benchmarks show <200ms first token latency
- ✅ Security audit passed

### Phase 5 (Week 9+):
- ✅ Kubernetes deployment successful
- ✅ Horizontal scaling works
- ✅ Production traffic handled

---

## Next Immediate Actions

1. **Extract realm-models** (1-2 days)
2. **Extract realm-compute-cpu** (1 day)
3. **Extract realm-compute-gpu** (1 day)
4. **Build realm-runtime** (2-3 days)
5. **Build realm-wasm** (2-3 days)
6. **Integration test** (1 day)

**Total Phase 1 Estimate**: 8-12 days

---

## Notes

- The architecture is finalized and proven via wasm-chord experiments
- All core components exist in wasm-chord and just need refactoring
- Memory64 + Candle integration is already working in wasm-chord
- Main work is extraction, reorganization, and API design
- Production features (multi-tenancy, observability) are new additions
