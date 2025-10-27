# Backend Selection Architecture

## Overview

Each WASM instance in Realm can independently select its compute backend (CPU, CUDA, Metal, WebGPU) based on configuration.

## Why Per-WASM Selection?

### Multi-Tenant Flexibility

```
realm-runtime (one host process)
│
├─ WASM Instance #1 (Customer A)
│  ├─ Config: { backend: "cuda" }
│  └─ Uses: NVIDIA GPU
│
├─ WASM Instance #2 (Customer B)
│  ├─ Config: { backend: "metal" }
│  └─ Uses: Apple Silicon GPU
│
├─ WASM Instance #3 (Customer C)
│  ├─ Config: { backend: "cpu" }
│  └─ Uses: CPU (cost-optimized)
│
└─ WASM Instance #4 (Customer D - Browser)
   ├─ Config: { backend: "webgpu" }
   └─ Uses: WebGPU
```

All instances run concurrently, sharing the same host but using different backends.

## Backend Configuration

### Client-Side Config

```typescript
// Customer A wants GPU
const realm = new Realm({
  apiKey: 'cust_abc123',
  backend: 'cuda'  // NVIDIA GPU
});

// Customer B wants CPU
const realm = new Realm({
  apiKey: 'cust_def456',  
  backend: 'cpu'  // CPU fallback
});
```

### WASM Initialization

```rust
// In realm.wasm
pub fn init(config: &Config) -> Result<()> {
    // Select backend based on config
    let backend = match config.backend {
        Backend::Cuda => host_register_cuda_backend()?,
        Backend::Metal => host_register_metal_backend()?,
        Backend::WebGPU => host_register_webgpu_backend()?,
        Backend::CPU => host_register_cpu_backend()?,
    };
    
    // Store in WASM instance state
    STATE.backend = Some(backend);
    Ok(())
}
```

## Host Runtime Responsibilities

### Register All Available Backends

```rust
// In realm-runtime
impl HostRuntime {
    fn new() -> Self {
        let mut backends = BackendRegistry::new();
        
        // Try CUDA
        if let Ok(cuda) = CudaBackend::new() {
            backends.register("cuda", Box::new(cuda));
        }
        
        // Try Metal
        if let Ok(metal) = MetalBackend::new() {
            backends.register("metal", Box::new(metal));
        }
        
        // Try WebGPU
        if let Ok(webgpu) = WebGpuBackend::new() {
            backends.register("webgpu", Box::new(webgpu));
        }
        
        // Always register CPU
        backends.register("cpu", Box::new(CpuBackend::new()));
        
        Self { backends, .. }
    }
}
```

### Per-WASM Backend Selection

```rust
// When WASM requests a backend
fn handle_backend_request(&mut self, wasm_id: u64, backend_name: &str) -> Result<BackendHandle> {
    let backend = self.backends.get(backend_name)
        .ok_or(Error::UnknownBackend)?;
    
    // Each WASM gets its own handle
    let handle = self.create_backend_handle(wasm_id, backend)?;
    
    Ok(handle)
}
```

## Use Cases

### 1. Cost Optimization
```typescript
// Customer on tight budget
realm.configure({ backend: 'cpu' });
// Pay CPU rates ($0.10/hour vs $1/hour)
```

### 2. Performance Priority
```typescript
// Customer needs speed
realm.configure({ backend: 'cuda' });
// Get GPU acceleration (60-100 tok/s)
```

### 3. Platform-Specific
```typescript
// Customer on macOS
realm.configure({ backend: 'metal' });
// Uses Apple Silicon GPU automatically
```

### 4. Browser Deployment
```typescript
// Customer in browser
realm.configure({ backend: 'webgpu' });
// Runs inference in browser with WebGPU
```

### 5. Auto-Fallback
```typescript
realm.configure({ 
  backend: 'cuda',
  fallback: 'cpu'  // If CUDA unavailable
});
// Tries GPU, falls back to CPU
```

## Implementation Details

### Backend Trait

```rust
pub trait ComputeBackend: Send + Sync {
    fn matmul(&self, a: &[f32], b: &[f32], shape: MatMulShape) -> Result<Vec<f32>>;
    fn matmul_transposed(&self, a: &[f32], b: &[f32], shape: MatMulShape) -> Result<Vec<f32>>;
    fn fused_dequant_matmul_q4k(&self, ...) -> Result<Vec<f32>>;
    fn fused_dequant_matmul_q5k(&self, ...) -> Result<Vec<f32>>;
    fn fused_dequant_matmul_q6k(&self, ...) -> Result<Vec<f32>>;
    fn fused_dequant_matmul_q8k(&self, ...) -> Result<Vec<f32>>;
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
}
```

### Per-WASM State

```rust
pub struct WasmInstanceState {
    pub config: Config,
    pub backend: Option<Box<dyn ComputeBackend>>,
    pub model_handle: Option<ModelHandle>,
    pub kv_cache: KVCache,
    // ...
}

// Each WASM instance has its own state
impl HostRuntime {
    fn get_instance_state(&self, wasm_id: u64) -> &WasmInstanceState;
}
```

## Performance Characteristics

| Backend | First Token | Tokens/sec | Use Case |
|---------|-------------|------------|----------|
| CUDA (A100) | ~100ms | 125-200 | Production SaaS |
| CUDA (RTX 4090) | ~200ms | 60-100 | High-end consumer |
| Metal (M2 Max) | ~500ms | 25-33 | Mac users |
| WebGPU | ~300ms | 40-80 | Browser |
| CPU (16-core) | ~2000ms | 6-10 | Cost-optimized |

## Economic Impact

### Revenue Opportunities

1. **Tiered Pricing**
   - GPU: $1/hour
   - CPU: $0.10/hour
   - 10x price difference based on backend

2. **Platform Licensing**
   - CUDA license for enterprise
   - Metal for macOS customers
   - WebGPU for browser deployments

3. **Flexible Contracts**
   - Customer chooses performance vs cost
   - Easy upsell from CPU → GPU
   - Transparent pricing model

### Competitive Advantage

- **vLLM**: One global backend choice
- **Ollama**: Manual backend selection
- **Realm**: **Per-instance dynamic selection** ← YOU WIN!

## Conclusion

Per-WASM backend selection is a key differentiator that enables:
- ✅ Multi-tenant resource optimization
- ✅ Platform-specific performance
- ✅ Cost-conscious deployment options
- ✅ Automatic fallback capabilities
- ✅ Tiered pricing strategies

This is what makes Realm superior to existing solutions!


