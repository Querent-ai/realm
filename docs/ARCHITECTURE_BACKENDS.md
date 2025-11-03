# Backend Architecture - Expert Engineering Design

## Current Issues

1. **WASM creates its own backends**: Currently WASM generates backends inside `generate()`, duplicating host logic
2. **Model ID is sequential**: No validation or derivation from model metadata
3. **Backend capabilities unknown**: WASM doesn't know what backends are available on host

## Proposed Solution

### 1. HOST-Side Backend Management

**Initialize all available backends in HOST:**
- CPU Backend (CandleCpuBackend)
- GPU Backend (if CUDA/Metal/WebGPU available)
- Fallback chain: GPU → CPU → Naive

**Expose via FFI:**
- `realm_get_backend_type()` - Returns available backend (0=CPU, 1=GPU, 2=None)
- `realm_matmul_cpu()` - Matrix multiply via CPU backend
- `realm_matmul_gpu()` - Matrix multiply via GPU backend (if available)

### 2. Model ID Enhancement

**Current**: Sequential ID (`NEXT_MODEL_ID++`)

**Proposed**: Hybrid approach
- Keep sequential for uniqueness
- Add validation: Hash model name/config → checksum
- Return model_id + checksum + metadata
- WASM validates model_id matches expected model

**Benefits:**
- Prevents model ID collisions
- Enables model identity verification
- Supports multi-tenant with same model

### 3. Backend Dispatch in WASM

**Instead of creating backends in WASM:**
```rust
// ❌ Current (bad)
let cpu_backend = CandleCpuBackend::new(); // Duplicates host logic

// ✅ Proposed (good)
let backend_type = realm_get_backend_type();
match backend_type {
    1 => use realm_matmul_gpu(),
    0 => use realm_matmul_cpu(),
    _ => use naive_matmul(),
}
```

## Implementation Plan

### Phase 1: Backend Host Functions

1. Add backend detection to `Memory64Runtime`
2. Expose `realm_get_backend_type()` FFI function
3. Create `realm_matmul_*()` host functions that use host backends

### Phase 2: Model ID Validation

1. Compute model hash from metadata
2. Return `(model_id, model_hash, model_info)` from `realm_store_model()`
3. Add `realm_validate_model_id()` function

### Phase 3: WASM Refactor

1. Remove backend creation from `generate()`
2. Use host-provided backend functions
3. Validate model_id before use

