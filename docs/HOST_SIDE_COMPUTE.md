# Host-Side Computation Architecture

## The Problem We're Solving

**Current (WRONG)**: WASM loads 262MB weights into memory → OOM  
**Solution (CORRECT)**: Weights stay in HOST, computation happens on HOST

## Correct Architecture

```
WASM (Lightweight)
├─ Tokenization
├─ Embedding lookup (small)
├─ Control flow (generation loop)
└─ Calls HOST functions

HOST (Heavy)
├─ Weight storage (637MB quantized)
├─ Layer computation (attention + FFN)
├─ GPU acceleration
└─ Returns only activations to WASM
```

## Host Functions Needed

### 1. `realm_forward_layer` (PRIMARY)
**Purpose**: Complete transformer layer forward pass on HOST

**Signature**:
```rust
fn realm_forward_layer(
    model_id: u32,
    layer_idx: u32,
    hidden_states_ptr: u32,  // Input from WASM
    hidden_states_len: u32,  // f32 elements
    position: u32,           // Position for KV cache
    kv_cache_id: u64,        // KV cache identifier
    out_ptr: u32,            // Output to WASM
) -> i32
```

**What it does**:
1. Reads hidden_states from WASM
2. Loads weights from HOST storage (quantized, never dequantized to WASM)
3. Uses fused dequant+matmul on HOST (with GPU)
4. Computes attention block (norm → QKV → attention → output proj)
5. Computes FFN block (norm → gate/up → swiglu → down)
6. Adds residuals
7. Writes output to WASM

**Key**: Weights NEVER enter WASM memory!

### 2. `realm_forward_embedding`
**Purpose**: Embed token IDs using HOST-stored embeddings

### 3. `realm_forward_lm_head`
**Purpose**: Project final hidden state to vocabulary logits

### 4. KV Cache Management
**Options**:
- Option A: Store KV cache in HOST (per model_id + layer_idx)
- Option B: Pass KV cache pointer from WASM (simpler, but WASM manages memory)

**Recommendation**: Option A - store in HOST for simplicity

## Implementation Status

- ✅ Placeholder added
- ⏳ Need to complete:
  1. Get config from model storage
  2. Load weights from HOST (use quantized directly)
  3. Implement full forward pass
  4. KV cache management

## Benefits

1. **Memory**: WASM only holds activations (~50MB vs 2.5GB)
2. **Performance**: GPU acceleration for all compute
3. **Scalability**: Can run models of any size
4. **Multi-tenant**: Share weights across WASM instances

