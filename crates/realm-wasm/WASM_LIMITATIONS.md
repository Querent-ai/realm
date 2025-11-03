# WASM Limitations and Solutions

## Current Status

✅ **Working:**
- WASM module compiles successfully
- JS bindings generate correctly
- GGUF parsing works
- Tokenizer creation works
- API is fully functional

❌ **Memory Limitation:**
- Large models (> 200MB) fail to load due to memory constraints
- TinyLlama 1.1B (637MB quantized) requires ~2GB+ when loaded as f32 arrays
- WASM has practical memory limits that prevent loading full model weights upfront

## Root Cause

In `Model::new()`, we pre-allocate all weight matrices:
```rust
// These allocations happen upfront:
token_embeddings: vec![0.0; vocab_size * hidden_size],  // ~256MB for TinyLlama
lm_head: vec![0.0; hidden_size * vocab_size],           // ~256MB
// Plus 22 layers with attention + FFN weights each          // ~1.5GB+
```

For TinyLlama:
- 32,000 vocab * 2,048 hidden = 64M floats = 256MB (embeddings)
- 22 layers × ~70MB/layer = ~1.5GB
- **Total: ~2GB+ of f32 arrays**

This exceeds WASM's practical memory limits in most browsers/Node.js.

## Solutions

### 1. Use Smaller Models (Immediate)

Test with tiny models (< 100MB):
- 100M parameter models
- Heavily quantized versions (Q2_K)
- Custom micro-models for testing

### 2. Memory64 Support (Medium-term)

Enable WASM Memory64 for >4GB addressing:
- Requires `memory64` feature in realm-wasm
- Not all browsers support it yet
- Good for Node.js environments

### 3. Lazy Weight Loading (Recommended)

Don't allocate weight arrays upfront:
```rust
// Instead of:
token_embeddings: vec![0.0; size],

// Use:
token_embeddings: Vec::new(),  // Allocate during load_from_gguf()
```

Benefits:
- Only allocate space for weights that are actually loaded
- Allows streaming/chunked loading
- Memory usage matches actual model size

### 4. Quantized In-Memory Representation

Keep weights in quantized format (Q4_K_M) instead of f32:
- 4x memory savings
- Dequantize on-the-fly during inference
- Trades memory for compute

## Recommended Approach

**Phase 1:** Lazy loading (don't pre-allocate)
- Modify `Model::new()` to not allocate weight vectors
- Allocate during `load_from_gguf()` based on actual tensor sizes
- This should allow TinyLlama to load

**Phase 2:** Keep quantized weights in memory
- Store weights in original quantized format
- Dequantize during matmul operations
- Reduces memory footprint by 4x

**Phase 3:** Memory64 for very large models
- Enable Memory64 feature flag
- Support models > 4GB

## Testing the Fix

To verify the WASM module works end-to-end, we need either:
1. A very small model (< 100MB) for testing
2. Implement lazy loading (#3 above)

The current codebase architecture is correct - it's purely a memory allocation strategy issue.

## Current Debug Output

```
loadModel: received 668788096 bytes       ✓ Model bytes received
loadModel: header parsed                   ✓ GGUF parsing works
loadModel: tokenizer created, vocab_size=32000  ✓ Tokenizer works
loadModel: creating model...
Model::new - starting
Model::new - creating 22 layers
[PANIC] - Out of memory during layer allocation
```

The panic happens in the layer creation loop, confirming it's a memory allocation issue.
