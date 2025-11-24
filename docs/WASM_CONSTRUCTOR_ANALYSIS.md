# WASM Constructor Analysis

## Current Situation

### What We Need
The `Realm` struct in WASM is **required** even with HOST-side storage because it holds:
- `model_id: Option<u32>` - Handle to HOST storage
- `tokenizer: Option<Tokenizer>` - For encoding/decoding
- `transformer_config: Option<TransformerConfig>` - Model configuration
- `kv_caches: Option<Vec<KVCache>>` - KV caches for inference
- `config: WasmGenerationConfig` - Generation settings

So we **DO need** the constructor, but it's currently failing.

### The Problem
The `realm_new` constructor call fails with WASM trap errors, even though:
- ✅ Signature is correctly detected: `1 params, 0 results`
- ✅ Memory is properly allocated via `__wbindgen_malloc`
- ✅ Pointer is validated (within bounds)
- ✅ Typed function calls are used
- ✅ All imports are stubbed

### Root Cause Hypothesis
The constructor returns `Result<Realm, JsError>`. In wasm-bindgen:
- Success: Struct is written to the passed pointer
- Error: Should throw/return error code, but we're not handling it

The `(u32) -> ()` signature suggests the pointer is an in-out parameter where the struct is stored. Errors might be handled via:
1. Exception throwing (not available in Wasmtime)
2. Return error code (but signature shows 0 results)
3. Special error handling mechanism

## Proposed Solutions

### Option 1: Continue Debugging (Current Approach)
**Pros:**
- Keeps wasm-bindgen compatibility
- Works with existing WASM builds
- Maintains JavaScript compatibility

**Cons:**
- Complex error handling
- May require deep wasm-bindgen knowledge
- Time-consuming

**Next Steps:**
1. Check if `__wbg_init` needs to be called first
2. Verify all required stubs are present
3. Try different memory allocation strategies
4. Check if Result<Realm, JsError> needs special handling

### Option 2: Simplify - Use Static/Global Realm Instance
**Pros:**
- Avoids constructor complexity
- Simpler code
- Faster initialization

**Cons:**
- Only one Realm instance per WASM module
- Less flexible

**Implementation:**
- Create Realm instance at module initialization
- Store in static/global variable
- Methods access the global instance

### Option 3: Bypass wasm-bindgen for Server Builds
**Pros:**
- Full control over calling convention
- No wasm-bindgen complexity
- Can use simple C-style exports

**Cons:**
- Requires separate WASM build
- Loses JavaScript compatibility
- More code to maintain

**Implementation:**
- Add `#[no_mangle]` exports for server builds
- Use simple function signatures
- Direct WASM function calls

### Option 4: Use Zero Pointer (Quick Test)
**Pros:**
- Simplest to try
- Might work if wasm-bindgen handles it

**Cons:**
- Likely to fail (null pointer)
- Not a real solution

**Implementation:**
- Try calling `realm_new` with pointer `0`
- See if wasm-bindgen allocates internally

## Recommendation

**Try Option 1 first** (continue debugging), but with a **time limit**:
1. Add more detailed trap logging
2. Check if `__wbg_init` is actually being called
3. Verify all stubs are correct
4. Try calling with `this_ptr = 0` as a test

If Option 1 doesn't work within reasonable time, **switch to Option 2** (static instance):
- Much simpler
- Works for server use case (single tenant per WASM instance)
- Can be implemented quickly

## Current Code Status

The constructor calling code (~200 lines) is:
- ✅ Well-structured
- ✅ Handles multiple signature patterns
- ✅ Has good error logging
- ❌ Not working (WASM trap)

**Decision Point:**
- If we can fix the constructor call → Keep current code
- If we can't fix it quickly → Simplify to static instance

