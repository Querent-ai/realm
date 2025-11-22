# What's Wrong with loadModel?

## The Problem Chain

### 1. Constructor Fails
```
realm_new(pointer) → "out of bounds memory access" trap
```

**What should happen:**
- Constructor initializes Realm struct at the pointer
- Sets: `model: None`, `tokenizer: None`, `model_id: None`, etc.
- Struct is ready to use

**What actually happens:**
- Constructor fails with trap
- Memory at pointer is allocated but **uninitialized**
- Struct fields are garbage/uninitialized

### 2. loadModel Tries to Use Uninitialized Struct

**Rust Code:**
```rust
pub fn load_model(&mut self, model_bytes: &[u8]) -> Result<(), JsError> {
    // Line 363: Tries to access self.model
    self.model = Some(model);
    // Line 364: Tries to access self.tokenizer  
    self.tokenizer = Some(tokenizer);
    // Line 365: Tries to access self.model_id
    self.model_id = Some(model_id as u32);
    // etc.
}
```

**WASM Function Call:**
```rust
// We call: realm_loadModel(this_ptr, model_ptr, model_len)
// Where this_ptr points to uninitialized memory
```

**What Happens:**
1. `loadModel` receives `this_ptr` pointing to uninitialized Realm struct
2. When it tries to access `self.model`, it's reading from uninitialized memory
3. This causes "out of bounds memory access" or invalid memory access
4. WASM trap occurs

## Root Cause

**The Realm struct is not initialized!**

The struct needs these fields properly initialized:
```rust
pub struct Realm {
    model: Option<Model>,              // ❌ Uninitialized
    tokenizer: Option<Tokenizer>,      // ❌ Uninitialized  
    model_id: Option<u32>,             // ❌ Uninitialized
    transformer_config: Option<...>,   // ❌ Uninitialized
    config: WasmGenerationConfig,     // ❌ Uninitialized
    kv_caches: Option<Vec<...>>,       // ❌ Uninitialized
}
```

When `loadModel` tries to do:
```rust
self.model = Some(model);  // Writing to uninitialized Option<Model>
```

It's trying to write to memory that hasn't been properly set up, causing the trap.

## Why Constructor Fails

The `realm_new` constructor signature is `(u32) -> ()`:
- Takes a pointer where to store the struct
- Returns nothing (void)
- Should initialize the struct at that pointer

But it fails with "out of bounds memory access", which suggests:
1. The pointer we're passing might be invalid
2. Or wasm-bindgen's initialization code is trying to access memory it shouldn't
3. Or there's a missing initialization step

## The Fix Needed

**Option 1: Fix Constructor**
- Make `realm_new` actually work
- Properly initialize the Realm struct
- Then `loadModel` will work

**Option 2: Manual Initialization**
- Don't use constructor
- Manually write the struct fields to memory
- Initialize each field properly
- Then call methods

**Option 3: Different Approach**
- Use raw exports instead of wasm-bindgen methods
- Bypass the struct entirely
- Call functions directly

## Current Workaround (Why It Doesn't Work)

```rust
// We allocate memory
let realm_this = malloc(200);  // ✅ Memory allocated

// We try constructor (fails)
realm_new(realm_this);  // ❌ Fails, struct not initialized

// We continue anyway
loadModel(realm_this, ...);  // ❌ Fails because struct is uninitialized
```

The workaround allocates memory but doesn't initialize it, so `loadModel` still fails.

## Summary

**The Issue:**
- Constructor fails → Realm struct not initialized
- `loadModel` tries to use uninitialized struct → Trap
- Both fail because the struct isn't properly set up

**The Solution:**
- Need to either fix the constructor OR manually initialize the struct
- Once the struct is initialized, `loadModel` should work

