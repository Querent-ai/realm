# WASM Integration Fix Plan

## Problem Analysis

### Current Issue
- **WASM trap when calling `realm_new` constructor**
- **Root Cause**: WASM module built with `wasm-pack build --target web` expects JavaScript runtime APIs
- **Missing APIs**: `web_sys::console`, `js_sys::Function`, `console_error_panic_hook`

### What's Happening
1. ✅ **WASM module loads successfully** - All imports are satisfied
2. ✅ **All `wbg::` imports are stubbed** - Automatic extraction works
3. ❌ **Constructor traps** - Calls `console_error_panic_hook::set_once()` which uses `web_sys::console`
4. ❌ **Code uses `web_sys::console::log_1`** throughout - No JavaScript console in Wasmtime

### Code Dependencies
- `console_error_panic_hook::set_once()` - Needs JavaScript console
- `web_sys::console::log_1()` - Used for logging throughout
- `js_sys::Function` - Used for callbacks

## Solution Options

### Option 1: Stub JavaScript Runtime APIs (Quick Fix) ⚡
**Pros**: Fast, minimal changes, works with existing WASM
**Cons**: Logging won't work, but that's acceptable for server

**Implementation**:
1. Add stubs for `web_sys` and `js_sys` imports in `runtime_manager.rs`
2. Stub `console.log`, `console.error`, etc. as no-ops
3. Stub `Function` callbacks as no-ops

**Files to modify**:
- `crates/realm-server/src/runtime_manager.rs` - Add web_sys/js_sys stubs

### Option 2: Build Server-Specific WASM (Proper Fix) 🎯
**Pros**: Clean, no runtime overhead, proper separation
**Cons**: Requires conditional compilation, two WASM builds

**Implementation**:
1. Add `server` feature flag to `realm-wasm`
2. Conditionally compile out `web_sys` usage when `server` feature is enabled
3. Use `tracing` or direct logging instead of `web_sys::console`
4. Build with: `wasm-pack build --target no-modules --no-default-features --features server`

**Files to modify**:
- `crates/realm-wasm/Cargo.toml` - Add `server` feature
- `crates/realm-wasm/src/lib.rs` - Conditional compilation for web_sys
- `Makefile` - Add server WASM build target

### Option 3: Use Raw WASM Exports (No wasm-bindgen) 🔧
**Pros**: Full control, no JavaScript dependencies
**Cons**: Major refactor, lose wasm-bindgen conveniences

**Implementation**:
1. Export functions directly without wasm-bindgen
2. Use `#[no_mangle]` and raw function exports
3. Manual memory management

## Recommended Approach: **Option 1 + Option 2 Hybrid**

### Phase 1: Quick Fix (Option 1) - Get it working NOW
1. Add stubs for `web_sys` and `js_sys` imports
2. Extract all imports from WASM module (not just `wbg::`)
3. Stub `web_sys::console` functions as no-ops
4. Stub `js_sys::Function` as no-ops
5. Test e2e passes

### Phase 2: Proper Fix (Option 2) - Clean it up
1. Add `server` feature to `realm-wasm`
2. Conditionally disable `web_sys` when `server` feature is enabled
3. Use `tracing` for logging in server mode
4. Build separate WASM for server use
5. Update Makefile to build server WASM

## Implementation Plan for Phase 1 (Quick Fix)

### Step 1: Extract ALL imports, not just `wbg::`
```rust
// In runtime_manager.rs, extract ALL imports
for import in wasm_module.imports() {
    let module = import.module();
    match module {
        "wbg" => { /* existing wbg stubs */ }
        "web_sys" => { /* stub web_sys functions */ }
        "js_sys" => { /* stub js_sys functions */ }
        _ => { /* handle other modules */ }
    }
}
```

### Step 2: Stub web_sys::console functions
```rust
// Stub console.log, console.error, etc.
linker.func_wrap("web_sys", "console_log_1", |_caller, _ptr: u32, _len: u32| {});
linker.func_wrap("web_sys", "console_error_1", |_caller, _ptr: u32, _len: u32| {});
// ... etc for all console functions
```

### Step 3: Stub js_sys::Function
```rust
// Stub Function::new, Function::call, etc.
linker.func_wrap("js_sys", "Function_new", |_caller, _ptr: u32| -> u32 { 0 });
```

### Step 4: Handle console_error_panic_hook
```rust
// Stub panic hook setup
linker.func_wrap("console_error_panic_hook", "set_once", |_caller| {});
```

## Files to Modify

1. **`crates/realm-server/src/runtime_manager.rs`**
   - Extract all imports (not just `wbg::`)
   - Add stubs for `web_sys`, `js_sys`, `console_error_panic_hook`
   - Handle all import modules dynamically

2. **`Makefile`** (optional for Phase 2)
   - Add `wasm-server` target that builds with `--target no-modules --no-default-features --features server`

## Testing

1. Run `make e2e` - should pass all tests
2. Verify no WASM traps in logs
3. Verify model loads successfully
4. Verify generation works

## Expected Outcome

After Phase 1:
- ✅ E2E tests pass
- ✅ No WASM traps
- ✅ Model loads and generates correctly
- ⚠️ Logging goes to void (acceptable for server)

After Phase 2:
- ✅ Clean separation of web vs server WASM
- ✅ Proper logging in server mode
- ✅ No unnecessary JavaScript dependencies

