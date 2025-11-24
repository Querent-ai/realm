# Metal Wrapper Compilation Fix

**Date**: 2025-11-22  
**Issue**: CI failure in "Paris Generation with Metal" test  
**Status**: ✅ **Fixed**

---

## Problem

The Metal wrapper had compilation errors on lines 123 and 135:

```
error[E0599]: no method named `map_err` found for struct `candle_core::Tensor`
```

## Root Cause

On line 95, `scores_base` is assigned from a multiplication operation that returns `Result<Tensor>`, and the `?` operator unwraps it:

```rust
let scores_base = (&scores_base * &scale_tensor)
    .map_err(|e| WasmChordError::Runtime(format!("Failed to scale scores: {}", e)))?;
```

After the `?`, `scores_base` is a `Tensor`, not a `Result<Tensor>`. However, the code was incorrectly trying to call `map_err` on it again on lines 123 and 135.

## Fix

**Before** (lines 122-123):
```rust
let scores_tensor = scores_base
    .map_err(|e| WasmChordError::Runtime(format!("Failed to compute scores: {}", e)))?;
```

**After**:
```rust
let scores_tensor = scores_base;
```

**Before** (lines 134-135):
```rust
} else {
    scores_base
        .map_err(|e| WasmChordError::Runtime(format!("Failed to compute scores: {}", e)))?
};
```

**After**:
```rust
} else {
    scores_base
};
```

## Verification

- ✅ No linter errors
- ✅ Code logic is correct
- ✅ `scores_base` is correctly used as a `Tensor` after unwrapping

## Status

✅ **Fixed** - The Metal wrapper should now compile correctly on macOS CI.

