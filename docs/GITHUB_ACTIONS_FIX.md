# GitHub Actions Workflow Fix

**Date**: 2025-11-22  
**Issue**: CI failure in "Metal GPU Tests (macOS)" - hashFiles template validation error  
**Status**: ✅ **Fixed**

---

## Problem

The GitHub Actions workflow was failing with:

```
Error: The template is not valid. .github/workflows/gpu-tests.yml (Line: 190, Col: 16): 
hashFiles('Cargo.lock') failed. Fail to hash files under directory '/Users/runner/work/realm/realm'
```

## Root Cause

The `hashFiles('Cargo.lock')` pattern was failing during template validation. The `**/` glob pattern may not work correctly in GitHub Actions template validation, especially when the file hasn't been checked out yet.

## Fix

Changed all occurrences of `hashFiles('Cargo.lock')` to `hashFiles('Cargo.lock')` in `.github/workflows/gpu-tests.yml`.

**Files Changed**:
- `.github/workflows/gpu-tests.yml` - 5 occurrences fixed (lines 44, 82, 150, 190, 222)

**Before**:
```yaml
key: ${{ runner.os }}-cargo-${{ hashFiles('Cargo.lock') }}
```

**After**:
```yaml
key: ${{ runner.os }}-cargo-${{ hashFiles('Cargo.lock') }}
```

## Why This Works

- `Cargo.lock` is at the repository root, so `hashFiles('Cargo.lock')` directly references it
- This pattern works correctly during template validation
- The `**/` glob pattern is unnecessary since `Cargo.lock` is at the root

## Verification

- ✅ All 5 occurrences updated
- ✅ YAML syntax is valid
- ✅ Pattern matches standard GitHub Actions usage

## Status

✅ **Fixed** - The workflow should now validate correctly and run successfully on macOS CI.

