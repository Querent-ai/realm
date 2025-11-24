
---

## Update: hashFiles Syntax Error (Nov 24, 2025)

### Problem

Found incorrect `hashFiles()` syntax in workflow files:

```yaml
# Incorrect (multiple arguments)
key: ${{ runner.os }}-cargo-${{ hashFiles('Cargo.toml', 'crates/*/Cargo.toml') }}
```

**Error**: "The template is not valid... hashFiles('Cargo.toml, crates/*/Cargo.toml') failed"

**Root Cause**: `hashFiles()` takes a single glob pattern, not multiple comma-separated arguments.

### Solution

Fixed all occurrences across 4 workflow files:

```yaml
# Correct (single glob pattern)
key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.toml') }}
```

### Files Fixed

1. **`.github/workflows/ci.yml`** - 19 instances
2. **`.github/workflows/gpu-tests.yml`** - 5 instances
3. **`.github/workflows/pr-benchmark.yml`** - 2 instances
4. **`.github/workflows/paris-regression.yml`** - 3 instances

**Total**: 29 hashFiles patterns fixed

### Pattern Evolution

| Version | Pattern | Status |
|---------|---------|--------|
| v1 (original) | `hashFiles('**/Cargo.lock')` | ❌ Cargo.lock gitignored |
| v2 (attempted) | `hashFiles('Cargo.toml', 'crates/*/Cargo.toml')` | ❌ Invalid syntax |
| v3 (correct) | `hashFiles('**/Cargo.toml')` | ✅ Works correctly |

---

**Final Status**: ✅ All hashFiles patterns now use correct syntax
