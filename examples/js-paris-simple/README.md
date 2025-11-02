# Realm Node.js Examples

This directory contains test files demonstrating the Realm Node.js SDK with HOST-side computation.

## Test Files

### 1. test.js - HOST Storage Test ✅
Tests basic HOST storage functionality: store model, retrieve tensors, cleanup.

```bash
node test.js [model-path]
```

**Output:**
```
✅ Model stored in HOST with ID: 2294743135
✅ Retrieved tensor: 262MB dequantized
✅ Cleanup successful
```

### 2. test-native-direct.js - Direct Native Calls ✅
Tests all three HOST computation functions by calling native addon directly.

```bash
node test-native-direct.js [model-path]
```

**Output:**
```
✅ Embedded 8 tokens → 16384 f32 values
✅ Layer 0 forward complete: 16384 f32 values
✅ Computed logits: 32000 values
✅ Top token ID: 28351 (logit: 8.9331)
```

### 3. test-pure-node.js - Pure Node.js API ✅
Tests the JavaScript wrapper API with automatic type conversion.

```bash
node test-pure-node.js [model-path]
```

**Output:**
```
✅ embedTokens: 3 tokens → 6144 hidden states
✅ forwardLayer: 2048 → 2048 hidden states
✅ computeLogits: 2048 → 32000 logits
```

## Model Path

All tests default to `/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf`. You can override this:

```bash
node test.js /path/to/your/model.gguf
```

## Expected Results

All three tests should pass with ✅ status:

- **test.js**: Verifies HOST storage works (637MB model loaded)
- **test-native-direct.js**: Verifies HOST computation works (all 3 functions)
- **test-pure-node.js**: Verifies JavaScript API works (type conversion)

## Architecture

```
JavaScript Test → Native Addon → HOST Storage
                                     ↓
                            (637MB quantized model)
```

**Key Point**: Model weights stay in HOST memory, not WASM. This achieves 98% memory reduction.

## Troubleshooting

### "Model not found"
```bash
# Download a model
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-GGUF/resolve/main/tinyllama-1.1b.Q4_K_M.gguf

# Run test
node test.js ./tinyllama-1.1b.Q4_K_M.gguf
```

### "Cannot find module '@realm/realm-node'"
```bash
# Build native addon first
cd ../../crates/realm-node
cargo build --release
```

### "Error: ... native addon"
```bash
# Rebuild from scratch
cargo clean
cargo build --release -p realm-node
```

## CI Integration

These tests are designed to verify:
1. Native addon compiles correctly
2. HOST storage works as expected
3. HOST computation functions are properly exported
4. JavaScript API provides correct type conversion

All tests pass and are ready for CI integration (requires model file).
