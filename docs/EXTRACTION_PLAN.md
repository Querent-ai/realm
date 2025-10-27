# Realm Extraction Plan

Systematically extract proven components from wasm-chord to build Realm.ai

## 🎯 Goal

Build a production-ready Realm.ai by extracting and adapting components from the wasm-chord experimental codebase.

## 📋 Extraction Checklist

### Phase 1: Core Foundation (Days 1-2) ✅ DONE
- [x] Repository structure
- [x] Cargo workspace
- [x] realm-core crate (extracted from wasm-chord-core)
- [x] Professional README
- [x] Logo design

### Phase 2: Core Components (Days 3-4)
- [ ] Extract GGUF parser
- [ ] Extract tokenizer (BPE, SentencePiece)
- [ ] Extract tensor loader
- [ ] Extract quantization support (Q4/Q5/Q6/Q8)
- [ ] Build verification test ("Paris" test)

### Phase 3: Model Architecture (Days 5-6)
- [ ] Extract transformer layers
- [ ] Extract attention mechanism
- [ ] Extract FFN (feed-forward network)
- [ ] Extract RMS normalization
- [ ] Extract KV cache management
- [ ] Build model forward pass

### Phase 4: Backends (Days 7-8)
- [ ] Extract Naive CPU backend
- [ ] Extract Candle CPU backend
- [ ] Extract Candle CUDA backend
- [ ] Extract Candle Metal backend
- [ ] Create unified backend trait

### Phase 5: Runtime & Memory64 (Days 9-10)
- [ ] Extract Memory64 layer manager
- [ ] Extract async prefetch logic
- [ ] Extract LRU eviction system
- [ ] Integrate with backend selection

### Phase 6: Integration & Testing (Days 11-12)
- [ ] End-to-end test ("Paris" generation)
- [ ] Performance benchmarks
- [ ] Memory usage verification
- [ ] Multi-model support test

## 📂 File Mapping

### From wasm-chord → Realm

```
wasm-chord-core/              → realm-core/
├── formats/gguf.rs          → crates/realm-core/src/formats/gguf.rs
├── tokenizer.rs             → crates/realm-core/src/tokenizer.rs
├── tensor_loader.rs         → crates/realm-core/src/tensor_loader.rs
└── quant/                   → crates/realm-core/src/quant/

wasm-chord-runtime/          → realm-models/ + realm-runtime/
├── transformer/
│   ├── model.rs             → crates/realm-models/src/model.rs
│   ├── layer.rs             → crates/realm-models/src/layer.rs
│   ├── attention.rs         → crates/realm-models/src/attention.rs
│   └── ffn.rs               → crates/realm-models/src/ffn.rs
├── memory64_layer_manager.rs → crates/realm-runtime/src/memory64.rs
└── context.rs               → crates/realm-runtime/src/context.rs

wasm-chord-cpu/              → realm-compute-cpu/
├── naive_backend.rs         → crates/realm-compute-cpu/src/naive.rs
├── candle_cpu_backend.rs    → crates/realm-compute-cpu/src/candle.rs
└── fused.rs                 → crates/realm-compute-cpu/src/fused.rs

wasm-chord-gpu/              → realm-compute-gpu/
├── lib.rs                   → crates/realm-compute-gpu/src/lib.rs
└── candle_backend.rs        → crates/realm-compute-gpu/src/candle.rs
```

## 🔄 Extraction Process

### Step 1: Copy Files
```bash
# Copy core components
cp wasm-chord/crates/wasm-chord-core/src/tokenizer.rs \
   realm/crates/realm-core/src/

# Copy model architecture
cp wasm-chord/crates/wasm-chord-runtime/src/transformer/*.rs \
   realm/crates/realm-models/src/
```

### Step 2: Update Imports
Replace `wasm_chord_*` with `realm_*`:
```bash
sed -i 's/wasm_chord_core/realm_core/g' *.rs
sed -i 's/wasm_chord_cpu/realm_compute_cpu/g' *.rs
sed -i 's/wasm_chord_gpu/realm_compute_gpu/g' *.rs
```

### Step 3: Build & Test
```bash
cd realm
cargo build --release
cargo test --release
```

### Step 4: Integration Test
```bash
# Run "Paris" test
cargo run --release --example capital-test \
  --model tinyllama-1.1b.Q4_K_M.gguf
```

## 📊 Success Criteria

Each phase is complete when:
- ✅ Code compiles without errors
- ✅ All tests pass
- ✅ "Paris" test succeeds
- ✅ Documentation updated

## 🚨 Important Notes

1. **Keep Proven Components**: Only extract what we've verified works
2. **Remove Experimental Code**: Skip debugging code, verbose logging
3. **Update Branding**: Change all references from wasm-chord → realm
4. **Clean Architecture**: Remove unused dependencies, simplify APIs
5. **Production Focus**: Add error handling, metrics, observability

## 🎯 Starting Now

Let's begin with Phase 2: Core Components extraction!


