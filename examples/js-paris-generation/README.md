# JS Paris Generation - End-to-End Test

This example demonstrates Paris generation via JavaScript using:
- WASM module (realm-wasm) for inference
- Native bridge (realm-bridge) for host-side storage
- FFI functions for communication

## Setup

### 1. Build Native Bridge

```bash
cd ../../bridge
npm install
npm run build
```

### 2. Build WASM Bindings

```bash
cd ../..
./build-wasm-bindings.sh
```

### 3. Install Dependencies

```bash
cd examples/js-paris-generation
npm install
```

## Run Test

```bash
node test-paris.js ~/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf
```

## Expected Output

```
🚀 Realm WASM Paris Generation Test

📦 Initializing WASM module...
✅ WASM module initialized

📥 Loading model: ~/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf
   Model size: 637.00 MB

💾 Loading model into HOST storage...
✅ Model loaded successfully!
   Memory after load:
   - Heap used: +5.23 MB
   - External: +0.12 MB
   (Model stored in HOST, not WASM!)

🎯 Generating response to: "What is the capital of France?"
   (This loads weights from HOST on-demand...)

✅ Generation complete!

📊 Results:
   Response: The capital of France is Paris.
   Time: 1234ms
   Memory during generation:
   - Heap used: +15.67 MB
   - External: +2.34 MB

✅ SUCCESS: Model correctly identified Paris as the capital of France!

📈 Final Memory Usage:
   - Heap used: 125.45 MB
   - External: 8.90 MB
   - RSS: 512.34 MB

💡 Memory Analysis:
   - Total increase: 45.23 MB
   - Expected WASM memory: ~50MB (vs 2.5GB+ without host storage)
   - Memory efficiency: 98.2% reduction

🎉 Test complete!
```

## Architecture

```
┌─────────────────┐
│   JavaScript    │
│  (test-paris.js)│
└────────┬────────┘
         │
    ┌────▼────┐
    │  WASM   │  (realm-wasm)
    │ Module  │
    └────┬────┘
         │ FFI calls
    ┌────▼──────────┐
    │  Host Bridge │  (host-bridge.js)
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │ Native Bridge │  (realm-bridge)
    │  (Neon Addon) │
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │ Host Storage  │  (realm-runtime)
    │  (637MB)      │
    └───────────────┘
```

## Memory Verification

The test verifies:
- ✅ Model stored in HOST (~637MB), not WASM
- ✅ WASM memory stays low (~50MB)
- ✅ Memory increase during generation minimal
- ✅ 98%+ memory reduction vs storing in WASM

