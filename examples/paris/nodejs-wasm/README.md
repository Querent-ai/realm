# Paris Generation - Node.js WASM (Local)

This example demonstrates using Realm's **WASM module directly in Node.js** (no server):

- **Question**: "What is the capital of France?"
- **Expected Answer**: "Paris"

## What This Shows

- ✅ WASM module running locally in Node.js
- ✅ Host-side storage (model in HOST, not WASM)
- ✅ 98% memory reduction vs traditional approach
- ✅ No server required (runs locally)

## Prerequisites

1. **Build WASM module**:
   ```bash
   cd crates/realm-wasm
   wasm-pack build --target nodejs --release
   ```

2. **Model file** (TinyLlama Q4_K_M or similar)

## Run

```bash
cd examples/paris/nodejs-wasm
node index.js /path/to/model.gguf
```

## Expected Output

```
🚀 Realm Paris Generation - Node.js WASM (Local)

📦 Initializing WASM module...
✅ WASM module initialized

📥 Loading model: /path/to/model.gguf
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
```

## Architecture

```
┌─────────────────────────┐
│  Node.js Script        │
│  (this example)        │
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│  WASM Module           │
│  (realm-wasm)          │
└──────────┬──────────────┘
           │ Host Functions
┌──────────▼──────────────┐
│  HOST Storage          │
│  (Native Rust)         │
│  (Model in RAM)        │
└─────────────────────────┘
```

## Notes

- **No server required** - runs completely locally
- **98% memory reduction** - model in HOST, not WASM
- **Perfect for edge** - lightweight WASM module
- **Multi-tenant ready** - each tenant can have isolated WASM

