# Quick Start - Realm Node.js SDK

## Status: Ready to Build and Test

All code is complete. Follow these steps:

### 1. Build Native Addon (realm-node)

```bash
cd crates/realm-node
npm install
npm run build
# This creates native.node
```

### 2. Build WASM Bindings (Already Done!)

WASM bindings are already in `sdks/nodejs/pkg/` from previous build.

### 3. Test SDK

```bash
cd sdks/nodejs
npm install
node test-paris.js ~/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf
```

## What's Complete

✅ **realm-node crate**: Native addon code (needs build)
✅ **WASM bindings**: Generated in `sdks/nodejs/pkg/`
✅ **SDK wrapper**: Complete (`sdks/nodejs/index.js`)
✅ **Test script**: Ready (`sdks/nodejs/test-paris.js`)

## Architecture

- **WASM Module**: Lightweight inference (~50MB)
- **Native Addon**: Host-side storage (`realm-node`)
- **SDK**: JavaScript wrapper that ties everything together

## Next Steps

1. Build native addon (`cd crates/realm-node && npm run build`)
2. Run test (`cd sdks/nodejs && node test-paris.js <model>`)

