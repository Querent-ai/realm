# Realm WASM

WebAssembly module for running Realm inference in browsers and Node.js.

## Quick Start

### Build

```bash
# For Node.js
wasm-pack build --target nodejs --release

# For browsers
wasm-pack build --target web --release
```

### Test

```bash
node test-bindings.js
```

## Features

- ✅ Full GGUF model loading from bytes
- ✅ Real transformer inference (not simulation)
- ✅ Configurable generation parameters
- ✅ TypeScript definitions included
- ✅ Optimized WASM binary (395KB)
- ✅ Bulk memory and nontrapping float-to-int enabled

## API

```typescript
class Realm {
  constructor();
  loadModel(bytes: Uint8Array): void;
  isLoaded(): boolean;
  generate(prompt: string): string;
  setConfig(config: WasmGenerationConfig): void;
  vocabSize(): number;
  getModelConfig(): string;
}

class WasmGenerationConfig {
  max_tokens: number;
  temperature: number;
  top_p: number;
  top_k: number;
  repetition_penalty: number;
}
```

## Documentation

See [WASM_GUIDE.md](../../WASM_GUIDE.md) for complete usage guide with examples.

## Notes

- WASM uses NaiveCpuBackend (pure Rust) instead of Candle for compatibility
- Binary size is kept small through aggressive optimization
- wasm-opt is configured with bulk-memory and nontrapping-float-to-int features
