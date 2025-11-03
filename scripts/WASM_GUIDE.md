# WASM Integration Guide

This guide explains how to build and use the Realm WASM module for browser and Node.js environments.

## Building WASM

### Using wasm-pack (Recommended)

For production builds with optimizations and JS bindings:

**For Node.js:**
```bash
cd crates/realm-wasm
wasm-pack build --target nodejs --release
```

**For Browser/Web:**
```bash
cd crates/realm-wasm
wasm-pack build --target web --release
```

This creates a `pkg/` directory with:
- `realm_wasm_bg.wasm` - The compiled WASM binary (395KB optimized)
- `realm_wasm.js` - JavaScript bindings
- `realm_wasm.d.ts` - TypeScript definitions
- `package.json` - NPM package metadata

### Direct cargo build (Advanced)

For development without JS bindings:

```bash
cargo build -p realm-wasm --target wasm32-unknown-unknown --release
```

Note: This only produces the `.wasm` file without JS bindings. Use wasm-pack for production.

## Using in Node.js

### Basic Initialization

```javascript
const fs = require('fs');
const realmWasm = require('./crates/realm-wasm/pkg/realm_wasm.js');

(async () => {
    // Load WASM binary
    const wasmBytes = fs.readFileSync('./crates/realm-wasm/pkg/realm_wasm_bg.wasm');

    // Initialize the WASM module
    await realmWasm.default(wasmBytes);
    console.log('Realm WASM initialized successfully!');

    // Create Realm instance
    const realm = new realmWasm.Realm();
    console.log('Realm instance created');

    // Load model (pass GGUF file bytes)
    const modelBytes = fs.readFileSync('./path/to/model.gguf');
    await realm.loadModel(modelBytes);
    console.log('Model loaded');

    // Generate text
    const response = await realm.generate('What is the capital of France?');
    console.log('Response:', response);
})();
```

### Configuration

```javascript
const config = new realmWasm.WasmGenerationConfig();
config.max_tokens = 200;
config.temperature = 0.7;
config.top_p = 0.9;
config.top_k = 40;
config.repetition_penalty = 1.1;

realm.setConfig(config);
```

## Using in Browser

```html
<!DOCTYPE html>
<html>
<head>
    <title>Realm WASM Demo</title>
</head>
<body>
    <script type="module">
        import init, { Realm, WasmGenerationConfig } from './pkg/realm_wasm.js';

        async function main() {
            // Initialize WASM module
            await init();

            // Create Realm instance
            const realm = new Realm();

            // Load model from URL or user upload
            const response = await fetch('/path/to/model.gguf');
            const modelBytes = new Uint8Array(await response.arrayBuffer());
            await realm.loadModel(modelBytes);

            // Generate text
            const result = await realm.generate('Hello, world!');
            console.log('Generated:', result);
        }

        main();
    </script>
</body>
</html>
```

## API Reference

### Realm

Main class for inference.

#### Constructor

```javascript
const realm = new Realm();
```

#### Methods

- `loadModel(bytes: Uint8Array): Promise<void>` - Load a GGUF model from bytes
- `isLoaded(): boolean` - Check if model is loaded
- `generate(prompt: string): Promise<string>` - Generate text from prompt
- `setConfig(config: WasmGenerationConfig): void` - Set generation config
- `vocabSize(): Promise<number>` - Get model vocabulary size
- `getModelConfig(): Promise<string>` - Get model config as JSON string

### WasmGenerationConfig

Configuration for text generation.

#### Properties

- `max_tokens: number` - Maximum tokens to generate (default: 100)
- `temperature: number` - Sampling temperature (default: 0.7)
- `top_p: number` - Nucleus sampling threshold (default: 0.9)
- `top_k: number` - Top-k sampling limit (default: 40)
- `repetition_penalty: number` - Repetition penalty (default: 1.1)

## Performance Notes

### Memory Considerations

- GGUF models can be large (1-10GB+)
- Browser: Use streaming or chunked loading for large models
- Node.js: Can handle larger models in memory
- Consider using smaller quantized models (Q4_K_M, Q5_K_M) for browsers

### Optimization Tips

1. **Use quantized models** - Q4_K_M offers good quality/size tradeoff
2. **Enable bulk memory** - Already configured in `wasm-pack.toml`
3. **Lazy loading** - Load model layers on-demand (Memory64 support)
4. **Worker threads** - Run inference in Web Workers to avoid blocking UI

## Troubleshooting

### "Module not found" error

Make sure you're serving the WASM files with correct MIME types:
- `.wasm` → `application/wasm`
- `.js` → `application/javascript`

### Cross-Origin issues

WASM requires same-origin or proper CORS headers. For local development:

```bash
# Simple Python HTTP server
python3 -m http.server 8000
```

### Out of memory

For large models in browsers:
1. Use smaller quantized models
2. Enable Memory64 feature (experimental)
3. Implement streaming/chunked loading

## Example: Complete Node.js Script

```javascript
#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const realmWasm = require('./crates/realm-wasm/pkg/realm_wasm.js');

async function main() {
    // Initialize WASM
    const wasmPath = path.join(__dirname, 'crates/realm-wasm/pkg/realm_wasm_bg.wasm');
    await realmWasm.default(fs.readFileSync(wasmPath));

    // Create instance
    const realm = new realmWasm.Realm();

    // Configure generation
    const config = new realmWasm.WasmGenerationConfig();
    config.max_tokens = 50;
    config.temperature = 0.0; // Greedy/deterministic
    realm.setConfig(config);

    // Load model
    const modelPath = process.argv[2] || './model.gguf';
    console.log(`Loading model: ${modelPath}`);
    const modelBytes = fs.readFileSync(modelPath);
    await realm.loadModel(new Uint8Array(modelBytes));
    console.log(`Model loaded. Vocab size: ${await realm.vocabSize()}`);

    // Generate
    const prompt = 'What is the capital of France?';
    console.log(`\nPrompt: ${prompt}`);
    const response = await realm.generate(prompt);
    console.log(`Response: ${response}`);
}

main().catch(console.error);
```

## Next Steps

- See `examples/paris-generation/js/` for a complete working example
- Check `crates/realm-wasm/src/lib.rs` for API implementation details
- Read `WORKSPACE_GUIDE.md` for dependency management
