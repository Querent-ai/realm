# Realm Node.js SDK

Complete SDK for running LLM inference via WASM with host-side storage.

## Installation

```bash
cd sdks/nodejs
npm install
npm run build
```

## Usage

```javascript
import Realm from '@realm/sdk';

const realm = new Realm();
await realm.initialize();

// Load model (stores in HOST, not WASM)
await realm.loadModel('/path/to/model.gguf');

// Generate
const response = realm.generate("What is the capital of France?");
console.log(response); // "The capital of France is Paris."
```

## Testing

```bash
npm test
# or
node test-paris.js ~/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf
```

## Architecture

- **WASM Module**: Lightweight inference (~50MB)
- **Host Storage**: Models stored in Node.js (~637MB per model)
- **Memory Efficiency**: 98% reduction vs storing in WASM

