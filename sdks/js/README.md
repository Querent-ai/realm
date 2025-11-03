# Realm.ai JavaScript/TypeScript SDK

Official JavaScript/TypeScript SDK for Realm multi-tenant LLM inference runtime.

**Uses WASM-based architecture** - Each tenant gets isolated WASM execution with shared GPU compute.

## Features

✅ **WASM-based inference** - Direct WASM bindings for local inference  
✅ **Model Registry** - Multiple models, switch between them  
✅ **TypeScript Support** - Complete type definitions  
✅ **HOST-side Storage** - Models stored in native memory, not WASM  
✅ **Shared GPU** - Multiple models share GPU resources  

## Installation

```bash
npm install @realm-ai/sdk
```

## Quick Start

### Basic Usage

```typescript
import { Realm } from '@realm-ai/sdk';
import * as fs from 'fs';

// Initialize Realm
const realm = new Realm({
  mode: 'local',
  defaultModel: 'llama-7b',  // Optional: default model
});

// Load model from GGUF file
const modelBytes = fs.readFileSync('./models/llama-2-7b.gguf');
await realm.loadModel(modelBytes, 'llama-7b');

// Generate text
const response = await realm.generate('What is the capital of France?', {
  maxTokens: 50,
  temperature: 0.7,
});

console.log(response.text); // "The capital of France is Paris."
console.log(response.model); // "llama-7b"
```

### Multiple Models

```typescript
// Load multiple models
await realm.loadModel(fs.readFileSync('./models/llama-7b.gguf'), 'llama-7b');
await realm.loadModel(fs.readFileSync('./models/llama-13b.gguf'), 'llama-13b');

// Switch between models
realm.useModel('llama-7b');
const response1 = await realm.generate('Hello!');

realm.useModel('llama-13b');
const response2 = await realm.generate('Hello!');

// Or specify model per request
const response3 = await realm.generate('Hello!', {
  model: 'llama-7b',  // Use specific model
  maxTokens: 100,
});
```

### Model Registry

```typescript
// List available models
const models = realm.getModels();
console.log(models);
// [
//   { id: 'llama-7b', name: 'llama-7b', loaded: true },
//   { id: 'llama-13b', name: 'llama-13b', loaded: true }
// ]

// Check if model is loaded
if (realm.isModelLoaded('llama-7b')) {
  realm.useModel('llama-7b');
}
```

### Configuration

```typescript
// Set generation config
realm.setConfig({
  maxTokens: 200,
  temperature: 0.7,
  topK: 50,
  topP: 0.9,
  repetitionPenalty: 1.1,
});

// Or per-request
const response = await realm.generate('Hello!', {
  maxTokens: 100,
  temperature: 0.8,
});
```

## API Reference

### `Realm`

Main class for WASM-based inference.

#### Constructor

```typescript
new Realm(options?: RealmOptions)
```

**Options:**
- `mode?: 'local' | 'server'` - SDK mode (default: `'local'`)
- `defaultModel?: string` - Default model ID to use
- `wasmPath?: string` - Path to WASM module (optional)
- `endpoint?: string` - Server endpoint (for 'server' mode, future)

#### Methods

##### `init(wasmBytes?: Uint8Array): Promise<void>`

Initialize WASM module. Called automatically, but can be called explicitly.

##### `loadModel(modelBytes: Uint8Array, modelId?: string): Promise<string>`

Load a model from GGUF bytes. Returns model ID.

##### `useModel(modelId: string): void`

Switch to a model from the registry.

##### `generate(prompt: string, options?: GenerationConfig): Promise<GenerationResponse>`

Generate text from a prompt.

**Options:**
- `model?: string` - Model to use (overrides defaultModel)
- `maxTokens?: number` - Maximum tokens
- `temperature?: number` - Temperature (0.0-2.0)
- `topK?: number` - Top-k sampling
- `topP?: number` - Top-p sampling
- `repetitionPenalty?: number` - Repetition penalty

##### `getModels(): ModelInfo[]`

Get list of models in registry.

##### `getCurrentModel(): string | null`

Get current model ID.

##### `isModelLoaded(modelId?: string): boolean`

Check if a model is loaded.

##### `setConfig(config: GenerationConfig): void`

Set generation configuration.

##### `vocabSize(): number`

Get model vocabulary size.

##### `getModelConfig(): any`

Get model configuration as JSON.

##### `dispose(): void`

Free resources and cleanup.

## Architecture

Realm uses **WASM sandboxing** with **HOST-side storage**:

```
JavaScript → WASM Module → Host Functions → GPU/Memory64
                              ↓
                         Shared Resources
```

- **Models** stored in HOST memory (Memory64)
- **WASM** handles orchestration (tokenization, sampling)
- **GPU** shared across all WASM instances
- **Multiple models** can be loaded simultaneously

## Requirements

- Node.js >= 18.0.0
- WASM bindings from `realm-wasm/pkg/` (included in package)

## License

MIT OR Apache-2.0
