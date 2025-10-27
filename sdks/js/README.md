# Realm JavaScript/TypeScript SDK

Official JavaScript/TypeScript SDK for Realm multi-tenant LLM inference runtime.

## Installation

```bash
npm install @realm-ai/sdk
```

## Usage

### Basic Example

```typescript
import { Realm } from '@realm-ai/sdk';

const realm = new Realm({
  modelPath: './models/llama-2-7b-chat.gguf',
  maxTokens: 100,
  temperature: 0.7
});

const response = await realm.generate({
  prompt: 'Hello, how are you?',
  maxTokens: 50
});

console.log(response.text);
```

### Streaming

```typescript
const stream = realm.generateStream({
  prompt: 'Write a story about...'
});

for await (const token of stream) {
  process.stdout.write(token);
}
```

### Chat Completions

```typescript
const response = await realm.chat({
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is the capital of France?' }
  ]
});

console.log(response.message.content);
```

## API Reference

### `Realm`

Main class for interacting with Realm runtime.

#### Constructor

```typescript
new Realm(options: RealmOptions)
```

**Options:**
- `modelPath: string` - Path to GGUF model file
- `maxTokens?: number` - Maximum tokens to generate (default: 100)
- `temperature?: number` - Sampling temperature (default: 0.7)
- `topK?: number` - Top-k sampling (default: 50)
- `topP?: number` - Top-p (nucleus) sampling (default: 0.9)
- `gpu?: boolean` - Use GPU acceleration (default: false)

#### Methods

##### `generate(options: GenerateOptions): Promise<GenerateResponse>`

Generate text completion.

**Options:**
- `prompt: string` - Input prompt
- `maxTokens?: number` - Override default maxTokens
- `temperature?: number` - Override default temperature
- `topK?: number` - Override default topK
- `topP?: number` - Override default topP

**Returns:**
- `text: string` - Generated text
- `tokens: number` - Number of tokens generated
- `finishReason: string` - Why generation stopped

##### `generateStream(options: GenerateOptions): AsyncIterable<string>`

Generate text with streaming.

##### `chat(options: ChatOptions): Promise<ChatResponse>`

Chat completion (OpenAI-compatible).

##### `loadModel(modelPath: string): Promise<void>`

Load a different model.

##### `unload(): Promise<void>`

Unload the current model and free resources.

## Advanced Usage

### Multi-Tenant Deployment

```typescript
import { RealmServer } from '@realm-ai/sdk/server';

const server = new RealmServer({
  port: 8080,
  modelPath: './models/model.gguf',
  maxTenants: 8
});

server.listen();
```

### Using with Express

```typescript
import express from 'express';
import { Realm } from '@realm-ai/sdk';

const app = express();
const realm = new Realm({ modelPath: './model.gguf' });

app.post('/generate', async (req, res) => {
  const { prompt } = req.body;
  const response = await realm.generate({ prompt });
  res.json(response);
});

app.listen(3000);
```

## Implementation Status

⚠️ **Note**: This SDK is under active development. Current status:

- [ ] Core API design
- [ ] N-API bindings to Rust runtime
- [ ] Streaming support
- [ ] Chat completions
- [ ] Multi-tenant server
- [ ] TypeScript definitions
- [ ] Unit tests
- [ ] Integration tests
- [ ] Documentation

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

Dual licensed under MIT OR Apache-2.0, at your option.
