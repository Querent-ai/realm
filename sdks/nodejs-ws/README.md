# Realm WebSocket Client SDK (Node.js/TypeScript)

Official WebSocket client SDK for connecting to Realm.ai multi-tenant LLM inference server.

## Installation

```bash
npm install @realm-ai/ws-client
```

## Quick Start

```typescript
import { RealmWebSocketClient } from "@realm-ai/ws-client";

const client = new RealmWebSocketClient({
  url: "ws://localhost:8080",
  apiKey: "your-api-key", // Optional
  tenantId: "my-tenant",  // Optional
});

// Connect
await client.connect();

// Generate text
const result = await client.generate({
  prompt: "What is the capital of France?",
  max_tokens: 50,
});

console.log(result.text);

// Disconnect
client.disconnect();
```

## Features

- ✅ **Type-safe API** - Full TypeScript support
- ✅ **Authentication** - API key support
- ✅ **Multi-tenant** - Tenant ID isolation
- ✅ **Streaming** - Real-time token streaming
- ✅ **Reconnection** - Automatic reconnection on disconnect
- ✅ **Error handling** - Comprehensive error handling
- ✅ **Rate limiting** - Handles rate limit errors gracefully

## API Reference

### Constructor

```typescript
new RealmWebSocketClient(options?: {
  url?: string;                    // WebSocket URL (default: "ws://localhost:8080")
  apiKey?: string;                 // API key for authentication
  tenantId?: string;               // Tenant ID for multi-tenancy
  reconnect?: boolean;              // Auto-reconnect (default: true)
  reconnectInterval?: number;       // Reconnect interval in ms (default: 5000)
  timeout?: number;                // Request timeout in ms (default: 30000)
})
```

### Methods

#### `connect(): Promise<void>`

Connect to the WebSocket server. Automatically authenticates if API key is provided.

#### `generate(options: GenerationOptions): Promise<GenerationResult>`

Generate text from a prompt.

```typescript
const result = await client.generate({
  prompt: "Hello, world!",
  max_tokens: 100,
  temperature: 0.7,
  stream: false, // Set to true for streaming
});
```

#### `executePipeline(pipelineName: string, input: PipelineInput): Promise<any>`

Execute a multi-model pipeline.

```typescript
const result = await client.executePipeline("rag-pipeline", {
  query: "What is machine learning?",
  context: "Machine learning is...",
});
```

#### `health(): Promise<HealthStatus>`

Check server health status.

#### `metadata(): Promise<RuntimeMetadata>`

Get runtime metadata (available functions, parameters, etc.).

#### `disconnect(): void`

Disconnect from the server.

#### `isConnected(): boolean`

Check if connected to the server.

### Events

```typescript
// Listen for connection events
client.on("connected", () => {
  console.log("Connected!");
});

client.on("disconnected", () => {
  console.log("Disconnected!");
});

client.on("error", (error) => {
  console.error("Error:", error);
});

// Remove listener
client.off("connected", callback);
```

## Error Handling

The SDK handles various error types:

```typescript
try {
  await client.generate({ prompt: "Hello" });
} catch (error) {
  if (error.code === "RATE_LIMIT_EXCEEDED") {
    console.log("Rate limited. Retry after:", error.retryAfter);
  } else if (error.code === "UNAUTHORIZED") {
    console.log("Authentication failed");
  } else {
    console.error("Error:", error.message);
  }
}
```

## Examples

See `src/examples/` for complete examples:

- `basic.ts` - Basic usage (connect, generate, pipeline)

## License

MIT OR Apache-2.0

