# Realm WebSocket Client SDK (Python)

Official WebSocket client SDK for connecting to Realm.ai multi-tenant LLM inference server.

## Installation

```bash
pip install realm-ws-client
```

Or from source:

```bash
cd sdks/python-ws
pip install -e .
```

## Quick Start

```python
import asyncio
from realm import RealmWebSocketClient

async def main():
    client = RealmWebSocketClient(
        url="ws://localhost:8080",
        api_key="your-api-key",  # Optional
        tenant_id="my-tenant",    # Optional
    )

    # Connect
    await client.connect()

    # Generate text
    result = await client.generate({
        "prompt": "What is the capital of France?",
        "max_tokens": 50,
    })

    print(result["text"])

    # Disconnect
    await client.disconnect()

asyncio.run(main())
```

## Features

- ✅ **Type-safe API** - Full type hints
- ✅ **Async support** - Built on asyncio
- ✅ **Authentication** - API key support
- ✅ **Multi-tenant** - Tenant ID isolation
- ✅ **Streaming** - Real-time token streaming
- ✅ **Reconnection** - Automatic reconnection on disconnect
- ✅ **Error handling** - Comprehensive error handling
- ✅ **Rate limiting** - Handles rate limit errors gracefully

## API Reference

### Constructor

```python
RealmWebSocketClient(
    url: str = "ws://localhost:8080",
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None,
    reconnect: bool = True,
    reconnect_interval: float = 5.0,
    timeout: float = 30.0,
)
```

### Methods

#### `async connect() -> None`

Connect to the WebSocket server. Automatically authenticates if API key is provided.

#### `async generate(options: GenerationOptions) -> GenerationResult`

Generate text from a prompt.

```python
result = await client.generate({
    "prompt": "Hello, world!",
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": False,  # Set to True for streaming
})
```

#### `async execute_pipeline(pipeline_name: str, input_data: PipelineInput) -> Any`

Execute a multi-model pipeline.

```python
result = await client.execute_pipeline("rag-pipeline", {
    "query": "What is machine learning?",
    "context": "Machine learning is...",
})
```

#### `async health() -> Dict[str, Any]`

Check server health status.

#### `async metadata() -> Dict[str, Any]`

Get runtime metadata (available functions, parameters, etc.).

#### `async disconnect() -> None`

Disconnect from the server.

#### `is_connected() -> bool`

Check if connected to the server.

### Events

```python
# Listen for connection events
client.on("connected", lambda: print("Connected!"))
client.on("disconnected", lambda: print("Disconnected!"))
client.on("error", lambda e: print(f"Error: {e}"))

# Remove listener
client.off("connected", callback)
```

## Error Handling

The SDK handles various error types:

```python
try:
    await client.generate({"prompt": "Hello"})
except Exception as e:
    if hasattr(e, "code") and e.code == "RATE_LIMIT_EXCEEDED":
        print(f"Rate limited. Retry after: {e.retry_after}")
    elif hasattr(e, "code") and e.code == "UNAUTHORIZED":
        print("Authentication failed")
    else:
        print(f"Error: {e}")
```

## Examples

See `examples/` directory for complete examples:

- `basic.py` - Basic usage (connect, generate, pipeline)

## License

MIT OR Apache-2.0

