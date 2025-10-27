# Realm Python SDK

Official Python SDK for Realm multi-tenant LLM inference runtime.

## Installation

```bash
pip install realm-ai
```

## Usage

### Basic Example

```python
from realm import Realm

realm = Realm(
    model_path="./models/llama-2-7b-chat.gguf",
    max_tokens=100,
    temperature=0.7
)

response = realm.generate(
    prompt="Hello, how are you?",
    max_tokens=50
)

print(response.text)
```

### Streaming

```python
for token in realm.generate_stream(prompt="Write a story about..."):
    print(token, end="", flush=True)
```

### Chat Completions

```python
response = realm.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.message.content)
```

### Async Support

```python
import asyncio
from realm import AsyncRealm

async def main():
    realm = AsyncRealm(model_path="./model.gguf")

    response = await realm.generate(prompt="Hello!")
    print(response.text)

    # Streaming
    async for token in realm.generate_stream(prompt="Tell me a joke"):
        print(token, end="", flush=True)

asyncio.run(main())
```

## API Reference

### `Realm`

Main class for interacting with Realm runtime.

#### Constructor

```python
Realm(
    model_path: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    gpu: bool = False
)
```

#### Methods

##### `generate(prompt: str, **kwargs) -> GenerateResponse`

Generate text completion.

**Arguments:**
- `prompt` (str): Input prompt
- `max_tokens` (int, optional): Override default
- `temperature` (float, optional): Override default
- `top_k` (int, optional): Override default
- `top_p` (float, optional): Override default

**Returns:**
- `GenerateResponse` with attributes:
  - `text` (str): Generated text
  - `tokens` (int): Number of tokens
  - `finish_reason` (str): Why generation stopped

##### `generate_stream(prompt: str, **kwargs) -> Iterator[str]`

Generate text with streaming.

##### `chat(messages: List[Dict], **kwargs) -> ChatResponse`

Chat completion (OpenAI-compatible).

##### `load_model(model_path: str) -> None`

Load a different model.

##### `unload() -> None`

Unload the current model.

### `AsyncRealm`

Async version of `Realm` class.

## Advanced Usage

### FastAPI Integration

```python
from fastapi import FastAPI
from realm import Realm
from pydantic import BaseModel

app = FastAPI()
realm = Realm(model_path="./model.gguf")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
async def generate(req: GenerateRequest):
    response = realm.generate(
        prompt=req.prompt,
        max_tokens=req.max_tokens
    )
    return {"text": response.text}

# Run with: uvicorn main:app --reload
```

### Context Manager

```python
with Realm(model_path="./model.gguf") as realm:
    response = realm.generate("Hello!")
    print(response.text)
# Model automatically unloaded
```

## Implementation Status

⚠️ **Note**: This SDK is under active development. Current status:

- [ ] Core API design
- [ ] PyO3 bindings to Rust runtime
- [ ] Sync API
- [ ] Async API
- [ ] Streaming support
- [ ] Chat completions
- [ ] Type hints
- [ ] Unit tests
- [ ] Integration tests
- [ ] Documentation

## Building from Source

```bash
# Install Rust and maturin
pip install maturin

# Build and install
cd sdks/python
maturin develop
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

Dual licensed under MIT OR Apache-2.0, at your option.
