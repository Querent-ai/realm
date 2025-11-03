# Realm.ai Python SDK

Official Python SDK for Realm multi-tenant LLM inference runtime.

**Status**: Planning phase - Architecture decision needed

## Architecture Options

### Option 1: HTTP Client (Recommended to start)
- Simplest implementation
- Works cross-platform
- Can connect to `realm-runtime server`
- No native dependencies

### Option 2: PyO3 Bindings
- Direct access to Rust runtime
- Best performance
- Requires Rust toolchain
- Native dependencies

### Option 3: WASM via wasmer-python
- Same WASM as JavaScript SDK
- Cross-platform
- Some overhead

## Planned API

```python
from realm import Realm

# Initialize
realm = Realm(
    mode='local',           # or 'server'
    default_model='llama-7b'
)

# Load model
realm.load_model(model_bytes, model_id='llama-7b')

# Generate
response = realm.generate(
    'What is the capital of France?',
    model='llama-7b',  # Optional, uses default_model
    max_tokens=50,
    temperature=0.7
)

# Model registry
models = realm.get_models()
realm.use_model('llama-7b')
```

## Implementation Status

⚠️ **Planning phase** - Need to decide on architecture:
- [ ] Choose implementation approach (HTTP client vs PyO3 vs wasmer)
- [ ] Implement Realm class
- [ ] Model registry support
- [ ] Type hints
- [ ] Examples
- [ ] Tests

## Next Steps

1. Decide on implementation approach
2. Implement based on chosen approach
3. Test with existing WASM module or HTTP server
