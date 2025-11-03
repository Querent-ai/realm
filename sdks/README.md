# Realm.ai SDKs

Official SDKs for Realm multi-tenant LLM inference runtime.

## JavaScript/TypeScript SDK

✅ **Production-Ready** - WASM-based architecture

```bash
cd js
npm install
npm run build
```

**Features:**
- WASM wrapper
- Model registry (multiple models)
- Full TypeScript support
- Examples included

See [js/README.md](js/README.md) for details.

## Python SDK

✅ **HTTP Client Ready** - Works with server mode

```bash
cd python
pip install -e .
```

**Features:**
- HTTP client for server mode
- Error handling
- Retry logic
- Examples included

See [python/README.md](python/README.md) for details.

## Status

- ✅ JavaScript SDK: Complete, ready for testing
- ✅ Python SDK: HTTP client complete, ready when server exists

Both SDKs follow Realm's WASM-based multi-tenant architecture!
