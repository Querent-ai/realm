# Realm.ai SDKs

Official SDKs for Realm multi-tenant LLM inference runtime.

## WebSocket Client SDKs (Recommended)

### Node.js/TypeScript WebSocket Client ✅

**Status**: ✅ **Production-Ready**

```bash
cd nodejs-ws
npm install
npm run build
```

**Features:**
- ✅ Full TypeScript support
- ✅ WebSocket connection to Realm server
- ✅ API key authentication
- ✅ Multi-tenant support
- ✅ Streaming support
- ✅ Automatic reconnection

**Installation:**
```bash
npm install @realm-ai/ws-client
```

See [nodejs-ws/README.md](nodejs-ws/README.md) for details.

---

### Python WebSocket Client ✅

**Status**: ✅ **Production-Ready**

```bash
cd python-ws
pip install -e .
```

**Features:**
- ✅ Full async/await support
- ✅ WebSocket connection to Realm server
- ✅ API key authentication
- ✅ Multi-tenant support
- ✅ Streaming support
- ✅ Automatic reconnection

**Installation:**
```bash
pip install realm-ws-client
```

See [python-ws/README.md](python-ws/README.md) for details.

---

## Legacy SDKs

### JavaScript/TypeScript SDK (WASM)

**Status**: ✅ **Local WASM Mode**

```bash
cd js
npm install
npm run build
```

**Features:**
- WASM wrapper for local inference
- Model registry (multiple models)
- Full TypeScript support

**Note**: For server mode, use the WebSocket client SDKs above.

See [js/README.md](js/README.md) for details.

---

### Python SDK (HTTP)

**Status**: ⚠️ **HTTP Only (Server uses WebSocket)**

```bash
cd python
pip install -e .
```

**Features:**
- HTTP client (legacy)
- Error handling
- Retry logic

**Note**: For server mode, use the WebSocket client SDK above.

See [python/README.md](python/README.md) for details.

---

## Quick Start

### Node.js/TypeScript

```typescript
import { RealmWebSocketClient } from "@realm-ai/ws-client";

const client = new RealmWebSocketClient({
  url: "ws://localhost:8080",
  apiKey: "your-api-key",
});

await client.connect();

const result = await client.generate({
  prompt: "Hello, world!",
  max_tokens: 50,
});

console.log(result.text);
```

### Python

```python
import asyncio
from realm import RealmWebSocketClient

async def main():
    client = RealmWebSocketClient(url="ws://localhost:8080")
    await client.connect()
    
    result = await client.generate({
        "prompt": "Hello, world!",
        "max_tokens": 50,
    })
    
    print(result["text"])
    await client.disconnect()

asyncio.run(main())
```

---

## Status Summary

| SDK | Type | Status | Use Case |
|-----|------|--------|----------|
| **Node.js WebSocket** | WebSocket Client | ✅ Ready | Production server |
| **Python WebSocket** | WebSocket Client | ✅ Ready | Production server |
| JavaScript/TypeScript | WASM | ✅ Ready | Local inference |
| Python HTTP | HTTP Client | ⚠️ Legacy | Not recommended |

---

## Recommendation

**For production server usage**: Use the WebSocket client SDKs (Node.js or Python).

**For local WASM inference**: Use the JavaScript/TypeScript SDK.

---

## License

MIT OR Apache-2.0
