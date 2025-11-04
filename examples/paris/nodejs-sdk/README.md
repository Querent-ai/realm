# Paris Generation - Node.js WebSocket SDK

This example demonstrates using Realm's **Node.js WebSocket SDK** to generate "Paris":

- **Question**: "What is the capital of France?"
- **Expected Answer**: "Paris"

## What This Shows

- ✅ Node.js WebSocket client
- ✅ Connecting to Realm server
- ✅ Text generation via WebSocket
- ✅ Server-based inference

## Prerequisites

1. **Realm server must be running** (see `examples/paris/server/`)
2. **Model must be loaded** on the server

## Setup

```bash
cd examples/paris/nodejs-sdk
npm install
```

## Run

```bash
# Start server first (in another terminal)
cd ../../paris/server
cargo run --release

# Then run this example
cd ../../paris/nodejs-sdk
node index.js
```

## Environment Variables

```bash
export REALM_URL="ws://localhost:8080"          # Server URL
export REALM_API_KEY="your-api-key"            # Optional
export REALM_MODEL="tinyllama-1.1b.Q4_K_M.gguf" # Model name
export REALM_TENANT_ID="my-tenant"             # Optional (auto-assigned)
```

## Expected Output

```
🚀 Realm Paris Generation - Node.js SDK

📡 Connecting to Realm server...
✅ Connected!

🏥 Checking server health...
   Status: healthy

🎯 Generating response to: "What is the capital of France?"
   (Expected: "Paris")

✅ Generation complete!

📝 Response: The capital of France is Paris.

✅ SUCCESS: Model correctly identified Paris as the capital of France!

📊 Statistics:
   Input tokens: 15
   Output tokens: 6
   Model: tinyllama-1.1b.Q4_K_M.gguf
   Tenant ID: abc123-def456-...

👋 Disconnected
```

## Architecture

```
┌─────────────────────────┐
│  Node.js Client        │
│  (this example)        │
└──────────┬──────────────┘
           │ WebSocket
┌──────────▼──────────────┐
│  Realm Server           │
│  (WebSocket server)     │
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│  realm-runtime          │
│  (WASM + GPU/CPU)      │
└─────────────────────────┘
```

## Notes

- Requires **running server** (unlike native or WASM examples)
- Uses **WebSocket** for real-time streaming
- Supports **authentication** via API keys
- **Multi-tenant** ready (per-tenant isolation)

