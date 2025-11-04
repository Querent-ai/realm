# Paris Generation - Server Example

This example demonstrates running a **Realm WebSocket server** and connecting with clients:

- **Question**: "What is the capital of France?"
- **Expected Answer**: "Paris"

## What This Shows

- ✅ Realm WebSocket server setup
- ✅ Model loading on server
- ✅ Client connections (Node.js and Python examples)
- ✅ End-to-end server + client flow

## Structure

```
server/
├── start-server.sh       # Server startup script
├── client-nodejs.js      # Node.js client example
├── client-python.py      # Python client example
└── README.md             # This file
```

## Run Server

```bash
cd examples/paris/server

# Option 1: Use the script
./start-server.sh

# Option 2: Manual
cargo run --release --bin realm-server -- \
    --host 127.0.0.1 \
    --port 8080 \
    --model /path/to/model.gguf
```

## Run Clients

### Node.js Client

```bash
# In another terminal
cd examples/paris/nodejs-sdk
node index.js
```

### Python Client

```bash
# In another terminal
cd examples/paris/python-sdk
python main.py
```

## Expected Flow

1. **Server starts** and loads model
2. **Client connects** via WebSocket
3. **Client sends** "What is the capital of France?"
4. **Server generates** "Paris"
5. **Client receives** response

## Architecture

```
┌─────────────────────────┐
│  Client (Node.js/Python)│
└──────────┬──────────────┘
           │ WebSocket
┌──────────▼──────────────┐
│  Realm Server           │
│  (WebSocket + WASM)     │
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│  realm-runtime          │
│  (GPU/CPU inference)   │
└─────────────────────────┘
```

## Notes

- Server handles **multiple clients** (multi-tenant)
- Each client gets **isolated WASM runtime**
- **GPU acceleration** if available
- **Authentication** via API keys (optional)

